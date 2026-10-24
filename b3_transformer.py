# B3 - small pretrained transformer (DistilBERT) on IMDB reviews
#
# B2 showed that learning word meanings from scratch with 22k reviews didn't beat
# my bag of words logreg (90.6%). DistilBERT is a smaller version of BERT that was
# already pretrained on loads of English text (wikipedia + books), so it should know
# what words mean and how "not" changes things before it ever sees a review.
# Same idea as resnet in A2, just for text.
#
# What i compare (all on the same test reviews):
#   B1 logreg          - retrained here on all 25k reviews so the test set is the same
#   sst-2 distilbert   - a distilbert someone else already finetuned for sentiment on
#                        short movie sentences (SST-2). i don't train it at all
#   my finetune        - plain distilbert finetuned by me on only 4,000 imdb reviews
#
# To keep it to ~10-15 min on my mac i only test on 5,000 random test reviews
# (--test-size 25000 for all of them), and reviews get cut at 256 tokens.
# First run downloads the two models (~260 MB each) from hugging face.
#
# first try (256 tokens, 4k reviews) my finetune only got 88.2%, below logreg. so i added
# --max-len and --head-tail to see if cutting off the end of the reviews was the problem:
# python b3_transformer.py --train-size 8000 --max-len 384 --head-tail
#
# needs imdb.csv + b1_bow_imdb.py in the same folder
# pip install transformers
# python b3_transformer.py --quick
# python b3_transformer.py

import argparse
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"   # stops a warning from the tokenizer

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from b1_bow_imdb import load_imdb

BASE = "distilbert/distilbert-base-uncased"
SST2 = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MAX_LEN = 256
HEAD_TAIL = False   # set by --head-tail
torch.manual_seed(0)


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode(tok, texts, device):
    if not HEAD_TAIL:
        # normal way: keep the first MAX_LEN tokens and cut the rest
        return tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
    # head + tail: keep the first 128 tokens and the last ones. people usually say
    # what they actually think at the end of a review, so cutting the end loses that
    keep_tail = MAX_LEN - 128 - 2
    rows = []
    for ids in tok(texts, add_special_tokens=False, verbose=False)["input_ids"]:
        if len(ids) > MAX_LEN - 2:
            ids = ids[:128] + ids[-keep_tail:]
        rows.append([tok.cls_token_id] + ids + [tok.sep_token_id])
    n = max(len(r) for r in rows)
    input_ids = torch.tensor([r + [tok.pad_token_id] * (n - len(r)) for r in rows])
    mask = torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in rows])
    return {"input_ids": input_ids.to(device), "attention_mask": mask.to(device)}


@torch.no_grad()
def predict_proba(model, tok, texts, device, bs=32):
    model.eval()
    if device.type == "mps":
        torch.mps.empty_cache()   # learned this one in B2
    probs = []
    for i in range(0, len(texts), bs):
        enc = encode(tok, texts[i:i + bs], device)
        probs.append(F.softmax(model(**enc).logits, 1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def finetune(model, tok, texts, labels, val_texts, val_labels, device, epochs, bs=16, lr=2e-5):
    # small lr because the weights are already good, i just want to nudge them
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(texts) + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best, best_state = 0, None
    step = 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(len(texts))
        losses = []
        for i in range(0, len(texts), bs):
            idx = perm[i:i + bs]
            enc = encode(tok, [texts[k] for k in idx], device)
            y = torch.tensor(labels[idx]).to(device)
            loss = F.cross_entropy(model(**enc).logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{steps}  loss {np.mean(losses[-50:]):.3f}  ({time.time() - t0:.0f}s)")
        val = ((predict_proba(model, tok, val_texts, device) > 0.5) == val_labels).mean()
        mark = ""
        if val > best:
            # keep the best weights on the cpu so they don't take gpu memory
            best, mark = val, "  <- best"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep}  val {val:.1%}  ({time.time() - t0:.0f}s){mark}")
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run to check it works")
    ap.add_argument("--train-size", type=int, default=4000, help="reviews for my finetune")
    ap.add_argument("--test-size", type=int, default=5000, help="test reviews to score")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256, help="tokens per review (max 512)")
    ap.add_argument("--head-tail", action="store_true", help="keep start + end of long reviews")
    args = ap.parse_args()
    global MAX_LEN, HEAD_TAIL
    MAX_LEN, HEAD_TAIL = args.max_len, args.head_tail
    if args.quick:
        args.train_size, args.test_size, args.epochs = 300, 300, 1

    device = pick_device()
    print(f"device: {device}")
    xtr, ytr, xte, yte = load_imdb()
    rng = np.random.default_rng(0)
    te_i = rng.choice(len(xte), args.test_size, replace=False)
    xte, yte = [xte[k] for k in te_i], yte[te_i]
    print(f"testing everything on the same {len(xte):,} test reviews, "
          f"{MAX_LEN} tokens per review{' (head + tail)' if HEAD_TAIL else ''}")
    results, probs = {}, {}

    # 1. baseline: my B1 model, trained on all 25k
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
    logreg = LogisticRegression(C=5, max_iter=2000).fit(vec.fit_transform(xtr), ytr)
    p = logreg.predict_proba(vec.transform(xte))[:, 1]
    results["B1 logreg (25k reviews)"] = ((p > 0.5) == yte).mean()
    print(f"\nB1 logreg: {results['B1 logreg (25k reviews)']:.1%} ({time.time() - t0:.0f}s)")

    tries = ["this movie was not good", "this movie was not bad at all",
             "i expected it to be boring but it was great", "great actors, terrible script"]
    probs["logreg"] = logreg.predict_proba(vec.transform(tries))[:, 1]

    # 2. distilbert already finetuned on SST-2, no training by me
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(SST2)
    sst2 = AutoModelForSequenceClassification.from_pretrained(SST2).to(device)
    p = predict_proba(sst2, tok, xte, device)
    results["sst-2 distilbert (0 imdb reviews)"] = ((p > 0.5) == yte).mean()
    probs["sst-2"] = predict_proba(sst2, tok, tries, device)
    print(f"sst-2 distilbert, no training: {results['sst-2 distilbert (0 imdb reviews)']:.1%} ({time.time() - t0:.0f}s)")
    del sst2

    # 3. my own finetune of plain distilbert
    tr_i = rng.choice(len(xtr), args.train_size + 500, replace=False)
    texts = [xtr[k] for k in tr_i[:args.train_size]]
    labels = ytr[tr_i[:args.train_size]]
    val_texts = [xtr[k] for k in tr_i[args.train_size:]]
    val_labels = ytr[tr_i[args.train_size:]]
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"\n--- finetuning distilbert ({n / 1e6:.0f}M weights) on {len(texts):,} reviews, "
          f"{args.epochs} epochs, 500 for validation")
    model = finetune(model, tok, texts, labels, val_texts, val_labels, device, args.epochs)
    t0 = time.time()
    p = predict_proba(model, tok, xte, device)
    name = f"my distilbert finetune ({len(texts):,} reviews)"
    results[name] = ((p > 0.5) == yte).mean()
    probs["mine"] = predict_proba(model, tok, tries, device)
    print(f"tested in {time.time() - t0:.0f}s")

    print(f"\ntest accuracy (same {len(xte):,} reviews for all):")
    print(f"  {'B2 avg embeddings / bilstm':40s} ~87%   (from b2, full test set)")
    for k, a in results.items():
        print(f"  {k:40s} {a:6.1%}")

    print("\nmy test sentences, P(positive)            logreg  sst-2  mine")
    for i, s in enumerate(tries):
        print(f"  {s:45s} {probs['logreg'][i]:.2f}   {probs['sst-2'][i]:.2f}   {probs['mine'][i]:.2f}")


if __name__ == "__main__":
    main()
