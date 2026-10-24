# B2 - word embeddings + reading words in order (IMDB reviews again)
#
# In B1 bag of words got 90.6%, but it thought "not bad at all" was negative,
# because it has no idea about word order. Here i try neural networks where
# every word gets an embedding, a vector of 100 numbers that the network learns.
# Words used in similar ways should end up with similar vectors.
#
# 2 models:
#   avg embeddings - average all the word vectors in a review, then one linear layer.
#                    still ignores order, kind of a neural bag of words
#   bilstm         - an LSTM reads the review word by word, forwards and backwards,
#                    so "not" can change what comes after it
# Baseline is my B1 logreg (words + pairs) at 90.6%.
#
# At the end i also check which words got similar embeddings, e.g. what is closest to "awful".
#
# needs imdb.csv from b1 (it reuses b1's loading code)
# python b2_embeddings.py --quick   (small run just to see if it works)
# python b2_embeddings.py

import argparse
import copy
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from b1_bow_imdb import clean, load_imdb

VOCAB_SIZE = 30000
MAX_LEN = 300      # reviews average ~230 words, so i cut them at 300
EMB_DIM = 100
BATCH = 64
torch.manual_seed(0)
np.random.seed(0)


def pick_device(name):
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text)


def build_vocab(texts):
    counts = Counter(w for t in texts for w in tokenize(t))
    # 0 is for padding, 1 is for words that aren't in the vocab
    words = [w for w, c in counts.most_common(VOCAB_SIZE - 2) if c >= 3]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(texts, vocab):
    out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in tokenize(t)][:MAX_LEN]
        out[i, :len(ids)] = ids
    return torch.tensor(out)


class AvgEmbeddings(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(EMB_DIM, 2)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()       # ignore the padding
        avg = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(avg))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMB_DIM, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        out, _ = self.lstm(self.drop(self.emb(x)))     # one output per word
        # for each feature take the max over all the words (skipping padding)
        out = out.masked_fill((x == 0).unsqueeze(-1), -1e4)
        return self.fc(self.drop(out.max(1).values))


@torch.no_grad()
def predict_proba(model, x, bs=100):
    # bs used to be 500 but the lstm ran out of gpu memory on the test set,
    # so smaller batches + clearing the mps cache first
    model.eval()
    if x.device.type == "mps":
        torch.mps.empty_cache()
    try:
        return torch.cat([F.softmax(model(x[i:i + bs]), 1)[:, 1] for i in range(0, len(x), bs)])
    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise
        # still out of memory -> do the predictions on the cpu instead, slower but works
        print("  (gpu out of memory, predicting on cpu instead)")
        model_cpu = copy.deepcopy(model).cpu()
        xc = x.cpu()
        p = torch.cat([F.softmax(model_cpu(xc[i:i + bs]), 1)[:, 1] for i in range(0, len(xc), bs)])
        return p.to(x.device)


def train(name, model, x, y, xv, yv, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- {name}: {n_params:,} weights ({VOCAB_SIZE * EMB_DIM:,} of them are the embeddings), {epochs} epochs")
    best, best_state = 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(x), device=x.device)
        correct = 0
        for i in range(0, len(x), BATCH):
            idx = perm[i:i + BATCH]
            out = model(x[idx])
            loss = F.cross_entropy(out, y[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # lstms can blow up without this
            opt.step()
            correct += (out.argmax(1) == y[idx]).sum().item()
        val = ((predict_proba(model, xv) > 0.5).long() == yv).float().mean().item()
        mark = ""
        if val > best:
            best, best_state, mark = val, copy.deepcopy(model.state_dict()), "  <- best"
        print(f"epoch {ep:2d}  train {correct / len(x):.1%}  val {val:.1%}  ({time.time() - t0:.0f}s){mark}")
    model.load_state_dict(best_state)
    return model


def nearest_words(model, vocab, words, k=6):
    emb = F.normalize(model.emb.weight.detach().cpu(), dim=1)
    id2word = {i: w for w, i in vocab.items()}
    print("\nclosest words by embedding (cosine similarity):")
    for w in words:
        if w not in vocab:
            continue
        sims = emb @ emb[vocab[w]]
        top = sims.argsort(descending=True)[1:k + 1].tolist()
        print(f"  {w:10s} -> {', '.join(id2word.get(i, '?') for i in top)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run to check it works")
    ap.add_argument("--device", default=None, help="force cpu or mps")
    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")

    xtr, ytr, xte, yte = load_imdb()
    if args.quick:
        rng = np.random.default_rng(0)
        i, j = rng.choice(len(xtr), 3000, replace=False), rng.choice(len(xte), 2000, replace=False)
        xtr, ytr = [xtr[k] for k in i], ytr[i]
        xte, yte = [xte[k] for k in j], yte[j]

    # hold out 10% of the training reviews for validation
    perm = np.random.permutation(len(xtr))
    n_val = len(xtr) // 10
    val_i, tr_i = perm[:n_val], perm[n_val:]
    vocab = build_vocab([xtr[k] for k in tr_i])
    print(f"vocab: {len(vocab):,} words (+ padding and unknown)")

    x_all = encode(xtr, vocab).to(device)
    y_all = torch.tensor(ytr).to(device)
    tr_i, val_i = torch.tensor(tr_i, device=device), torch.tensor(val_i, device=device)
    x, y = x_all[tr_i], y_all[tr_i]
    xv, yv = x_all[val_i], y_all[val_i]
    x_te, y_te = encode(xte, vocab).to(device), torch.tensor(yte).to(device)
    print(f"train {len(x):,}  val {len(xv):,}  test {len(x_te):,} reviews, each cut/padded to {MAX_LEN} words")

    q = args.quick
    avg = train("avg embeddings", AvgEmbeddings(len(vocab) + 2).to(device), x, y, xv, yv,
                epochs=2 if q else 8, lr=2e-3)
    lstm = train("bilstm", BiLSTM(len(vocab) + 2).to(device), x, y, xv, yv,
                 epochs=1 if q else 5, lr=1e-3)

    print(f"\ntest accuracy ({len(x_te):,} reviews):")
    print(f"  {'B1 logreg words + pairs':26s}  90.6%   (from b1, for comparison)")
    for name, m in [("avg embeddings", avg), ("bilstm", lstm)]:
        a = ((predict_proba(m, x_te) > 0.5).long() == y_te).float().mean().item()
        print(f"  {name:26s} {a:6.1%}")

    nearest_words(avg, vocab, ["awful", "great", "boring", "funny", "excellent", "worst", "actor"])

    tries = ["this movie was not good", "this movie was not bad at all",
             "i expected it to be boring but it was great", "great actors, terrible script"]
    xs = encode([clean(s) for s in tries], vocab).to(device)
    p_avg, p_lstm = predict_proba(avg, xs).tolist(), predict_proba(lstm, xs).tolist()
    print("\nmy test sentences, P(positive)   avg   bilstm")
    for s, a, b in zip(tries, p_avg, p_lstm):
        print(f"  {s:45s} {a:.2f}  {b:.2f}")


if __name__ == "__main__":
    main()
