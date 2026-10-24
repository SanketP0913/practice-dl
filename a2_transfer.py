# A2 - transfer learning with a pretrained ResNet18
#
# In A1 my CNN from scratch got 88% on CIFAR-10. Now i want to see what happens
# if i start from a network that already learned from ~1.2 million ImageNet
# photos, instead of starting from random weights.
#
# What i try:
#   1. frozen    - keep the resnet exactly as it is and only train a new last layer
#                  on top of its features
#   2. finetune  - train the whole resnet a little bit more on cifar
#   3. small data - pretend i only have 2,000 labelled photos. my cnn from scratch
#                  vs finetuned resnet. i think this is where pretraining helps most
# Baseline is my A1 cnn (it loads a1_cnn.pt and tests it again).
#
# resnet was trained on big images (224x224) so i resize cifar from 32x32 to 128x128.
# first run downloads the resnet18 weights (~45 MB).
#
# needs a1_cnn_cifar.py and a1_cnn.pt in the same folder, i reuse stuff from A1.
# python a2_transfer.py --quick   (small run just to see if it works)
# python a2_transfer.py           (full run, probably 10-15 min on a mac)

import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from a1_cnn_cifar import augment, cnn_model, load_cifar, pick_device

# the colour mean/std of imagenet, resnet expects inputs normalised with these
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def make_prep(mean, std, size, device):
    # A1 normalised with cifar mean/std, so undo that, redo it the imagenet way,
    # then make the image bigger
    m, s = mean.to(device), std.to(device)
    im, ist = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    def prep(x):
        x = (x * s + m - im) / ist
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return prep


def pretrained_resnet():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)   # swap the 1000 imagenet classes for my 10
    return model


@torch.no_grad()
def predict(model, x, prep=None, bs=250):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        if prep:
            xb = prep(xb)
        out.append(model(xb).argmax(1))
    return torch.cat(out)


def acc(preds, y):
    return (preds == y).float().mean().item()


def train(name, model, x, y, xv, yv, epochs, lr, prep=None, use_aug=False, bs=128):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)
    steps = epochs * ((len(x) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    print(f"\n--- {name}: {sum(p.numel() for p in params):,} trainable weights, "
          f"{len(x):,} images, {epochs} epochs")
    best, best_state = 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(x), device=x.device)
        correct = 0
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            xb, yb = x[idx], y[idx]
            if use_aug:
                xb = augment(xb)
            if prep:
                xb = prep(xb)
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            correct += (out.argmax(1) == yb).sum().item()
        val = acc(predict(model, xv, prep), yv)
        mark = ""
        if val > best:
            best, best_state, mark = val, copy.deepcopy(model.state_dict()), "  <- best"
        print(f"epoch {ep:2d}  train {correct / len(x):.1%}  val {val:.1%}  ({time.time() - t0:.0f}s){mark}")
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def get_features(backbone, x, prep, bs=250):
    backbone.eval()
    return torch.cat([backbone(prep(x[i:i + bs])) for i in range(0, len(x), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run to check it works")
    ap.add_argument("--size", type=int, default=128, help="resize cifar images to this")
    ap.add_argument("--small", type=int, default=2000, help="how many images in the small data test")
    args = ap.parse_args()
    q = args.quick

    device = pick_device()
    print(f"device: {device}")
    (x, y), (xv, yv), (x_te, y_te), (mean, std) = load_cifar(device, q)
    prep = make_prep(mean, std, args.size, device)
    print(f"train {len(x):,}  val {len(xv):,}  test {len(x_te):,} images, resized to {args.size}x{args.size}")
    results = {}

    # baseline: my cnn from A1, trained on all 45k
    if os.path.exists("a1_cnn.pt"):
        cnn = cnn_model().to(device)
        cnn.load_state_dict(torch.load("a1_cnn.pt", map_location=device))
        results["A1 cnn from scratch"] = acc(predict(cnn, x_te), y_te)
        print(f"\nA1 cnn on the test set: {results['A1 cnn from scratch']:.1%}")
    else:
        print("\n(no a1_cnn.pt found, run a1_cnn_cifar.py first if you want the A1 baseline)")

    # 1. frozen resnet. features don't change so i compute them once, then train
    #    just a linear layer on them (512 numbers per image -> 10 classes)
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.to(device)
    t0 = time.time()
    f_tr, f_va, f_te = (get_features(backbone, a, prep) for a in (x, xv, x_te))
    print(f"\ngot resnet features for all images in {time.time() - t0:.0f}s, shape {tuple(f_tr.shape)}")
    head = train("frozen resnet + new last layer", nn.Linear(512, 10).to(device),
                 f_tr, y, f_va, yv, epochs=3 if q else 20, lr=1e-3, bs=256)
    results["frozen resnet"] = acc(predict(head, f_te), y_te)
    del backbone, f_tr, f_va, f_te

    # 2. finetune everything. lower lr than A1 so i don't wreck the pretrained weights
    model = train("finetune whole resnet", pretrained_resnet().to(device),
                  x, y, xv, yv, epochs=1 if q else 3, lr=5e-4, prep=prep, use_aug=True)
    results["finetuned resnet"] = acc(predict(model, x_te, prep), y_te)
    del model

    # 3. small data: same 2,000 images for both
    n = 500 if q else args.small
    xs, ys = x[:n], y[:n]
    cnn_small = train(f"my cnn from scratch, only {n} images", cnn_model().to(device),
                      xs, ys, xv, yv, epochs=3 if q else 40, lr=3e-3, use_aug=True)
    results[f"cnn from scratch ({n} imgs)"] = acc(predict(cnn_small, x_te), y_te)
    res_small = train(f"finetune resnet, only {n} images", pretrained_resnet().to(device),
                      xs, ys, xv, yv, epochs=1 if q else 10, lr=5e-4, prep=prep, use_aug=True)
    results[f"finetuned resnet ({n} imgs)"] = acc(predict(res_small, x_te, prep), y_te)

    print("\ntest accuracy:")
    print(f"{'just guessing':32s} {0.1:6.1%}")
    for name, a in results.items():
        print(f"{name:32s} {a:6.1%}")

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results)
    bars = ax.barh(names, [results[k] for k in names], color="tab:blue")
    ax.bar_label(bars, labels=[f"{results[k]:.1%}" for k in names], padding=3)
    ax.axvline(0.1, color="gray", ls="--", label="guessing")
    ax.set(xlim=(0, 1.05), xlabel="test accuracy", title="A2 - pretrained resnet vs my cnn from scratch")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("a2_results.png", dpi=120)
    print("\nsaved a2_results.png")


if __name__ == "__main__":
    main()
