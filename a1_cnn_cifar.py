# A1 - CNN on CIFAR-10
#
# I wanted to see if a CNN is actually better than just flattening the image
# and throwing it into a normal network. So I train 3 models on the same data:
#   linear  - basically logistic regression on the raw pixels (my baseline)
#   mlp     - a couple of hidden layers, but it still sees the image as one long list
#   cnn     - 3x3 filters that slide over the image
# Random guessing would get 10% since there are 10 classes.
#
# CIFAR-10 = 60k small photos (32x32), 10 classes. torchvision downloads it
# the first time into ./data (around 170 MB).
#
# I split the 50k train images into 45k train + 5k validation. Validation is
# for picking the best epoch, and the 10k test set only gets used at the end.
#
# pip install torch torchvision matplotlib
# python a1_cnn_cifar.py --quick   (small run just to see if it works)
# python a1_cnn_cifar.py           (full run, around 5-10 min on a mac)

import argparse
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import CIFAR10

CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
BATCH = 128
torch.manual_seed(0)
np.random.seed(0)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")   # mac gpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cifar(device, quick):
    # whole dataset fits in memory so i just keep it as tensors, no DataLoader
    train = CIFAR10("data", train=True, download=True)
    test = CIFAR10("data", train=False, download=True)

    def to_tensor(ds):
        x = torch.tensor(ds.data).permute(0, 3, 1, 2).float() / 255   # pytorch wants (N, 3, 32, 32)
        return x, torch.tensor(ds.targets)

    x_tr, y_tr = to_tensor(train)
    x_te, y_te = to_tensor(test)
    mean = x_tr.mean(dim=(0, 2, 3), keepdim=True)    # mean/std per colour channel
    std = x_tr.std(dim=(0, 2, 3), keepdim=True)
    x_tr, x_te = (x_tr - mean) / std, (x_te - mean) / std

    perm = torch.randperm(len(x_tr))
    val_idx, tr_idx = perm[:5000], perm[5000:]
    if quick:
        tr_idx, val_idx = tr_idx[:5000], val_idx[:1000]
        x_te, y_te = x_te[:2000], y_te[:2000]
    to = lambda *t: [a.to(device) for a in t]
    return (to(x_tr[tr_idx], y_tr[tr_idx]), to(x_tr[val_idx], y_tr[val_idx]),
            to(x_te, y_te), (mean, std))


def augment(x):
    # flip half the images and shift each one by up to 4 pixels.
    # this way the cnn sees slightly different pictures every epoch
    b = x.size(0)
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    dy = torch.randint(0, 9, (b,), device=x.device)
    dx = torch.randint(0, 9, (b,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (dy[:, None] + ar)[:, None, :, None]          # B,1,32,1
    cols = (dx[:, None] + ar)[:, None, None, :]          # B,1,1,32
    bi = torch.arange(b, device=x.device)[:, None, None, None]
    ci = torch.arange(3, device=x.device)[None, :, None, None]
    return xp[bi, ci, rows, cols]


def linear_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))


def mlp_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 512), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 10),
    )


def conv_block(c_in, c_out):
    # 2 conv layers then maxpool cuts the size in half
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.MaxPool2d(2),
    )


def cnn_model():
    return nn.Sequential(
        conv_block(3, 32),     # 32x32 -> 16x16
        conv_block(32, 64),    # 16x16 -> 8x8
        conv_block(64, 128),   # 8x8   -> 4x4
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),   # average each channel -> 128 numbers
        nn.Dropout(0.3),
        nn.Linear(128, 10),
    )


@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.cat([model(x[i:i + 1000]).argmax(1) for i in range(0, len(x), 1000)])


def train(name, model, train_data, val_data, epochs, use_aug):
    x, y = train_data
    xv, yv = val_data
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    steps = epochs * ((len(x) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    best_acc, best_state, curve = 0, None, []
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- {name}: {n_params:,} weights, {epochs} epochs, augmentation={'on' if use_aug else 'off'}")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(x), device=x.device)
        correct = 0
        for i in range(0, len(x), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x[idx], y[idx]
            if use_aug:
                xb = augment(xb)
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            correct += (out.argmax(1) == yb).sum().item()
        tr_acc = correct / len(x)
        val_acc = (predict(model, xv) == yv).float().mean().item()
        curve.append(val_acc)
        mark = ""
        # keep the weights from the best epoch on validation
        if val_acc > best_acc:
            best_acc, best_state, mark = val_acc, copy.deepcopy(model.state_dict()), "  <- best"
        print(f"epoch {ep:2d}  train {tr_acc:.1%}  val {val_acc:.1%}  ({time.time() - t0:.0f}s){mark}")
    model.load_state_dict(best_state)
    return model, curve, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run to check it works")
    ap.add_argument("--epochs", type=int, default=15, help="epochs for the CNN")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    train_data, val_data, (x_te, y_te), (mean, std) = load_cifar(device, args.quick)
    print(f"train {len(train_data[0]):,}  val {len(val_data[0]):,}  test {len(x_te):,} images")

    base_ep = 2 if args.quick else 8
    cnn_ep = 2 if args.quick else args.epochs
    runs = [
        ("Linear", linear_model(), base_ep, False),
        ("MLP", mlp_model(), base_ep, False),
        ("CNN", cnn_model(), cnn_ep, True),
    ]
    results, curves = {}, {}
    for name, model, ep, aug in runs:
        model, curve, n_params = train(name, model.to(device), train_data, val_data, ep, aug)
        preds = predict(model, x_te)
        results[name] = (model, preds, (preds == y_te).float().mean().item(), n_params)
        curves[name] = curve

    majority = np.bincount(y_te.cpu().numpy()).max() / len(y_te)
    print("\ntest accuracy:")
    print(f"{'just guessing':26s} {majority:6.1%}")
    for name, (_, _, acc, n_params) in results.items():
        print(f"{name:26s} {acc:6.1%}   ({n_params:,} weights)")

    cnn, cnn_preds, _, _ = results["CNN"]
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_te.cpu().numpy(), cnn_preds.cpu().numpy()):
        cm[t, p] += 1
    print("\ncnn accuracy per class:")
    for i, c in enumerate(CLASSES):
        row = cm[i].copy()
        row[i] = -1
        print(f"  {c:6s} {cm[i, i] / cm[i].sum():6.1%}   mostly confused with {CLASSES[row.argmax()]}")

    torch.save(cnn.state_dict(), "a1_cnn.pt")

    # plot: val curves, confusion matrix, and some pictures it got wrong
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 8, height_ratios=[3, 1])
    ax = fig.add_subplot(grid[0, :4])
    for name, curve in curves.items():
        ax.plot(range(1, len(curve) + 1), curve, marker="o", label=name)
    ax.axhline(0.1, color="gray", ls="--", label="guessing")
    ax.set(title="Validation accuracy per epoch", xlabel="epoch", ylabel="accuracy")
    ax.legend()

    ax = fig.add_subplot(grid[0, 4:])
    ax.imshow(cm, cmap="Blues")
    ax.set(title="CNN confusion matrix (test)", xlabel="predicted", ylabel="true",
           xticks=range(10), yticks=range(10))
    ax.set_xticklabels(CLASSES, rotation=60)
    ax.set_yticklabels(CLASSES)

    wrong = (cnn_preds != y_te).nonzero().flatten()[:8]
    imgs = (x_te[wrong].cpu() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    for k, (i, img) in enumerate(zip(wrong.tolist(), imgs)):
        ax = fig.add_subplot(grid[1, k])
        ax.imshow(img)
        ax.set_title(f"true {CLASSES[y_te[i]]}\nsaid {CLASSES[cnn_preds[i]]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("A1 - CNN vs linear vs MLP on CIFAR-10")
    fig.tight_layout()
    fig.savefig("a1_cnn_results.png", dpi=120)
    print("\nsaved a1_cnn_results.png and a1_cnn.pt")


if __name__ == "__main__":
    main()
