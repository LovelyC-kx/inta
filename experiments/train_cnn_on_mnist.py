"""
Train the CNN feature extractor on MNIST (standard 10-class classification).

Without this step, EncodingPipeline uses a CNN with **random weights**, which
means the (8, 3, 3) feature maps it produces are essentially random projections
of the image — not real image features. Encoders then turn these random
features into stim pulses, so the resulting stim has no meaningful relationship
to the image content.

After training:
  - CNN reaches ~98% val acc on MNIST
  - Feature maps are real image-derived features
  - Encoders produce stim that actually encodes the image
  - Downstream decoder has a chance to learn

Usage:
  python experiments/train_cnn_on_mnist.py                     # default 10 epochs
  python experiments/train_cnn_on_mnist.py --n-epochs 20       # longer training
  python experiments/train_cnn_on_mnist.py --device cuda       # GPU

Output:
  checkpoints/cnn_mnist_<timestamp>.pt
  experiment_log.jsonl entry
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.data as torch_data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.cnn_feature_extractor import CNNFeatureExtractor, train_feature_extractor
from utils.experiment_log import log_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-channels", type=int, default=8,
                        help="Number of output feature map channels (= number of stim electrodes)")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Max training epochs (default 10; early-stops at convergence)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Wang et al. used 5e-4 with Adam")
    parser.add_argument("--mnist-root", default="./data/mnist",
                        help="Where to download/cache MNIST")
    parser.add_argument("--out", default=None,
                        help="Output .pt path. Default: checkpoints/cnn_mnist_<ts>.pt")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--patience", type=int, default=3,
                        help="Early-stop patience (epochs without val loss improvement)")
    args = parser.parse_args()

    from torchvision import datasets, transforms

    print(f"Loading MNIST from {args.mnist_root} (downloads if needed)...")
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root=args.mnist_root, train=True,
                               download=True, transform=transform)
    val_set = datasets.MNIST(root=args.mnist_root, train=False,
                             download=True, transform=transform)

    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)
    val_loader = torch_data.DataLoader(val_set, batch_size=args.batch_size,
                                        shuffle=False, num_workers=0)

    print(f"  Train: {len(train_set)} images, Val: {len(val_set)} images")
    print(f"  n_channels={args.n_channels}, n_epochs={args.n_epochs}, "
          f"lr={args.lr}, batch_size={args.batch_size}, device={args.device}\n")

    print("Training...")
    model, history = train_feature_extractor(
        train_loader, val_loader,
        n_channels=args.n_channels,
        n_classes=10,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=args.device,
        early_stopping_patience=args.patience,
    )

    final_train_acc = history["train_acc"][-1]
    final_val_acc = history["val_acc"][-1]
    n_epochs_actual = len(history["train_acc"])

    print()
    print("Per-epoch summary:")
    print(f"  {'epoch':>6} {'train_loss':>11} {'val_loss':>10} {'train_acc':>11} {'val_acc':>9}")
    for i in range(n_epochs_actual):
        print(f"  {i+1:>6} {history['train_loss'][i]:>11.4f} "
              f"{history['val_loss'][i]:>10.4f} {history['train_acc'][i]:>11.4f} "
              f"{history['val_acc'][i]:>9.4f}")

    out_path = args.out
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"checkpoints/cnn_mnist_{ts}.pt"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": model.state_dict(),
        "n_channels": args.n_channels,
        "n_classes": 10,
        "input_size": 28,
        "history": history,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "n_epochs_actual": n_epochs_actual,
        "trained_at": datetime.now().isoformat(),
    }, out_path)

    print(f"\nFinal train acc: {final_train_acc:.2%}")
    print(f"Final val acc:   {final_val_acc:.2%}")
    print(f"Saved: {out_path}")

    log_experiment(
        script="train_cnn_on_mnist.py",
        args=vars(args),
        verdict="PASS" if final_val_acc > 0.90 else "WARN (val acc < 90%)",
        summary={
            "n_channels": int(args.n_channels),
            "n_epochs_actual": int(n_epochs_actual),
            "final_train_acc": round(float(final_train_acc), 4),
            "final_val_acc": round(float(final_val_acc), 4),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
        },
        data=str(out_path),
    )


if __name__ == "__main__":
    main()
