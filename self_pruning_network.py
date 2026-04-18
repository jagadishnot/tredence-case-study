
import math
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 0.  Global config
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPARSE_THRESH = 0.5   # gate below this → weight is "pruned"

print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# 1.  PrunableLinear Layer
# ---------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """
    Custom linear layer with per-weight learnable gates.

    Each weight w_ij has a matching gate score s_ij (same tensor shape).

    Forward pass
    ------------
        gates         = sigmoid(gate_scores)     ∈ (0, 1)
        pruned_weight = weight  ⊙  gates         element-wise
        output        = x @ pruned_weight.T + bias

    Gradient flow
    -------------
    sigmoid and ⊙ are both differentiable, so autograd propagates gradients
    to BOTH `weight` and `gate_scores` automatically — no custom backward
    or straight-through estimator required.

    gate_score → -∞  ⟹  gate → 0  ⟹  weight effectively removed.
    gate_score → +∞  ⟹  gate → 1  ⟹  weight fully active.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_parameters()

    def _init_parameters(self):
        # Standard Kaiming init for weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # gate_scores = +2  →  sigmoid(+2) ≈ 0.88
        # Gates start mostly open so L1 penalty has clear gates to close.
        nn.init.constant_(self.gate_scores, 2.0)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores)   # (out, in)
        pruned_weight = self.weight * gates               # (out, in)
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Detached gate values for inspection."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ---------------------------------------------------------------------------
# 2.  Self-Pruning Network
# ---------------------------------------------------------------------------

class SelfPruningNet(nn.Module):
    """
    Feed-forward CIFAR-10 classifier using only PrunableLinear layers.

    Architecture
    ------------
    Flatten  →  3072
    PrunableLinear(3072, 1024) + BN + ReLU + Dropout
    PrunableLinear(1024,  512) + BN + ReLU + Dropout
    PrunableLinear( 512,  256) + BN + ReLU + Dropout
    PrunableLinear( 256,   10)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),

            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        Mean of all gate values across every PrunableLinear layer.

        Why MEAN not SUM?
        -----------------
        The network has ~4 million gates.  sum ≈ 4e6 × 0.88 ≈ 3.5e6 initially.
        Even λ = 1e-5 would make the sparsity term ≈ 35 >> CE loss ≈ 2.3,
        swamping the classification signal completely.
        mean() keeps the loss in (0, 1) regardless of model size, so λ
        directly expresses 'how much sparsity relative to classification'.

        Why L1 (absolute value / mean) and not L2?
        -------------------------------------------
        L1 gradient w.r.t. gate = sign(gate) = +1  (gates are always positive).
        This CONSTANT gradient keeps pushing gates all the way to 0 even when
        they are already tiny.  L2 gradient = 2 × gate → shrinks to 0 near 0,
        so L2 can never fully zero out a gate.
        """
        all_gates = torch.cat(
            [torch.sigmoid(l.gate_scores).view(-1) for l in self.prunable_layers()]
        )
        return all_gates.mean()   # scalar ∈ (0, 1)

    def compute_sparsity(self, threshold: float = SPARSE_THRESH) -> float:
        """
        Percentage of gates below `threshold` (default 0.5).
        Gate < 0.5  →  more closed than open  →  weight effectively pruned.
        Using 0.5 gives non-zero readings from epoch 1, making progress visible.
        """
        total = pruned = 0
        for layer in self.prunable_layers():
            g      = layer.get_gates()
            total  += g.numel()
            pruned += (g < threshold).sum().item()
        return 100.0 * pruned / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 3.  Data loading
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf)

    # num_workers=0 avoids Windows multiprocessing issues
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    test_loader  = DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 4.  Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     SelfPruningNet,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    lam:       float,
) -> Tuple[float, float, float]:
    """
    One full pass over training data.

    Total Loss = CrossEntropyLoss  +  λ × mean(all_gates)

    Returns (avg_total_loss, avg_ce_loss, avg_sparsity_loss).
    """
    model.train()
    tot = ce_sum = sp_sum = 0.0
    n   = len(loader)

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        logits     = model(images)
        ce_loss    = F.cross_entropy(logits, labels)
        sp_loss    = model.sparsity_loss()          # ∈ (0, 1)
        total_loss = ce_loss + lam * sp_loss

        total_loss.backward()
        optimizer.step()

        tot    += total_loss.item()
        ce_sum += ce_loss.item()
        sp_sum += sp_loss.item()

    return tot / n, ce_sum / n, sp_sum / n


# ---------------------------------------------------------------------------
# 5.  Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: SelfPruningNet, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds    = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# 6.  Single experiment
# ---------------------------------------------------------------------------

def run_experiment(
    lam:          float,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    epochs:       int   = 25,
    lr:           float = 1e-3,
) -> Tuple[float, float, SelfPruningNet]:
    """Train a fresh model with the given λ and return (accuracy, sparsity, model)."""
    print(f"\n{'='*65}")
    print(f"  Experiment  λ = {lam}  |  epochs = {epochs}  |  lr = {lr}")
    print(f"{'='*65}")

    model     = SelfPruningNet(dropout=0.3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss, ce_loss, sp_loss = train_one_epoch(
            model, train_loader, optimizer, lam)
        scheduler.step()
        elapsed = time.time() - t0

        acc      = evaluate(model, test_loader)
        sparsity = model.compute_sparsity()

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Loss {total_loss:.4f} "
              f"(CE {ce_loss:.4f} | SP {sp_loss:.4f}) | "
              f"Acc {acc:.2f}% | "
              f"Sparsity {sparsity:.1f}% | "
              f"{elapsed:.1f}s")

    final_acc      = evaluate(model, test_loader)
    final_sparsity = model.compute_sparsity()
    print(f"\n  ► Final Test Accuracy : {final_acc:.2f}%")
    print(f"  ► Final Sparsity      : {final_sparsity:.2f}%")
    return final_acc, final_sparsity, model


# ---------------------------------------------------------------------------
# 7.  Gate distribution plot
# ---------------------------------------------------------------------------

def plot_gate_distribution(model: SelfPruningNet, lam: float, save_path: str):
    """
    Histogram of all gate values.
    Successful pruning → bimodal: large spike near 0 + cluster above 0.5.
    """
    all_gates = np.concatenate(
        [l.get_gates().cpu().numpy().ravel() for l in model.prunable_layers()])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_gates, bins=120, color="#3A86FF", edgecolor="white", linewidth=0.3)
    ax.axvline(SPARSE_THRESH, color="red", linestyle="--", linewidth=2.0,
               label=f"Prune threshold = {SPARSE_THRESH}")
    ax.set_xlabel("Gate Value  (sigmoid output)", fontsize=13)
    ax.set_ylabel("Number of Weights", fontsize=13)
    ax.set_title(
        f"Gate Value Distribution  (λ = {lam})\n"
        "Spike near 0 → pruned weights  |  Cluster > 0.5 → active weights",
        fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Gate distribution plot saved → {save_path}")


# ---------------------------------------------------------------------------
# 8.  Results table
# ---------------------------------------------------------------------------

def print_results_table(results: List[Tuple[float, float, float]]):
    print("\n\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Lambda':<14} {'Test Accuracy (%)':<22} {'Sparsity (%)'}")
    print("-" * 60)
    for lam, acc, sparsity in results:
        print(f"  {lam:<14.2f} {acc:<22.2f} {sparsity:.2f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 9.  Main
# ---------------------------------------------------------------------------

def main():
    EPOCHS     = 25       # epochs per experiment  (3 × 25 = 75 total)
    BATCH_SIZE = 128
    LR         = 1e-3
    LAMBDAS    = [0.1, 0.5, 2.0]   # low / medium / high
    BEST_LAM   = 0.5               # model used for gate distribution plot

    print("\n" + "="*65)
    print("  Self-Pruning Neural Network — CIFAR-10")
    print(f"  Lambdas: {LAMBDAS}  |  Epochs each: {EPOCHS}  |  Device: {DEVICE}")
    print("="*65)

    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    results    = []
    best_model = None

    for lam in LAMBDAS:
        acc, sparsity, model = run_experiment(
            lam=lam, train_loader=train_loader,
            test_loader=test_loader, epochs=EPOCHS, lr=LR)
        results.append((lam, acc, sparsity))
        if lam == BEST_LAM:
            best_model = model

    print_results_table(results)
    plot_gate_distribution(best_model, BEST_LAM, "gate_distribution.png")

    print("\nAll done! Push these files to GitHub:")
    print("  ✅  self_pruning_network.py")
    print("  ✅  REPORT.md")
    print("  ✅  gate_distribution.png")


if __name__ == "__main__":
    main()
