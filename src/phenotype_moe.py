"""
phenotype_moe.py

Module B of the end-to-end architecture: attention-based gating (soft
clustering) + a bank of experts (MoE), trained jointly on the frozen
embedding z produced by stage A (SAITS -> TS2Vec -> autoencoder).

No warmup, no re-init of the prototypes mid-training: a single continuous
training run, a single optimizer, a single total loss = weighted sum of
L_pred + L_cluster (L_ent + L_compact + L_margin) + L_balance.

All training/evaluation functions take tensors that already come from a
single train/val/test split done once upstream (see main).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------------- #
# 1. Gating = soft clustering (attention over learned prototypes)
# --------------------------------------------------------------------------- #

class PrototypeGating(nn.Module):
    """
    Projects z into a space h, then computes cosine similarity between h and
    K learned prototypes (free vectors, not tied to any real patient).
    g = softmax(cos(h, mu_k) / tau): membership weights over the K groups,
    each row sums to 1 (soft clustering).
    """

    def __init__(self, input_dim, hidden_dim=64, n_clusters=4, temperature=0.3):
        super().__init__()
        self.n_clusters = n_clusters
        self.temperature = temperature

        # Projection z -> h. No final ReLU: we want to be able to explore the
        # whole space (negative cosines included), not just the positive orthant.
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Prototypes = free vectors, learned by gradient like everything else.
        self.prototypes = nn.Parameter(torch.randn(n_clusters, hidden_dim) * 0.1)

    def forward(self, z):
        """
        z : (B, input_dim)
        Returns g (B, K) and h (B, hidden_dim), the projected representation
        (used for L_compact/L_margin and for optional diagnostics).
        """
        h = self.proj(z)
        h_norm = F.normalize(h, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)

        cos_sim = h_norm @ proto_norm.T  # (B, K)
        g = F.softmax(cos_sim / self.temperature, dim=-1)
        return g, h_norm

    @torch.no_grad()
    def init_prototypes_from_z(self, z_train, random_state=42):
        """
        Optional initialization: cosine KMeans on raw z (not on h, so it can
        be used before any training). Prototypes remain free vectors
        afterwards, learned by gradient. NOT called by default in
        train_phenotype_moe (V1 = no special init); enable only if training
        without warmup shows a cluster collapse.
        """
        z_np = F.normalize(z_train, dim=-1).cpu().numpy()
        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=random_state)
        km.fit(z_np)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        # Centers live in input_dim, not hidden_dim: only safe to copy
        # directly if hidden_dim == input_dim. Otherwise, initializing after
        # a few epochs on h remains preferable (see warmup discussion).
        if centers.shape[1] == self.prototypes.shape[1]:
            self.prototypes.data.copy_(centers)
        else:
            raise ValueError(
                "init_prototypes_from_z assumes hidden_dim == input_dim; "
                "otherwise, initialize after a few epochs on h (see warmup discussion)."
            )


# --------------------------------------------------------------------------- #
# 2. Expert bank
# --------------------------------------------------------------------------- #

class ExpertBank(nn.Module):
    """K small binary MLPs, one per group, trained simultaneously."""

    def __init__(self, input_dim, n_experts=4, hidden=(64, 32), dropout=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden[0], hidden[1]),
                nn.ReLU(),
                nn.Linear(hidden[1], 1),
            )
            for _ in range(n_experts)
        ])

    def forward(self, z):
        """z : (B, input_dim) -> p_k (B, K), probabilities (sigmoid of logits)."""
        logits = torch.cat([expert(z) for expert in self.experts], dim=1)  # (B, K)
        return torch.sigmoid(logits)


# --------------------------------------------------------------------------- #
# 3. Assembly: gating + experts + losses
# --------------------------------------------------------------------------- #

class PhenotypeMoE(nn.Module):
    def __init__(self, input_dim, n_clusters=4, n_experts=None,
                 gating_hidden=64, expert_hidden=(64, 32), temperature=0.3,
                 margin=0.5):
        super().__init__()
        n_experts = n_experts or n_clusters
        assert n_experts == n_clusters, "one expert per group, for now."

        self.gating = PrototypeGating(input_dim, gating_hidden, n_clusters, temperature)
        self.experts = ExpertBank(input_dim, n_experts, expert_hidden)
        self.n_clusters = n_clusters
        self.margin = margin

    def forward(self, z):
        g, h = self.gating(z)          # (B,K), (B,H)
        p_k = self.experts(z)          # (B,K)
        p = (g * p_k).sum(dim=1)       # (B,)
        return p, g, p_k, h

    def compute_losses(self, p, g, h, y, lambdas):
        """
        Returns a dict with every term separated + 'total', so each can be
        logged independently (useful for spotting a collapse early).
        """
        eps = 1e-8
        K = self.n_clusters

        # --- L_pred: BCE on the final mixture ---
        l_pred = F.binary_cross_entropy(p.clamp(eps, 1 - eps), y.float())

        # --- L_ent: mean per-patient entropy (low = confident) ---
        l_ent = -(g * (g + eps).log()).sum(dim=1).mean()

        # --- L_compact: soft k-means in cosine space (h and prototypes already normalized) ---
        proto_norm = F.normalize(self.gating.prototypes, dim=-1)  # (K,H)
        cos_to_proto = h @ proto_norm.T                            # (B,K)
        l_compact = (g * (1 - cos_to_proto)).sum(dim=1).mean()

        # --- L_margin: pushes prototypes apart from each other ---
        sim_proto = proto_norm @ proto_norm.T  # (K,K), cos in [-1,1]
        dist_proto = 1 - sim_proto             # 0 = identical, 2 = opposite
        iu = torch.triu_indices(K, K, offset=1)
        pair_dist = dist_proto[iu[0], iu[1]]
        l_margin = F.relu(self.margin - pair_dist).pow(2).mean() if pair_dist.numel() > 0 else torch.tensor(0.0)

        # --- L_balance: KL(mean_g || uniform) over the batch ---
        g_bar = g.mean(dim=0)  # (K,)
        uniform = torch.full_like(g_bar, 1.0 / K)
        l_balance = F.kl_div((g_bar + eps).log(), uniform, reduction='sum')

        l_cluster = l_ent + l_compact + l_margin

        total = (
            lambdas.get('pred', 1.0) * l_pred
            + lambdas.get('cluster', 0.0) * l_cluster
            + lambdas.get('balance', 0.0) * l_balance
        )

        return {
            'total': total,
            'pred': l_pred.detach(),
            'ent': l_ent.detach(),
            'compact': l_compact.detach(),
            'margin': l_margin.detach() if torch.is_tensor(l_margin) else l_margin,
            'balance': l_balance.detach(),
        }


# --------------------------------------------------------------------------- #
# 4. Training (no warmup) and evaluation
# --------------------------------------------------------------------------- #

def train_phenotype_moe(model, z_train, y_train, z_val, y_val,
                         epochs=200, batch_size=64, lr=1e-3, weight_decay=1e-4,
                         lambdas=None, patience=20, device='cpu', verbose_every=10):
    """
    Single training loop, no warmup phase and no prototype re-init:
    lambdas stay constant from start to end.

    z_train, z_val : (N, input_dim) torch.FloatTensor already on `device`
    y_train, y_val : (N,) torch.FloatTensor (0/1)

    Returns (model with best weights loaded, history); history holds the
    loss components and the val AUROC at every epoch, so a collapse can be
    caught early rather than only noticed at the end.
    """
    lambdas = lambdas or {'pred': 1.0, 'cluster': 0.5, 'balance': 1.0}
    model = model.to(device)
    z_train, y_train = z_train.to(device), y_train.to(device)
    z_val, y_val = z_val.to(device), y_val.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = z_train.shape[0]
    history = {k: [] for k in ['total', 'pred', 'ent', 'compact', 'margin',
                                'balance', 'val_auroc', 'g_bar', 'g_entropy']}

    best_auroc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_losses = {k: 0.0 for k in ['total', 'pred', 'ent', 'compact', 'margin', 'balance']}
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            zb, yb = z_train[idx], y_train[idx]

            p, g, p_k, h = model(zb)
            losses = model.compute_losses(p, g, h, yb, lambdas)

            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()

            for k in epoch_losses:
                v = losses[k]
                epoch_losses[k] += v.item() if torch.is_tensor(v) else v
            n_batches += 1

        for k in epoch_losses:
            history[k].append(epoch_losses[k] / n_batches)

        # --- Val evaluation: AUROC + anti-collapse diagnostics ---
        model.eval()
        with torch.no_grad():
            p_val, g_val, _, _ = model(z_val)
            val_auroc = roc_auc_score(y_val.cpu().numpy(), p_val.cpu().numpy())
            g_bar = g_val.mean(dim=0).cpu().numpy()
            g_entropy = -(g_val * (g_val + 1e-8).log()).sum(dim=1).mean().item()

        history['val_auroc'].append(val_auroc)
        history['g_bar'].append(g_bar)
        history['g_entropy'].append(g_entropy)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"Epoch {epoch+1:3d} | total {history['total'][-1]:.4f} | "
                  f"pred {history['pred'][-1]:.4f} | cluster(ent+comp+marg) "
                  f"{history['ent'][-1]+history['compact'][-1]+history['margin'][-1]:.4f} | "
                  f"balance {history['balance'][-1]:.4f} | val AUROC {val_auroc:.4f} | "
                  f"g_bar {np.round(g_bar, 2)}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


from sklearn.metrics import confusion_matrix, classification_report
@torch.no_grad()
def evaluate_phenotype_moe(model, z, y, threshold=0.5, device='cpu'):
    """
    A single forward pass, no weight update. Use on val (during tuning) and
    only once on test (at the very end).
    Returns a metrics dict + the g/p arrays for further analysis (e.g.
    clinical characterization of the groups).
    """
    model.eval()
    z = z.to(device)
    p, g, p_k, h = model(z)

    p_np = p.cpu().numpy()
    g_np = g.cpu().numpy()
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)

    y_pred = (p_np >= threshold).astype(int)

    cm = confusion_matrix(y_np, y_pred)
    report = classification_report(y_np, y_pred, zero_division=0)

    # Extraction TP, FP, TN, FN (si classification binaire)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"Confusion Matrix -> TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    else:
        print(f"Confusion Matrix:\n{cm}")
        
    print(f"\nClassification Report (F1-score par classe):\n{report}")

    auroc = roc_auc_score(y_np, p_np)
    assignments = g_np.argmax(axis=1)
    g_bar = g_np.mean(axis=0)
    g_entropy = -(g_np * np.log(g_np + 1e-8)).sum(axis=1).mean()
    max_g = g_np.max(axis=1).mean()  # close to 1 -> near-hard clustering; close to 1/K -> fully mixed

    print(f"AUROC: {auroc:.4f}")
    print(f"Group distribution (g_bar): {np.round(g_bar, 3)}")
    print(f"Mean entropy of g: {g_entropy:.4f}")
    print(f"Mean confidence (max_k g_k): {max_g:.4f}")

    return {
        'auroc': auroc,
        'g_bar': g_bar,
        'g_entropy': g_entropy,
        'max_g': max_g,
        'p': p_np,
        'g': g_np,
        'assignments': assignments,
    }



def plot_training_curves(history):
    """Monitoring plots: losses, val AUROC, group distribution, entropy."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(history['pred'], label='L_pred')
    axes[0, 0].plot(history['ent'], label='L_ent')
    axes[0, 0].plot(history['compact'], label='L_compact')
    axes[0, 0].plot(history['margin'], label='L_margin')
    axes[0, 0].plot(history['balance'], label='L_balance')
    axes[0, 0].set_title("Loss components (train)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history['val_auroc'], color='darkorange')
    axes[0, 1].set_title("AUROC (validation)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(alpha=0.3)

    g_bar_arr = np.array(history['g_bar'])  # (epochs, K)
    for k in range(g_bar_arr.shape[1]):
        axes[1, 0].plot(g_bar_arr[:, k], label=f'Group {k}')
    axes[1, 0].axhline(1 / g_bar_arr.shape[1], color='gray', linestyle='--', label='Uniform')
    axes[1, 0].set_title("Mean group distribution (g_bar) — validation")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history['g_entropy'], color='green')
    axes[1, 1].set_title("Mean entropy of g (validation)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()