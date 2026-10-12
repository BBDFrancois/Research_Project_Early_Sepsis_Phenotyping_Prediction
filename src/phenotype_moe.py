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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, roc_auc_score, precision_recall_curve, silhouette_score)


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

@torch.no_grad()
def evaluate_phenotype_moe_plus(model, z, y, history=None, threshold=0.5, device='cpu'):
    model.eval()
    z_tensor = torch.FloatTensor(z).to(device) if not torch.is_tensor(z) else z.to(device)
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)

    # Inférence
    p, g, p_k, h = model(z_tensor)
    p_np = p.cpu().numpy()
    g_np = g.cpu().numpy()
    h_np = h.cpu().numpy()
    
    y_pred = (p_np >= threshold).astype(int)
    assignments = g_np.argmax(axis=1)

    # --- 1. RAPPORTS TEXTUELS ---
    print("="*50)
    print("      DIAGNOSTIC GLOBAL : Phenotype MoE")
    print("="*50)
    print("\n[Rapport de Classification]")
    print(classification_report(y_np, y_pred, zero_division=0))

    # Calcul du Silhouette Score (possible uniquement si > 1 cluster utilisé)
    if len(np.unique(assignments)) > 1:
        sil_score = silhouette_score(h_np, assignments)
    else:
        sil_score = -1.0
        
    print("\n[Métriques de Clustering]")
    print(f"Score de Silhouette        : {sil_score:.4f} (proche de 1 = clusters très nets)")
    print(f"Entropie moyenne (H)       : {-(g_np * np.log(g_np + 1e-8)).sum(axis=1).mean():.4f}")
    print(f"Confiance maximale moyenne : {g_np.max(axis=1).mean():.4f}")
    print(f"Répartition des patients   : {np.bincount(assignments, minlength=model.n_clusters)}")

    # --- 2. TABLEAU DE BORD VISUEL ---
    fig = plt.figure(figsize=(18, 10))

    # A. Matrice de Confusion (Heatmap)
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_np, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=['Sain (0)', 'Sepsis (1)'], yticklabels=['Sain (0)', 'Sepsis (1)'])
    ax1.set_title("Matrice de Confusion")
    ax1.set_ylabel("Vérité Terrain")
    ax1.set_xlabel("Prédiction")

    # B. Courbe ROC
    ax2 = plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(y_np, p_np)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {roc_auc:.3f}')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_title("Courbe ROC")
    ax2.set_xlabel("Taux de Faux Positifs")
    ax2.set_ylabel("Taux de Vrais Positifs")
    ax2.legend(loc="lower right")

    # C. Courbe Precision-Recall
    ax3 = plt.subplot(2, 3, 3)
    precision, recall, _ = precision_recall_curve(y_np, p_np)
    pr_auc = auc(recall, precision)
    ax3.plot(recall, precision, color='purple', lw=2, label=f'AUPRC = {pr_auc:.3f}')
    ax3.set_title("Courbe Precision-Recall")
    ax3.set_xlabel("Recall (Sensibilité)")
    ax3.set_ylabel("Precision (VPP)")
    ax3.legend(loc="lower left")

    # D. Projection t-SNE de l'espace caché (Phénotypes)
    ax4 = plt.subplot(2, 3, 4)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    h_tsne = tsne.fit_transform(h_np)
    scatter = ax4.scatter(h_tsne[:, 0], h_tsne[:, 1], c=assignments, cmap='Set1', alpha=0.7, s=15)
    ax4.set_title("t-SNE : Phénotypes découverts")
    legend1 = ax4.legend(*scatter.legend_elements(), title="Expert assigné", loc="best")
    ax4.add_artist(legend1)

    # E & F. Évolution de l'entraînement (si history est fourni)
    if history is not None:
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(history['total'], label='Loss Totale', color='black', lw=2)
        ax5.plot(history['pred'], label='L_pred (BCE)', linestyle='--')
        ax5.set_title("Dynamique de la Fonction de Perte (Train)")
        ax5.set_xlabel("Époques")
        ax5.legend()

        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(history['val_auroc'], label='AUROC (Validation)', color='darkorange', lw=2)
        ax6.set_title("Performance de Validation")
        ax6.set_xlabel("Époques")
        ax6.legend()
    else:
        # Si on passe un Test set sans historique, on masque ces graphes
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

    plt.tight_layout()
    plt.show()

    return p_np, assignments