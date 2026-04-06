import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report,
                             adjusted_mutual_info_score, silhouette_score, accuracy_score)


class AttentionGatingNetwork(nn.Module):
    """
    Attention-based gating network that learns to route patient embeddings to clusters.
    A feature attention gate re-weights input dimensions before projecting to a query space.
    Similarity against learnable cluster prototype keys produces routing logits.
    """

    def __init__(self, input_dim, n_clusters=4):
        super(AttentionGatingNetwork, self).__init__()
        self.n_clusters = n_clusters

        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.query_net = nn.Linear(input_dim, 64)
        self.cluster_keys = nn.Parameter(torch.randn(n_clusters, 64))

    def forward(self, x):
        gate = self.feature_gate(x)
        x_gated = x * gate
        query = F.relu(self.query_net(x_gated))
        logits = torch.matmul(query, self.cluster_keys.T)
        return logits, gate

    def predict_detailed(self, X):
        """
        Runs inference and returns logits, gate weights, and query embeddings.
        Accepts a numpy array and returns numpy arrays.
        """
        self.eval()
        device = next(self.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            gate = self.feature_gate(X_tensor)
            query = F.relu(self.query_net(X_tensor * gate))
            logits = torch.matmul(query, self.cluster_keys.T)

        return logits.cpu().numpy(), gate.cpu().numpy(), query.cpu().numpy()

    def fit_model(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.001):
        """
        Trains the gating network using cross-entropy loss on cluster assignments.
        Tracks train loss and validation accuracy at each epoch.
        Returns a history dictionary with train_loss and val_acc lists.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.long).to(device)

        history = {'train_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            logits, _ = self(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_logits, _ = self(X_v)
                preds = torch.argmax(val_logits, dim=1)
                acc = (preds == y_v).float().mean()

            history['train_loss'].append(loss.item())
            history['val_acc'].append(acc.item())

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Val Acc: {acc.item():.4f}")

        return history

    @staticmethod
    def plot_transition_matrix(y_true, logits, title="Transition Matrix Past -> Future"):
        """Plots a normalized confusion matrix between true cluster labels and gating predictions."""
        y_pred = np.argmax(logits, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens")
        plt.ylabel("True Cluster (Future)")
        plt.xlabel("Predicted Cluster (Past)")
        plt.title(title)
        plt.show()

        print("Performance Report")
        print(classification_report(y_true, y_pred))

    @staticmethod
    def evaluate_performance(y_true, logits, query, stage_name="Test"):
        """
        Computes AMI, silhouette score, and accuracy for gating predictions.
        Returns a dictionary with ami, silhouette, and accuracy values.
        """
        y_pred = np.argmax(logits, axis=1)

        ami = adjusted_mutual_info_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        if len(np.unique(y_pred)) > 1:
            sil = silhouette_score(query, y_pred)
        else:
            sil = 0.0

        print(f"Scores ({stage_name})")
        print(f"AMI (Adjusted Mutual Info): {ami:.4f}")
        print(f"Silhouette (Latent Space):  {sil:.4f}")
        print(f"Accuracy (Recovery Rate):   {acc:.4f}")

        return {"ami": ami, "silhouette": sil, "accuracy": acc}
