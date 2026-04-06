import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, classification_report,
                             roc_auc_score, average_precision_score)
from sklearn.calibration import calibration_curve
from scipy.special import softmax


class AttentionSepsisMoE(BaseEstimator, ClassifierMixin):
    """
    Mixture of Experts classifier for sepsis prediction.
    Uses a pre-trained AttentionGatingNetwork to route each patient to one of n_experts MLP classifiers.
    Each expert is trained on the subset of patients routed to it.
    """

    def __init__(self, gating_net, n_experts=4, hidden_layer_sizes=(64, 32), learning_rate=1e-2, random_state=42):
        self.gating_net = gating_net
        self.n_experts = n_experts
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.experts_ = [
            MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                learning_rate_init=self.learning_rate,
                max_iter=500,
                early_stopping=True,
                random_state=self.random_state + i
            ) for i in range(self.n_experts)
        ]
        self.is_fitted = False

    def fit(self, X, y):
        """
        Trains each expert on its assigned patient subset as determined by gating weights.
        If an expert receives fewer than two classes, it falls back to training on the full dataset.
        """
        print(f"Training {self.n_experts} experts via attention gating")

        logits, _, _ = self.gating_net.predict_detailed(X)
        gating_weights = softmax(logits, axis=1)
        assignments = np.argmax(gating_weights, axis=1)

        for k in range(self.n_experts):
            mask = (assignments == k)
            X_k, y_k = X[mask], y[mask]

            if len(np.unique(y_k)) < 2:
                print(f"Expert {k}: insufficient class diversity, falling back to full dataset.")
                self.experts_[k].fit(X, y)
            else:
                print(f"Expert {k}: training on {len(X_k)} patients.")
                self.experts_[k].fit(X_k, y_k)

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """
        Computes weighted sepsis probability by combining each expert's output
        with the gating network's routing weights.
        Returns a 1D array of sepsis probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained.")

        logits, _, _ = self.gating_net.predict_detailed(X)
        gating_weights = softmax(logits, axis=1)

        expert_probs = np.zeros((X.shape[0], self.n_experts))
        for k in range(self.n_experts):
            expert_probs[:, k] = self.experts_[k].predict_proba(X)[:, 1]

        final_proba = np.sum(gating_weights * expert_probs, axis=1)
        return final_proba

    def predict(self, X, threshold=0.5):
        """Returns binary predictions using the specified probability threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)


class SepsisMoEDiagnostic:
    """
    Diagnostic toolkit for evaluating an AttentionSepsisMoE model.
    Computes global metrics and generates plots for model performance,
    attention-based routing analysis, and probability calibration.
    """

    def __init__(self, model, X_test, y_test, feature_names=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names

        self.y_prob = model.predict_proba(X_test)
        self.y_pred = (self.y_prob >= 0.5).astype(int)

        self.logits, self.gate, self.query = model.gating_net.predict_detailed(X_test)
        self.weights = softmax(self.logits, axis=1)

    def run_full_diagnosis(self):
        """Runs the complete diagnostic: prints global metrics and generates all diagnostic plots."""
        print("Full Diagnostic: Attention-based MoE")

        auroc = roc_auc_score(self.y_test, self.y_prob)
        auprc = average_precision_score(self.y_test, self.y_prob)

        print("Global Prediction Results")
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print("Classification Report")
        print(classification_report(self.y_test, self.y_pred, target_names=['Non-Sepsis', 'Sepsis']))

        self.plot_global_performance(auroc, auprc)
        self.audit_attention_gating()
        self.plot_calibration()

    def plot_global_performance(self, auroc, auprc):
        """Plots confusion matrix, ROC curve, and Precision-Recall curve side by side."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_title("Global Confusion Matrix")
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("MoE Prediction")

        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        axes[1].plot(fpr, tpr, label=f'AUC = {auroc:.3f}', color='darkorange', lw=2)
        axes[1].plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
        axes[1].set_title("ROC Curve")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(loc="lower right")

        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
        axes[2].plot(recall, precision, label=f'PR AUC = {auprc:.3f}', color='green', lw=2)
        axes[2].set_title("Precision-Recall Curve")
        axes[2].set_xlabel("Recall (Sepsis detected)")
        axes[2].set_ylabel("Precision (True alarms)")
        axes[2].legend(loc="lower left")

        plt.tight_layout()
        plt.show()

    def audit_attention_gating(self):
        """Plots patient routing distribution across experts and top feature attention weights."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        assignments = np.argmax(self.weights, axis=1)
        sns.countplot(x=assignments, ax=axes[0], palette="viridis")
        axes[0].set_title("Patient Distribution Across Experts")
        axes[0].set_xlabel("Expert ID")
        axes[0].set_ylabel("Number of patients")

        avg_gate = np.mean(self.gate, axis=0)
        indices = np.argsort(avg_gate)[-10:]

        names = [self.feature_names[i] if self.feature_names else f"Feature {i}" for i in indices]
        axes[1].barh(range(len(indices)), avg_gate[indices], color='skyblue')
        axes[1].set_yticks(range(len(indices)))
        axes[1].set_yticklabels(names)
        axes[1].set_title("Top 10 Features by Attention Gate Weight")
        axes[1].set_xlabel("Mean attention weight (Sigmoid)")

        plt.tight_layout()
        plt.show()

    def plot_calibration(self):
        """Plots a calibration curve comparing predicted probabilities to observed positive rates."""
        prob_true, prob_pred = calibration_curve(self.y_test, self.y_prob, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='MoE Model', color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.title("Probability Calibration Curve")
        plt.xlabel("Predicted probability (model confidence)")
        plt.ylabel("Observed positive fraction (ground truth)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
