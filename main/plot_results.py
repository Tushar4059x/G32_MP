"""Plot federated learning training results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_metrics():
    """Load and plot metrics from training history."""
    
    # Load metrics
    metrics_file = Path("metrics_history.json")
    if not metrics_file.exists():
        print("❌ No metrics history found. Run training first!")
        return
    
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    if not metrics["rounds"]:
        print("❌ No metrics data available!")
        return
    
    rounds = metrics["rounds"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Federated Learning Training Results - Pneumonia Detection', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy over rounds
    ax1 = axes[0, 0]
    ax1.plot(rounds, metrics["accuracy"], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Precision and Recall
    ax2 = axes[0, 1]
    ax2.plot(rounds, metrics["precision"], 'g-o', label='Precision', linewidth=2, markersize=6)
    ax2.plot(rounds, metrics["recall"], 'r-s', label='Recall', linewidth=2, markersize=6)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: F1-Score
    ax3 = axes[0, 2]
    ax3.plot(rounds, metrics["f1_score"], 'm-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('F1-Score', fontsize=12)
    ax3.set_title('F1-Score (Harmonic Mean)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Specificity
    ax4 = axes[1, 0]
    ax4.plot(rounds, metrics["specificity"], 'c-o', linewidth=2, markersize=6)
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Specificity', fontsize=12)
    ax4.set_title('Specificity (True Negative Rate)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # Plot 5: AUC-ROC
    ax5 = axes[1, 1]
    ax5.plot(rounds, metrics["auc_roc"], 'orange', marker='o', linewidth=2, markersize=6)
    ax5.set_xlabel('Round', fontsize=12)
    ax5.set_ylabel('AUC-ROC', fontsize=12)
    ax5.set_title('AUC-ROC Score', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    ax5.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Excellent (0.9)')
    ax5.axhline(y=0.8, color='y', linestyle='--', alpha=0.5, label='Good (0.8)')
    ax5.legend(fontsize=9)
    
    # Plot 6: Loss
    ax6 = axes[1, 2]
    ax6.plot(rounds, metrics["loss"], 'darkred', marker='o', linewidth=2, markersize=6)
    ax6.set_xlabel('Round', fontsize=12)
    ax6.set_ylabel('Loss', fontsize=12)
    ax6.set_title('Evaluation Loss', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = "training_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    
    # Print final metrics summary
    print("\n" + "="*80)
    print("📊 FINAL METRICS SUMMARY (Round {})".format(rounds[-1]))
    print("="*80)
    print(f"  Accuracy:    {metrics['accuracy'][-1]:.4f} ({metrics['accuracy'][-1]*100:.2f}%)")
    print(f"  Precision:   {metrics['precision'][-1]:.4f} ({metrics['precision'][-1]*100:.2f}%)")
    print(f"  Recall:      {metrics['recall'][-1]:.4f} ({metrics['recall'][-1]*100:.2f}%) ⭐")
    print(f"  F1-Score:    {metrics['f1_score'][-1]:.4f}")
    print(f"  Specificity: {metrics['specificity'][-1]:.4f} ({metrics['specificity'][-1]*100:.2f}%)")
    print(f"  AUC-ROC:     {metrics['auc_roc'][-1]:.4f}")
    print(f"  Loss:        {metrics['loss'][-1]:.4f}")
    print("="*80)
    
    # Print improvement summary
    if len(rounds) > 1:
        print("\n📈 IMPROVEMENT FROM ROUND 1 TO ROUND {}".format(rounds[-1]))
        print("="*80)
        print(f"  Accuracy:    {metrics['accuracy'][0]:.4f} → {metrics['accuracy'][-1]:.4f} ({(metrics['accuracy'][-1] - metrics['accuracy'][0])*100:+.2f}%)")
        print(f"  Precision:   {metrics['precision'][0]:.4f} → {metrics['precision'][-1]:.4f} ({(metrics['precision'][-1] - metrics['precision'][0])*100:+.2f}%)")
        print(f"  Recall:      {metrics['recall'][0]:.4f} → {metrics['recall'][-1]:.4f} ({(metrics['recall'][-1] - metrics['recall'][0])*100:+.2f}%)")
        print(f"  F1-Score:    {metrics['f1_score'][0]:.4f} → {metrics['f1_score'][-1]:.4f} ({(metrics['f1_score'][-1] - metrics['f1_score'][0])*100:+.2f}%)")
        print(f"  Specificity: {metrics['specificity'][0]:.4f} → {metrics['specificity'][-1]:.4f} ({(metrics['specificity'][-1] - metrics['specificity'][0])*100:+.2f}%)")
        print(f"  AUC-ROC:     {metrics['auc_roc'][0]:.4f} → {metrics['auc_roc'][-1]:.4f} ({(metrics['auc_roc'][-1] - metrics['auc_roc'][0])*100:+.2f}%)")
        print(f"  Loss:        {metrics['loss'][0]:.4f} → {metrics['loss'][-1]:.4f} ({(metrics['loss'][-1] - metrics['loss'][0]):+.4f})")
        print("="*80)
    
    plt.show()


if __name__ == "__main__":
    plot_metrics()
