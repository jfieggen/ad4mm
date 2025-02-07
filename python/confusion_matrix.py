import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrices(matrices, labels, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), dpi=300)  # Adjusted for column width with high DPI
    axes = axes.flatten()
    
    for ax, (matrix, label) in zip(axes, zip(matrices, labels)):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=["No Myeloma", "Myeloma"], 
                    yticklabels=["No Myeloma", "Myeloma"], ax=ax, annot_kws={"size": 10})
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Ensuring fit with high DPI
    plt.show()

# Define the confusion matrices
lof_matrix = np.array([[7447, 3126], [5, 22]])
iforest_matrix = np.array([[9211, 1362], [9, 18]])
seqxgb_matrix = np.array([[10564, 9], [23, 4]])
xgb_matrix = np.array([[10153, 420], [8, 19]])

# Labels for each model
labels = ["Local Outlier Factor", "Isolation Forest", "SeqXGB", "XGBoost"]

# Output path
output_path = "/well/clifton/users/ncu080/ad4mm/outputs/performace_metrics/confusion_matrices.png"

# Plot the matrices
plot_confusion_matrices([lof_matrix, iforest_matrix, seqxgb_matrix, xgb_matrix], labels, output_path)
