import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# ==========================================
# 1. Load Resources
# ==========================================
print("Loading model and tools...")
try:
    # Load the trained model
    model = keras.models.load_model('examples/tourism_model.keras')
    # Load the scaler used in training
    scaler = joblib.load('scaler.pkl')
    # Load the column list to ensure alignment
    model_columns = joblib.load('model_columns.pkl')
    print("Model and tools loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")
    print("Please run class5_training.py first!")
    exit()

model.summary()

# ==========================================
# 2. Load Test Data
# ==========================================
print("\nLoading test data...")
# Load the test data created by the training script
df_test = pd.read_csv('/opt/project/data/class5_test_data.csv')

print(f"Test dataset shape: {df_test.shape}")
if 'ProdTaken' in df_test.columns:
    print(f"\nTarget distribution:\n{df_test['ProdTaken'].value_counts()}")

# Separate features and target
if 'ProdTaken' in df_test.columns:
    X_test = df_test.drop('ProdTaken', axis=1)
    y_test = df_test['ProdTaken']
else:
    X_test = df_test
    y_test = None

# ==========================================
# 3. Preprocessing
# ==========================================
# Since class5_test_data.csv comes from the training split of class5data_cleaned2.csv,
# it is already One-Hot Encoded. We just need to ensure column alignment and scaling.

# Align columns (Critical step for robustness)
# 1. Add missing columns with 0
for col in model_columns:
    if col not in X_test.columns:
        X_test[col] = 0

# 2. Drop extra columns not in model
X_test = X_test[model_columns]

# 3. Scale features using the loaded scaler
# Note: We use the scaler fitted on training data, which is best practice
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. Prediction
# ==========================================
print("\nMaking predictions...")
y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# ==========================================
# 5. Evaluation & Visualization
# ==========================================
print("\n" + "="*60)
print("INFERENCE RESULTS")
print("="*60)

if y_test is not None:
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    try:
        auc_score = roc_auc_score(y_test, y_pred_probs)
        print(f"AUC Score: {auc_score:.4f}")
    except:
        print("AUC Score: Could not calculate")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Tourism Prediction', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Add stats text box
    tn, fp, fn, tp = cm.ravel()
    stats_text = f'True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}'
    stats_text += f'\n\nAccuracy: {accuracy:.4f}'
    plt.text(2.5, 0.5, stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='center')

    plt.tight_layout()
    output_path = 'output/5assignment5_confusion_matrix.jpg'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix plot saved to: {output_path}")

    # Plot Normalized Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                cbar_kws={'label': 'Percentage'})
    plt.title('Normalized Confusion Matrix - Tourism Prediction', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    output_path_normalized = 'output/5assignment5_confusion_matrix_normalized.jpg'
    plt.savefig(output_path_normalized, dpi=300, bbox_inches='tight')
    print(f"Normalized confusion matrix plot saved to: {output_path_normalized}")

    # Save predictions to CSV
    results_df = X_test.copy()
    results_df['predicted_label'] = y_pred
    results_df['prediction_probability'] = y_pred_probs.flatten()
    results_df['true_label'] = y_test
    results_df['correct_prediction'] = (results_df['true_label'] == results_df['predicted_label'])

    output_csv_path = 'output/assignement5_inference_results.csv'
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to: {output_csv_path}")

else:
    # No ground truth
    print("No ground truth available. Saving predictions only.")
    results_df = X_test.copy()
    results_df['predicted_label'] = y_pred
    results_df['prediction_probability'] = y_pred_probs.flatten()
    output_csv_path = 'output/predictions.csv'
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to: {output_csv_path}")

print("\n" + "="*60)
print("Inference completed successfully!")
print("="*60)