import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

# ==========================================
# 0. Helper Function: Split Data
# ==========================================
def split_csv_data(input_file, sample_output, remainder_output, fraction=0.10, random_state=42):
    """
    Reads a CSV, randomly extracts a fraction (default 10%) to a new file,
    and saves the remaining data to a separate file.
    """
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found.")
        return

    try:
        print(f"Reading {input_file} for splitting...")
        df_full = pd.read_csv(input_file)
        
        # Randomly sample the specified fraction
        sample_df = df_full.sample(frac=fraction, random_state=random_state)
        
        # Create the remainder
        remaining_df = df_full.drop(sample_df.index)
        
        # Save files
        sample_df.to_csv(sample_output, index=False)
        print(f"Success! {len(sample_df)} rows ({fraction*100:.1f}%) saved to '{sample_output}'")
        
        remaining_df.to_csv(remainder_output, index=False)
        print(f"Success! {len(remaining_df)} rows saved to '{remainder_output}'")
        
    except Exception as e:
        print(f"An error occurred during splitting: {e}")

# ==========================================
# 1. Prepare Data (Split off Test Set)
# ==========================================
# Define file paths
source_file = '/opt/project/data/class5data_cleaned2.csv'
if not os.path.exists(source_file):
    # Fallback if running locally and file is in current folder
    source_file = 'class5data_cleaned2.csv'

test_file_name = '/opt/project/data/class5_test_data.csv'
remaining_file_name = '/opt/project/data/class5_remaining_data.csv'

# RUN THE SPLIT
# This creates the "unseen" test data and the "remaining" training data
split_csv_data(
    input_file=source_file, 
    sample_output=test_file_name, 
    remainder_output=remaining_file_name, 
    fraction=0.10,
    random_state=42
)

# ==========================================
# 2. Load TRAINING Data
# ==========================================
print("\nLoading remaining data for training...")
# KEY CHANGE: We load the 'remaining' file, not the original source
df = pd.read_csv(remaining_file_name)

# ==========================================
# 3. Preprocessing
# ==========================================
# Drop identifier column if it exists
if 'Index' in df.columns:
    df = df.drop('Index', axis=1)

# Drop 'ProductPitched' if it exists
if 'ProductPitched' in df.columns:
    df = df.drop('ProductPitched', axis=1)

# Define Target and Features
if 'ProdTaken' in df.columns:
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']
else:
    print("Error: Target column 'ProdTaken' not found.")
    exit()

# Ensure numeric
X = X.astype(float)
print(f"Features: {X.shape[1]}")

# ==========================================
# 4. Train/Validation Split
# ==========================================
# We split the REMAINING data. 
# 80% used for Training, 20% used for internal Evaluation (Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ==========================================
# 5. Scaling
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler and Column Names for Inference consistency
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')
print("Saved scaler and column list.")

# ==========================================
# 6. Handle Imbalance (Class Weights)
# ==========================================
# This calculates how much to penalize the model for missing a buyer
#weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#class_weights_dict = dict(enumerate(weights))
#print(f"Calculated Class Weights: {class_weights_dict}")

# ==========================================
# 7. Build Keras Model
# ==========================================
model = keras.Sequential([
    # Input Layer
    keras.layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dropout(0.3), 

    # Hidden Layers

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.25),
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),

    # Output Layer
    keras.layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 8. Train
# ==========================================
print("\nTraining...")
# validation_split here splits the Training set (from step 4) further for epoch validation
history = model.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=64,
    validation_split=0.2, 
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='loss', 
            mode='min',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='accuracy',
            mode='max',
            factor=0.5,
            patience=7,
            min_lr=0.00001,
            verbose=1
        ),
    ]
)

# ==========================================
# 9. Evaluate (Internal Validation)
# ==========================================
print("\nEvaluating on Validation Set (part of the 90%)...")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_probs = model.predict(X_test_scaled)
y_pred = (y_pred_probs > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 10. Save Model
# ==========================================
if not os.path.exists('examples'):
    os.makedirs('examples')
    
model.save('examples/tourism_model.keras')
print("Model saved as 'tourism_model.keras'")