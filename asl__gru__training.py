import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
import os
import sys
import string
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

LOG_DIR = os.path.join(
    "logs", 
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
os.makedirs(LOG_DIR, exist_ok=True)

# --- GPU Setup ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Use only the first GPU and enable memory growth
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"--- GPU Detected! Training on: {gpus[0].name} ---")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("--- GPU NOT Detected. Training will use CPU. ---")

# --- Model Parameters ---
SEQUENCE_LEN = 30
MAX_HANDS = 2
LANDMARK_COUNT = 21
FEATURES_PER_HAND = LANDMARK_COUNT * 2
FEATURES_PER_FRAME = FEATURES_PER_HAND * MAX_HANDS # 84 Features
TOTAL_INPUT_FEATURES = SEQUENCE_LEN * FEATURES_PER_FRAME # 2520

# Sign Labels
SIGN_LABELS = list(string.ascii_uppercase) + ['HELLO', 'GOODBYE', 'I_LOVE_YOU']
NUM_CLASSES = len(SIGN_LABELS)

# File Paths
DATA_DIR = 'raw_data_2hand'
OUTPUT_CSV = 'asl_sequence_data_2hand.csv'
DATA_PATH = os.path.join(DATA_DIR, OUTPUT_CSV)
MODEL_SAVE_PATH = "model/asl_dynamic_gru_classifier_2hand.keras"


# --- Data Loading and Preprocessing ---

print("\n--- Loading and preprocessing sequence data for GRU training ---")

try:
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Training data file not found at {DATA_PATH}.")
        print("Please run the data collection script first.")
        sys.exit(1)

    data_df = pd.read_csv(DATA_PATH)
    data_df = data_df[data_df['label'] < NUM_CLASSES]

    # Separate labels and features
    y_labels = data_df['label'].values.astype(np.int32)
    X_features = data_df.iloc[:, 1:].values.astype(np.float32)

    if X_features.shape[1] != TOTAL_INPUT_FEATURES:
        print(f"Error: Feature count mismatch. Expected {TOTAL_INPUT_FEATURES}, got {X_features.shape[1]}.")
        sys.exit(1)

except Exception as e:
    print(f"Fatal Error during data loading: {e}")
    sys.exit(1)

# Split the data into training, validation, and test sets (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_features, y_labels, test_size=0.4, random_state=42, stratify=y_labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Reshape Data for GRU: (samples, sequence_length, features_per_step) -> (None, 30, 84)
X_train_seq = X_train.reshape((-1, SEQUENCE_LEN, FEATURES_PER_FRAME))
X_val_seq = X_val.reshape((-1, SEQUENCE_LEN, FEATURES_PER_FRAME))
X_test_seq = X_test.reshape((-1, SEQUENCE_LEN, FEATURES_PER_FRAME))

print(f"Data ready. Training samples: {len(X_train_seq)}, Validation: {len(X_val_seq)}, Test: {len(X_test_seq)}")


# --- Model Definition ---

def create_sequence_classifier(num_classes):
    """
    Constructs the Sequential model for dynamic sequence classification using GRU layers.
    """
    model = Sequential([
        # GRU layer to process the 30-frame sequence
        GRU(128, return_sequences=False, input_shape=(SEQUENCE_LEN, FEATURES_PER_FRAME), name='sequence_processor'),

        Dropout(0.4, name='drop_seq'),

        Dense(64, activation='relu', name='intermediate_dense'),

        Dropout(0.4, name='drop_dense'),

        # Output Layer: Softmax activation for probability distribution over classes.
        Dense(num_classes, activation='softmax', name='sign_output')
    ], name="asl_2hand_dynamic_gru_classifier")
    return model

model = create_sequence_classifier(NUM_CLASSES)
model.summary()


# --- Training Configuration and Execution ---

# Ensure the model save directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Callbacks for robust training
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_loss', save_best_only=True, verbose=1, mode='min'
)

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, verbose=1, restore_best_weights=True
)
tb_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n--- Model compiled. Initiating training for {NUM_CLASSES} classes ---")

EPOCHS = 100
BATCH_SIZE = 32

history = model.fit(
    X_train_seq, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_seq, y_val),
    callbacks=[cp_callback, es_callback, tb_callback],
    verbose=2 # Prints one line per epoch
)

print("\n--- Sequence Training Concluded ---")

# Evaluate on the unseen test set
loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
print(f"Final Test Loss: {loss:.4f}")
print(f"Final Test Accuracy: {acc:.4f}")

# --- Utility Code for Plotting (Commented out by default) ---

"""
# Create results directory
os.makedirs('results', exist_ok=True)

# ACCURACY PLOT GENERATION
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('results/figure_5_accuracy.png')

# LOSS PLOT GENERATION
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('results/figure_5_loss.png')

# CONFUSION MATRIX GENERATION
y_pred_probs = model.predict(X_test_seq, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
class_labels = SIGN_LABELS

plt.figure(figsize=(18, 16))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

disp.plot(cmap=plt.cm.viridis, xticks_rotation=90, values_format='d', ax=plt.gca())
plt.title(f'Figure 6: Confusion Matrix for {NUM_CLASSES} Sign Classes')
plt.xlabel('Predicted Sign')
plt.ylabel('True Sign')
plt.tight_layout()

plt.savefig('results/figure_6_confusion_matrix.png')
# plt.show() # Uncomment to display plots
"""
