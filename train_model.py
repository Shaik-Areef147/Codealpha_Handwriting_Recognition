"""
train_model.py
--------------
Trains a CNN on MNIST (digits 0-9) and saves model.h5
Run this ONCE locally before deploying, then commit model.h5 to Git.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

print("TensorFlow version:", tf.__version__)

# ─── 1. Load and preprocess MNIST ────────────────────────────────────────────
print("\n[1/4] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1] and add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encode labels  (10 classes: digits 0-9)
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

print(f"    Training samples : {x_train.shape[0]}")
print(f"    Test samples     : {x_test.shape[0]}")
print(f"    Image shape      : {x_train.shape[1:]}")

# ─── 2. Build CNN ─────────────────────────────────────────────────────────────
print("\n[2/4] Building CNN model...")

model = models.Sequential([
    # Block 1 – learn low-level features (edges, corners)
    layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                  input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2 – learn higher-level features (curves, strokes)
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 3 – deep feature extraction
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    # Classifier head
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")   # 10 digit classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ─── 3. Train ─────────────────────────────────────────────────────────────────
print("\n[3/4] Training model (this takes ~3-5 minutes)...")

# Callbacks: reduce LR on plateau + early stopping
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
)

history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[lr_scheduler, early_stop],
    verbose=1
)

# ─── 4. Evaluate & Save ───────────────────────────────────────────────────────
print("\n[4/4] Evaluating and saving model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n    Test Loss     : {test_loss:.4f}")
print(f"    Test Accuracy : {test_acc * 100:.2f}%")

model.save("model.h5")
print("\n    model.h5 saved successfully!")
print("    Next step: commit model.h5 to Git and push to GitHub.")
