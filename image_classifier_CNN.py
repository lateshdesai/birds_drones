import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -------------------------------------------------------------
# 1. PATH SETUP
# -------------------------------------------------------------
DATA_DIR = r"C:\Birds_Drones\Data"
train_dir = os.path.join(DATA_DIR, "train")
valid_dir = os.path.join(DATA_DIR, "valid")
test_dir = os.path.join(DATA_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -------------------------------------------------------------
# 2. Dataset Inspection: Count images per class
# -------------------------------------------------------------
def count_images(directory):
    counts = {}
    for cls in os.listdir(directory):
        cls_path = os.path.join(directory, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len(os.listdir(cls_path))
    return counts

print("\n--- IMAGE COUNTS ---")
print("Train:", count_images(train_dir))
print("Valid:", count_images(valid_dir))
print("Test:", count_images(test_dir))

# Detect imbalance
train_counts = count_images(train_dir)
print("\nClass Imbalance Check:")
for cls, num in train_counts.items():
    print(f"{cls}: {num} images")

# -------------------------------------------------------------
# Visualize sample images
# -------------------------------------------------------------
def show_sample_images(directory):
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))
    classes = os.listdir(directory)

    for i, cls in enumerate(classes[:2]):  # show first 2 classes
        img_name = os.listdir(os.path.join(directory, cls))[0]
        img_path = os.path.join(directory, cls, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(img)
        axes[i].set_title(cls)
        axes[i].axis("off")
    plt.show()

print("\nShowing sample images from TRAIN set...")
show_sample_images(train_dir)

# -------------------------------------------------------------
# 3. DATA AUGMENTATION + NORMALIZATION
# -------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=25,
    zoom_range=0.20,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# -------------------------------------------------------------
# 4. CNN MODEL DEFINITION
# -------------------------------------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------------------------
# 5. CALLBACKS: EarlyStopping + ModelCheckpoint
# -------------------------------------------------------------
checkpoint_path = "best_model.h5"

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# -------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=25,
    callbacks=[early_stop, checkpoint]
)

# -------------------------------------------------------------
# 6. Save Accuracy/Loss Plots
# -------------------------------------------------------------
plt.figure(figsize=(10,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

print("\nSaved accuracy/loss plots as training_curves.png")

# -------------------------------------------------------------
# 7. TEST EVALUATION
# -------------------------------------------------------------
pred_prob = model.predict(test_generator)
predictions = (pred_prob > 0.5).astype(int)

true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=class_names))

print("\nBest model saved as best_model.h5")
