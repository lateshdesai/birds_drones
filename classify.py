import cv2
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MODEL_PATH = r"C:\Birds_Drones\best_model.h5"   # update if needed
IMG_SIZE = (224, 224)

# These should match the order in your training generator
CLASS_NAMES = ["bird", "drone"]

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # shape: (1, 224, 224, 3)

    return img

# ---------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------
def predict(image_path):
    img = preprocess_image(image_path)
    pred_prob = model.predict(img)[0][0]

    if pred_prob > 0.5:
        predicted_class = CLASS_NAMES[1]  # drone
        confidence = pred_prob
    else:
        predicted_class = CLASS_NAMES[0]  # bird
        confidence = 1 - pred_prob

    print("\nPrediction Result:")
    print(f"Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    return predicted_class, confidence


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example image path — change this to test another image:
    image_path = r"C:\Users\Latesh's Acer\Downloads\drone.jpg"

    predict(image_path)
