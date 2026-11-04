import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.utils import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
import json
import gradio as gr
from PIL import Image

# --- Configuration ---
IMG_SIZE = 128
NUM_CLASSES = 70
model_path = 'best_dog_breed_model.keras'

# --- Load class names ---
class_names_mapping = None
try:
    with open('class_indices.json','r',encoding='utf-8') as f:
        class_indices = json.load(f)
        class_names_mapping = {v: k for k, v in class_indices.items()}
    print("Class names loaded from class_indices.json")
except FileNotFoundError:
    print("Error: class_indices.json not found.")
    print("Please provide class names manually.")

# --- Load model ---
try:
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Preprocess a PIL image for Gradio ---
def preprocess_pil_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# --- Gradio prediction function ---
def gradio_predict(image):

    if image is None:
        return "No image provided", None

    processed_image = preprocess_pil_image(image)

    predictions = model.predict(processed_image)[0]

    if class_names_mapping:

        predicted_class_index = int(np.argmax(predictions))
        predicted_breed = class_names_mapping.get(predicted_class_index, "Unknown Breed")

        confidence = float(np.max(predictions) * 100)

        # Top five predictions
        top_5_indices = np.argsort(predictions)[-5:][::-1]

        top_results = {
            class_names_mapping.get(int(i), "Unknown Breed"): float(predictions[i])
            for i in top_5_indices
        }

        result_text = f"{predicted_breed}"

        return result_text,f"{confidence:.2f}%", top_results

    else:
        return "Class names not available", None


# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue='purple'), title="Dog Breed Classifier") as app:

    gr.Markdown("# 🐶 Dog Breed Classifier")
    gr.Markdown("Upload a dog image and the AI will predict its breed.")

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Dog Image"
            )

            predict_btn = gr.Button(
                "Predict Breed",
                variant="primary"
            )

            clear_btn = gr.Button("Clear")

        with gr.Column():

            breed_output = gr.Textbox(
                label="Predicted Breed"
            )

            confidence_output = gr.Textbox(
                label="Confidence"
            )

            probabilities = gr.Label(
                num_top_classes=5,
                label="Top Predictions"
            )

    predict_btn.click(
        fn=gradio_predict,
        inputs=image_input,
        outputs=[breed_output, confidence_output, probabilities]
    )

    clear_btn.click(
        lambda: (None, "", "", None),
        None,
        [image_input, breed_output, confidence_output, probabilities]
    )

    gr.Examples(
        examples=[
            "images/dog1.jpg",
            "images/dog2.jpg"
        ],
        inputs=image_input
    )


app.launch()