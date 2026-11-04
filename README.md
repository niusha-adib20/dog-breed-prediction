# Dog Breed Prediction

A MobileNetV2 transfer-learning project that classifies dog-breed images. The repository includes a trained model and a Gradio application for immediate predictions. Retraining is a separate workflow that requires the dataset and pretrained MobileNetV2 weights.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Objectives](#project-objectives)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Dataset Description](#dataset-description)
- [Dataset Source](#dataset-source)
- [Pretrained Weights](#pretrained-weights)
- [Project Structure](#project-structure)
- [Model Training Process](#model-training-process)
- [Model Performance](#model-performance)
- [Prediction Application](#prediction-application)
- [Quick Start: Run the Pretrained Model](#quick-start-run-the-pretrained-model)
- [Installation](#installation)
- [Reproduce Training from Scratch](#reproduce-training-from-scratch)
- [Dataset Setup](#dataset-setup)
- [Pretrained Weights Setup](#pretrained-weights-setup)
- [Existing Model vs. Retraining](#existing-model-vs-retraining)
- [Known Issues](#known-issues)
- [Future Improvements](#future-improvements)

## Project Overview

The project uses a MobileNetV2 image feature extractor with a custom classification head to predict a dog breed from an uploaded image. The trained model and its class-index mapping are stored in `app/`, and `app/app.py` provides a browser-based Gradio interface.

## Project Objectives

- Classify dog images into the classes represented by the trained model.
- Reuse MobileNetV2 ImageNet weights through transfer learning.
- Provide an interactive application that displays a predicted breed, confidence, and the top five class probabilities.
- Keep training materials and the application artifacts organized separately.

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy and Pandas
- Matplotlib and ipympl for notebook plotting support
- Pillow for image loading
- Gradio for the prediction interface
- JupyterLab for running the notebooks

OpenCV and scikit-learn are not imported by the inspected application or notebooks and are not included in `requirements.txt`.

## Model Architecture

The training notebook builds a MobileNetV2 base with input shape `128 × 128 × 3`, `include_top=False`, and locally loaded ImageNet weights. The base is frozen. A custom head applies `GlobalAveragePooling2D`, a 256-unit ReLU dense layer, dropout at `0.5`, and a softmax output layer sized from the training generator's class mapping.

Training uses Adam, categorical cross-entropy, and accuracy as the metric. The image generator rescales pixel values by `1/255`; the training generator also applies random rotations up to 20 degrees, zoom up to 0.2, and horizontal flips. The batch size is 32 and the maximum epoch count is 50. Early stopping monitors validation loss with patience 5 and restores the best weights. ModelCheckpoint monitors validation accuracy and saves the best model to `app/best_dog_breed_model.keras`.

## Dataset Description

The project uses separate `train/`, `valid/`, and `test/` image folders and a `dogs.csv` file. The notebook reads CSV columns named `filepaths`, `labels`, and `data_label`; relative image paths are resolved from `data/`. The advertised dataset contains 70 dog-breed classes. The current project CSV has a label discrepancy described in [Known Issues](#known-issues).

The dataset is not included in this repository. Download it separately if you want to retrain the model.

## Dataset Source

The dataset was downloaded from Kaggle: [70 Dog Breeds Image Data Set](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

## Pretrained Weights

The training notebook loads ImageNet-pretrained MobileNetV2 weights from this file:

`mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5`

[Download the official weights file](https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5).

The weights are used by the training notebook for transfer learning and are excluded from the repository. Place the downloaded file in `weights/` before training. The final dog-breed model is already included under `app/`; the original MobileNetV2 weights are not needed to run the existing Gradio application.

## Project Structure

The actual project layout is:

```text
dogs_breed/
├── app/
│   ├── app.py
│   ├── best_dog_breed_model.keras
│   ├── class_indices.json
│   └── images/
│       ├── dog1.jpg
│       └── dog2.jpg
├── data/                         # Download separately; ignored by Git
│   ├── train/
│   ├── test/
│   ├── valid/
│   └── dogs.csv
├── notebooks/
│   ├── dog_breeds_MobileNetV2_updated.ipynb
│   └── dog_predict_breeds_updated.ipynb
├── weights/                      # Download separately; ignored by Git
│   └── mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
├── image/                        # Additional project images
├── .gitignore
├── README.md
└── requirements.txt
```

## Model Training Process

`notebooks/dog_breeds_MobileNetV2_updated.ipynb` loads `data/dogs.csv`, resolves image paths under `data/`, creates image generators, builds and trains the model, and evaluates it on the test generator. It writes the class-to-index mapping to `app/class_indices.json` and the best checkpoint to `app/best_dog_breed_model.keras`.

The class mapping comes from `train_data.class_indices`. The application reverses that mapping to convert the model's predicted class index into a breed name.

## Model Performance

The notebook contains code to calculate test loss and test accuracy, but the saved training notebook has no execution outputs. Training accuracy, validation accuracy, test accuracy, and corresponding losses are therefore not documented in the available notebook results. The prediction notebook's saved output is a single-image demonstration, not an evaluation metric. No precision, recall, or F1-score is reported.

## Prediction Application

`app/app.py` loads `best_dog_breed_model.keras` and `class_indices.json`, preprocesses the uploaded PIL image, and calls the model for a prediction. The interface shows the predicted breed, the top-class confidence as a percentage, and up to five class probabilities. Two example images are loaded from `app/images/`.

The application uses paths relative to its current working directory. Run it from inside `app/` so the model, class map, and sample-image paths resolve correctly.

## Quick Start: Run the Pretrained Model

This is the default and simplest workflow. The trained model, class mapping, application, and example images are included. You do not need to download the dataset or the original MobileNetV2 weights, and no retraining is required.

1. Clone the repository using its GitHub URL, then change into the cloned project directory:

   ```bash
   git clone <repository-url>
   cd dogs_breed
   ```

2. Install the dependencies as described in [Installation](#installation).
3. Start the app from its own directory so its relative model, class-map, and example-image paths resolve:

   ```bash
   cd app
   python app.py
   ```

4. Open the local URL printed by Gradio in your browser.
5. Upload a dog image to see the predicted breed, confidence, and top predictions.

The working directory must be `app/` because the current application uses relative paths. Replace `<repository-url>` with the URL of the published repository.

## Installation

Python 3.10 or 3.11 is recommended for the pinned TensorFlow/Keras versions in `requirements.txt`.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`; on macOS/Linux, use `source .venv/bin/activate`.

## Reproduce Training from Scratch

1. Install the project dependencies.
2. Download the dataset from the Kaggle link above and arrange it as described in [Dataset Setup](#dataset-setup).
3. Download the MobileNetV2 weights and place them as described in [Pretrained Weights Setup](#pretrained-weights-setup).
4. From the project root, start JupyterLab:

   ```bash
   jupyter lab
   ```

5. Open `notebooks/dog_breeds_MobileNetV2_updated.ipynb` and run its cells in order.
6. The notebook generates `app/class_indices.json` and `app/best_dog_breed_model.keras`. Those are the files used by `app/app.py`.

Training may take substantial time and compute, depending on the machine. This workflow uses the existing training configuration; it does not guarantee a particular accuracy.

## Dataset Setup

The training notebook expects the following paths relative to the project root:

```text
data/
├── train/
│   └── <class-name>/<image-file>
├── test/
│   └── <class-name>/<image-file>
├── valid/
│   └── <class-name>/<image-file>
└── dogs.csv
```

Place the extracted training, test, and validation image folders directly inside `data/`, and place `dogs.csv` in the same `data/` directory. The CSV must have `filepaths`, `labels`, and `data_label` columns. Its relative file paths should begin with the split directory, for example `train/Afghan/001.jpg`; the notebook resolves those paths under `data/`. Do not add an extra archive-wrapper directory between `data/` and these folders.

The current `.gitignore` excludes all of `data/`, including `dogs.csv`, so these files must be supplied locally for retraining and will not be committed.

## Pretrained Weights Setup

Create `weights/` at the project root if it does not exist. Download the official file linked above and place it at:

```text
weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
```

The training notebook checks for this exact file. The `.gitignore` excludes the entire `weights/` directory.

## Existing Model vs. Retraining

| Requirement | Run Existing Model | Retrain Model |
|---|---|---|
| Dataset download | Not required | Required |
| Pretrained weights | Not required | Required by the training notebook |
| Final trained model | Already included | Generated by the notebook |
| Class mapping | Already included | Generated by the notebook |
| Gradio application | Ready to run | Available after training outputs are generated |

## Known Issues

- **Unresolved class-label discrepancy:** the current CSV contains 71 distinct label strings, while the saved class mapping and model have 70 classes. The extra label, `American  Spaniel` (with two spaces), appears in the validation split. The notebook does not explicitly reconcile this split-only label with the 70 training classes, so validation/evaluation may fail due to a class-output mismatch. This should be corrected before retraining or treating validation results as reliable.
- **Training/inference preprocessing differs:** the training notebook rescales inputs to `[0, 1]`, while the prediction code uses MobileNetV2 `preprocess_input` on the image. These transformations are not the same. The model's reported performance should not be assumed to describe the current application preprocessing until this is reviewed.
- **No verified performance metrics are saved:** the training notebook's outputs are empty, so no test accuracy or loss can be cited from the repository state.
- **Application working directory matters:** run `app.py` from the `app/` directory as shown above; launching it from the repository root leaves the current relative artifact paths unresolved.

## Future Improvements

- Reconcile the extra validation label and ensure all splits use the training class mapping.
- Use the same MobileNetV2 preprocessing convention for training and inference, then retrain and evaluate.
- Save and document training, validation, and test metrics from a reproducible run.
- Make application artifact paths independent of the current working directory.
- Add a lightweight automated check for missing dataset files and class consistency.
