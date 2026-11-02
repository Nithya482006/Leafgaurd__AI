# LeafGuard AI — Plant Disease Identification with CNNs

A portfolio-ready Flask web application for screening leaf images with a convolutional neural network. It uses transfer learning with MobileNetV2, stores scan history in SQLite, and provides cautious next-step guidance.

> **Important:** This project copy includes a trained model and class names under `models/`. The included model was only partially trained and reached 55.5% validation accuracy; predictions may be unreliable and it has not been evaluated on the held-out test split. Predictions are educational screening aids, not a definitive diagnosis.

## Features

- Polished responsive web interface with leaf image preview and drag-and-drop upload
- CNN image classifier using TensorFlow/Keras and MobileNetV2 transfer learning
- Confidence score and class label
- Treatment / follow-up suggestions for selected classes and safe general guidance otherwise
- SQLite scan history
- Input validation for file type and upload size
- `/health` endpoint for checking whether the model is loaded

## Architecture

```text
Browser (HTML/CSS/JavaScript)
        |
        | multipart image upload
        v
Flask app (app.py) ---- SQLite (plantguard.db)
        |
        v
Image preprocessing (RGB, 224x224, normalized)
        |
        v
Trained MobileNetV2 CNN (models/plant_disease_model.keras)
        |
        v
Class + confidence ----> Recommendations ----> Result page
```

## 1. Requirements

- Python 3.10–3.12 recommended for TensorFlow compatibility. Check the TensorFlow install page if you use another Python version.
- 8 GB RAM recommended; a GPU is optional.
- Internet connection for installing dependencies and downloading pretrained MobileNetV2 weights during training.

Python 3.13 may not be supported by every TensorFlow release. If pip cannot install TensorFlow in your current Python, install Python 3.11 and create the environment with that interpreter.

## 2. Open the project

Extract the ZIP. In VS Code, open the `leafguard_ai` folder. Open Terminal in that folder.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Get a plant leaf dataset

Use the PlantVillage dataset or another dataset you are permitted to use. Check the dataset's license and citation requirements. PlantVillage is commonly used for educational plant-disease classification, but images taken in controlled conditions may not reflect real field conditions.

Arrange the images into class-named folders:

```text
dataset/
  train/
    Apple___Apple_scab/
      image1.jpg
    Apple___healthy/
      image2.jpg
    Tomato___Early_blight/
      image3.jpg
    Tomato___healthy/
      image4.jpg
  valid/                 # optional but recommended
    Apple___Apple_scab/
    Apple___healthy/
    Tomato___Early_blight/
    Tomato___healthy/
```

Each class folder should contain images of that class. If you have only one folder per class, put those class folders inside `dataset/train/`; the script creates a 15% validation split automatically. For a stronger evaluation, keep a separate test set that is not used for training or tuning.

## 4. Train the CNN

With the virtual environment active, run:

```bash
python train.py --data dataset --epochs 8 --fine-tune-epochs 3
```

Training may take a while depending on dataset size and computer performance. For a first smoke test, try fewer epochs, for example `--epochs 1 --fine-tune-epochs 0`.
To deliberately limit each epoch to a fixed number of batches (for example, on a CPU-only machine), add `--steps-per-epoch 863`. The model is saved after the shortened epoch completes; this is a partial training run, not a fully trained or independently evaluated model.

When training completes, these files are created:

- `models/plant_disease_model.keras`
- `models/class_names.json`

The class names are written in the exact order used by the training dataset. Do not manually reorder them.

## 5. Start the web app

From the `leafguard_ai` folder, run this single command in Windows PowerShell:

```powershell
py run.py
```

The launcher uses the project `.venv` when available, otherwise the existing `%LOCALAPPDATA%\LeafGuardAI\venv` environment. It starts the local server with the included model. Keep the terminal open while using the app; press Ctrl+C to stop it.

To start the app directly with an already activated environment, you can still run:

```bash
python app.py
```

Open the local address printed in the terminal (usually `http://127.0.0.1:5000`). Upload a clear leaf photo. After a successful prediction, the result page shows the predicted class, confidence, suggested next steps, and a record is added to the History page.

## 6. Check the app

Open `http://127.0.0.1:5000/health`. It reports whether the model is available. If it says model setup is needed, check that both model files exist under `models/`.

## Troubleshooting

- **TensorFlow install fails:** Use Python 3.11 in a fresh virtual environment. TensorFlow wheels depend on your operating system and Python version.
- **Model setup needed:** Train the model and check both files in `models/`.
- **Model deserialization mentions `GlorotUniform` axes:** The app handles newer Keras archives that add null `input_axes` or `output_axes` initializer settings. If loading still fails, use compatible TensorFlow/Keras versions for the model and app environment.
- **Out of memory / slow training:** Lower batch size in `train.py` (for example, change `BATCH_SIZE` or add a CLI option), reduce dataset size for a smoke test, or train on a machine with more memory.
- **Low accuracy:** Check class balance, image labels, lighting/background variation, and class mapping. Use a held-out test set and report precision, recall, F1-score, and a confusion matrix rather than only training accuracy.
- **Prediction seems wrong:** Models can fail on unfamiliar crops, backgrounds, multiple leaves, shadows, and diseases absent from training data. Confirm with an expert.

## Responsible use

- The classifier only knows the classes in its training dataset. It cannot reliably identify a disease outside those classes.
- Confidence is the model's score, not a calibrated probability that the diagnosis is correct.
- Do not use the app as the sole basis for pesticide or fungicide decisions. Product availability, approved uses, dosage, and safety rules vary by region and crop.
- The treatment notes are broad educational suggestions. Confirm disease and management with a local agricultural extension officer or qualified agronomist.
- Before deploying publicly, add authentication, CSRF protection, rate limits, upload cleanup, privacy controls, production configuration, and a proper model evaluation report.

## Suggested portfolio improvements

1. Add a separate held-out test evaluation and confusion matrix.
2. Show top-3 predictions rather than only the top class.
3. Add Grad-CAM visual explanations to highlight image regions influencing the CNN.
4. Add user accounts and downloadable PDF scan reports.
5. Compare MobileNetV2 with a custom CNN and report latency, model size, and test-set F1-score.
