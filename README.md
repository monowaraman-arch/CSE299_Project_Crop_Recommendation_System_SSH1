# Crop Recommendation System

## Overview
This project is a machine learning based crop recommendation system built with Python, Flask, Pandas, and scikit-learn. It predicts the most suitable crop from soil nutrient and environmental values entered by the user.

The application uses these 7 input features:
- Nitrogen (`N`)
- Phosphorus (`P`)
- Potassium (`K`)
- Temperature
- Humidity
- pH
- Rainfall

The predicted result is one of 22 crop classes, including Rice, Maize, Mango, Coffee, Coconut, and others.

## Features
- Flask web interface for entering field conditions
- Trained model loaded from saved `.pkl` artifacts
- Prediction result shown with a crop-specific image when available
- Form validation for required numeric values
- Automatic rebuilding of model artifacts from the CSV dataset if saved pickles are incompatible with the installed scikit-learn version

## Tech Stack
- Python
- Flask
- NumPy
- Pandas
- scikit-learn
- Bootstrap 5

## Main Files
- `app.py`: Flask app, model loading, preprocessing, prediction, and image lookup
- `templates/index.html`: Frontend form and prediction result UI
- `Crop Classification With Recommendation System.ipynb`: Notebook for dataset analysis, preprocessing, model comparison, and training
- `Crop_recommendation.csv`: Dataset used for training and evaluation
- `model.pkl`: Saved trained classifier
- `minmaxscaler.pkl`: Saved `MinMaxScaler`
- `standscaler.pkl`: Saved `StandardScaler`
- `static/crops/`: Crop images used by the UI

## Dataset Summary
The dataset in `Crop_recommendation.csv` contains:
- `2200` rows
- `8` columns
- `7` input features
- `1` target column: `label`
- `22` crop classes

The current dataset also appears to have:
- no missing values
- no duplicate rows
- balanced classes with `100` samples per crop

## Machine Learning Workflow
The notebook covers:
- dataset inspection
- class distribution and feature analysis
- label encoding
- feature scaling with `MinMaxScaler` and `StandardScaler`
- train/test split
- classifier comparison
- saving trained artifacts for deployment

Models compared in the notebook include:
- Logistic Regression
- Gaussian Naive Bayes
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Bagging
- AdaBoost
- Gradient Boosting

The deployed application rebuilds artifacts with a `RandomForestClassifier` when the existing pickle files cannot be loaded safely.

## Project Structure
```text
.
|-- app.py
|-- Crop Classification With Recommendation System.ipynb
|-- Crop_recommendation.csv
|-- model.pkl
|-- minmaxscaler.pkl
|-- standscaler.pkl
|-- requirements.txt
|-- README.md
|-- templates/
|   `-- index.html
`-- static/
    `-- crops/
```

## Installation
### Windows PowerShell
Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `python` is not available on your PATH, you can use the virtual environment interpreter directly after creating `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Application
Start the Flask app with:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open:

```text
http://127.0.0.1:5001
```

## Crop Images
Crop result images are loaded from `static/crops/`.

Supported file extensions:
- `.png`
- `.jpg`
- `.jpeg`
- `.webp`

Expected crop image base names in the app:
- `rice`
- `maize`
- `jute`
- `cotton`
- `coconut`
- `papaya`
- `orange`
- `apple`
- `muskmelon`
- `watermelon`
- `grapes`
- `mango`
- `banana`
- `pomegranate`
- `lentil`
- `blackgram`
- `mungbean`
- `mothbeans`
- `pigeonpeas`
- `kidneybeans`
- `chickpea`
- `coffee`

Notes:
- The repository currently includes many crop images in `static/crops/`, but not every expected filename is present.
- The folder currently contains `mothbean.png`, while the app looks for `mothbeans.*` for that crop class.
- The code supports a `default` fallback image, but this repository does not currently include a `default.png`, `default.jpg`, `default.jpeg`, or `default.webp` file.

## Notes
- The web app predicts one crop label from 7 user inputs.
- `app.py` runs with `debug=True`, which is fine for local development but should be disabled before deployment.
- The page also uses remote assets for Google Fonts, Bootstrap CDN, and background images, so those visuals depend on internet access.

## Requirements
Dependencies are listed in `requirements.txt`:
- `Flask>=3.0,<4.0`
- `numpy>=1.26,<3.0`
- `pandas>=2.2,<3.0`
- `scikit-learn>=1.4,<2.0`

## Contact
For project-related questions:
- `611noorsaeed@gmail.com`
