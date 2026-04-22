import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Crop_recommendation.csv"
STATIC_CROPS_DIR = BASE_DIR / "static" / "crops"


FIELD_NAMES = {
    "Nitrogen": "Nitrogen",
    "Phosporus": "Phosphorus",
    "Potassium": "Potassium",
    "Temperature": "Temperature",
    "Humidity": "Humidity",
    "Ph": "pH",
    "Rainfall": "Rainfall",
}

CROP_DICT = {
    1: "Rice",
    2: "Maize",
    3: "Jute",
    4: "Cotton",
    5: "Coconut",
    6: "Papaya",
    7: "Orange",
    8: "Apple",
    9: "Muskmelon",
    10: "Watermelon",
    11: "Grapes",
    12: "Mango",
    13: "Banana",
    14: "Pomegranate",
    15: "Lentil",
    16: "Blackgram",
    17: "Mungbean",
    18: "Mothbeans",
    19: "Pigeonpeas",
    20: "Kidneybeans",
    21: "Chickpea",
    22: "Coffee",
}

CROP_IMAGE_BASENAMES = {
    "Rice": "rice",
    "Maize": "maize",
    "Jute": "jute",
    "Cotton": "cotton",
    "Coconut": "coconut",
    "Papaya": "papaya",
    "Orange": "orange",
    "Apple": "apple",
    "Muskmelon": "muskmelon",
    "Watermelon": "watermelon",
    "Grapes": "grapes",
    "Mango": "mango",
    "Banana": "banana",
    "Pomegranate": "pomegranate",
    "Lentil": "lentil",
    "Blackgram": "blackgram",
    "Mungbean": "mungbean",
    "Mothbeans": "mothbeans",
    "Pigeonpeas": "pigeonpeas",
    "Kidneybeans": "kidneybeans",
    "Chickpea": "chickpea",
    "Coffee": "coffee",
}

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL_TO_CLASS = {
    "rice": 1,
    "maize": 2,
    "jute": 3,
    "cotton": 4,
    "coconut": 5,
    "papaya": 6,
    "orange": 7,
    "apple": 8,
    "muskmelon": 9,
    "watermelon": 10,
    "grapes": 11,
    "mango": 12,
    "banana": 13,
    "pomegranate": 14,
    "lentil": 15,
    "blackgram": 16,
    "mungbean": 17,
    "mothbeans": 18,
    "pigeonpeas": 19,
    "kidneybeans": 20,
    "chickpea": 21,
    "coffee": 22,
}


def load_pickle(filename):
    with open(BASE_DIR / filename, "rb") as file:
        return pickle.load(file)


def save_pickle(filename, obj):
    with open(BASE_DIR / filename, "wb") as file:
        pickle.dump(obj, file)


def build_artifacts_from_dataset():
    dataset = pd.read_csv(DATASET_PATH)
    features = dataset[FEATURE_COLUMNS].copy()
    labels = dataset["label"].str.strip().str.lower().map(LABEL_TO_CLASS)

    minmax_scaler = MinMaxScaler()
    scaled_features = minmax_scaler.fit_transform(features)

    standard_scaler = StandardScaler()
    normalized_features = standard_scaler.fit_transform(scaled_features)

    trained_model = RandomForestClassifier(random_state=42)
    trained_model.fit(normalized_features, labels)

    save_pickle("model.pkl", trained_model)
    save_pickle("minmaxscaler.pkl", minmax_scaler)
    save_pickle("standscaler.pkl", standard_scaler)

    return trained_model, minmax_scaler, standard_scaler


def artifacts_work(trained_model, minmax_scaler, standard_scaler):
    sample = pd.DataFrame(
        [
            {
                "N": 90.0,
                "P": 42.0,
                "K": 43.0,
                "temperature": 20.87974371,
                "humidity": 82.00274423,
                "ph": 6.502985292,
                "rainfall": 202.9355362,
            }
        ]
    )

    transformed = minmax_scaler.transform(sample)
    transformed = standard_scaler.transform(transformed)
    trained_model.predict(transformed)
    return True


def load_or_train_artifacts():
    try:
        trained_model = load_pickle("model.pkl")
        minmax_scaler = load_pickle("minmaxscaler.pkl")
        standard_scaler = load_pickle("standscaler.pkl")
        artifacts_work(trained_model, minmax_scaler, standard_scaler)
        return trained_model, minmax_scaler, standard_scaler
    except Exception:
        return build_artifacts_from_dataset()


model, ms, scaler_after_minmax = load_or_train_artifacts()


app = Flask(__name__)


def parse_form_values(form):
    values = {}
    for field_name in FIELD_NAMES:
        raw_value = form.get(field_name, "").strip()
        if not raw_value:
            raise ValueError(f"{FIELD_NAMES[field_name]} is required.")
        values[field_name] = float(raw_value)
    return values


def predict_crop(values):
    features = pd.DataFrame(
        [
            {
                "N": values["Nitrogen"],
                "P": values["Phosporus"],
                "K": values["Potassium"],
                "temperature": values["Temperature"],
                "humidity": values["Humidity"],
                "ph": values["Ph"],
                "rainfall": values["Rainfall"],
            }
        ],
    )

    transformed = ms.transform(features)
    if scaler_after_minmax is not None:
        transformed = scaler_after_minmax.transform(transformed)

    prediction = model.predict(transformed)[0]

    if isinstance(prediction, str):
        return prediction.title()
    if prediction in CROP_DICT:
        return CROP_DICT[prediction]

    raise ValueError("The trained model returned an unknown crop class.")


def get_crop_image_data(crop_name):
    basename = CROP_IMAGE_BASENAMES.get(crop_name, "default")
    filename = None

    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        candidate = f"{basename}{extension}"
        if (STATIC_CROPS_DIR / candidate).exists():
            filename = candidate
            break

    if filename is None:
        for extension in SUPPORTED_IMAGE_EXTENSIONS:
            candidate = f"default{extension}"
            if (STATIC_CROPS_DIR / candidate).exists():
                filename = candidate
                break

    if filename is None:
        filename = "default.jpg"

    return {
        "image_url": url_for("static", filename=f"crops/{filename}"),
        "is_default": filename.startswith("default."),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        form_values={},
        result=None,
        result_crop=None,
        result_image_url=None,
        result_image_is_default=False,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_values = {field_name: request.form.get(field_name, "") for field_name in FIELD_NAMES}

    try:
        parsed_values = parse_form_values(request.form)
        crop_name = predict_crop(parsed_values)
    except ValueError as exc:
        return render_template(
            "index.html",
            form_values=form_values,
            result=None,
            result_crop=None,
            result_image_url=None,
            result_image_is_default=False,
            error=str(exc),
        )
    except Exception:
        return render_template(
            "index.html",
            form_values=form_values,
            result=None,
            result_crop=None,
            result_image_url=None,
            result_image_is_default=False,
            error="Prediction failed. Check that the saved model and scaler files match the training pipeline.",
        )

    result = f"{crop_name} is the best crop to be cultivated right there."
    image_data = get_crop_image_data(crop_name)
    return render_template(
        "index.html",
        form_values=form_values,
        result=result,
        result_crop=crop_name,
        result_image_url=image_data["image_url"],
        result_image_is_default=image_data["is_default"],
        error=None,
    )


if __name__ == "__main__":
    app.run(debug=True)
