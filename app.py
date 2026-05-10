import json
import pickle
import sqlite3
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
from flask import Flask, Response, has_request_context, render_template, request, url_for
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Crop_recommendation.csv"
STATIC_CROPS_DIR = BASE_DIR / "static" / "crops"
HISTORY_DB_PATH = BASE_DIR / "prediction_history.db"


FIELD_NAMES = {
    "Nitrogen": "Nitrogen",
    "Phosporus": "Phosphorus",
    "Potassium": "Potassium",
    "Temperature": "Temperature",
    "Humidity": "Humidity",
    "Ph": "pH",
    "Rainfall": "Rainfall",
}

FIELD_LIMITS = {
    "Nitrogen": (0.0, 140.0),
    "Phosporus": (5.0, 145.0),
    "Potassium": (5.0, 205.0),
    "Temperature": (8.825674745, 43.67549305),
    "Humidity": (14.25803981, 99.98187601),
    "Ph": (3.504752314, 9.93509073),
    "Rainfall": (20.21126747, 298.5601175),
}

FIELD_UNITS = {
    "Temperature": " C",
    "Humidity": "%",
    "Rainfall": " mm",
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
    "Mothbeans": "mothbean",
    "Pigeonpeas": "pigeonpeas",
    "Kidneybeans": "kidneybeans",
    "Chickpea": "chickpea",
    "Coffee": "coffee",
}

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FEATURE_DISPLAY_NAMES = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "pH",
    "rainfall": "Rainfall",
}
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

CROP_NOTES = {
    "Rice": "Keep the field evenly moist and use reliable rainfall or irrigation during early growth.",
    "Maize": "Plant in well-drained soil and keep moisture steady during germination and tasseling.",
    "Jute": "Best suited to warm, humid fields with enough moisture during the vegetative stage.",
    "Cotton": "Use well-drained soil and avoid excess water around the root zone.",
    "Coconut": "Maintain warm, humid conditions and steady moisture for healthy palm growth.",
    "Papaya": "Choose warm, well-drained land and protect plants from waterlogging.",
    "Orange": "Use well-drained soil and maintain moderate moisture around the root zone.",
    "Apple": "Grow in cooler conditions with balanced moisture and good drainage.",
    "Muskmelon": "Prefer warm, sunny fields and avoid heavy watering near fruit maturity.",
    "Watermelon": "Use warm, well-drained soil and keep moisture consistent during fruit setting.",
    "Grapes": "Grow in well-drained soil with controlled moisture and enough sunlight.",
    "Mango": "Plant in warm conditions and avoid waterlogging during root development.",
    "Banana": "Maintain high moisture and nutrient-rich soil for steady vegetative growth.",
    "Pomegranate": "Prefer well-drained soil and avoid excessive humidity around the plants.",
    "Lentil": "Use moderate soil moisture and avoid waterlogging during flowering.",
    "Blackgram": "Grow in warm fields with moderate moisture and good drainage.",
    "Mungbean": "Use warm, well-drained soil and avoid standing water after rainfall.",
    "Mothbeans": "Suitable for warmer, drier fields with light to moderate rainfall.",
    "Pigeonpeas": "Use well-drained soil and keep moisture moderate during early growth.",
    "Kidneybeans": "Use well-drained soil and avoid heat or water stress during flowering.",
    "Chickpea": "Grow in cool, well-drained conditions with moderate rainfall.",
    "Coffee": "Maintain warm, humid conditions with steady rainfall and partial shade where possible.",
}


def load_crop_profiles():
    dataset = pd.read_csv(DATASET_PATH)
    dataset = dataset.copy()
    dataset["label"] = dataset["label"].str.strip().str.lower()
    grouped = dataset.groupby("label")[FEATURE_COLUMNS].agg(["min", "max", "mean"])

    profiles = {}
    for crop_label, row in grouped.iterrows():
        profiles[crop_label] = {
            feature: {
                "min": float(row[(feature, "min")]),
                "max": float(row[(feature, "max")]),
                "mean": float(row[(feature, "mean")]),
            }
            for feature in FEATURE_COLUMNS
        }

    return profiles


CROP_PROFILES = load_crop_profiles()


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


def init_prediction_history():
    with closing(sqlite3.connect(HISTORY_DB_PATH)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                nitrogen REAL NOT NULL,
                phosphorus REAL NOT NULL,
                potassium REAL NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                ph REAL NOT NULL,
                rainfall REAL NOT NULL,
                top_crop TEXT NOT NULL,
                top_confidence REAL NOT NULL,
                recommendations_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def serialize_recommendations(recommendations):
    return [
        {
            "rank": recommendation["rank"],
            "crop": recommendation["crop"],
            "confidence": recommendation["confidence"],
            "ideal_conditions": recommendation["ideal_conditions"],
            "cultivation_note": recommendation["cultivation_note"],
        }
        for recommendation in recommendations
    ]


def save_prediction_history(values, recommendations):
    if not recommendations:
        return

    init_prediction_history()
    top_recommendation = recommendations[0]
    with closing(sqlite3.connect(HISTORY_DB_PATH)) as connection:
        connection.execute(
            """
            INSERT INTO prediction_history (
                created_at,
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall,
                top_crop,
                top_confidence,
                recommendations_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                values["Nitrogen"],
                values["Phosporus"],
                values["Potassium"],
                values["Temperature"],
                values["Humidity"],
                values["Ph"],
                values["Rainfall"],
                top_recommendation["crop"],
                top_recommendation["confidence"],
                json.dumps(serialize_recommendations(recommendations)),
            ),
        )
        connection.commit()


def get_recent_predictions(limit=5):
    init_prediction_history()
    with closing(sqlite3.connect(HISTORY_DB_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall,
                top_crop,
                top_confidence,
                recommendations_json
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    recent_predictions = []
    for row in rows:
        image_data = get_crop_image_data(row["top_crop"])
        try:
            recommendations = json.loads(row["recommendations_json"])
        except json.JSONDecodeError:
            recommendations = []
        for recommendation in recommendations:
            recommendation["image_url"] = get_crop_image_data(recommendation["crop"])["image_url"]

        created_at = datetime.fromisoformat(row["created_at"]).strftime("%b %d, %Y %I:%M %p")
        recent_predictions.append(
            {
                "id": row["id"],
                "created_at": created_at,
                "top_crop": row["top_crop"],
                "top_confidence": format_limit(row["top_confidence"]),
                "image_url": image_data["image_url"],
                "inputs": {
                    "N": format_limit(row["nitrogen"]),
                    "P": format_limit(row["phosphorus"]),
                    "K": format_limit(row["potassium"]),
                    "Temp": report_field_value("Temperature", row["temperature"]),
                    "Humidity": report_field_value("Humidity", row["humidity"]),
                    "pH": format_limit(row["ph"]),
                    "Rainfall": report_field_value("Rainfall", row["rainfall"]),
                },
                "recommendations": recommendations,
            }
        )

    return recent_predictions


def render_index_template(form_values=None, recommendations=None, error=None):
    return render_template(
        "index.html",
        form_values=form_values or {},
        field_range_hints=field_range_hints(),
        recommendations=recommendations or [],
        error=error,
    )


def format_limit(value):
    rounded_value = round(value, 2)
    if rounded_value.is_integer():
        return str(int(rounded_value))
    return f"{rounded_value:g}"


def field_range_hint(field_name):
    min_value, max_value = FIELD_LIMITS[field_name]
    min_display = format_limit(min_value)
    max_display = format_limit(max_value)

    if field_name == "Humidity":
        return f"{min_display}% - {max_display}%"
    if field_name in FIELD_UNITS:
        return f"{min_display} - {max_display}{FIELD_UNITS[field_name]}"

    return f"{min_display} - {max_display}"


def field_range_hints():
    return {field_name: field_range_hint(field_name) for field_name in FIELD_NAMES}


def field_range_message(field_name):
    min_value, max_value = FIELD_LIMITS[field_name]
    unit = FIELD_UNITS.get(field_name, "")
    return (
        f"{FIELD_NAMES[field_name]} must be between "
        f"{format_limit(min_value)} and {format_limit(max_value)}{unit}."
    )


def report_field_value(field_name, value):
    if field_name == "Humidity":
        return f"{format_limit(value)}%"
    if field_name in FIELD_UNITS:
        return f"{format_limit(value)}{FIELD_UNITS[field_name]}"
    return format_limit(value)


def pdf_escape(text):
    return (
        str(text)
        .encode("latin-1", "replace")
        .decode("latin-1")
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def pdf_text_command(x, y, text, size=11, bold=False):
    font = "/F2" if bold else "/F1"
    escaped_text = pdf_escape(text)
    return f"BT {font} {size} Tf {x} {y} Td ({escaped_text}) Tj ET\n"


def build_pdf(lines):
    content = "".join(
        pdf_text_command(line["x"], line["y"], line["text"], line["size"], line["bold"])
        for line in lines
    ).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"endstream",
    ]

    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"

    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode("ascii")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return pdf


def append_pdf_line(lines, x, y, text, size=11, bold=False):
    lines.append({"x": x, "y": y, "text": text, "size": size, "bold": bold})
    return y - (size + 7)


def append_wrapped_pdf_text(lines, x, y, text, size=10, bold=False, width=92):
    wrapped_lines = wrap(str(text), width=width) or [""]
    for wrapped_line in wrapped_lines:
        y = append_pdf_line(lines, x, y, wrapped_line, size=size, bold=bold)
    return y


def build_pdf_report(values, recommendations):
    generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    lines = []

    y = 750
    y = append_pdf_line(lines, 50, y, "Crop Recommendation Report", size=20, bold=True)
    y = append_pdf_line(lines, 50, y, "CSE299 Crop Recommendation System", size=11)
    y = append_pdf_line(lines, 50, y, f"Generated: {generated_at}", size=10)

    y -= 20
    y = append_pdf_line(lines, 50, y, "Entered Field Values", size=14, bold=True)
    y -= 4

    field_items = list(FIELD_NAMES.items())
    left_x = 66
    right_x = 320
    row_y = y
    for index, (field_name, label) in enumerate(field_items):
        column_x = left_x if index % 2 == 0 else right_x
        current_y = row_y - ((index // 2) * 18)
        value = report_field_value(field_name, values[field_name])
        lines.append(
            {
                "x": column_x,
                "y": current_y,
                "text": f"{label}: {value}",
                "size": 10,
                "bold": False,
            }
        )

    y = row_y - (((len(field_items) + 1) // 2) * 18) - 18
    y = append_pdf_line(lines, 50, y, "Top 3 Crop Recommendations", size=14, bold=True)
    y = append_pdf_line(
        lines,
        50,
        y,
        "Ranked by model confidence using the entered soil and climate values.",
        size=10,
    )
    y -= 8

    for recommendation in recommendations:
        crop = recommendation["crop"]
        confidence = recommendation["confidence"]
        rank = recommendation["rank"]
        conditions = ", ".join(
            f"{condition['label']}: {condition['value']}"
            for condition in recommendation["ideal_conditions"]
        )

        y = append_pdf_line(
            lines,
            66,
            y,
            f"#{rank} {crop} - Confidence: {confidence}%",
            size=12,
            bold=True,
        )
        y = append_wrapped_pdf_text(
            lines,
            82,
            y,
            f"Ideal conditions: {conditions}",
            size=9,
            width=95,
        )
        y = append_wrapped_pdf_text(
            lines,
            82,
            y,
            f"Cultivation note: {recommendation['cultivation_note']}",
            size=9,
            width=95,
        )
        y -= 10

    y = append_pdf_line(lines, 50, 56, "Note: Use this report as decision support with local agronomy advice.", size=9)

    return build_pdf(lines)


def format_condition_range(min_value, max_value, unit="", decimals=1):
    return f"{min_value:.{decimals}f}-{max_value:.{decimals}f}{unit}"


def nutrient_condition(profile):
    return (
        f"N {format_condition_range(profile['N']['min'], profile['N']['max'], decimals=0)}, "
        f"P {format_condition_range(profile['P']['min'], profile['P']['max'], decimals=0)}, "
        f"K {format_condition_range(profile['K']['min'], profile['K']['max'], decimals=0)}"
    )


def build_ideal_conditions(crop_name):
    profile = CROP_PROFILES.get(crop_name.strip().lower())
    if not profile:
        return []

    return [
        {"label": "N-P-K", "value": nutrient_condition(profile)},
        {
            "label": "Temperature",
            "value": format_condition_range(
                profile["temperature"]["min"], profile["temperature"]["max"], " C"
            ),
        },
        {
            "label": "Humidity",
            "value": format_condition_range(
                profile["humidity"]["min"], profile["humidity"]["max"], "%"
            ),
        },
        {
            "label": "pH",
            "value": format_condition_range(
                profile["ph"]["min"], profile["ph"]["max"], decimals=2
            ),
        },
        {
            "label": "Rainfall",
            "value": format_condition_range(
                profile["rainfall"]["min"], profile["rainfall"]["max"], " mm"
            ),
        },
    ]


def crop_cultivation_note(crop_name):
    return CROP_NOTES.get(
        crop_name,
        "Use local agronomy guidance and keep soil, water, and climate conditions within the ideal range.",
    )


def parse_form_values(form):
    values = {}
    for field_name in FIELD_NAMES:
        raw_value = form.get(field_name, "").strip()
        if not raw_value:
            raise ValueError(f"{FIELD_NAMES[field_name]} is required.")

        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{FIELD_NAMES[field_name]} must be a number.") from exc

        if not np.isfinite(value):
            raise ValueError(f"{FIELD_NAMES[field_name]} must be a finite number.")

        min_value, max_value = FIELD_LIMITS[field_name]
        if value < min_value or value > max_value:
            raise ValueError(field_range_message(field_name))

        values[field_name] = value
    return values


def build_features(values):
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

    return transformed


def class_to_crop_name(prediction):
    if isinstance(prediction, str):
        return prediction.title()
    if prediction in CROP_DICT:
        return CROP_DICT[prediction]

    raise ValueError("The trained model returned an unknown crop class.")


def predict_top_crops(values, limit=3):
    transformed = build_features(values)

    if not hasattr(model, "predict_proba"):
        crop_name = class_to_crop_name(model.predict(transformed)[0])
        image_data = get_crop_image_data(crop_name)
        return [
            {
                "rank": 1,
                "crop": crop_name,
                "confidence": 100.0,
                "ideal_conditions": build_ideal_conditions(crop_name),
                "cultivation_note": crop_cultivation_note(crop_name),
                "image_url": image_data["image_url"],
                "image_is_default": image_data["is_default"],
            }
        ]

    probabilities = model.predict_proba(transformed)[0]
    classes = model.classes_
    top_indexes = np.argsort(probabilities)[::-1][:limit]
    recommendations = []

    for rank, index in enumerate(top_indexes, start=1):
        crop_name = class_to_crop_name(classes[index])
        image_data = get_crop_image_data(crop_name)
        recommendations.append(
            {
                "rank": rank,
                "crop": crop_name,
                "confidence": round(float(probabilities[index]) * 100, 2),
                "ideal_conditions": build_ideal_conditions(crop_name),
                "cultivation_note": crop_cultivation_note(crop_name),
                "image_url": image_data["image_url"],
                "image_is_default": image_data["is_default"],
            }
        )

    return recommendations


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

    image_path = f"crops/{filename}"
    return {
        "image_url": url_for("static", filename=image_path)
        if has_request_context()
        else f"/static/{image_path}",
        "is_default": filename.startswith("default."),
    }


def prepare_training_data(dataset):
    features = dataset[FEATURE_COLUMNS].copy()
    labels = dataset["label"].str.strip().str.lower().map(LABEL_TO_CLASS)
    return features, labels


def scaled_train_test_split(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    minmax_scaler = MinMaxScaler()
    x_train_scaled = minmax_scaler.fit_transform(x_train)
    x_test_scaled = minmax_scaler.transform(x_test)

    standard_scaler = StandardScaler()
    x_train_normalized = standard_scaler.fit_transform(x_train_scaled)
    x_test_normalized = standard_scaler.transform(x_test_scaled)
    return x_train_normalized, x_test_normalized, y_train, y_test


def score_current_artifacts(features, labels):
    transformed = ms.transform(features)
    if scaler_after_minmax is not None:
        transformed = scaler_after_minmax.transform(transformed)
    predictions = model.predict(transformed)
    return accuracy_score(labels, predictions)


def train_holdout_model(features, labels):
    x_train, x_test, y_train, y_test = scaled_train_test_split(features, labels)
    holdout_model = RandomForestClassifier(random_state=42)
    holdout_model.fit(x_train, y_train)
    predictions = holdout_model.predict(x_test)
    return accuracy_score(y_test, predictions)


def get_feature_importance():
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    importance_values = np.asarray(importances, dtype=float)
    total_importance = importance_values.sum()
    if total_importance <= 0:
        return []

    importance_rows = []
    for feature_column, importance in zip(FEATURE_COLUMNS, importance_values):
        percentage = float(round((importance / total_importance) * 100, 2))
        importance_rows.append(
            {
                "feature": FEATURE_DISPLAY_NAMES[feature_column],
                "importance": percentage,
            }
        )

    importance_rows = sorted(importance_rows, key=lambda row: row["importance"], reverse=True)
    max_importance = importance_rows[0]["importance"] if importance_rows else 0
    for row in importance_rows:
        row["bar_width"] = (
            float(round((row["importance"] / max_importance) * 100, 2))
            if max_importance
            else 0
        )

    return importance_rows


def comparison_models():
    return [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Support Vector Machine", SVC()),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("Bagging", BaggingClassifier(random_state=42)),
        ("AdaBoost", AdaBoostClassifier(random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        ("Extra Trees", ExtraTreesClassifier(random_state=42)),
    ]


@lru_cache(maxsize=1)
def get_model_comparison():
    dataset = pd.read_csv(DATASET_PATH)
    features, labels = prepare_training_data(dataset)
    x_train, x_test, y_train, y_test = scaled_train_test_split(features, labels)

    results = []
    for model_name, candidate_model in comparison_models():
        candidate_model.fit(x_train, y_train)
        predictions = candidate_model.predict(x_test)
        accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
        results.append({"model": model_name, "accuracy": accuracy})

    results = sorted(results, key=lambda item: item["accuracy"], reverse=True)
    best_accuracy = results[0]["accuracy"] if results else 0
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
        result["bar_width"] = (
            round((result["accuracy"] / best_accuracy) * 100, 2) if best_accuracy else 0
        )
        result["is_current"] = result["model"] == "Random Forest"

    return {
        "results": results,
        "best_model": results[0]["model"] if results else "N/A",
        "best_accuracy": best_accuracy,
        "dataset_rows": len(dataset),
        "test_size": len(y_test),
        "split_description": "80% training / 20% testing, random_state=42, stratified by crop label",
        "preprocessing": "MinMaxScaler followed by StandardScaler",
    }


@lru_cache(maxsize=1)
def get_model_info():
    dataset = pd.read_csv(DATASET_PATH)
    features, labels = prepare_training_data(dataset)
    label_counts = dataset["label"].str.strip().str.lower().value_counts().sort_index()
    crop_classes = [
        CROP_DICT.get(LABEL_TO_CLASS[label], label.title()) for label in label_counts.index
    ]

    return {
        "model_name": type(model).__name__,
        "dataset_rows": len(dataset),
        "dataset_columns": len(dataset.columns),
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": "label",
        "class_count": len(label_counts),
        "samples_per_class": int(label_counts.iloc[0]) if label_counts.nunique() == 1 else None,
        "feature_names": [FEATURE_DISPLAY_NAMES[feature] for feature in FEATURE_COLUMNS],
        "feature_importance": get_feature_importance(),
        "crop_classes": crop_classes,
        "holdout_accuracy": round(train_holdout_model(features, labels) * 100, 2),
        "artifact_accuracy": round(score_current_artifacts(features, labels) * 100, 2),
        "train_size": int(len(dataset) * 0.8),
        "test_size": len(dataset) - int(len(dataset) * 0.8),
        "split_description": "80% training / 20% testing, random_state=42, stratified by crop label",
        "preprocessing": ["MinMaxScaler", "StandardScaler"],
    }


@app.route("/")
def index():
    return render_index_template()


@app.route("/model-info")
def model_info():
    return render_template("model_info.html", model_info=get_model_info())


@app.route("/model-comparison")
def model_comparison():
    return render_template(
        "model_comparison.html",
        comparison=get_model_comparison(),
    )


@app.route("/prediction-history")
def prediction_history():
    return render_template(
        "prediction_history.html",
        recent_predictions=get_recent_predictions(limit=30),
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_values = {field_name: request.form.get(field_name, "") for field_name in FIELD_NAMES}

    try:
        parsed_values = parse_form_values(request.form)
        recommendations = predict_top_crops(parsed_values)
    except ValueError as exc:
        return render_index_template(form_values=form_values, error=str(exc))
    except Exception:
        return render_index_template(
            form_values=form_values,
            error="Prediction failed. Check that the saved model and scaler files match the training pipeline.",
        )

    save_prediction_history(parsed_values, recommendations)
    return render_index_template(form_values=form_values, recommendations=recommendations)


@app.route("/download-report", methods=["POST"])
def download_report():
    form_values = {field_name: request.form.get(field_name, "") for field_name in FIELD_NAMES}

    try:
        parsed_values = parse_form_values(request.form)
        recommendations = predict_top_crops(parsed_values)
    except ValueError as exc:
        return render_index_template(form_values=form_values, error=str(exc))
    except Exception:
        return render_index_template(
            form_values=form_values,
            error="Report download failed. Check that the saved model and scaler files match the training pipeline.",
        )

    report_pdf = build_pdf_report(parsed_values, recommendations)
    response = Response(report_pdf, mimetype="application/pdf")
    response.headers["Content-Disposition"] = (
        'attachment; filename="crop-recommendation-report.pdf"'
    )
    return response


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
