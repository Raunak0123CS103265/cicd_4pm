
# ==========================================
# IRIS CLASSIFICATION PROJECT
# Logistic Regression vs Random Forest
# ==========================================

# 1. Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

sns.set_theme(style="white")


# 2. Load dataset

dataset = pd.read_csv("iris.csv")

print("Original Dataset:")
print(dataset.head())

print("\nOriginal Columns:")
print(dataset.columns.tolist())


# 3. Clean column names

def clean_column_name(column):
    column = column.strip().lower()
    column = re.sub(r"\s*\(cm\)", "", column)
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")

dataset.columns = [
    clean_column_name(col) for col in dataset.columns
]

print("\nCleaned Columns:")
print(dataset.columns.tolist())


# 4. Remove unnecessary index column if present

dataset = dataset.loc[
    :, ~dataset.columns.str.startswith("unnamed")
]


# 5. Check target column

if "target" not in dataset.columns:
    raise ValueError(
        "Target column not found. "
        "Please check your iris.csv column names."
    )

# Convert target labels into numeric values if necessary

if not pd.api.types.is_numeric_dtype(dataset["target"]):
    encoder = LabelEncoder()
    dataset["target"] = encoder.fit_transform(dataset["target"])

print("\nTarget Classes:")
print(dataset["target"].value_counts())


# 6. Feature Engineering

dataset["sepal_length_width_ratio"] = (
    dataset["sepal_length"] / dataset["sepal_width"]
)

dataset["petal_length_width_ratio"] = (
    dataset["petal_length"] / dataset["petal_width"]
)


# 7. Select Features and Target

X = dataset.drop("target", axis=1)
y = dataset["target"]

feature_names = X.columns.tolist()

print("\nFeatures:")
print(feature_names)


# 8. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=44,
    stratify=y
)


# 9. Create Models

models = {
    "Logistic Regression": LogisticRegression(
        C=0.0001,
        solver="lbfgs",
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        random_state=42,
        criterion="entropy"
    )
}


# 10. Train and Evaluate Models

results = {}
predictions = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    predictions[name] = test_predictions

    # Accuracy
    train_accuracy = accuracy_score(
        y_train, train_predictions
    )

    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    # Other evaluation metrics
    precision = precision_score(
        y_test, test_predictions, average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, test_predictions, average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, test_predictions, average="weighted",
        zero_division=0
    )

    # Store results
    results[name] = {
        "Train Accuracy": train_accuracy * 100,
        "Test Accuracy": test_accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100
    }

    # Display results
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )


# 11. Confusion Matrix Plotting Function

def plot_confusion_matrix(y_true, y_pred, model_name):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=sorted(y.unique())
    )

    class_names = sorted(y.unique())

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(filename, dpi=120)
    plt.show()
    plt.close()


# 12. Plot Confusion Matrices

for name in models:

    plot_confusion_matrix(
        y_test,
        predictions[name],
        name
    )


# 13. Random Forest Feature Importance

rf_model = models["Random Forest"]

importances = rf_model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_df)


# Plot Feature Importance

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_df,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig("FeatureImportance.png", dpi=120)
plt.show()
plt.close()


# 14. Compare Model Performance

results_df = pd.DataFrame(results).T

print("\nModel Comparison:")
print(results_df.round(2))

results_df.to_csv("model_comparison.csv")


# 15. Save Scores to Text File

with open("scores.txt", "w") as score:

    score.write("IRIS CLASSIFICATION MODEL RESULTS\n")
    score.write("=" * 45 + "\n\n")

    for model_name, metrics in results.items():

        score.write(f"Model: {model_name}\n")

        for metric_name, value in metrics.items():

            score.write(
                f"{metric_name}: {value:.2f}%\n"
            )

        score.write("\n" + "-" * 40 + "\n\n")


print("\nProject Completed Successfully!")
print("Generated Files:")
print("1. logistic_regression_confusion_matrix.png")
print("2. random_forest_confusion_matrix.png")
print("3. FeatureImportance.png")
print("4. model_comparison.csv")
print("5. scores.txt")
