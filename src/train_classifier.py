# Evaluation Note:
# randomly splitting consecutive webcam frames can produce overly optimistic results 
# because neighbouring frames are highly correlated. Amore robust evaluation 
# should use separate recording sessions or unseen participants for testing.

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "gesture_landmarks.csv"


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # Features and target
    X = df.drop(columns=["label"])
    y = df["label"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train classifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.3f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        )
    )

    print("\nClass order:")
    print(model.classes_)


if __name__ == "__main__":
    main()