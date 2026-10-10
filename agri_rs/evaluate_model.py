
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from agri_rs.states import get_crop
from agri_rs.simulate import simulate_field, season_to_records
from agri_rs.service import infer


def main():
    # Load the maize model definition
    crop = get_crop("maize")

    # Generate a test season with known reference stages
    season = simulate_field(
        crop,
        cloud_prob=0.10,
        noise_scale=1.0,
        apply_district=False,
        seed=42,
    )

    # Keep the reference labels for evaluation.
    # The inference function uses the spectral indices, not these labels.
    records = season_to_records(season, include_labels=True)

    # Run the actual HMM inference pipeline
    result = infer("maize", records)

    # Actual stage labels are integer state positions.
    y_true = np.array([
        row["simulator_stage"] for row in records
    ])

    # Convert predicted stage IDs back to integer positions.
    stage_ids = [stage.id for stage in crop.stages]
    y_pred = np.array([
        stage_ids.index(row["stage_viterbi"])
        for row in result["timeline"]
    ])

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    print("\nMAIZE HMM EVALUATION")
    print("-" * 40)
    print(f"Number of observations: {len(y_true)}")
    print(f"Accuracy: {accuracy:.2%}")

    # Detailed precision, recall and F1-score
    stage_names = [stage.name for stage in crop.stages]

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(stage_names))),
        target_names=stage_names,
        zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(stage_names))),
    )

    print("\nConfusion matrix:")
    print(cm)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=stage_names,
        yticklabels=stage_names,
    )
    plt.xlabel("Predicted growth stage")
    plt.ylabel("Actual growth stage")
    plt.title("Maize HMM Confusion Matrix")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    print("\nConfusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()
