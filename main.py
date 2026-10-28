"""
PlantXAI-Net Main Entry Point
"""

from models.training_pipeline import main as training_main


def main():
    """
    Project Entry Point
    """

    print(
        "Welcome to PlantXAI-Net: Explainable AI for Plant Disease Detection & Recommendation"
    )

    print("Initializing Training Pipeline...\n")

    training_main()


if __name__ == "__main__":
    main()