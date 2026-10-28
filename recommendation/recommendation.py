# ==============================================================================
#
# PlantXAI-Net
#
# Phase 7
#
# Treatment Recommendation
#
# Part 1
#
# Recommendation Configuration
#
# ==============================================================================

from pathlib import Path
import pandas as pd


# ==============================================================================
# Project Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEVERITY_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "severity"
)

RECOMMENDATION_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "recommendations"
)

RECOMMENDATION_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FINAL_SEVERITY_FILE = (
    SEVERITY_RESULTS_DIR
    / "final_severity_results.csv"
)


RECOMMENDATION_FILE = (
    RECOMMENDATION_RESULTS_DIR
    / "treatment_recommendations.csv"
)


# ==============================================================================
# Recommendation Severity Levels
# ==============================================================================

SEVERITY_LEVELS = [

    "Mild",
    "Moderate",
    "Severe"

]


# ==============================================================================
# Recommendation Configuration
# ==============================================================================

CONFIDENCE_THRESHOLD = 0.70


# ==============================================================================
# Configuration Verification
# ==============================================================================

def verify_recommendation_configuration():

    print()

    print("=" * 80)

    print("PlantXAI-Net")

    print("Phase 7 - Treatment Recommendation")

    print("Part 1 - Recommendation Configuration")

    print("=" * 80)

    print()

    print(
        f"Project Root          : {PROJECT_ROOT}"
    )

    print(
        f"Severity Results      : {SEVERITY_RESULTS_DIR}"
    )

    print(
        f"Recommendation Results: "
        f"{RECOMMENDATION_RESULTS_DIR}"
    )

    print(
        f"Severity Input        : "
        f"{FINAL_SEVERITY_FILE}"
    )

    print()

    print(
        "Severity Levels       : "
        "Mild, Moderate, Severe"
    )

    print(
        f"Confidence Threshold  : "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print()

    print("=" * 80)

    print(
        "Phase 7 - Part 1 Configuration Ready"
    )

    print("=" * 80)

    print()


# ==============================================================================
# End of Part 1
# ==============================================================================

# ==============================================================================
#
# Phase 7
#
# Part 2
#
# Disease-wise Treatment Recommendation Mapping
#
# ==============================================================================


# ==============================================================================
# General Recommendation Templates
# ==============================================================================

GENERAL_PREVENTIVE = (
    "Remove severely affected plant material, "
    "maintain good field sanitation, avoid prolonged leaf wetness, "
    "and monitor nearby plants for disease spread."
)

MILD_ACTION = (
    "Monitor the plant closely, remove visibly affected leaves where practical, "
    "maintain sanitation, and improve airflow around the plant."
)

MODERATE_ACTION = (
    "Remove affected plant material where practical, improve airflow and sanitation, "
    "and consider an appropriate disease-control treatment according to local "
    "agricultural guidance and product labeling."
)

SEVERE_ACTION = (
    "Isolate or remove severely affected plant material where practical, "
    "improve field sanitation, and seek appropriate crop-management or "
    "agronomic guidance for disease-control treatment."
)


# ==============================================================================
# Disease-specific Management Guidance
# ==============================================================================

DISEASE_RECOMMENDATIONS = {

    "Apple___Apple_scab": {
        "management": (
            "Remove infected fallen leaves and affected plant material, "
            "maintain canopy airflow, and avoid prolonged leaf wetness."
        )
    },

    "Apple___Black_rot": {
        "management": (
            "Remove mummified fruit and infected plant material, "
            "maintain orchard sanitation, and prune affected branches where practical."
        )
    },

    "Apple___Cedar_apple_rust": {
        "management": (
            "Remove heavily affected plant material where practical, "
            "maintain orchard sanitation, and monitor for recurring symptoms."
        )
    },

    "Apple___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine monitoring, "
            "good sanitation, and preventive crop management."
        )
    },

    "Blueberry___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine monitoring "
            "and maintain good crop hygiene."
        )
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "management": (
            "Improve airflow, avoid excessive humidity around foliage, "
            "remove heavily affected material where practical, and monitor progression."
        )
    },

    "Cherry_(including_sour)___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine monitoring "
            "and maintain good sanitation."
        )
    },

    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
        "management": (
            "Maintain field sanitation, improve crop airflow where possible, "
            "and monitor disease progression."
        )
    },

    "Corn_(maize)___Common_rust_": {
        "management": (
            "Monitor disease progression, maintain crop health, "
            "and use locally recommended disease-management practices when necessary."
        )
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "management": (
            "Maintain field sanitation, monitor spread, and use locally "
            "recommended disease-management practices."
        )
    },

    "Corn_(maize)___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine crop monitoring "
            "and preventive management."
        )
    },

    "Grape___Black_rot": {
        "management": (
            "Remove infected fruit and plant material where practical, "
            "maintain vineyard sanitation, and improve canopy airflow."
        )
    },

    "Grape___Esca_(Black_Measles)": {
        "management": (
            "Remove or manage severely affected plant material where practical, "
            "maintain vineyard sanitation, and seek local agronomic guidance."
        )
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "management": (
            "Remove affected plant material where practical, improve canopy airflow, "
            "and maintain vineyard sanitation."
        )
    },

    "Grape___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine vineyard monitoring "
            "and sanitation."
        )
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "management": (
            "Monitor affected trees carefully, manage infected plant material "
            "according to local citrus-health guidance, and control disease vectors "
            "using locally recommended practices."
        )
    },

    "Peach___Bacterial_spot": {
        "management": (
            "Remove severely affected plant material where practical, "
            "maintain orchard sanitation, and reduce conditions favoring prolonged "
            "leaf wetness."
        )
    },

    "Peach___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine monitoring "
            "and orchard sanitation."
        )
    },

    "Pepper,_bell___Bacterial_spot": {
        "management": (
            "Remove affected plant material where practical, maintain sanitation, "
            "and avoid practices that spread moisture and contamination between plants."
        )
    },

    "Pepper,_bell___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine monitoring "
            "and maintain good crop hygiene."
        )
    },

    "Potato___Early_blight": {
        "management": (
            "Remove affected foliage where practical, maintain field sanitation, "
            "and monitor disease progression."
        )
    },

    "Potato___Late_blight": {
        "management": (
            "Remove or isolate severely affected plant material where practical, "
            "monitor closely for rapid spread, and seek locally recommended "
            "late-blight management guidance."
        )
    },

    "Potato___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine crop monitoring "
            "and preventive management."
        )
    },

    "Raspberry___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine monitoring "
            "and maintain good crop hygiene."
        )
    },

    "Soybean___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine crop monitoring "
            "and preventive field management."
        )
    },

    "Squash___Powdery_mildew": {
        "management": (
            "Improve airflow, remove heavily affected leaves where practical, "
            "and monitor disease development."
        )
    },

    "Strawberry___Leaf_scorch": {
        "management": (
            "Remove severely affected foliage where practical, "
            "maintain plant sanitation, and monitor symptom progression."
        )
    },

    "Strawberry___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine monitoring "
            "and maintain good crop hygiene."
        )
    },

    "Tomato___Bacterial_spot": {
        "management": (
            "Remove affected plant material where practical, maintain sanitation, "
            "and reduce practices that spread moisture between plants."
        )
    },

    "Tomato___Early_blight": {
        "management": (
            "Remove affected foliage where practical, improve airflow, "
            "and maintain field sanitation."
        )
    },

    "Tomato___Late_blight": {
        "management": (
            "Monitor closely for rapid disease progression, remove severely "
            "affected material where practical, and seek locally recommended "
            "late-blight management guidance."
        )
    },

    "Tomato___Leaf_Mold": {
        "management": (
            "Improve ventilation and airflow, reduce prolonged leaf wetness, "
            "and remove heavily affected material where practical."
        )
    },

    "Tomato___Septoria_leaf_spot": {
        "management": (
            "Remove affected foliage where practical, improve airflow, "
            "and maintain good field sanitation."
        )
    },

    "Tomato___Spider_mites_Two_spotted_spider_mite": {
        "management": (
            "Monitor the underside of leaves, remove heavily affected material "
            "where practical, and use locally appropriate integrated pest-management "
            "practices."
        )
    },

    "Tomato___Target_Spot": {
        "management": (
            "Remove affected foliage where practical, improve airflow, "
            "and maintain field sanitation."
        )
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "management": (
            "Remove severely affected plants where appropriate, monitor nearby plants, "
            "and manage insect vectors using locally recommended integrated "
            "pest-management practices."
        )
    },

    "Tomato___Tomato_mosaic_virus": {
        "management": (
            "Remove severely affected plants where appropriate, maintain sanitation, "
            "and prevent mechanical spread through contaminated tools or handling."
        )
    },

    "Tomato___healthy": {
        "management": (
            "No disease treatment is indicated. Continue routine monitoring "
            "and preventive crop management."
        )
    }
}


# ==============================================================================
# End of Part 2
# ==============================================================================
# ==============================================================================
#
# ==============================================================================
#
# Phase 7
#
# Part 3
#
# Recommendation Generation
#
# ==============================================================================


def generate_recommendation(
    disease: str,
    severity: str,
    confidence: float
) -> str:

    """
    Generate a disease- and severity-aware management recommendation.
    """

    # --------------------------------------------------------------------------
    # Low-confidence prediction
    # --------------------------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        return (
            "Prediction confidence is below the configured threshold. "
            "Recheck the image under clear lighting and consider "
            "expert/agronomic verification before taking treatment action."
        )


    # --------------------------------------------------------------------------
    # Healthy plant
    # --------------------------------------------------------------------------

    if disease.endswith("healthy"):

        return (
            "No disease treatment is indicated. "
            "Continue routine monitoring, sanitation, and preventive "
            "crop-management practices."
        )


    # --------------------------------------------------------------------------
    # Disease-specific lookup
    # --------------------------------------------------------------------------

    disease_information = DISEASE_RECOMMENDATIONS.get(
        disease
    )


    if disease_information is None:

        management = GENERAL_PREVENTIVE

    else:

        management = disease_information["management"]


    # --------------------------------------------------------------------------
    # Severity-specific action
    # --------------------------------------------------------------------------

    if severity == "Mild":

        severity_action = MILD_ACTION

    elif severity == "Moderate":

        severity_action = MODERATE_ACTION

    elif severity == "Severe":

        severity_action = SEVERE_ACTION

    else:

        severity_action = GENERAL_PREVENTIVE


    # --------------------------------------------------------------------------
    # Final recommendation
    # --------------------------------------------------------------------------

    return (
        f"{management} "
        f"{severity_action} "
        "Any chemical or biological treatment should be selected "
        "according to local agricultural guidance and the product label."
    )


# ==============================================================================
# Generate Recommendation Dataset
# ==============================================================================


def generate_recommendation_dataset() -> Path:

    """
    Generate treatment recommendations from the latest
    Phase 6 severity results from the full-test-set evaluation.

    The recommendation module does not claim treatment accuracy.
    It only generates decision-support recommendations from:
        - predicted disease
        - prediction confidence
        - Grad-CAM-derived severity proxy
    """


    # --------------------------------------------------------------------------
    # Verify Severity Results File
    # --------------------------------------------------------------------------

    if not FINAL_SEVERITY_FILE.exists():

        raise FileNotFoundError(

            f"Severity results not found: "
            f"{FINAL_SEVERITY_FILE}"

        )


    # --------------------------------------------------------------------------
    # Load Severity Results
    # --------------------------------------------------------------------------

    data = pd.read_csv(

        FINAL_SEVERITY_FILE

    )
    # --------------------------------------------------------------------------
    # Validate Number of Records
    # --------------------------------------------------------------------------

    if data.empty:

        raise RuntimeError(

            "Phase 6 severity results file is empty."

        )


    print(

        f"Severity Records Found : {len(data)}"

    )
# --------------------------------------------------------------------------
    # Required Columns
    # --------------------------------------------------------------------------

    required_columns = [

        "Image",

        "Actual",

        "Predicted",

        "Confidence",

        "Severity_Percent",

        "Severity_Level"

    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in data.columns

    ]


    if missing_columns:

        raise ValueError(

            "Required severity columns missing: "

            +

            ", ".join(missing_columns)

        )


    # --------------------------------------------------------------------------
    # Validate Severity Levels
    # --------------------------------------------------------------------------

    invalid_severity = (

        data[
            ~data["Severity_Level"]
            .isin(SEVERITY_LEVELS)
        ]

    )


    if not invalid_severity.empty:

        raise ValueError(

            "Invalid severity level(s) found: "

            +

            ", ".join(

                sorted(

                    invalid_severity[
                        "Severity_Level"
                    ]
                    .astype(str)
                    .unique()

                )

            )

        )


    # --------------------------------------------------------------------------
    # Normalize Confidence
    # --------------------------------------------------------------------------

    data["Confidence"] = pd.to_numeric(

        data["Confidence"],

        errors="coerce"

    )


    if data["Confidence"].isna().any():

        raise ValueError(

            "Invalid or missing confidence values "
            "were found in the severity results."

        )


    # --------------------------------------------------------------------------
    # Handle Confidence Stored as Percentage
    # --------------------------------------------------------------------------

    if (

        data["Confidence"]
        .dropna()
        .max()

        >

        1.0

    ):

        data["Confidence"] = (

            data["Confidence"]

            /

            100.0

        )


    # --------------------------------------------------------------------------
    # Validate Confidence Range
    # --------------------------------------------------------------------------

    if (

        (data["Confidence"] < 0.0).any()

        or

        (data["Confidence"] > 1.0).any()

    ):

        raise ValueError(

            "Confidence values must be between "
            "0.0 and 1.0."

        )


    # --------------------------------------------------------------------------
    # Generate Recommendations
    # --------------------------------------------------------------------------

    data["Recommendation"] = data.apply(

        lambda row:

        generate_recommendation(

            disease=str(

                row["Predicted"]

            ),

            severity=str(

                row["Severity_Level"]

            ),

            confidence=float(

                row["Confidence"]

            )

        ),

        axis=1

    )


    # --------------------------------------------------------------------------
    # Recommendation Priority
    # --------------------------------------------------------------------------

    priority_map = {

        "Mild":
            "Routine Monitoring",

        "Moderate":
            "Action Recommended",

        "Severe":
            "Immediate Attention"

    }


    data["Recommendation_Priority"] = (

        data["Severity_Level"]

        .map(

            priority_map

        )

        .fillna(

            "Review Required"

        )

    )


    # --------------------------------------------------------------------------
    # Confidence-Aware Status
    # --------------------------------------------------------------------------

    data["Confidence_Status"] = data[
        "Confidence"
    ].apply(

        lambda value:

        "Direct Guidance"

        if value >= CONFIDENCE_THRESHOLD

        else

        "Manual Verification Required"

    )


    # --------------------------------------------------------------------------
    # Save Results
    # --------------------------------------------------------------------------

    output_columns = [

        "Image",

        "Actual",

        "Predicted",

        "Confidence",

        "Severity_Percent",

        "Severity_Level",

        "Confidence_Status",

        "Recommendation_Priority",

        "Recommendation"

    ]


    output_file = (

        RECOMMENDATION_FILE

    )


    data[
        output_columns
    ].to_csv(

        output_file,

        index=False

    )


    # --------------------------------------------------------------------------
    # Summary Statistics
    # --------------------------------------------------------------------------

    total_records = len(data)


    direct_guidance = int(

        (

            data["Confidence"]

            >=

            CONFIDENCE_THRESHOLD

        ).sum()

    )


    manual_verification = (

        total_records

        -

        direct_guidance

    )


    mild_count = int(

        (

            data["Severity_Level"]

            ==

            "Mild"

        ).sum()

    )


    moderate_count = int(

        (

            data["Severity_Level"]

            ==

            "Moderate"

        ).sum()

    )


    severe_count = int(

        (

            data["Severity_Level"]

            ==

            "Severe"

        ).sum()

    )


    # --------------------------------------------------------------------------
    # Display Summary
    # --------------------------------------------------------------------------

    print()

    print("=" * 80)

    print(

        "PHASE 7 - PART 3"

    )

    print(

        "TREATMENT RECOMMENDATION GENERATION"

    )

    print("=" * 80)

    print()


    print(

        f"Images Processed       : "
        f"{total_records}"

    )


    print(

        f"Recommendations        : "
        f"{total_records}"

    )


    print()


    print(

        f"Mild Cases             : "
        f"{mild_count}"

    )


    print(

        f"Moderate Cases         : "
        f"{moderate_count}"

    )


    print(

        f"Severe Cases           : "
        f"{severe_count}"

    )


    print()


    print(

        f"Confidence Threshold   : "
        f"{CONFIDENCE_THRESHOLD:.2f}"

    )


    print(

        f"Direct Guidance       : "
        f"{direct_guidance}"

    )


    print(

        f"Manual Verification   : "
        f"{manual_verification}"

    )


    print()


    print(

        f"Output File            : "
        f"{output_file}"

    )


    print()

    print(

        "NOTE: Recommendation generation success "
        "does not represent treatment accuracy."

    )


    print(

        "Independent expert-labeled treatment outcomes "
        "were not used for recommendation validation."

    )


    print()

    print("=" * 80)

    print(

        "Phase 7 - Part 3 Completed Successfully"

    )

    print("=" * 80)

    print()


    return output_file


# ==============================================================================
# End of Part 3
# ==============================================================================
# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    verify_recommendation_configuration()

    generate_recommendation_dataset()