from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "open"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
EXPERIMENT_LOG_PATH = REPORTS_DIR / "experiment_log.csv"

TARGET = "stress_score"
ID_COL = "ID"
RANDOM_STATE = 42
N_SPLITS = 5
N_TARGET_BINS = 10

CATEGORICAL_MISSING_COLS = [
    "medical_history",
    "family_medical_history",
    "edu_level",
]

BASE_CATEGORICAL_COLS = [
    "gender",
    "activity",
    "smoke_status",
    "medical_history",
    "family_medical_history",
    "sleep_pattern",
    "edu_level",
    "mean_working_cat",
]

SCORECARD_BIN_SPECS = {
    "age": "age_bin",
    "bmi": "bmi_bin",
    "glucose": "glucose_bin",
    "cholesterol": "cholesterol_bin",
    "systolic_blood_pressure": "systolic_bp_bin",
    "diastolic_blood_pressure": "diastolic_bp_bin",
    "bone_density": "bone_density_bin",
}

INTERACTION_SPECS = [
    ("mean_working_cat", "sleep_pattern", "mean_working_cat__sleep_pattern"),
    ("mean_working_cat", "activity", "mean_working_cat__activity"),
    ("smoke_status", "activity", "smoke_status__activity"),
    ("gender", "bone_density_bin", "gender__bone_density_bin"),
    ("age_bin", "bone_density_bin", "age_bin__bone_density_bin"),
    (
        "medical_history",
        "family_medical_history",
        "medical_history__family_medical_history",
    ),
]

V2_CATEGORICAL_COLS = BASE_CATEGORICAL_COLS + list(SCORECARD_BIN_SPECS.values()) + [
    spec[2] for spec in INTERACTION_SPECS
]

V3_CATEGORICAL_COLS = V2_CATEGORICAL_COLS + ["mean_working_cat_v3"]

V4_INTERACTION_SPECS = [
    ("mean_working_cat_v3", "sleep_pattern", "mean_working_cat_v3__sleep_pattern"),
    ("mean_working_cat_v3", "activity", "mean_working_cat_v3__activity"),
]

V4_TARGET_ENCODING_COLS = [
    "mean_working_cat_v3",
    "sleep_pattern",
    "activity",
    "smoke_status",
    "medical_history",
    "family_medical_history",
    "mean_working_cat_v3__sleep_pattern",
    "mean_working_cat_v3__activity",
    "medical_history__family_medical_history",
]

V4_CATEGORICAL_COLS = V3_CATEGORICAL_COLS + [
    spec[2] for spec in V4_INTERACTION_SPECS
]

RAW_EXCLUDED_FEATURES = [ID_COL, TARGET, "mean_working"]
