import os

# Get project root directory (parent of configs/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BaseConfig:
    project_root = PROJECT_ROOT
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "outputs")

    device = "cuda"
    num_workers = 4
    seed = 42

    use_amp = True

    height = 1152
    width = 1440

    lead_name_to_label = {
        'None': 0,
        'I': 1,
        'aVR': 2,
        'V1': 3,
        'V4': 4,
        'II': 5,
        'aVL': 6,
        'V2': 7,
        'V5': 8,
        'III': 9,
        'aVF': 10,
        'V3': 11,
        'V6': 12,
        'II-rhythm': 13,
    }

    label_to_lead_name = {v: k for k, v in lead_name_to_label.items()}
