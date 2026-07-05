"""
dataset_config.py
==================
Central registry of dataset-specific settings for the MAC-CDZS reproduction.

WHY THIS FILE EXISTS:
The original authors' code hardcoded each dataset's paths and class counts
directly inside the training scripts via a long if/elif chain. That meant
adding a new dataset required editing the core training logic itself.

Here, adding a new dataset means adding ONE entry below. Nothing in
train_stage1.py or train_stage2.py needs to change.

WHAT EACH FIELD MEANS:
    class_num   : total number of classes in the target dataset's ground truth
    IR_task     : how many classes get a few labeled samples ("few-shot")
    R_task      : how many classes get ZERO labeled samples ("zero-shot")
                  Must satisfy: IR_task + R_task == class_num
    channel     : number of spectral bands in the hyperspectral image cube
                  (e.g. Indian Pines has 200 bands after preprocessing)
    data_path   : path to the .mat file containing the image cube
    label_path  : path to the .mat file containing ground-truth labels
    data_key    : the dict key inside the .mat file holding the image array.
                  Leave as None to let the loader auto-detect it -- only set
                  this if loading fails with a "key not found" error (this
                  was needed for Houston13's non-standard file format).
    label_key   : same idea, for the label array.

NOTE ON THE "Embedding" PARAMETER FROM THE ORIGINAL CODE:
The original ZSDAModel class took a per-dataset "Embedding" number used to
build a nn.Embedding layer (self.features). We traced every use of that
layer through the entire model file and confirmed it is NEVER actually
read anywhere -- the only code that touches it lives inside a fully
commented-out method (init_Graph). It's dead code left over from an
earlier version of the model. So it's been removed from this config
entirely; you don't need to supply it for new datasets.
"""

# Change this to wherever your Drive / data folder actually is.
DRIVE_ROOT = '/content/drive/MyDrive/MAC-CDZS_data'

DATASETS = {
    'IP': {  # Indian Pines
        'class_num': 16, 'IR_task': 10, 'R_task': 6,
        'channel': 200,
        'data_path': f'{DRIVE_ROOT}/IndianPines/indian_pines_corrected.mat',
        'label_path': f'{DRIVE_ROOT}/IndianPines/indian_pines_gt.mat',
        'data_key': None, 'label_key': None,
    },
    'SA': {  # Salinas
        'class_num': 16, 'IR_task': 10, 'R_task': 6,
        'channel': 204,
        'data_path': f'{DRIVE_ROOT}/Salinas/salinas_corrected.mat',
        'label_path': f'{DRIVE_ROOT}/Salinas/salinas_gt.mat',
        'data_key': None, 'label_key': None,
    },
    'UP': {  # Pavia University
        'class_num': 9, 'IR_task': 5, 'R_task': 4,
        'channel': 103,
        'data_path': f'{DRIVE_ROOT}/PaviaU/paviaU.mat',
        'label_path': f'{DRIVE_ROOT}/PaviaU/paviaU_gt.mat',
        'data_key': None, 'label_key': None,
    },
    'Ho': {  # Houston
        'class_num': 15, 'IR_task': 10, 'R_task': 5,
        'channel': 144,
        'data_path': '/content/drive/MyDrive/MAC-CDZS_data/Houston/Houston.mat',
        'label_path': '/content/drive/MyDrive/MAC-CDZS_data/Houston/Houston_GT.mat',
        # Houston13's .mat file uses non-standard key names, confirmed by
        # trial and error against the actual file -- everything else
        # auto-detects fine without these.
        'data_key': 'Houston', 'label_key': 'Houston_GT',
    },

    # --- Add your own dataset below, following the same pattern. ---
    # No other file needs to change.
    # 'MyDataset': {
    #     'class_num': 12, 'IR_task': 8, 'R_task': 4,
    #     'channel': 176,
    #     'data_path': f'{DRIVE_ROOT}/my_dataset.mat',
    #     'label_path': f'{DRIVE_ROOT}/my_dataset_gt.mat',
    #     'data_key': None, 'label_key': None,
    # },
}


def get_dataset_config(name):
    """Look up a dataset's settings by name, with validation."""
    if name not in DATASETS:
        available = ', '.join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}. "
            f"To use a different dataset, add a new entry to DATASETS "
            f"in dataset_config.py -- no other file needs to change."
        )
    cfg = DATASETS[name]
    if cfg['IR_task'] + cfg['R_task'] != cfg['class_num']:
        raise ValueError(
            f"Dataset '{name}': IR_task ({cfg['IR_task']}) + R_task "
            f"({cfg['R_task']}) must equal class_num ({cfg['class_num']})."
        )
    return cfg
