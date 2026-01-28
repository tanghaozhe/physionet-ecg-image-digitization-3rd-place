import os
from .base import BaseConfig


def _init_dirs(self, stage_dir):
    """Initialize stage-specific output directories."""
    self.stage_output_dir = os.path.join(self.output_dir, stage_dir, f"run{self.run}")
    self.checkpoint_dir = os.path.join(self.stage_output_dir, "checkpoints")
    self.log_dir = os.path.join(self.stage_output_dir, "logs")
    self.infer_dir = os.path.join(self.stage_output_dir, "pixels")
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    os.makedirs(self.log_dir, exist_ok=True)
    os.makedirs(self.infer_dir, exist_ok=True)


class Stage2Config_4352x1696(BaseConfig):
    stage_name = "stage2"
    
    # Training run config
    run = 14
    num_folds = 5
    train_folds = [1]
    
    # Weights & Biases
    use_wandb = False
    wandb_project = f"physionet-ecg_run{run}"
    wandb_entity = None

    checkpoint_path = None
    backbone = "resnet34.a3_in1k"
    pretrained = True
    encoder_dim = [64, 128, 256, 512]
    decoder_dim = [256, 128, 64, 32]
    use_coord_conv = True

    num_output_channels = 4

    batch_size = 2
    num_epochs = 40
    learning_rate = 1e-4
    weight_decay = 1e-5
    accumulation_steps = 1

    pixel_pos_weight = 10.0

    use_augmentation = True

    mask_dir = 'data/mask_4352x1696'
    rectified_dir = 'outputs/stage1/rectified_kaggle_data_4400x1700_INTER_CUBIC'

    crop_x_range = (0, 4352)
    crop_y_range = (0, 1696)
    time_range = (236, 4160)

    zero_mv_positions = [703.5, 987.5, 1271.5, 1531.5]
    mv_to_pixel = 79

    mv_limits = [
        [-2, 2], [-2, 2], [-4, 4], [-4, 4]
    ]

    save_frequency = 1

    def __init__(self):
        _init_dirs(self, "stage2")


class Stage2HRNetConfig_4352x1696(BaseConfig):
    stage_name = "stage2_hrnet"

    # Training run config
    run = 10
    num_folds = 5
    train_folds = [0]

    # Weights & Biases
    use_wandb = False
    wandb_project = f"physionet-ecg_hrnet_run{run}"
    wandb_entity = None

    checkpoint_path = None
    backbone = "hrnet_w32.ms_in1k"
    pretrained = True
    encoder_dim = [64, 128, 256, 512]  # HRNet Level 0-3
    decoder_dim = [256, 128, 64, 32]
    use_coord_conv = True

    num_output_channels = 4

    batch_size = 2
    num_epochs = 40
    learning_rate = 1e-4
    weight_decay = 1e-5
    accumulation_steps = 1

    pixel_pos_weight = 10.0

    use_augmentation = True

    mask_dir = 'data/mask_4352x1696'
    rectified_dir = 'outputs/stage1/rectified_kaggle_data_4400x1700_INTER_CUBIC'

    crop_x_range = (0, 4352)
    crop_y_range = (0, 1696)
    time_range = (236, 4160)

    zero_mv_positions = [703.5, 987.5, 1271.5, 1531.5]
    mv_to_pixel = 79

    mv_limits = [
        [-2, 2], [-2, 2], [-4, 4], [-4, 4]
    ]

    save_frequency = 1

    def __init__(self):
        _init_dirs(self, "stage2_hrnet")


class Stage2ConvNeXtV2Config_4352x1696(BaseConfig):
    stage_name = "stage2_convnextv2"

    # Training run config
    run = 15 #13 #15 # 8
    num_folds = 5
    train_folds = [4] # f1:g0, f2:g1, f3:g2, f4:g3

    # Weights & Biases
    use_wandb = False
    wandb_project = f"physionet-ecg_convnextv2_run{run}"
    wandb_entity = None

    checkpoint_path = "outputs/stage2_convnextv2/run13/checkpoints/fold_{}/epoch_0010.pth"
    backbone = "convnextv2_tiny.fcmae_ft_in1k"
    pretrained = True
    encoder_dim = [96, 192, 384, 768]  # ConvNeXt V2 Base Level 0-3
    decoder_dim = [256, 128, 64, 32]
    use_coord_conv = True

    num_output_channels = 4

    batch_size = 2
    num_epochs = 30
    learning_rate = 1e-4 # 2e-4
    weight_decay = 1e-5
    accumulation_steps = 1

    pixel_pos_weight = 10.0

    use_augmentation = True

    mask_dir = 'data/mask_4352x1696'
    rectified_dir = 'outputs/stage1/rectified_kaggle_data_4400x1700_INTER_CUBIC'
    # rectified_dir = 'outputs/stage1/rectified_kaggle_data_4400x1700'

    crop_x_range = (0, 4352)
    crop_y_range = (0, 1696)
    time_range = (236, 4160)

    zero_mv_positions = [703.5, 987.5, 1271.5, 1531.5]
    mv_to_pixel = 79

    mv_limits = [
        [-2, 2], [-2, 2], [-4, 4], [-4, 4]
    ]

    save_frequency = 1

    def __init__(self):
        _init_dirs(self, "stage2_convnextv2")



class Stage2EfficientNetV2Config_4352x1696(BaseConfig):
    stage_name = "stage2_efficientnetv2"

    # Training run config
    run = 12 # 11
    num_folds = 5
    train_folds = [0]

    # Weights & Biases
    use_wandb = False
    wandb_project = f"physionet-ecg_efficientnetv2_run{run}"
    wandb_entity = None

    checkpoint_path = None
    backbone = "efficientnetv2_rw_s.ra2_in1k"
    pretrained = True
    encoder_dim = [24, 48, 64, 160, 272]  # EfficientNetV2-S Level 0-4
    decoder_dim = [384, 192, 96, 48] # [256, 128, 64, 32]
    use_coord_conv = True

    num_output_channels = 4

    batch_size = 2
    num_epochs = 40
    learning_rate = 1e-4
    weight_decay = 1e-5
    accumulation_steps = 1

    pixel_pos_weight = 10.0

    use_augmentation = True

    mask_dir = 'data/mask_4352x1696'
    rectified_dir = 'outputs/stage1/rectified_kaggle_data_4400x1700_INTER_CUBIC'

    crop_x_range = (0, 4352)
    crop_y_range = (0, 1696)
    time_range = (236, 4160)

    zero_mv_positions = [703.5, 987.5, 1271.5, 1531.5]
    mv_to_pixel = 79

    mv_limits = [
        [-2, 2], [-2, 2], [-4, 4], [-4, 4]
    ]

    save_frequency = 1

    def __init__(self):
        _init_dirs(self, "stage2_efficientnetv2")