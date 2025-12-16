"""Configuration settings for the Portrait Converter."""

import torch

class Config:
    """Global configuration."""

    # Video settings
    OUTPUT_WIDTH = 1080  # 9:16 portrait width
    OUTPUT_HEIGHT = 1920  # 9:16 portrait height
    OUTPUT_ASPECT_RATIO = 9 / 16

    # Detection thresholds
    POSE_THRESHOLD = 0.50
    FACE_THRESHOLD = 0.95

    # Scene detection
    SCENE_THRESHOLD = 20.0  # Sensitivity for scene detection
    MIN_SCENE_LENGTH = 5  # Minimum frames per scene

    # Lip movement detection
    LIP_MOVEMENT_THRESHOLD = 0.02  # Threshold for lip movement variance
    SAMPLE_FRAMES = 30  # Number of frames to sample for lip movement

    # GPU settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Debug settings
    DEBUG_MODE = False
    DEBUG_FRAME_INTERVAL = 30  # Save debug frame every N frames

    # Crop padding (to ensure eyes/nose are well-framed)
    CROP_PADDING_TOP = 0.3  # 30% padding above eyes
    CROP_PADDING_BOTTOM = 0.5  # 50% padding below nose
    CROP_PADDING_SIDES = 0.15  # 15% padding on sides

    @classmethod
    def set_debug(cls, debug: bool):
        """Enable or disable debug mode."""
        cls.DEBUG_MODE = debug

    @classmethod
    def set_thresholds(cls, pose_threshold: float = None, face_threshold: float = None):
        """Set detection thresholds."""
        if pose_threshold is not None:
            cls.POSE_THRESHOLD = pose_threshold
        if face_threshold is not None:
            cls.FACE_THRESHOLD = face_threshold
