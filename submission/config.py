class Config:
    """Configuration class for player re-identification system"""
    
    # Model settings
    MODEL_PATH = "models/yolo_model.pt"
    CONFIDENCE_THRESHOLD = 0.5
    
    # Tracking settings
    MAX_DISAPPEARED = 15
    SIMILARITY_THRESHOLD = 0.5
    TRAJECTORY_HISTORY = 20
    FEATURE_HISTORY = 5
    
    # Feature extraction settings
    COLOR_BINS = 32
    
    # Similarity weights
    COLOR_WEIGHT = 0.5
    SHAPE_WEIGHT = 0.3
    SPATIAL_WEIGHT = 0.2
    
    # Video processing settings
    PROCESS_EVERY_N_FRAMES = 1  # Process every frame
    MAX_FRAMES = None  # Process all frames
    
    # Output settings
    DRAW_TRACKS = True
    SAVE_FEATURES = False
    VERBOSE = True
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), value)
        return config
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            key.lower(): value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }