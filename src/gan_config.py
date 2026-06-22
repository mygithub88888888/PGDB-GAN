import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_ROOT = "./data"

    BATCH_SIZE = 8
    NUM_EPOCHS = 2000
    LEARNING_RATE = 2e-4

    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    CHECKPOINT_INTERVAL = 100

    IMG_SIZE = 256

config = Config()