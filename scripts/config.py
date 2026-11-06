class Config:
    SEED = 42
    DEVICE = "cuda" 

    # данные
    DATA_DIR = "data"
    VAL_SIZE = 0.15  # доля train, которая уходит в val
    OUTLIER_IDS = ["dish_1560974769"]  # ошибка разметки

    # модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "efficientnet_b3"
    TEXT_MODEL_UNFREEZE = "encoder.layer.10|encoder.layer.11"
    IMAGE_MODEL_UNFREEZE = "blocks.5|blocks.6|conv_head|bn2"
    HIDDEN_DIM = 256
    DROPOUT = 0.2

    # входы и таргет
    IMG_SIZE = 300
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD = (0.229, 0.224, 0.225)
    MAX_TEXT_LEN = 128
    MASS_SCALE = 1000  # масса подаётся в модель в килограммах
    PREDICT_DENSITY = True  # True: ккал на грамм; False: сотни ккал на порцию
    TARGET_SCALE = 100  

    # обучение
    LOSS = "l1"  
    SMOOTH_L1_BETA = 50 
    BATCH_SIZE = 16
    EPOCHS = 30
    USE_SCHEDULER = True  
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 1e-3
    NUM_WORKERS = 2

    SAVE_PATH = "models/best_model.pt"
