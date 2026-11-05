"""Конфиг для обучения мультимодальной модели предсказания калорийности."""


class Config:
    SEED = 42
    DEVICE = "cuda"  # если CUDA недоступна, train переключится на cpu

    # данные
    DATA_DIR = "data"
    VAL_SIZE = 0.15  # доля train, которая уходит в валидацию
    OUTLIER_IDS = ["dish_1560974769"]  # ошибка разметки, найдена в EDA

    # модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "efficientnet_b3"
    # размораживаем только слои, имя которых начинается с этих префиксов
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
    PREDICT_DENSITY = False  # True: модель предсказывает ккал на грамм, потом умножаем на массу
    TARGET_SCALE = 100  # при PREDICT_DENSITY=False модель предсказывает калории в сотнях ккал

    # обучение
    LOSS = "l1"  # "l1" или "smooth_l1" (Huber)
    SMOOTH_L1_BETA = 50  # до какой ошибки в ккал лосс квадратичный
    BATCH_SIZE = 4  # маленький батч, чтобы не словить OOM при разморозке энкодеров
    EPOCHS = 20
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 1e-3
    NUM_WORKERS = 2

    # куда сохранять веса лучшей по val MAE эпохи
    SAVE_PATH = "models/best_model.pt"
