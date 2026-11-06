import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def ingredients_to_text(ingr_ids, id2name):
    names = [id2name[int(i.replace("ingr_", ""))] for i in ingr_ids.split(";")]
    return ", ".join(n for n in names if n != "deprecated")


def load_data(cfg):
    dish = pd.read_csv(os.path.join(cfg.DATA_DIR, "dish.csv"))
    ingredients = pd.read_csv(os.path.join(cfg.DATA_DIR, "ingredients.csv"))
    id2name = dict(zip(ingredients["id"], ingredients["ingr"]))
    dish["ingredients"] = dish["ingredients"].apply(lambda s: ingredients_to_text(s, id2name))

    test_df = dish[dish["split"] == "test"]
    train_df = dish[(dish["split"] == "train") & ~dish["dish_id"].isin(cfg.OUTLIER_IDS)]
    val_df = train_df.sample(frac=cfg.VAL_SIZE, random_state=cfg.SEED)
    train_df = train_df.drop(val_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_transforms(cfg, train=False):
    aug = []
    if train:
        aug = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0, p=0.5),
        ]
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        *aug,
        A.Normalize(mean=cfg.IMG_MEAN, std=cfg.IMG_STD),
        ToTensorV2(),
    ], seed=cfg.SEED)


class NutritionDataset(Dataset):
    def __init__(self, df, cfg, transforms, shuffle_ingredients=False):
        self.df = df
        self.data_dir = cfg.DATA_DIR
        self.transforms = transforms
        self.shuffle_ingredients = shuffle_ingredients

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.data_dir, "images", str(row["dish_id"]), "rgb.png")
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image=np.array(image))["image"]

        # порядок ингредиентов ничего не значит
        text = row["ingredients"]
        if self.shuffle_ingredients and text:
            text = ", ".join(np.random.permutation(text.split(", ")))

        return {
            "image": image,
            "text": text,
            "mass": float(row["total_mass"]),
            "calories": float(row["total_calories"]),
        }


def collate_fn(batch, tokenizer, max_len):
    tokens = tokenizer(
        [item["text"] for item in batch],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "mass": torch.tensor([item["mass"] for item in batch], dtype=torch.float32),
        "calories": torch.tensor([item["calories"] for item in batch], dtype=torch.float32),
    }
