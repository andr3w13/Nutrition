import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
from functools import partial

from scripts.dataset import NutritionDataset, collate_fn, get_transforms, load_data


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_requires_grad(module, unfreeze_pattern=""):
    prefixes = tuple(unfreeze_pattern.split("|")) if unfreeze_pattern else ()
    for name, param in module.named_parameters():
        param.requires_grad = name.startswith(prefixes) if prefixes else False


class MultimodalModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(cfg.TEXT_MODEL_NAME)
        self.image_encoder = timm.create_model(cfg.IMAGE_MODEL_NAME, pretrained=True, num_classes=0)

        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, cfg.HIDDEN_DIM)
        self.image_projection = nn.Linear(self.image_encoder.num_features, cfg.HIDDEN_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.HIDDEN_DIM + 1, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.HIDDEN_DIM, 1),
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        image_emb = self.image_encoder(image)
        x = torch.cat([
            self.text_projection(text_emb),
            self.image_projection(image_emb),
            mass.unsqueeze(1),
        ], dim=1)
        return self.classifier(x).squeeze(1)


def predict(model, batch, cfg, device):
    """Возвращает предсказанные калории для батча."""
    mass = batch["mass"].to(device)
    out = model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["image"].to(device),
        mass / cfg.MASS_SCALE,
    )
    # выход модели переводим в калории: либо ккал на грамм * масса, либо сотни ккал * 100
    return out * mass if cfg.PREDICT_DENSITY else out * cfg.TARGET_SCALE


def get_loss(cfg):
    if cfg.LOSS == "smooth_l1":
        return nn.SmoothL1Loss(beta=cfg.SMOOTH_L1_BETA)
    return nn.L1Loss()


def get_dataloaders(cfg, tokenizer):
    train_df, val_df, test_df = load_data(cfg)
    collate = partial(collate_fn, tokenizer=tokenizer, max_len=cfg.MAX_TEXT_LEN)
    loader_args = dict(batch_size=cfg.BATCH_SIZE, collate_fn=collate, num_workers=cfg.NUM_WORKERS,
                       worker_init_fn=seed_worker)

    train_ds = NutritionDataset(train_df, cfg, get_transforms(cfg, train=True), shuffle_ingredients=True)
    val_ds = NutritionDataset(val_df, cfg, get_transforms(cfg))
    test_ds = NutritionDataset(test_df, cfg, get_transforms(cfg))

    generator = torch.Generator().manual_seed(cfg.SEED)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    mae = torchmetrics.MeanAbsoluteError().to(device)
    rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
    for batch in loader:
        preds = predict(model, batch, cfg, device)
        target = batch["calories"].to(device)
        mae.update(preds, target)
        rmse.update(preds, target)
    return {"mae": mae.compute().item(), "rmse": rmse.compute().item()}


def train(cfg):
    seed_everything(cfg.SEED)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print("Устройство", device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)
    model = MultimodalModel(cfg).to(device)
    set_requires_grad(model.text_encoder, cfg.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_encoder, cfg.IMAGE_MODEL_UNFREEZE)

    optimizer = AdamW([
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR},
        {"params": model.text_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.image_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.classifier.parameters(), "lr": cfg.CLASSIFIER_LR},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS) if cfg.USE_SCHEDULER else None
    criterion = get_loss(cfg)
    train_loader, val_loader, _ = get_dataloaders(cfg, tokenizer)

    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
    best_mae = float("inf")
    history = []

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = predict(model, batch, cfg, device)
            loss = criterion(preds, batch["calories"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(preds)
        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, cfg, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"]})
        print(f"Эпоха {epoch}/{cfg.EPOCHS} | train loss {train_loss:.2f} | "
              f"val MAE {val_metrics['mae']:.2f} | val RMSE {val_metrics['rmse']:.2f}")

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            torch.save(model.state_dict(), cfg.SAVE_PATH)

    print(f"Лучший val MAE {best_mae:.2f}")
    model.load_state_dict(torch.load(cfg.SAVE_PATH, map_location=device))
    return model, history
