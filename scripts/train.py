#!/usr/bin/env python
"""Main training entry point."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from tqdm import tqdm

from aapr.utils.config import get_config, Config
from aapr.utils.seed import set_seed
from aapr.utils.device import get_device
from aapr.utils.logging import setup_logger, get_writer
from aapr.data.cremad import CremaDDataset
from aapr.data.mderma import MDERMADataset
from aapr.data.tame import TAMEDataset
from aapr.data.utils import create_dataloaders
from aapr.features.mel_spectrogram import MelSpectrogramExtractor
from aapr.features.ssl_embeddings import SSLEmbeddingExtractor
from aapr.features.feature_cache import CachedFeatureDataset
from aapr.models.privacy_filter import PrivacyFilter
from aapr.models.task_model import TaskModel
from aapr.models.adversary import MultiHeadAdversary
from aapr.data.combined import CombinedEmotionDataset
from aapr.models.teacher import TeacherModel
from aapr.training.trainer import Trainer


DATASET_MAP = {
    "cremad": CremaDDataset,
    "mderma": MDERMADataset,
    "tame": TAMEDataset,
}


def build_dataset(cfg):
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    sample_rate = ds_cfg.get("sample_rate", 16000)
    max_len = ds_cfg.get("max_length_sec", 5.0)

    if name == "combined":
        # Joint CREMA-D + MDER-MA with 4-class common emotion space
        return CombinedEmotionDataset(
            cremad_root=ds_cfg.get("cremad_root", "data/raw/cremad"),
            mderma_root=ds_cfg.get("mderma_root", "data/raw/mderma"),
            sample_rate=sample_rate,
            max_length_sec=max_len,
        )

    cls = DATASET_MAP[name]
    kwargs = {"root": ds_cfg["root"], "sample_rate": sample_rate, "max_length_sec": max_len}
    if name == "tame" and "num_pain_bins" in ds_cfg:
        kwargs["num_pain_bins"] = ds_cfg["num_pain_bins"]
    return cls(**kwargs)


def build_feature_extractor(cfg):
    feat_cfg = cfg["feature"]
    if feat_cfg["type"] == "melspec":
        return MelSpectrogramExtractor(
            sample_rate=cfg["dataset"].get("sample_rate", 16000),
            n_fft=feat_cfg.get("n_fft", 2048),
            hop_length=feat_cfg.get("hop_length", 512),
            n_mels=feat_cfg.get("n_mels", 128),
        )
    elif feat_cfg["type"] == "hubert":
        return SSLEmbeddingExtractor(
            model_name=feat_cfg.get("hubert_model", "facebook/hubert-base-ls960"),
            freeze_ssl=feat_cfg.get("freeze_ssl", True),
        )
    else:
        raise ValueError(f"Unknown feature type: {feat_cfg['type']}")


def pretrain_teacher(cfg, train_loader, feature_extractor, device, logger):
    """Pre-train a teacher model on raw (unfiltered) mel features.

    The teacher is a deeper MLP that acts as an emotion recognition oracle.
    After pre-training it is frozen and used to generate soft targets for the
    student (privacy_filter + task_model) during adversarial training.
    """
    train_cfg = cfg["training"]
    teacher_epochs = train_cfg.get("teacher_pretrain_epochs", 30)
    teacher_lr = train_cfg.get("teacher_lr", 1e-3)
    filter_cfg = cfg["model"]["filter"]

    teacher = TeacherModel(
        input_dim=filter_cfg.get("input_dim", 128),
        hidden_dim=256,
        num_classes=cfg["dataset"].get("num_utility_classes", 6),
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(teacher.parameters(), lr=teacher_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=teacher_epochs, eta_min=teacher_lr * 0.01
    )
    criterion = torch.nn.CrossEntropyLoss()

    logger.info(f"Pre-training teacher for {teacher_epochs} epochs...")

    for epoch in range(teacher_epochs):
        teacher.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc=f"Teacher epoch {epoch}", leave=False):
            waveform = batch["waveform"].to(device)
            labels = batch["utility_label"].to(device)

            with torch.no_grad():
                features = feature_extractor(waveform)

            logits = teacher(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        logger.info(
            f"Teacher epoch {epoch} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Train Acc: {correct / total:.4f}"
        )

    teacher.freeze()
    logger.info("Teacher pre-training complete and frozen.")
    return teacher


def main():
    cfg = get_config()
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("aapr", log_file=str(output_dir / "train.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Dataset / dataloaders
    use_cache = cfg["feature"].get("use_cache", False)
    if use_cache:
        from aapr.data.utils import collate_fn
        from torch.utils.data import DataLoader
        cache_dir = cfg["feature"]["cache_dir"]
        train_ds = CachedFeatureDataset(cache_dir, "train")
        val_ds = CachedFeatureDataset(cache_dir, "val")
        loaders = {
            "train": DataLoader(train_ds, batch_size=cfg["dataset"]["batch_size"],
                                shuffle=True, collate_fn=collate_fn, drop_last=True),
            "val": DataLoader(val_ds, batch_size=cfg["dataset"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn),
        }
        feature_extractor = None
        input_dim = cfg["model"]["filter"]["input_dim"]
    else:
        dataset = build_dataset(cfg)
        loaders = create_dataloaders(
            dataset,
            batch_size=cfg["dataset"].get("batch_size", 32),
            train_ratio=cfg["dataset"].get("train_ratio", 0.7),
            val_ratio=cfg["dataset"].get("val_ratio", 0.15),
            num_workers=cfg["dataset"].get("num_workers", 0),
            seed=cfg.get("seed", 42),
        )
        feature_extractor = build_feature_extractor(cfg)
        input_dim = feature_extractor.output_dim

    # Models
    filter_cfg = cfg["model"]["filter"]
    privacy_filter = PrivacyFilter(
        input_dim=input_dim,
        hidden_dim=filter_cfg.get("hidden_dim", 256),
        output_dim=filter_cfg.get("output_dim", 128),
        num_layers=filter_cfg.get("num_layers", 3),
        use_vib=filter_cfg.get("use_vib", True),
        vib_beta=filter_cfg.get("vib_beta", 0.001),
        dropout=filter_cfg.get("dropout", 0.1),
    )

    task_cfg = cfg["model"]["task"]
    num_classes = (
        dataset.num_utility_classes if not use_cache
        else cfg["dataset"].get("num_utility_classes", 6)
    )
    task_model = TaskModel(
        input_dim=filter_cfg.get("output_dim", 128),
        hidden_dim=task_cfg.get("hidden_dim", 256),
        num_classes=num_classes,
        dropout=task_cfg.get("dropout", 0.2),
    )
    logger.info(f"Task model: {num_classes} classes ({getattr(dataset, 'utility_label_names', [])})")

    adv_cfg = cfg["model"]["adversary"]
    # Auto-derive speaker_id head size from the loaded dataset so it works for
    # any dataset (cremad=91, mderma=varies, combined=cremad+mderma speakers).
    default_heads = dict(adv_cfg.get("heads", {"gender": 2, "speaker_id": 91}))
    if not use_cache:
        default_heads["speaker_id"] = dataset.num_speakers
    adversary = MultiHeadAdversary(
        input_dim=filter_cfg.get("output_dim", 128),
        trunk_dim=adv_cfg.get("trunk_dim", 128),
        heads=default_heads,
        dropout=adv_cfg.get("dropout", 0.3),
    )
    logger.info(f"Adversary heads: {default_heads}")

    # Teacher pre-training (student-teacher distillation)
    train_cfg = cfg["training"]
    teacher = None
    if train_cfg.get("use_teacher", False) and feature_extractor is not None:
        feature_extractor.to(device)
        teacher = pretrain_teacher(cfg, loaders["train"], feature_extractor, device, logger)
        # Save teacher checkpoint
        teacher_path = output_dir / "teacher.pt"
        torch.save(teacher.state_dict(), teacher_path)
        logger.info(f"Teacher saved to {teacher_path}")

    # Main adversarial training
    num_epochs = train_cfg.get("num_epochs", 100)
    writer = get_writer(cfg["output"].get("tensorboard_dir", str(output_dir / "tensorboard")))

    trainer = Trainer(
        privacy_filter=privacy_filter,
        task_model=task_model,
        adversary=adversary,
        feature_extractor=feature_extractor,
        teacher=teacher,
        device=device,
        lr_main=train_cfg.get("lr_main", 1e-3),
        lr_adversary=train_cfg.get("lr_adversary", 5e-4),
        lambda_privacy=train_cfg.get("lambda_privacy", 1.0),
        lambda_warmup_epochs=train_cfg.get("lambda_warmup_epochs", 10),
        adversary_refresh_interval=train_cfg.get("adversary_refresh_interval", 20),
        adversary_retrain_epochs=train_cfg.get("adversary_retrain_epochs", 5),
        checkpoint_dir=cfg["output"].get("checkpoint_dir", str(output_dir / "checkpoints")),
        use_cached_features=use_cache,
        distillation_alpha=train_cfg.get("distillation_alpha", 0.0),
        distillation_temperature=train_cfg.get("distillation_temperature", 4.0),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        num_epochs=num_epochs,
    )

    final_metrics = trainer.fit(
        loaders["train"], loaders["val"],
        num_epochs=num_epochs,
        writer=writer,
    )

    logger.info(f"Training complete. Final val metrics: {final_metrics}")
    writer.close()


if __name__ == "__main__":
    main()
