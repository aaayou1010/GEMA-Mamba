
import os
import sys
import warnings

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from configs.config_setting import setting_config
from datasets.dataset import ACDC_dataset, LIDCCropGenerator, LIDCROIDataset, RandomGenerator
from gema_engine import test_one_epoch, train_one_epoch_isic, train_one_epoch_sy_ac, val_one_epoch_lidc
from gema_utils import get_logger, get_optimizer, get_scheduler, log_config_info, set_seed, test_single_volume
from loader import isic_loader
from models.GEMAMamba.GEMAMamba import GEMAMamba

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ISIC_DATASETS = {"isic2017", "isic2018"}
SUPPORTED_DATASETS = ISIC_DATASETS | {"acdc", "lidc"}


def is_isic_dataset(name):
    return name in ISIC_DATASETS


def build_common_dirs(config):
    log_dir = os.path.join(config.work_dir, "log")
    checkpoint_dir = os.path.join(config.work_dir, "checkpoints")
    outputs_dir = os.path.join(config.work_dir, "outputs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    return log_dir, checkpoint_dir, outputs_dir


def build_model(config):
    model_cfg = config.model_config
    model = GEMAMamba(
        num_classes=model_cfg["num_classes"],
        input_channels=model_cfg["input_channels"],
        depths=model_cfg["depths"],
        depths_decoder=model_cfg["depths_decoder"],
        drop_path_rate=model_cfg["drop_path_rate"],
        load_ckpt_path=model_cfg["load_ckpt_path"],
    )
    model.load_from()
    model = torch.nn.DataParallel(model.cuda(), device_ids=[0], output_device=0)
    return model


def build_isic_loaders(config):
    train_dataset = isic_loader(path_Data=config.data_path, train=True)
    val_dataset = isic_loader(path_Data=config.data_path, train=False)
    test_dataset = isic_loader(path_Data=config.data_path, train=False, Test=True)
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.num_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=True,
        ),
        "val_dataset": val_dataset,
    }


def build_acdc_loaders(config):
    train_dataset = ACDC_dataset(
        base_dir=config.data_path,
        list_dir=config.list_dir,
        split="train",
        transform=transforms.Compose([RandomGenerator(output_size=[config.input_size_h, config.input_size_w])]),
    )
    val_dataset = ACDC_dataset(
        base_dir=config.volume_path,
        list_dir=config.list_dir,
        split="test_vol",
        transform=None,
    )
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.num_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=True,
        ),
        "val_dataset": val_dataset,
    }


def build_lidc_loaders(config):
    train_dataset = LIDCROIDataset(
        base_dir=config.data_path,
        list_dir=config.list_dir,
        split="train",
        transform=transforms.Compose([LIDCCropGenerator(output_size=[config.input_size_h, config.input_size_w])]),
    )
    val_dataset = LIDCROIDataset(
        base_dir=config.data_path,
        list_dir=config.list_dir,
        split="valid",
        transform=None,
    )
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.num_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=False,
        ),
        "val_dataset": val_dataset,
    }


def build_dataloaders(config):
    if is_isic_dataset(config.datasets_name):
        return build_isic_loaders(config)
    if config.datasets_name == "acdc":
        return build_acdc_loaders(config)
    if config.datasets_name == "lidc":
        return build_lidc_loaders(config)
    raise ValueError(f"Unsupported dataset for this refactored entry: {config.datasets_name}")


def fast_validate_acdc(val_dataset, val_loader, model, epoch, logger, config):
    metric_sum = None
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(val_loader)):
            image, label, case_name = batch["image"], batch["label"], batch["case_name"][0]
            metric_i = test_single_volume(
                image,
                label,
                model,
                classes=config.model_config["num_classes"],
                patch_size=[config.input_size_h, config.input_size_w],
                test_save_path=None,
                case=case_name,
                z_spacing=config.z_spacing,
                val_or_test=False,
            )
            metric_i = np.array(metric_i, dtype=np.float32)
            metric_sum = metric_i if metric_sum is None else metric_sum + metric_i
            logger.info(
                "idx %d case %s mean_dice %f mean_hd95 %f"
                % (batch_index, case_name, float(metric_i.mean(axis=0)[0]), float(metric_i.mean(axis=0)[1]))
            )
    metric_mean = metric_sum / len(val_dataset)
    performance = metric_mean.mean(axis=0)[0]
    mean_hd95 = metric_mean.mean(axis=0)[1]
    logger.info(f"val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}")
    return performance, mean_hd95


def load_resume_if_available(model, optimizer, scheduler, resume_path, logger):
    state = {
        "start_epoch": 1,
        "best_score": 0.0,
    }
    if not os.path.exists(resume_path):
        return state

    checkpoint = torch.load(resume_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        state["start_epoch"] = checkpoint["epoch"] + 1
        state["best_score"] = checkpoint.get("best_score", 0.0)
        logger.info(f"Resumed from {resume_path} at epoch {checkpoint['epoch']}.")
    except RuntimeError as exc:
        logger.info(f"Skip resume because checkpoint is incompatible: {exc}")
    return state


def save_latest_checkpoint(model, optimizer, scheduler, epoch, train_loss, best_score, checkpoint_dir):
    torch.save(
        {
            "epoch": epoch,
            "loss": train_loss,
            "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        os.path.join(checkpoint_dir, "latest.pth"),
    )


def maybe_run_validation(config, loaders, model, criterion, epoch, logger, outputs_dir):
    if is_isic_dataset(config.datasets_name):
        if epoch <= 100 or epoch % 20 != 0:
            return None
        _, miou, dice = test_one_epoch(loaders["test"], model, criterion, logger, config)
        return {"score": dice, "filename": f"epoch{epoch}-miou{miou:.4f}-dsc{dice:.4f}.pth"}

    if config.datasets_name == "acdc":
        if epoch <= 10 or epoch % 10 != 0:
            return None
        mean_dice, mean_hd95 = fast_validate_acdc(loaders["val_dataset"], loaders["val"], model, epoch, logger, config)
        return {"score": mean_dice, "filename": f"epoch{epoch}-mean_dice{mean_dice:.4f}-mean_hd95{mean_hd95:.4f}.pth"}

    if config.datasets_name == "lidc":
        if epoch <= 100 or epoch % 20 != 0:
            return None
        save_path = os.path.join(outputs_dir, f"val_epoch_{epoch}")
        _, miou, dice = val_one_epoch_lidc(loaders["val"], model, criterion, epoch, logger, config, save_path=save_path)
        return {"score": dice, "filename": f"epoch{epoch}-miou{miou:.4f}-dice{dice:.4f}.pth"}

    return None


def train_one_epoch_for_dataset(config, loaders, model, criterion, optimizer, scheduler, epoch, logger, scaler):
    if is_isic_dataset(config.datasets_name):
        return train_one_epoch_isic(loaders["train"], model, criterion, optimizer, scheduler, epoch, logger, config, scaler=scaler)
    return train_one_epoch_sy_ac(loaders["train"], model, criterion, optimizer, scheduler, epoch, logger, config, scaler=scaler)


def plot_loss_curve(loss_values, work_dir):
    plt.figure()
    axis = plt.gca()
    axis.plot(loss_values)
    axis.set_title("Training Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    plt.savefig(os.path.join(work_dir, "loss.png"))
    plt.close()


def validate_dataset_support(config):
    if config.datasets_name not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"This refactored train entry only supports: {supported}. Got: {config.datasets_name}")


def main():
    config = setting_config
    config.add_argument_config()
    config.set_datasets()
    config.set_opt_sch()
    validate_dataset_support(config)

    log_dir, checkpoint_dir, outputs_dir = build_common_dirs(config)
    logger = get_logger("train", log_dir)
    log_config_info(config, logger)

    sys.path.append(config.work_dir + "/")
    set_seed(config.seed)
    torch.cuda.empty_cache()

    loaders = build_dataloaders(config)
    model = build_model(config)
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    resume_state = load_resume_if_available(
        model,
        optimizer,
        scheduler,
        os.path.join(checkpoint_dir, "latest.pth"),
        logger,
    )
    start_epoch = resume_state["start_epoch"]
    best_score = resume_state["best_score"]

    loss_values = []
    for epoch in tqdm(range(start_epoch, config.epochs + 1)):
        torch.cuda.empty_cache()
        train_loss = train_one_epoch_for_dataset(
            config,
            loaders,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            scaler,
        )
        loss_values.append(train_loss)
        plot_loss_curve(loss_values, config.work_dir)
        validation_result = maybe_run_validation(config, loaders, model, criterion, epoch, logger, outputs_dir)

        if validation_result is not None and validation_result["score"] > best_score:
            best_score = validation_result["score"]
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, validation_result["filename"]))

        save_latest_checkpoint(model, optimizer, scheduler, epoch, train_loss, best_score, checkpoint_dir)


if __name__ == "__main__":
    main()
