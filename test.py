
import os
import warnings

import torch
from torch.utils.data import DataLoader

from configs.config_setting import setting_config
from datasets.dataset import ACDC_dataset, LIDCROIDataset
from gema_engine import test_one_epoch, test_sy_ac, val_one_epoch_lidc
from gema_utils import get_logger, log_config_info, set_seed
from loader import isic_loader
from models.GEMAMamba.GEMAMamba import GEMAMamba

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ISIC_DATASETS = {"isic2017", "isic2018"}
SUPPORTED_DATASETS = ISIC_DATASETS | {"acdc", "lidc"}


def is_isic_dataset(name):
    return name in ISIC_DATASETS


def validate_dataset_support(config):
    if config.datasets_name not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"This refactored test entry only supports: {supported}. Got: {config.datasets_name}")


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
    return torch.nn.DataParallel(model.cuda(), device_ids=[0], output_device=0)


def build_test_loader(config):
    if is_isic_dataset(config.datasets_name):
        dataset = isic_loader(path_Data=config.data_path, train=False, Test=True)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=True,
        )
        return dataset, loader

    if config.datasets_name == "acdc":
        dataset = ACDC_dataset(base_dir=config.volume_path, list_dir=config.list_dir, split="test_vol", transform=None)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=True,
        )
        return dataset, loader

    if config.datasets_name == "lidc":
        dataset = LIDCROIDataset(base_dir=config.data_path, list_dir=config.list_dir, split="test", transform=None)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=False,
        )
        return dataset, loader

    raise ValueError(f"Unsupported dataset for testing: {config.datasets_name}")


def load_model_weights(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)


def main():
    config = setting_config
    config.add_argument_config()
    config.set_datasets()
    config.set_opt_sch()
    validate_dataset_support(config)

    log_dir, _, outputs_dir = build_common_dirs(config)
    logger = get_logger("test", log_dir)
    log_config_info(config, logger)

    set_seed(config.seed)
    torch.cuda.empty_cache()

    model = build_model(config)
    load_model_weights(model, config.best_model_path)
    model.eval()

    dataset, loader = build_test_loader(config)
    criterion = config.criterion

    if is_isic_dataset(config.datasets_name):
        with torch.no_grad():
            test_one_epoch(loader, model, criterion, logger, config)
        return

    if config.datasets_name == "acdc":
        with torch.no_grad():
            test_sy_ac(dataset, loader, model, logger, config, test_save_path=outputs_dir, val_or_test=True)
        return

    if config.datasets_name == "lidc":
        with torch.no_grad():
            loss, miou, dice = val_one_epoch_lidc(
                loader,
                model,
                criterion,
                epoch=0,
                logger=logger,
                config=config,
                save_path=outputs_dir,
            )
            logger.info(f"test of best model, loss: {loss:.4f}, miou: {miou:.4f}, dice: {dice:.4f}")


if __name__ == "__main__":
    main()
