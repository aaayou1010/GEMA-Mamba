import re
from pathlib import Path

import h5py
import numpy as np


SRC_SLICE_DIR = Path("/home/nihongyou/data/ACDC_raw/ACDC_training_slices")
SRC_VOLUME_DIR = Path("/home/nihongyou/data/ACDC_raw/ACDC_training_volumes")

DST_ROOT = Path("/home/nihongyou/data/ACDC")
DST_TRAIN = DST_ROOT / "train"
DST_VALID = DST_ROOT / "valid"
DST_TEST = DST_ROOT / "test"

LIST_DIR = Path("/home/nihongyou/projects/GEMAMamba-main/lists/lists_ACDC")
TRAIN_LIST = LIST_DIR / "train.txt"
VALID_LIST = LIST_DIR / "valid.txt"
TEST_LIST = LIST_DIR / "test_vol.txt"


def get_patient_id(path: Path):
    match = re.search(r"patient(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def read_h5_img_label(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

        image_key = None
        label_key = None

        for candidate in ["image", "img"]:
            if candidate in keys:
                image_key = candidate
                break

        for candidate in ["label", "mask"]:
            if candidate in keys:
                label_key = candidate
                break

        if image_key is None or label_key is None:
            raise KeyError(f"{h5_path} keys={keys}, cannot find image/img and label/mask")

        image = f[image_key][:]
        label = f[label_key][:]

    return image.astype(np.float32), label.astype(np.uint8)


def convert_h5_to_npz(src_h5: Path, dst_npz: Path):
    image, label = read_h5_img_label(src_h5)
    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_npz, img=image, label=label)


def main():
    for directory in [DST_TRAIN, DST_VALID, DST_TEST, LIST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    train_names = []
    valid_names = []
    test_names = []

    slice_files = sorted(SRC_SLICE_DIR.glob("*.h5"))

    for src_h5 in slice_files:
        if "(1)" in src_h5.name:
            continue

        patient_id = get_patient_id(src_h5)
        if patient_id is None:
            print(f"[skip unknown name] {src_h5.name}")
            continue

        dst_name = src_h5.with_suffix(".npz").name

        if 1 <= patient_id <= 70:
            convert_h5_to_npz(src_h5, DST_TRAIN / dst_name)
            train_names.append(dst_name)
        elif 71 <= patient_id <= 80:
            convert_h5_to_npz(src_h5, DST_VALID / dst_name)
            valid_names.append(dst_name)

    volume_files = sorted(SRC_VOLUME_DIR.glob("*.h5"))

    for src_h5 in volume_files:
        if "(1)" in src_h5.name:
            continue

        patient_id = get_patient_id(src_h5)
        if patient_id is None:
            print(f"[skip unknown name] {src_h5.name}")
            continue

        dst_name = src_h5.with_suffix(".npz").name

        if 81 <= patient_id <= 100:
            convert_h5_to_npz(src_h5, DST_TEST / dst_name)
            test_names.append(dst_name)

    TRAIN_LIST.write_text("\n".join(train_names) + "\n", encoding="utf-8")
    VALID_LIST.write_text("\n".join(valid_names) + "\n", encoding="utf-8")
    TEST_LIST.write_text("\n".join(test_names) + "\n", encoding="utf-8")

    print(f"train slices: {len(train_names)}")
    print(f"valid slices: {len(valid_names)}")
    print(f"test volumes: {len(test_names)}")
    print(f"saved to: {DST_ROOT}")
    print(f"lists saved to: {LIST_DIR}")


if __name__ == "__main__":
    main()
