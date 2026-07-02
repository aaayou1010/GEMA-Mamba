import random
import shutil
from pathlib import Path

import numpy as np
import pylidc as pl
from pylidc.utils import consensus
from scipy.ndimage import zoom


if not hasattr(np, "int"):
    np.int = int


DICOM_ROOT = Path("/mnt/f/LIDC-IDRI/LIDC-IDRI/LIDC-IDRI/manifest-1600709154662/LIDC-IDRI")

OUTPUT_ROOT = Path("/home/nihongyou/data/LIDC_IDRI")
LIST_DIR = Path("/home/nihongyou/projects/GEMAMamba-main/lists/lists_LIDC")

TRAIN_DIR = OUTPUT_ROOT / "train"
VALID_DIR = OUTPUT_ROOT / "valid"
TEST_DIR = OUTPUT_ROOT / "test"

HU_MIN = -1200
HU_MAX = 600
CONSENSUS_LEVEL = 0.5
RANDOM_SEED = 42

# Preprocess strategy:
# 1. Resample the full CT volume and nodule mask to 0.5 x 0.5 x 0.5 mm.
# 2. Use the nodule center to define a fixed 256 x 256 crop in the axial plane.
# 3. Keep only the single positive slice closest to the nodule center.
# 4. Apply basic HU clipping/normalization before the center crop.
TARGET_SPACING = (0.5, 0.5, 0.5)  # (y, x, z) in mm
OUTPUT_PATCH_SIZE = 256
MAX_SLICES_PER_NODULE = 3
MIN_NODULE_DIAMETER_MM = 8.0  # Set to None to keep all nodules.

# Safety switch: clear previous train/valid/test outputs before regenerating.
RESET_OUTPUT_DIRS = True

# Set to None for full dataset generation.
MAX_SCANS = None


def normalize_ct(image_2d):
    image_2d = np.clip(image_2d, HU_MIN, HU_MAX)
    image_2d = (image_2d - HU_MIN) / (HU_MAX - HU_MIN)
    return image_2d.astype(np.float32)

def infer_spacing_3d(scan):
    spacing_xy = getattr(scan, "pixel_spacing", None)
    spacing_z = getattr(scan, "slice_spacing", None)

    dicoms = None
    try:
        dicoms = scan.load_all_dicom_images(verbose=False)
    except TypeError:
        try:
            dicoms = scan.load_all_dicom_images()
        except Exception:
            dicoms = None
    except Exception:
        dicoms = None

    if isinstance(spacing_xy, (tuple, list, np.ndarray)) and len(spacing_xy) >= 2:
        spacing_y = float(spacing_xy[0])
        spacing_x = float(spacing_xy[1])
    elif isinstance(spacing_xy, (int, float)):
        spacing_y = float(spacing_xy)
        spacing_x = float(spacing_xy)
    elif dicoms:
        pixel_spacing = getattr(dicoms[0], "PixelSpacing", None)
        if pixel_spacing is not None and len(pixel_spacing) >= 2:
            spacing_y = float(pixel_spacing[0])
            spacing_x = float(pixel_spacing[1])
        else:
            spacing_y = 1.0
            spacing_x = 1.0
    else:
        spacing_y = 1.0
        spacing_x = 1.0

    if isinstance(spacing_z, (int, float)) and float(spacing_z) > 0:
        spacing_z = float(spacing_z)
    elif dicoms and len(dicoms) > 1:
        z_positions = []
        for dcm in dicoms:
            position = getattr(dcm, "ImagePositionPatient", None)
            if position is not None and len(position) >= 3:
                z_positions.append(float(position[2]))
        if len(z_positions) >= 2:
            z_positions = sorted(z_positions)
            diffs = np.diff(z_positions)
            diffs = diffs[np.abs(diffs) > 1e-6]
            spacing_z = float(np.median(np.abs(diffs))) if diffs.size > 0 else 1.0
        else:
            slice_thickness = getattr(dicoms[0], "SliceThickness", None)
            spacing_z = float(slice_thickness) if slice_thickness is not None else 1.0
    elif dicoms:
        slice_thickness = getattr(dicoms[0], "SliceThickness", None)
        spacing_z = float(slice_thickness) if slice_thickness is not None else 1.0
    else:
        spacing_z = 1.0

    return spacing_y, spacing_x, spacing_z


def crop_centered_2d(image_2d, mask_2d, center_y, center_x, crop_size, image_pad_value=HU_MIN):
    half = crop_size // 2
    h, w = image_2d.shape

    center_y = int(round(center_y))
    center_x = int(round(center_x))

    y1 = center_y - half
    x1 = center_x - half
    y2 = y1 + crop_size
    x2 = x1 + crop_size

    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - h)
    pad_right = max(0, x2 - w)

    if pad_top or pad_bottom or pad_left or pad_right:
        image_2d = np.pad(
            image_2d,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=image_pad_value,
        )
        mask_2d = np.pad(
            mask_2d,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        y1 += pad_top
        y2 += pad_top
        x1 += pad_left
        x2 += pad_left

    return image_2d[y1:y2, x1:x2], mask_2d[y1:y2, x1:x2]


def build_global_mask(volume_shape, cbbox, cmask):
    global_mask = np.zeros(volume_shape, dtype=np.uint8)
    target = global_mask[cbbox]
    if target.shape != cmask.shape:
        min_shape = tuple(min(a, b) for a, b in zip(target.shape, cmask.shape))
        if any(s <= 0 for s in min_shape):
            return None
        target_slices = tuple(slice(0, s) for s in min_shape)
        target[target_slices] = np.maximum(target[target_slices], cmask[target_slices])
        global_mask[cbbox] = target
    else:
        global_mask[cbbox] = np.maximum(target, cmask)
    return global_mask


def resample_volume(volume_hwd, zoom_factors, order):
    return zoom(volume_hwd, zoom_factors, order=order)


def select_center_slices(positive_slices, center_z, max_slices):
    if len(positive_slices) <= max_slices:
        return positive_slices

    positive_slices = sorted(positive_slices)
    ranked = sorted(positive_slices, key=lambda z: (abs(z - center_z), z))
    selected = sorted(ranked[:max_slices])
    return selected


def make_nodule_rois(scan, volume_hwd):
    rois = []
    nodules = scan.cluster_annotations()
    spacing_y, spacing_x, spacing_z = infer_spacing_3d(scan)

    zoom_factors = (
        spacing_y / TARGET_SPACING[0],
        spacing_x / TARGET_SPACING[1],
        spacing_z / TARGET_SPACING[2],
    )
    resampled_volume = resample_volume(volume_hwd, zoom_factors, order=1)

    for nodule_idx, anns in enumerate(nodules):
        if not anns:
            continue

        try:
            cmask, cbbox, _ = consensus(anns, clevel=CONSENSUS_LEVEL)
        except Exception as exc:
            print(f"[warning] consensus failed for {scan.patient_id}, nodule {nodule_idx}: {exc}")
            continue

        try:
            cmask = cmask.astype(np.uint8)
            if cmask.size == 0:
                print(f"[warning] empty consensus mask for {scan.patient_id}, nodule {nodule_idx}")
                continue

            coords = np.argwhere(cmask > 0)
            if coords.size == 0:
                print(f"[warning] no positive voxels in consensus mask for {scan.patient_id}, nodule {nodule_idx}")
                continue

            local_center_y, local_center_x, local_center_z = coords.mean(axis=0).tolist()

            global_y0 = cbbox[0].start or 0
            global_x0 = cbbox[1].start or 0
            global_z0 = cbbox[2].start or 0

            center_y = global_y0 + local_center_y
            center_x = global_x0 + local_center_x
            center_z = global_z0 + local_center_z

            global_mask = build_global_mask(volume_hwd.shape, cbbox, cmask)
            if global_mask is None:
                print(
                    f"[warning] invalid consensus shape for {scan.patient_id}, "
                    f"nodule {nodule_idx}: target={volume_hwd.shape}, cmask={cmask.shape}"
                )
                continue

            resampled_mask = resample_volume(global_mask, zoom_factors, order=0)
            resampled_mask = (resampled_mask > 0).astype(np.uint8)

            center_y_rs = center_y * zoom_factors[0]
            center_x_rs = center_x * zoom_factors[1]
            center_z_rs = center_z * zoom_factors[2]

            positive_slices = np.where(resampled_mask.any(axis=(0, 1)))[0].tolist()
            if not positive_slices:
                center_slice = int(round(center_z_rs))
                center_slice = max(0, min(center_slice, resampled_mask.shape[2] - 1))
                positive_slices = [center_slice]
            positive_slices = select_center_slices(
                positive_slices,
                center_z_rs,
                MAX_SLICES_PER_NODULE,
            )

            original_coords = np.argwhere(global_mask > 0)
            y_min, x_min, z_min = original_coords.min(axis=0)
            y_max, x_max, z_max = original_coords.max(axis=0)
            max_diameter_mm = max(
                (y_max - y_min + 1) * spacing_y,
                (x_max - x_min + 1) * spacing_x,
                (z_max - z_min + 1) * spacing_z,
            )
            if MIN_NODULE_DIAMETER_MM is not None and max_diameter_mm < MIN_NODULE_DIAMETER_MM:
                continue

            for slice_idx in positive_slices:
                image_slice = resampled_volume[:, :, slice_idx]
                mask_slice = resampled_mask[:, :, slice_idx]
                image_slice = normalize_ct(image_slice)
                crop_img, crop_mask = crop_centered_2d(
                    image_slice,
                    mask_slice,
                    center_y=center_y_rs,
                    center_x=center_x_rs,
                    crop_size=OUTPUT_PATCH_SIZE,
                    image_pad_value=0.0,
                )
                crop_mask = (crop_mask > 0).astype(np.uint8)

                rois.append({
                    "nodule_idx": nodule_idx,
                    "slice_idx": int(slice_idx),
                    "center_y_rs": float(center_y_rs),
                    "center_x_rs": float(center_x_rs),
                    "center_z_rs": float(center_z_rs),
                    "spacing_y": TARGET_SPACING[0],
                    "spacing_x": TARGET_SPACING[1],
                    "spacing_z": TARGET_SPACING[2],
                    "max_diameter_mm": float(max_diameter_mm),
                    "img": crop_img.astype(np.float32),
                    "label": crop_mask.astype(np.uint8),
                })
        except Exception as exc:
            print(f"[warning] roi build failed for {scan.patient_id}, nodule {nodule_idx}: {exc}")
            continue

    return rois


def save_roi_npz(dst_dir, names, patient_id, roi):
    filename = f"{patient_id}_nodule_{roi['nodule_idx']:02d}_slice_{roi['slice_idx']:03d}.npz"
    np.savez_compressed(
        dst_dir / filename,
        img=roi["img"].astype(np.float32),
        label=roi["label"].astype(np.uint8),
        center_y=np.float32(roi["center_y_rs"]),
        center_x=np.float32(roi["center_x_rs"]),
        center_z=np.float32(roi["center_z_rs"]),
        max_diameter_mm=np.float32(roi["max_diameter_mm"]),
        spacing_y=np.float32(roi["spacing_y"]),
        spacing_x=np.float32(roi["spacing_x"]),
        spacing_z=np.float32(roi["spacing_z"]),
        slice_index=np.int32(roi["slice_idx"]),
    )
    names.append(filename)


def reset_output_dirs():
    for dst_dir in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
    for list_path in [LIST_DIR / "train.txt", LIST_DIR / "valid.txt", LIST_DIR / "test.txt"]:
        if list_path.exists():
            list_path.unlink()


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if RESET_OUTPUT_DIRS:
        reset_output_dirs()

    for dst_dir in [TRAIN_DIR, VALID_DIR, TEST_DIR, LIST_DIR]:
        dst_dir.mkdir(parents=True, exist_ok=True)

    local_patient_ids = sorted([
        path.name for path in DICOM_ROOT.glob("LIDC-IDRI-*")
        if path.is_dir()
    ])

    if MAX_SCANS is not None:
        local_patient_ids = local_patient_ids[:MAX_SCANS]

    print(f"local patient count: {len(local_patient_ids)}")

    random.shuffle(local_patient_ids)

    total = len(local_patient_ids)
    n_train = int(total * 0.7)
    n_valid = int(total * 0.15)

    train_ids = set(local_patient_ids[:n_train])
    valid_ids = set(local_patient_ids[n_train:n_train + n_valid])

    train_names = []
    valid_names = []
    test_names = []

    for idx, patient_id in enumerate(local_patient_ids, 1):
        if patient_id in train_ids:
            split = "train"
            dst_dir = TRAIN_DIR
            names = train_names
        elif patient_id in valid_ids:
            split = "valid"
            dst_dir = VALID_DIR
            names = valid_names
        else:
            split = "test"
            dst_dir = TEST_DIR
            names = test_names

        print(f"[{idx}/{len(local_patient_ids)}] {patient_id} -> {split}")

        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
        if scan is None:
            print(f"[skip] no pylidc record for {patient_id}")
            continue

        try:
            volume_hwd = scan.to_volume(verbose=False)
        except Exception as exc:
            print(f"[skip] cannot load volume {patient_id}: {exc}")
            continue

        try:
            nodule_rois = make_nodule_rois(scan, volume_hwd)
        except Exception as exc:
            print(f"[skip] cannot build nodule rois {patient_id}: {exc}")
            continue

        if not nodule_rois:
            print(f"[skip] no valid nodule roi found for {patient_id}")
            continue

        print(f"  kept roi count: {len(nodule_rois)}")
        for roi in nodule_rois:
            save_roi_npz(dst_dir, names, patient_id, roi)

    (LIST_DIR / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (LIST_DIR / "valid.txt").write_text("\n".join(valid_names) + "\n", encoding="utf-8")
    (LIST_DIR / "test.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")

    print("Done.")
    print(f"train rois: {len(train_names)}")
    print(f"valid rois: {len(valid_names)}")
    print(f"test rois: {len(test_names)}")
    print(f"output root: {OUTPUT_ROOT}")
    print(f"list dir: {LIST_DIR}")
    print(
        "preprocess settings: "
        f"target_spacing={TARGET_SPACING[0]}x{TARGET_SPACING[1]}x{TARGET_SPACING[2]}mm, "
        f"crop_size={OUTPUT_PATCH_SIZE}x{OUTPUT_PATCH_SIZE}, "
        f"center_positive_slices={MAX_SLICES_PER_NODULE}, "
        f"min_nodule_diameter_mm={MIN_NODULE_DIAMETER_MM}, "
        f"hu_clip_range=({HU_MIN}, {HU_MAX})"
    )


if __name__ == "__main__":
    main()
