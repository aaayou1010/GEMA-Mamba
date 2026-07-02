import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_split(image_dir, mask_dir, image_size=256):
    image_files = sorted([f for f in os.listdir(image_dir)
                         if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    images = []
    masks = []

    for image_file in tqdm(image_files, desc=f"Loading {os.path.basename(image_dir)}", unit="img"):
        base_name = os.path.splitext(image_file)[0]
        mask_file = f"{base_name}_segmentation.png"
        mask_path = os.path.join(mask_dir, mask_file)
        image_path = os.path.join(image_dir, image_file)

        if not os.path.exists(mask_path):
            print(f"Warning: mask not found for {image_file}")
            continue

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if image_size is not None:
            image = image.resize((image_size, image_size), Image.BILINEAR)
            mask = mask.resize((image_size, image_size), Image.NEAREST)

        images.append(np.array(image, dtype=np.uint8))
        masks.append(np.array(mask, dtype=np.uint8))

    if not images:
        raise ValueError(f"No valid image-mask pairs found in {image_dir}")

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    return images, masks



root = "/home/nihongyou/data/ISIC2017"

train_img_dir = os.path.join(root, "train", "images")
train_mask_dir = os.path.join(root, "train", "masks")

train_images, train_masks = load_split(train_img_dir, train_mask_dir, image_size=256)

np.save(os.path.join(root, "data_train.npy"), train_images)
np.save(os.path.join(root, "mask_train.npy"), train_masks)

val_img_dir = os.path.join(root, "val", "images")   
val_mask_dir = os.path.join(root, "val", "masks")


val_images, val_masks = load_split(val_img_dir, val_mask_dir, image_size=256)

np.save(os.path.join(root, "data_val.npy"), val_images)
np.save(os.path.join(root, "mask_val.npy"), val_masks)


test_img_dir = os.path.join(root, "test", "images")
test_mask_dir = os.path.join(root, "test", "masks")

test_images, test_masks = load_split(test_img_dir, test_mask_dir, image_size=256)

np.save(os.path.join(root, "data_test.npy"), test_images)
np.save(os.path.join(root, "mask_test.npy"), test_masks)

print("\n" + "=" * 60)
print("train images:", train_images.shape, "masks:", train_masks.shape)
print("val   images:", val_images.shape,   "masks:", val_masks.shape)
print("test  images:", test_images.shape,  "masks:", test_masks.shape)
