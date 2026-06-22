import os
import numpy as np
from PIL import Image
import json

def generate_face_mask(image_path, annotation_path, mask_dir):
    """

    Args:
        image_path (str):
        annotation_path (str):
        mask_dir (str):
    """
    img = Image.open(image_path)
    w, h = img.size

    mask = np.zeros((h, w), dtype=np.uint8)

    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            coords = list(map(int, line.strip().split()))
            if len(coords) != 4:
                continue
            x_min, y_min, x_max, y_max = coords
            mask[y_min:y_max, x_min:x_max] = 1

    img_name = os.path.basename(image_path)
    mask_name = img_name.replace('.jpg', '_mask.npy').replace('.png', '_mask.npy')
    np.save(os.path.join(mask_dir, mask_name), mask)

def batch_process(image_dir, annotation_dir, mask_dir):
    """

    Args:
        image_dir (str):
        annotation_dir (str):
        mask_dir (str):
    """
    os.makedirs(mask_dir, exist_ok=True)

    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_name)
        annot_name = os.path.splitext(img_name)[0] + '.txt'
        annot_path = os.path.join(annotation_dir, annot_name)

        if not os.path.exists(annot_path):
            print(f"Warning:  {annot_name} ")
            continue

        generate_face_mask(img_path, annot_path, mask_dir)
        print(f"Generated mask for {img_name}")

if __name__ == '__main__':
    image_dir = "./datasets/data_choose_enhance/image"
    annotation_dir = "./datasets/data_choose_enhance/label"
    mask_dir = "./datasets/data_choose_enhance/mask"

    batch_process(image_dir, annotation_dir, mask_dir)