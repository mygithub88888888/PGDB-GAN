import os
import json
import shutil
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_dir', default='./datasets/DarkFace_Train_2021/image', help='')
    parser.add_argument('--label_dir', default='./datasets/DarkFace_Train_2021/label', help='')
    parser.add_argument('--output_image_dir', default='./datasets/data_choose/image', help='')
    parser.add_argument('--output_label_dir', default='./datasets/data_choose/label', help='')
    parser.add_argument('--min_face_ratio', type=float, default=0.005, help='0.0050.5%')
    parser.add_argument('--num_images', type=int, default=500, help='500')
    parser.add_argument('--test_mode', action='store_true', help='10')
    parser.add_argument('--visualize', action='store_true', help='')
    args = parser.parse_args()
    return args

def calculate_face_ratio(image_path, bbox):
    """"""
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        x1, y1, x2, y2 = bbox

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width <= 0 or face_height <= 0:
            return 0

        face_area = face_width * face_height
        image_area = img_width * img_height
        return face_area / image_area
    except Exception as e:
        print(f" {image_path}: {e}")
        return 0

def visualize_faces(image_path, bboxes, ratio):
    """"""
    img = cv2.imread(image_path)
    if img is None:
        return

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(img, f"Face Ratio: {ratio:.4f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter_faces(args):
    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_label_dir, exist_ok=True)

    qualified_images = []

    label_files = os.listdir(args.label_dir)

    if args.test_mode:
        print(f"[] 10")
        label_files = label_files[:10]

    for label_file in tqdm(label_files, desc=""):
        if not label_file.endswith('.txt'):
            if args.test_mode:
                print(f"txt{label_file}")
            continue

        base_name = os.path.splitext(label_file)[0]
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_file = None
        for ext in image_extensions:
            img_path = os.path.join(args.image_dir, f'{base_name}{ext}')
            if os.path.exists(img_path):
                image_file = f'{base_name}{ext}'
                break

        if not image_file:
            if args.test_mode:
                print(f"{label_file}{base_name} + ")
            continue

        image_path = os.path.join(args.image_dir, image_file)
        label_path = os.path.join(args.label_dir, label_file)

        if args.test_mode:
            print(f"\n{image_file}{label_file}")

        bboxes = []
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if args.test_mode:
                    print(f"{label_file}{lines}")
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 4:
                        if args.test_mode:
                            print(f"4{line}")
                        continue
                    x1, y1, x2, y2 = map(int, parts)
                    bboxes.append([x1, y1, x2, y2])
                    if args.test_mode:
                        print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        except Exception as e:
            if args.test_mode:
                print(f" {label_path} : {e}")
            continue

        if not bboxes:
            if args.test_mode:
                print(f"{label_file}")
            continue

        ratios = [calculate_face_ratio(image_path, bbox) for bbox in bboxes]
        max_ratio = max(ratios)

        if args.test_mode:
            print(f"{ratios}{max_ratio}")

        if args.visualize:
            visualize_faces(image_path, bboxes, max_ratio)

        if max_ratio >= args.min_face_ratio:
            qualified_images.append((image_path, label_path, max_ratio))
            if args.test_mode:
                print(f" {max_ratio} >= {args.min_face_ratio}")
        else:
            if args.test_mode:
                print(f" {max_ratio} < {args.min_face_ratio}")

    qualified_images.sort(key=lambda x: x[2], reverse=True)
    selected_images = qualified_images[:args.num_images]

    for img_path, label_path, ratio in tqdm(selected_images, desc=""):
        img_base_name = os.path.basename(img_path)
        label_base_name = os.path.basename(label_path)

        output_img_path = os.path.join(args.output_image_dir, img_base_name)
        output_label_path = os.path.join(args.output_label_dir, label_base_name)

        shutil.copy(img_path, output_img_path)
        shutil.copy(label_path, output_label_path)

    print(f"\n {len(selected_images)} ")
    print(f": {args.output_image_dir}")
    print(f": {args.output_label_dir}")

if __name__ == "__main__":
    args = parse_args()

    print("\n=====  =====")
    print(f": {args.image_dir}")
    print(f": {args.label_dir}")
    print(f": {args.output_image_dir}")
    print(f": {args.output_label_dir}")
    print(f": {args.min_face_ratio}")
    print(f": {args.num_images}")
    print(f": {args.test_mode}")
    print(f": {args.visualize}")
    print("===================\n")

    filter_faces(args)