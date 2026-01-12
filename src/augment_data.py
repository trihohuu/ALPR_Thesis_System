import albumentations as A
import cv2
import os
import glob
import shutil
import yaml
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 

TRAIN_DIR_RAW = os.path.join(PROJECT_ROOT, "data", "processed", "train")
IMG_DIR_RAW = os.path.join(TRAIN_DIR_RAW, "images")
LBL_DIR_RAW = os.path.join(TRAIN_DIR_RAW, "labels")

TRAIN_DIR_AUG = os.path.join(PROJECT_ROOT, "data", "processed", "train_aug")
IMG_DIR_AUG = os.path.join(TRAIN_DIR_AUG, "images")
LBL_DIR_AUG = os.path.join(TRAIN_DIR_AUG, "labels")

YAML_PATH_ORIG = os.path.join(PROJECT_ROOT, "data", "processed", "data.yaml")
YAML_PATH_AUG = os.path.join(PROJECT_ROOT, "data", "processed", "data_aug.yaml")

AUGMENT_TIMES = 2

transform = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.HueSaturationValue(p=1),
        A.CLAHE(p=1),
    ], p=0.7),

    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1),
        A.MotionBlur(blur_limit=(3, 7), p=1),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1),
    ], p=0.5),

    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.5),
    
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

def load_yolo_labels(txt_path):
    bboxes = []
    classes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                bboxes.append(coords)
                classes.append(cls)
    return bboxes, classes

def save_yolo_labels(txt_path, bboxes, classes):
    with open(txt_path, 'w') as f:
        for cls, bbox in zip(classes, bboxes):
            bbox = [max(0.0, min(1.0, x)) for x in bbox]
            line = f"{cls} {' '.join(map(str, bbox))}\n"
            f.write(line)

def create_aug_yaml():
    """Tạo file data_aug.yaml trỏ tới folder mới"""
    if not os.path.exists(YAML_PATH_ORIG):
        print("Không tìm thấy data.yaml gốc!")
        return

    with open(YAML_PATH_ORIG, 'r') as f:
        data = yaml.safe_load(f)

    data['train'] = os.path.abspath(IMG_DIR_AUG)
    data['val'] = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "processed", "valid", "images"))
    data['test'] = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "processed", "test", "images"))

    with open(YAML_PATH_AUG, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"--> Đã tạo file config mới: {YAML_PATH_AUG}")

def main():
    if os.path.exists(TRAIN_DIR_AUG):
        shutil.rmtree(TRAIN_DIR_AUG)
    os.makedirs(IMG_DIR_AUG)
    os.makedirs(LBL_DIR_AUG)

    image_paths = glob.glob(os.path.join(IMG_DIR_RAW, "*.*"))
    print(f"Tìm thấy {len(image_paths)} ảnh gốc. Bắt đầu xử lý...")

    for img_path in tqdm(image_paths):
        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]
        ext = os.path.splitext(fname)[1]
        
        txt_name = base_name + ".txt"
        txt_path = os.path.join(LBL_DIR_RAW, txt_name)

        shutil.copy(img_path, os.path.join(IMG_DIR_AUG, fname))
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(LBL_DIR_AUG, txt_name))

        image = cv2.imread(img_path)
        bboxes, classes = load_yolo_labels(txt_path)

        if not bboxes: continue

        for i in range(AUGMENT_TIMES):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=classes)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_classes = transformed['class_labels']

                if len(aug_bboxes) > 0:
                    new_fname = f"{base_name}_aug_{i}{ext}"
                    new_txt_name = f"{base_name}_aug_{i}.txt"

                    cv2.imwrite(os.path.join(IMG_DIR_AUG, new_fname), aug_img)
                    save_yolo_labels(os.path.join(LBL_DIR_AUG, new_txt_name), aug_bboxes, aug_classes)
            except Exception as e:
                pass 

    create_aug_yaml()
    print("xong")

if __name__ == "__main__":
    main()