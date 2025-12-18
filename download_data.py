import os 
import shutil 
from roboflow import Roboflow

api_key = "rhPOY4I3qrx0L0aHX9rB"
workspace = "vietnam-license"
PROJECT = "vietnam-license-plate-hjswj"
version = 2
format = "yolov8"

dest_dir = os.path.join("data", "processed")

def download():
    rf = Roboflow(api_key = api_key)
    project = rf.workspace(workspace).project(PROJECT)
    dataset = project.version(version).download(model_format = format, location = "temp_download")

    subfolders = ['train', 'valid', 'test']
    files = ['data.yaml']

    for folder in subfolders: 
        src_path = os.path.join("temp_download", folder) 
        dest_path = os.path.join(dest_dir, folder)

        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)

        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
        
    for file in files:
        src_path = os.path.join("temp_download", file)
        dest_path = os.path.join(dest_dir, file)
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)

    if os.path.exists("temp_download"):
        shutil.rmtree("temp_download")

    yaml_path = os.path.join(dest_dir, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            content = f.read()
        
        new_content = content.replace("test: ../test/images", f"test: test/images")
        new_content = new_content.replace("train: ../train/images", f"train: train/images")
        new_content = new_content.replace("val: ../valid/images", f"val: valid/images")
        
        with open(yaml_path, 'w') as f:
            f.write(new_content)
        print("Fix đường dẫn trong file data.yaml")

if __name__ == "__main__":
    download()