import sys
import cv2
import numpy as np
from pydantic import BaseModel
import os
import csv
import ultralytics
from ultralytics.engine.results import Results

# กำหนดคีย์พอยต์ตามโมเดล YOLOv8
class Keypoints(BaseModel):
    NOSE: int = 0
    LEFT_EYE: int = 1
    RIGHT_EYE: int = 2
    LEFT_EAR: int = 3
    RIGHT_EAR: int = 4
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW: int = 7
    RIGHT_ELBOW: int = 8
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    LEFT_HIP: int = 11
    RIGHT_HIP: int = 12
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 14
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16

# คลาสตรวจจับท่าทางเพื่อจัดการการโหลดโมเดลและการสกัดคีย์พอยต์
class PoseDetector:
    def __init__(self, model_name='yolov8x-pose'):
        self.model_name = model_name
        self.keypoints = Keypoints()
        self._load_model()

    def _load_model(self):
        if not self.model_name.endswith('-pose'):
            sys.exit('โมเดลไม่ใช่ YOLOv8 pose')
        self.model = ultralytics.YOLO(model=self.model_name)

    def extract_keypoints(self, keypoint_array: np.ndarray) -> list:
        return [keypoint_array[getattr(self.keypoints, key)].tolist() for key in self.keypoints.__fields__]

    def get_keypoints(self, results: Results) -> list:
        keypoint_array = results.keypoints.xyn.cpu().numpy()[0]
        return self.extract_keypoints(keypoint_array)

    def save_keypoints(self, image_name: str, label: str, keypoints: list, filepath: str):
        row = [image_name, label]
        for point in keypoints:
            if isinstance(point, list):
                row.extend(point)
            else:
                row.append(point)
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def detect(self, image_path: str) -> Results:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"ไม่พบภาพที่ {image_path}")
        results = self.model.predict(image, save=False)[0]
        return results


def determine_label(image_path: str) -> str:
    labels = {
        "วงบน": "Wong Bon",
        "วงกลาง": "Wong Klang",
        "วงล่าง": "Wong Lang",
        "จีบหงาย": "Chip Ngai",
        "จีบคว่ำ": "Chip Khwam",
        "จีบปกหน้า": "Chip Pok Na",
        "จีบปกข้าง": "Chip Pok Khang",
        "ล่อแก้ว": "Lo Kaeo",
        "ประเท้า": "Pra Thao",
        "ยกเท้า": "Yok Thao",
        "ก้าวหน้า": "Kaona",
        "ก้าวข้าง": "Kao Khang",
        "กระทุ้งเท้า": "KraThung Thao",
        "กระดกหลัง": "Kradok Lang",
        "กระดกเสี้ยว": "Kradok Siao",
        "เดี่ยวเท้า": "Diao Thao",
        "ซอยเท้า": "Soi Thao",
        "ขยั่นเท้า": "Yan Thao",
        "ลักคอ": "Lak Kho",
        "เอียงศรีษะ": "Iang Sisa",
    }
    
    for keyword, label in labels.items():
        if keyword in image_path:
            return label
    return "Other Label"

def process_images(input_dirs: list, output_file: str, detector: PoseDetector):
    header = [
        "image_name", "label", "nose_x", "nose_y", "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y", 
        "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y", "left_shoulder_x", "left_shoulder_y", 
        "right_shoulder_x", "right_shoulder_y", "left_elbow_x", "left_elbow_y", "right_elbow_x", "right_elbow_y", 
        "left_wrist_x", "left_wrist_y", "right_wrist_x", "right_wrist_y", "left_hip_x", "left_hip_y", 
        "right_hip_x", "right_hip_y", "left_knee_x", "left_knee_y", "right_knee_x", "right_knee_y", 
        "left_ankle_x", "left_ankle_y", "right_ankle_x", "right_ankle_y"
    ]
    
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    for input_dir in input_dirs:
        for root, dirs, files in os.walk(input_dir):
            for image_file in files:
                if image_file.endswith(('.png', '.jpg')):
                    image_path = os.path.join(root, image_file)
                    label = determine_label(image_path)
                    try:
                        results = detector.detect(image_path)
                        keypoints = detector.get_keypoints(results)
                        detector.save_keypoints(image_file, label, keypoints, output_file)
                        print(f"บันทึกคีย์พอยต์สำหรับ {image_file} พร้อม label '{label}' ไปที่ {output_file}")
                    except Exception as e:
                        print(f"เกิดข้อผิดพลาดในการประมวลผล {image_file}: {e}")

if __name__ == '__main__':
    input_directories = ["dataset\\1.วงบนพระนาง\\images",
                         "dataset\\2.วงกลางพระนาง\\images",
                         "dataset\\3.วงล่างพระนาง\\images",
                         "dataset\\4.จีบหงาย\\images",
                         "dataset\\5.จีบคว่ำ\\images",
                         "dataset\\6.จีบปกหน้า\\images",
                         "dataset\\7.จีบปกข้าง\\images",
                         "dataset\\8.ล่อแก้ว\\images",
                         "dataset\\9.ประเท้า\\images",
                         "dataset\\10.ยกเท้า\\images",
                         "dataset\\11.ก้าวหน้าพระนาง ซ-ข\\images",
                         "dataset\\12.ก้าวข้างพระนาง ซ-ข\\images",
                         "dataset\\13.กระทุ้งเท้าพระนาง ซ-ข\\images",
                         "dataset\\14.กระดกหลังพระนาง ซ-ข\\images",
                         "dataset\\15.กระดกเสี้ยวพระนาง ซ-ข\\images",
                         "dataset\\16.เดี่ยวเท้า ซ-ข\\images",
                         "dataset\\17.ซอยเท้า พระนาง\\images",
                         "dataset\\18.ขยั่นเท้าพระนาง ซ-ข\\images",
                         "dataset\\19.ลักคอ ซ-ข\\images",
                         "dataset\\20.เอียงศรีษะ ซ-ข\\images"
                         ]  # ใส่โฟลเดอร์อื่นๆ ที่ต้องการ

    output_file = 'Randomforest\\data\\keypoints.csv'
    pose_detector = PoseDetector()

    process_images(input_directories, output_file, pose_detector)
