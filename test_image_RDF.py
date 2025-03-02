import cv2
import pickle
import numpy as np
from ultralytics import YOLO


# 1. โหลดรูปภาพ
image_path = 'Randomforest\\images_test\\images.jpg'  # ใส่ path ของรูปภาพ
image = cv2.imread(image_path)

# 2. ใช้ YOLOv8 เพื่อดึง keypoints
yolo_model_path = 'yolov8x-pose.pt'  # ใส่ path ของโมเดล YOLOv8 ที่เทรนไว้
yolo_model = YOLO(yolo_model_path)

# รัน YOLOv8 บนรูปภาพ
results = yolo_model(image)

# ดึง keypoints และย้ายข้อมูลจาก GPU ไปยัง CPU
keypoints = results[0].keypoints.xy.cpu().numpy()  # เข้าถึงค่า (x, y) และแปลงเป็น NumPy array

# 3. แปลง keypoints เป็น array แบบ 1D
# ใช้ flatten เพื่อเปลี่ยนเป็น 1D array
# ตัดฟีเจอร์ที่เกินออกจาก keypoints (เลือกแค่ 34 ฟีเจอร์ที่จำเป็น)
keypoints_selected = keypoints.flatten()[:34]  # เลือกแค่ 34 ฟีเจอร์แรก
# คำนวณ translation และ scale normalization

def normalize_keypoints(keypoints):
    shoulder_left = keypoints[5 * 2], keypoints[5 * 2 + 1]
    shoulder_right = keypoints[6 * 2], keypoints[6 * 2 + 1]
    hip_left = keypoints[11 * 2], keypoints[11 * 2 + 1]
    hip_right = keypoints[12 * 2], keypoints[12 * 2 + 1]
    
    shoulder_center = ((shoulder_left[0] + shoulder_right[0]) / 2, (shoulder_left[1] + shoulder_right[1]) / 2)
    hip_center = ((hip_left[0] + hip_right[0]) / 2, (hip_left[1] + hip_right[1]) / 2)
    
    translation = np.array([(shoulder_center[0] + hip_center[0]) / 2, (shoulder_center[1] + hip_center[1]) / 2])
    keypoints[::2] -= translation[0]
    keypoints[1::2] -= translation[1]
    
    shoulder_hip_distance = np.linalg.norm(np.array(shoulder_center) - np.array(hip_center))
    scale_factor = 1 / shoulder_hip_distance
    keypoints[::2] *= scale_factor
    keypoints[1::2] *= scale_factor

    return keypoints

# ทำการ normalize keypoints
normalized_keypoints = normalize_keypoints(keypoints_selected)

# แปลง keypoints ที่ normalized เป็น 1D array สำหรับการทำนาย
test_image_feature = normalized_keypoints.flatten()

# 4. โหลดโมเดล RandomForest
rf_model_path = 'Randomforest\\random_forest_model.pkl'  # ใส่ path ของโมเดล RandomForest
with open(rf_model_path, 'rb') as file:
    rf = pickle.load(file)


# 5. รีเชฟข้อมูลให้อยู่ในรูปแบบที่ RandomForest คาดหวัง
test_image_feature = test_image_feature.reshape(1, -1)  # ใช้ reshape เพื่อให้เป็น 2D array

# 6. ทำนายคลาสด้วย RandomForest
print(f'Model expects {rf.n_features_in_} features.')
print(f'Test data has {test_image_feature.shape[1]} features.')  # ตอนนี้ test_image_feature เป็น 2D แล้ว

# 7. ทำนายและแสดงผลลัพธ์
predicted_class = rf.predict(test_image_feature)
print(f'Predicted class: {predicted_class[0]}')

# 8. แสดงความน่าจะเป็นของแต่ละคลาส
class_probabilities = rf.predict_proba(test_image_feature)
print(f'Class probabilities: {class_probabilities}')
