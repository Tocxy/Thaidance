import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import accuracy_score

# 1. โหลดรูปภาพ
image_path = 'Randomforest\images_test\นางรำNU\\4.จีบหงาย\\4.จีบหงาย(นางรำมอนอ)หลัง2.jpg'  # ใส่ path ของรูปภาพ
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
class_order = rf.classes_
print("Class order in model:", class_order)
# 7. ทำนายและแสดงผลลัพธ์
predicted_class = rf.predict(test_image_feature)
print(f'Predicted class: {predicted_class[0]}')

# 8. แสดงความน่าจะเป็นของแต่ละคลาส
class_probabilities = rf.predict_proba(test_image_feature)
# หาค่าคลาสที่มีความน่าจะเป็นสูงสุด
predicted_class_index = np.argmax(class_probabilities)
predicted_confidence = np.max(class_probabilities)  # ค่าความมั่นใจของคลาสที่มากที่สุด

# Mapping ของ labels (อัปเดตให้ตรงกับโมเดลของคุณ)
class_labels = {
    0: "Chip Khwam", 1: "Chip Ngai", 2: "Chip Pok Khang", 
    3: "Chip Pok Na", 4: "Diao Thao", 5: "Iang Sisa", 
    6: "Kao Khang", 7: "Kaona", 8: "KraThung Thao", 
    9: "Kradok Lang", 10: "Kradok Siao", 11: "Lak Kho",
    12: "Lo Kaeo", 13: "Pra Thao", 14: "Soi Thao", 
    15: "Wong Bon", 16: "Wong Klang", 17: "Wong Lang", 
    18: "Yan Thao", 19: "Yok Thao"
}

# ดึงชื่อคลาสที่พยากรณ์ได้
predicted_label = class_labels.get(predicted_class_index, "Unknown")

# แสดงผลลัพธ์ที่ความถูกต้องมากสุด
for class_name, probability in zip(class_order, class_probabilities[0]):
    print(f'{class_name}: {probability:.4f}')

print(f'Predicted class  : {predicted_label} (MAX Confidence: {predicted_confidence:.4f})')

# 9. ใส่ค่าคลาสที่แท้จริงที่ต้องการทดสอบ 
# คลาสที่ต้องการตรวจสอบ
true_class = ["Chip Ngai"]

# ตรวจสอบว่า `true_class` อยู่ในลิสต์ของคลาสที่โมเดลรู้จักหรือไม่
if true_class[0] in class_order:
    # หาตำแหน่ง index ของ `true_class` ในโมเดล
    true_class_index = np.where(class_order == true_class[0])[0][0]
    
    # ดึงค่าความน่าจะเป็นของ `true_class`
    true_class_probability = class_probabilities[0][true_class_index]

    print(f"True class: {true_class[0]}")
    print(f"Probability of '{true_class[0]}' in model: {true_class_probability:.4f}")
else:
    print(f"True class '{true_class[0]}' is not in the model's class list.")

