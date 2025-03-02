import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# 1. โหลดโมเดลที่บันทึกไว้
model_path = 'Randomforest\\random_forest_model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("โหลดโมเดลสำเร็จ!")
except Exception as e:
    print(f"ไม่สามารถโหลดโมเดลได้: {e}")
    exit()

# 2. โหลดข้อมูลสำหรับการทดสอบ
test_file_path = 'Randomforest\\data\\test_data.csv'
try:
    test_data = pd.read_csv(test_file_path)
    print("โหลดข้อมูลทดสอบสำเร็จ!")
except Exception as e:
    print(f"ไม่สามารถโหลดข้อมูลทดสอบได้: {e}")
    exit()

# แยกฟีเจอร์และเลเบลจากข้อมูลทดสอบ
keypoints_columns = [col for col in test_data.columns if col not in ['image_name', 'label']]
features = test_data[keypoints_columns]
labels = test_data['label']

# 3. ทดสอบข้อมูล **ไม่ Normalize**
try:
    raw_predictions = model.predict(features)
    raw_accuracy = accuracy_score(labels, raw_predictions)
    print(f"ความแม่นยำของโมเดลบนข้อมูล **ไม่ Normalize**: {raw_accuracy:.2f}")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการทดสอบข้อมูลดิบ: {e}")

# 4. ทดสอบข้อมูล **Normalize**
# คำนวณจุดศูนย์กลาง (Translation Normalization)
def calculate_center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

center_shoulder_x, center_shoulder_y = calculate_center(
    features['left_shoulder_x'], features['left_shoulder_y'],
    features['right_shoulder_x'], features['right_shoulder_y']
)
center_hip_x, center_hip_y = calculate_center(
    features['left_hip_x'], features['left_hip_y'],
    features['right_hip_x'], features['right_hip_y']
)
center_x, center_y = calculate_center(center_shoulder_x, center_shoulder_y, center_hip_x, center_hip_y)

# ปรับ Translation Normalization
normalized_features = features.copy()
for col in normalized_features.columns:
    if '_x' in col:
        normalized_features[col] -= center_x
    elif '_y' in col:
        normalized_features[col] -= center_y

# คำนวณระยะสูงและปรับ Scale Normalization
def calculate_high(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

high = calculate_high(center_shoulder_x, center_shoulder_y, center_hip_x, center_hip_y)
normalized_features = normalized_features.div(high, axis=0)

# ทดสอบข้อมูล Normalize
try:
    normalized_predictions = model.predict(normalized_features)
    normalized_accuracy = accuracy_score(labels, normalized_predictions)
    print(f"ความแม่นยำของโมเดลบนข้อมูล **Normalize**: {normalized_accuracy:.2f}")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการทดสอบข้อมูล Normalize: {e}")

# 5. เปรียบเทียบผลลัพธ์
print("\n=== สรุปผลการเปรียบเทียบ ===")
print(f"ความแม่นยำ (ไม่ Normalize): {raw_accuracy:.2f}")
print(f"ความแม่นยำ (Normalize): {normalized_accuracy:.2f}")


