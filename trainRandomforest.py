import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Load CSV
file_path = 'Randomforest\\data\\train_data.csv'  # แทนที่ด้วย path ของไฟล์ .csv
data = pd.read_csv(file_path)

# แยกฟีเจอร์และเลเบล
keypoints_columns = [col for col in data.columns if col not in ['image_name', 'label']]
features = data[keypoints_columns]
labels = data['label']

# 2. Translation Normalization
def calculate_center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

# คำนวณจุดศูนย์กลางระหว่างไหล่และสะโพก
center_shoulder_x, center_shoulder_y = calculate_center(
    features['left_shoulder_x'], features['left_shoulder_y'],
    features['right_shoulder_x'], features['right_shoulder_y']
)
center_hip_x, center_hip_y = calculate_center(
    features['left_hip_x'], features['left_hip_y'],
    features['right_hip_x'], features['right_hip_y']
)
center_x, center_y = calculate_center(center_shoulder_x, center_shoulder_y, center_hip_x, center_hip_y)

# ปรับ translation normalization
for col in features.columns:
    if '_x' in col:
        features[col] -= center_x
    elif '_y' in col:
        features[col] -= center_y

# 3. Scale Normalization
def calculate_high(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# ใช้ระยะระหว่างหัวไหล่และสะโพกเป็นตัวคำนวณ scale
high = calculate_high(center_shoulder_x, center_shoulder_y, center_hip_x, center_hip_y)

# ปรับ scale normalization
features = features.div(high, axis=0)

# 4. Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42) ### เพิ่ม n_estimators 100+
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"จำนวนข้อมูลทั้งหมด: {len(features)}")
print(f"จำนวนข้อมูลเทรน: {len(X_train)}")
print(f"จำนวนข้อมูลทดสอบ: {len(X_test)}")
# แสดงจำนวนฟีเจอร์และชื่อคอลัมน์หลังการปรับข้อมูล
print("\nหลังการปรับข้อมูล (Normalization):")
print(f"จำนวนฟีเจอร์ทั้งหมดหลังปรับข้อมูล: {features.shape[1]}")
print(f"ชื่อคอลัมน์หลังปรับข้อมูล: {features.columns.tolist()}")

# แสดงจำนวนฟีเจอร์ที่สำคัญที่สุดจาก Random Forest
importances = model.feature_importances_
important_features = sorted(zip(features.columns, importances), key=lambda x: x[1], reverse=True)

print("\nฟีเจอร์ที่สำคัญที่สุดตาม Random Forest:")
for feature, importance in important_features:
    print(f"{feature}: {importance:.4f}")

# 5. บันทึกโมเดลที่เทรนเสร็จแล้ว
model_save_path = 'Randomforest\\random_forest_model.pkl'
try:
    with open(model_save_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"โมเดลถูกบันทึกเรียบร้อยที่: {model_save_path}")
except Exception as e:
    print(f"Error saving model: {e}")