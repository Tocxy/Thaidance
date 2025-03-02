from flask import Flask, request, render_template, jsonify
import cv2
import os
import numpy as np
import pickle
from ultralytics import YOLO
import warnings
import base64

app = Flask(__name__)

# โหลดโมเดล YOLOv8 และ RandomForest
try:
    yolo_model_path = 'yolov8x-pose.pt'  # Path ของ YOLOv8
    rf_model_path = 'random_forest_model.pkl'  # Path ของ Random Forest
    yolo_model = YOLO(yolo_model_path)
    with open(rf_model_path, 'rb') as file:
        rf = pickle.load(file)
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

# ฟังก์ชัน Normalize Keypoints
def normalize_keypoints(keypoints):
    try:
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
    except Exception as e:
        raise ValueError(f"Error normalizing keypoints: {e}")

# ฟังก์ชันสำหรับวาด keypoints บนภาพ
def draw_keypoints(image, keypoints):
    height, width = image.shape[:2]
    for i in range(0, len(keypoints), 2):
        x, y = int(keypoints[i]), int(keypoints[i + 1])  # ใช้พิกัดจริงของภาพ
        if 0 <= x < width and 0 <= y < height:  # ตรวจสอบว่า keypoints อยู่ในภาพ
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # วาดวงกลมสีแดง
    return image

# แปลงภาพที่มีการวาด keypoints เป็น Base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Route สำหรับหน้าเว็บ
@app.route('/')
def index():
    return render_template('index.html')

# Route สำหรับอัปโหลดและประมวลผลภาพ
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # บันทึกภาพชั่วคราว
        image_path = os.path.join('temp', file.filename)
        file.save(image_path)

        # อ่านและประมวลผลภาพ
        image = cv2.imread(image_path)
        results = yolo_model(image)
        keypoints = results[0].keypoints.xy.cpu().numpy().flatten()[:34]  # ใช้พิกัดจริง
        print("Extracted Keypoints:", keypoints)

        # วาด keypoints บนภาพ
        image_with_keypoints = draw_keypoints(image, keypoints)

        if len(keypoints) < 34:
            return jsonify({'error': 'Insufficient keypoints detected'}), 400

        normalized_keypoints = normalize_keypoints(keypoints)
        print("Normalized Keypoints:", normalized_keypoints)

        # วาด keypoints บนภาพ
        image_with_keypoints = draw_keypoints(image, normalized_keypoints)

        # บันทึกภาพเพื่อตรวจสอบ
        cv2.imwrite("debug_keypoints.jpg", image_with_keypoints)

        # แปลงภาพเป็น Base64
        processed_image = encode_image_to_base64(image_with_keypoints)

        # ทำนายด้วย Random Forest
        test_image_feature = normalized_keypoints.reshape(1, -1)
        predicted_class = rf.predict(test_image_feature)[0]
        class_probabilities = rf.predict_proba(test_image_feature)

        # กำหนด path สำหรับไฟล์คำอธิบาย
        description_file_path = f'Web\class_description\{predicted_class}.txt'

        # ตรวจสอบว่าไฟล์คำอธิบายมีอยู่
        description = "Description not available"
        try:
            with open(description_file_path, 'r', encoding='utf-8-sig') as file:
                description = file.read().strip()
        except UnicodeDecodeError:
            try:
                with open(description_file_path, 'r', encoding='latin-1') as file:
                    description = file.read().strip()
            except Exception as e:
                app.logger.error(f"Error reading description file: {description_file_path}, Error: {e}")

        # ส่งข้อมูลที่จำเป็นกลับไปยัง frontend
        return jsonify({
            'predicted_class': predicted_class,
            'description': description,
            'class_probabilities': [float(prob) for prob in class_probabilities[0].tolist()],
            'processed_image': processed_image,  # ส่งภาพที่ประมวลผลแล้ว
            'keypoints': normalized_keypoints.tolist()  # ส่ง keypoints กลับไป
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

if __name__ == '__main__':
    app.run(debug=True)