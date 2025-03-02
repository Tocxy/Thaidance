import pandas as pd
from sklearn.model_selection import train_test_split

# โหลดข้อมูลจากไฟล์ CSV
file_path = "Randomforest\\data\\keypoints.csv"  
data = pd.read_csv(file_path)

# แบ่งข้อมูลเป็น train (70%) และ temp (30%)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)

# แบ่ง temp เป็น val (15%) และ test (15%)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# ตรวจสอบขนาดของแต่ละชุดข้อมูล
print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

# บันทึกแต่ละชุดข้อมูลลงไฟล์ CSV
train_data.to_csv("Randomforest\\data\\train_data.csv", index=False)
val_data.to_csv("Randomforest\\data\\val_data.csv", index=False)
test_data.to_csv("Randomforest\\data\\test_data.csv", index=False)

print("การแบ่งข้อมูลเสร็จสมบูรณ์ และบันทึกไฟล์เรียบร้อย!")
