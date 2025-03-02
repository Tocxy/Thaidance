import gdown

# URL ของไฟล์บน Google Drive
url = "https://drive.google.com/file/d/1lJno40JNurT-G9RF5qzAi4UoThF4n0pM/view?usp=sharing"
output = "model.zip"  # ชื่อไฟล์ที่ต้องการบันทึก

# ดาวน์โหลดไฟล์
gdown.download(url, output, quiet=False)