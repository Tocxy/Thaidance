import zipfile

# ฟังก์ชันสำหรับแตกไฟล์ ZIP
def unzip_file(zip_filename):
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall()  # แตกไฟล์ทั้งหมดไปยังโฟลเดอร์ปัจจุบัน
            print(f"Unzipped {zip_filename} successfully!")
    except FileNotFoundError:
        print(f"Error: {zip_filename} not found!")
    except zipfile.BadZipFile:
        print(f"Error: {zip_filename} is not a valid ZIP file!")

# แตกไฟล์ ZIP ทั้งสองไฟล์
unzip_file("model_RDF.zip")
unzip_file("model_yolo.zip")