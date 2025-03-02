
    document.addEventListener('DOMContentLoaded', () => {
        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('image');
        const fileNameDisplay = document.getElementById('file-name');
        const imagePreview = document.getElementById('image-preview');
        const resultDiv = document.getElementById('result');
        const loadingDiv = document.getElementById('loading');
        const keypointImage = document.getElementById('keypoint-image');
        const descriptionDiv = document.getElementById('description');
        const processedPreview = document.getElementById('processed-preview');

        // แสดงภาพที่ผู้ใช้อัปโหลด
        fileInput.addEventListener('change', function () {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                fileNameDisplay.textContent = file.name;

                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function (e) {
                        imagePreview.src = e.target.result;
                        imagePreview.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                } else {
                    alert('กรุณาอัปโหลดไฟล์ภาพที่ถูกต้อง');
                    imagePreview.style.display = 'none';
                }
            } else {
                fileNameDisplay.textContent = "ยังไม่ได้เลือกไฟล์";
                imagePreview.style.display = 'none';
            }
        });

        // ส่งคำขอไปยังเซิร์ฟเวอร์
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const formData = new FormData(form);

            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = "<p>กำลังประมวลผล...</p>";
            descriptionDiv.innerHTML = "";
            processedPreview.style.display = 'none';
            keypointImage.style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();
                loadingDiv.style.display = 'none';

                if (data.error) {
                    resultDiv.innerHTML = `<p class="error">ข้อผิดพลาด: ${data.error}</p>`;
                } else {
                    resultDiv.innerHTML = `<p class="success">ผลการทำนาย: ${data.predicted_class}</p>`;
                    descriptionDiv.innerHTML = `<h3>รายละเอียด:</h3><p>${data.description}</p>`;

                    // แสดงภาพที่ประมวลผล
                    if (data.keypoint_image_url) {
                        keypointImage.src = data.keypoint_image_url;
                        keypointImage.style.display = 'block';
                    }

                    // แสดงภาพที่ถูกประมวลผล
                    if (data.processed_image) {
                        processedPreview.src = `data:image/jpeg;base64,${data.processed_image}`;
                        processedPreview.style.display = 'block';
                    }
                }
            } catch (error) {
                loadingDiv.style.display = 'none';
                resultDiv.innerHTML = `<p class="error">ข้อผิดพลาด: ${error.message}</p>`;
            }
        });
    });

