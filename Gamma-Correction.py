import os
from PIL import Image
import numpy as np
import matplotlib.pylab as plt

# Đường dẫn tới thư mục chứa ảnh
image_directory = r'C:\Users\khang\Documents\Khang.207CT40341\Exercise'

# Danh sách các đuôi mở rộng file ảnh phổ biến
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

print("--- Bắt đầu xử lý Thay đổi chất lượng ảnh với Power law (Gamma-Correction) ---")

# Giá trị gamma (có thể điều chỉnh)
gamma = 0.5 # Thử với các giá trị khác như 0.5, 2.0, 5.0 để xem sự thay đổi

for filename in os.listdir(image_directory):
    file_path = os.path.join(image_directory, filename)

    # Chỉ xử lý các file là ảnh
    if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
        print(f"Đang xử lý ảnh: {filename}")
        try:
            # Mở ảnh dưới dạng ảnh xám (grayscale)
            img = Image.open(file_path).convert('L')

            # Chuyển đổi ảnh PIL sang mảng NumPy
            im_1 = np.asarray(img)

            # Chuyển đổi kiểu dữ liệu sang float để tính toán
            bl = im_1.astype(float)

            # Tìm giá trị pixel tối đa trong ảnh (thường là 255 cho ảnh 8-bit)
            max_pixel_value = np.max(bl)

            if max_pixel_value == 0: # Tránh chia cho 0 nếu ảnh hoàn toàn đen
                print(f"Cảnh báo: Ảnh {filename} hoàn toàn đen, không thể thực hiện Gamma Correction.")
                continue

            # Chuẩn hóa giá trị pixel về khoảng [0, 1]
            normalized_image = bl / max_pixel_value

            # Xử lý trường hợp log(0)
            if np.any(normalized_image <= 0):
                # Thay thế các giá trị 0 hoặc âm bằng một số rất nhỏ dương
                normalized_image = np.where(normalized_image <= 0, 1e-10, normalized_image)

            # Tính toán biểu thức trung gian cho gamma correction theo tài liệu
            # Lưu ý: Công thức trong tài liệu có vẻ là biến thể, công thức chuẩn là (input/255)^gamma * 255
            # Tôi sẽ giữ công thức từ tài liệu của bạn:
            intermediate_exp = np.log(normalized_image) * gamma
            gamma_corrected_pixels = np.exp(intermediate_exp) * 255.0

            # Chuyển đổi về kiểu số nguyên 8-bit (0-255)
            c1 = gamma_corrected_pixels.astype(np.uint8)
            d = Image.fromarray(c1)

            # Hiển thị ảnh gốc và ảnh đã xử lý
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f"Ảnh gốc - {filename}")
            axes[0].axis('off')

            axes[1].imshow(d, cmap='gray')
            axes[1].set_title(f"Gamma={gamma} - {filename}")
            axes[1].axis('off')

            plt.suptitle("Thay đổi chất lượng ảnh với Power law (Gamma-Correction)")
            plt.show()

        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {e}")

print("--- Hoàn thành xử lý Thay đổi chất lượng ảnh với Power law (Gamma-Correction) ---")