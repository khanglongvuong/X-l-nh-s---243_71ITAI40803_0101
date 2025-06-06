import os
from PIL import Image
import numpy as np
import matplotlib.pylab as plt

# Đường dẫn tới thư mục chứa ảnh
image_directory = r'C:\Users\khang\Documents\Khang.207CT40341\Exercise'

# Danh sách các đuôi mở rộng file ảnh phổ biến
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

print("--- Bắt đầu xử lý Thay đổi cường độ điểm ảnh với Log Transformation ---")

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

            # Tìm giá trị pixel tối đa trong ảnh
            max_pixel_value = np.max(bl)

            # Tránh trường hợp log(1+0) hoặc log(1+max_pixel_value) = log(1) = 0 nếu ảnh đen hoàn toàn
            if max_pixel_value == 0:
                print(f"Cảnh báo: Ảnh {filename} hoàn toàn đen, không thể thực hiện Log Transformation theo công thức này.")
                continue

            # Thực hiện phép biến đổi log theo công thức trong tài liệu
            # C = (128.0 * np.log(1 + bl)) / np.log(1 + b2)
            # Trong đó b2 là max_pixel_value

            # Đảm bảo các đối số của log là dương
            log_arg_bl = 1 + bl
            log_arg_max_pixel_value = 1 + max_pixel_value

            if np.any(log_arg_bl <= 0) or log_arg_max_pixel_value <= 0:
                print(f"Cảnh báo: Đối số của hàm log không hợp lệ cho {filename}. Đang điều chỉnh.")
                log_arg_bl = np.where(log_arg_bl <= 0, 1e-10, log_arg_bl)
                log_arg_max_pixel_value = np.where(log_arg_max_pixel_value <= 0, 1e-10, log_arg_max_pixel_value)

            # Tránh chia cho log(1) = 0 nếu max_pixel_value = 0
            if np.log(log_arg_max_pixel_value) == 0:
                print(f"Cảnh báo: Log của giá trị tối đa là 0 cho {filename}. Không thể thực hiện Log Transformation.")
                continue

            C = (128.0 * np.log(log_arg_bl)) / np.log(log_arg_max_pixel_value)

            # Chuyển đổi về kiểu số nguyên 8-bit (0-255)
            c1 = C.astype(np.uint8)
            d = Image.fromarray(c1)

            # Hiển thị ảnh gốc và ảnh đã xử lý
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f"Ảnh gốc - {filename}")
            axes[0].axis('off')

            axes[1].imshow(d, cmap='gray')
            axes[1].set_title(f"Log Transformed - {filename}")
            axes[1].axis('off')

            plt.suptitle("Thay đổi cường độ điểm ảnh với Log Transformation")
            plt.show()

        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {e}")

print("--- Hoàn thành xử lý Thay đổi cường độ điểm ảnh với Log Transformation ---")