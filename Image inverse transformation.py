import os
from PIL import Image
import numpy as np
import matplotlib.pylab as plt

# Đường dẫn tới thư mục chứa ảnh
image_directory = r'C:\Users\khang\Documents\Khang.207CT40341\Exercise'

# Danh sách các đuôi mở rộng file ảnh phổ biến
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

print("--- Bắt đầu xử lý Biến đổi cường độ ảnh (Image inverse transformation) ---")

for filename in os.listdir(image_directory):
    file_path = os.path.join(image_directory, filename)

    # Chỉ xử lý các file là ảnh
    if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
        print(f"Đang xử lý ảnh: {filename}")
        try:
            # Mở ảnh dưới dạng ảnh nhị phân (binary image) như trong tài liệu
            img = Image.open(file_path).convert('1')

            # Chuyển đổi ảnh PIL sang mảng NumPy
            im_1 = np.asarray(img)

            # Thực hiện phép biến đổi nghịch đảo
            im_2 = 255 - im_1

            # Chuyển đổi mảng NumPy trở lại ảnh PIL
            new_img = Image.fromarray(im_2.astype(np.uint8)) # Đảm bảo kiểu dữ liệu là uint8

            # Hiển thị ảnh gốc và ảnh đã xử lý
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f"Ảnh gốc - {filename}")
            axes[0].axis('off')

            axes[1].imshow(new_img, cmap='gray')
            axes[1].set_title(f"Ảnh nghịch đảo - {filename}")
            axes[1].axis('off')

            plt.suptitle("Biến đổi cường độ ảnh (Image Inverse Transformation)")
            plt.show()

        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {e}")

print("--- Hoàn thành xử lý Biến đổi cường độ ảnh (Image inverse transformation) ---")