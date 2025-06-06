import os
from PIL import Image
import numpy as np
import matplotlib.pylab as plt

# Đường dẫn tới thư mục chứa ảnh
image_directory = r'C:\Users\khang\Documents\Khang.207CT40341\Exercise'

# Danh sách các đuôi mở rộng file ảnh phổ biến
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

print("--- Bắt đầu xử lý Thay đổi ảnh với Contrast Stretching ---")

for filename in os.listdir(image_directory):
    file_path = os.path.join(image_directory, filename)

    # Chỉ xử lý các file là ảnh
    if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
        print(f"Đang xử lý ảnh: {filename}")
        try:
            # Mở ảnh dưới dạng ảnh xám (grayscale)
            img = Image.open(file_path).convert('L')

            # Chuyển đổi ảnh PIL sang mảng NumPy
            im1 = np.asarray(img)

            # Tìm giá trị pixel nhỏ nhất và lớn nhất trong ảnh
            a = im1.min()
            b = im1.max()
            print(f"Giá trị Min/Max gốc cho {filename}: {a}, {b}")

            # Kiểm tra nếu tất cả các pixel có cùng giá trị (ảnh đơn sắc)
            if a == b:
                print(f"Cảnh báo: Tất cả pixel trong ảnh {filename} có cùng giá trị. Contrast Stretching bị bỏ qua.")
                # Hiển thị ảnh gốc và bỏ qua phần stretching nếu không có sự khác biệt
                plt.figure(figsize=(6, 6))
                plt.imshow(img, cmap='gray')
                plt.title(f"Ảnh gốc (Không thể Contrast Stretching) - {filename}")
                plt.axis('off')
                plt.suptitle("Thay đổi ảnh với Contrast Stretching")
                plt.show()
                continue

            # Chuyển đổi kiểu dữ liệu sang float để tính toán
            c = im1.astype(float)

            # Thực hiện phép biến đổi Contrast Stretching
            # Công thức: S = (R - R_min) * (255 / (R_max - R_min))
            # Trong đó R là giá trị pixel gốc, R_min là min, R_max là max
            im2 = 255 * (c - a) / (b - a)

            # Chuyển đổi mảng NumPy trở lại ảnh PIL và đảm bảo kiểu dữ liệu uint8
            im3 = Image.fromarray(im2.astype(np.uint8))

            # Hiển thị ảnh gốc và ảnh đã xử lý
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f"Ảnh gốc - {filename}")
            axes[0].axis('off')

            axes[1].imshow(im3, cmap='gray')
            axes[1].set_title(f"Contrast Stretched - {filename}")
            axes[1].axis('off')

            plt.suptitle("Thay đổi ảnh với Contrast Stretching")
            plt.show()

        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {e}")

print("--- Hoàn thành xử lý Thay đổi ảnh với Contrast Stretching ---")