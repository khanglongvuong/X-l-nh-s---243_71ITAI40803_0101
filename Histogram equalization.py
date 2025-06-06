import os
from PIL import Image
import numpy as np
import matplotlib.pylab as plt

# Đường dẫn tới thư mục chứa ảnh
image_directory = r'C:\Users\khang\Documents\Khang.207CT40341\Exercise'

# Danh sách các đuôi mở rộng file ảnh phổ biến
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

print("--- Bắt đầu xử lý Histogram equalization ---")

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

            # Làm phẳng mảng 2D thành 1D để dễ tính toán histogram
            bl = im1.flatten()

            # Tính toán histogram
            hist, bins = np.histogram(im1, 256, [0, 255])

            # Tính hàm phân phối tích lũy (Cumulative Distribution Function - CDF)
            cdf = hist.cumsum()

            # Tạo masked array để bỏ qua các giá trị CDF bằng 0 (vùng ảnh không có pixel)
            cdf_m = np.ma.masked_equal(cdf, 0)

            # Kiểm tra trường hợp cdf_m.min() hoặc cdf.max() - cdf_m.min() bằng 0
            if cdf_m.min() is np.ma.masked or (cdf.max() - cdf_m.min()) == 0:
                print(f"Cảnh báo: Histogram equalization không thể thực hiện cho {filename} do CDF phẳng hoặc không có giá trị.")
                continue

            # Thực hiện phép cân bằng histogram
            num_cdf_m = (cdf_m - cdf_m.min()) * 255
            den_cdf_m = (cdf.max() - cdf_m.min())
            cdf_m_normalized = num_cdf_m / den_cdf_m

            # Điền lại các giá trị đã bị mask bằng 0 và chuyển về kiểu uint8
            cdf = np.ma.filled(cdf_m_normalized, 0).astype('uint8')

            # Ánh xạ các giá trị pixel cũ sang giá trị pixel mới từ CDF đã cân bằng
            im2 = cdf[bl]

            # Định hình lại mảng 1D về kích thước ảnh gốc
            im3 = np.reshape(im2, im1.shape)

            # Chuyển đổi mảng NumPy trở lại ảnh PIL
            im4 = Image.fromarray(im3)

            # Hiển thị ảnh gốc và ảnh đã xử lý
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f"Ảnh gốc - {filename}")
            axes[0].axis('off')

            axes[1].imshow(im4, cmap='gray')
            axes[1].set_title(f"Cân bằng Histogram - {filename}")
            axes[1].axis('off')

            plt.suptitle("Cân bằng Histogram (Histogram Equalization)")
            plt.show()

        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {e}")

print("--- Hoàn thành xử lý Histogram equalization ---")