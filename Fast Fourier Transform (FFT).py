import os
from PIL import Image
import numpy as np
import scipy.fftpack # Đúng với import trong tài liệu
import matplotlib.pylab as plt

# Đường dẫn tới thư mục chứa ảnh
image_directory = r'C:\Users\khang\Documents\Khang.207CT40341\Exercise'

# Danh sách các đuôi mở rộng file ảnh phổ biến
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

print("--- Bắt đầu xử lý Biến đổi ảnh với Fast Fourier Transform (FFT) ---")

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

            # Thực hiện FFT
            # scipy.fftpack.fft2 trả về một mảng phức hợp
            f_transform = scipy.fftpack.fft2(im1)

            # Tính toán biên độ (magnitude) của phổ tần số (để hiển thị)
            # abs() lấy giá trị tuyệt đối (biên độ) của số phức
            c = abs(f_transform)

            # Dịch chuyển tần số 0 (DC component) về trung tâm phổ
            d = scipy.fftpack.fftshift(c)

            # Để hiển thị phổ tần số tốt hơn, thường dùng log scale
            # Thêm 1 để tránh log(0)
            d_display = np.log(1 + d)

            # Chuẩn hóa về khoảng 0-255 để hiển thị dưới dạng ảnh 8-bit
            # Cần chuẩn hóa d_display trước khi chuyển sang uint8 để đảm bảo hiển thị đúng
            d_display_normalized = (d_display - d_display.min()) / (d_display.max() - d_display.min()) * 255
            im3 = Image.fromarray(d_display_normalized.astype(np.uint8))

            # Hiển thị ảnh gốc và phổ FFT
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f"Ảnh gốc - {filename}")
            axes[0].axis('off')

            axes[1].imshow(im3, cmap='gray')
            axes[1].set_title(f"Phổ FFT (Log Scale) - {filename}")
            axes[1].axis('off')

            plt.suptitle("Biến đổi ảnh với Fast Fourier Transform (FFT)")
            plt.show()

        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {e}")

print("--- Hoàn thành xử lý Biến đổi ảnh với Fast Fourier Transform (FFT) ---")