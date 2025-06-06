import os
from PIL import Image
import numpy as np
import scipy.fftpack
import math
import matplotlib.pylab as plt

# Đường dẫn tới thư mục chứa ảnh
image_directory = r'C:\Users\khang\Documents\Khang.207CT40341\Exercise'

# Danh sách các đuôi mở rộng file ảnh phổ biến
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

print("--- Bắt đầu xử lý Lọc ảnh với Butterworth Lowpass Filter ---")

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

            # Thực hiện FFT (giữ kết quả phức hợp)
            f_transform = scipy.fftpack.fft2(im1)

            # Dịch chuyển tần số 0 (DC component) về trung tâm phổ
            shifted_f_transform = scipy.fftpack.fftshift(f_transform)

            M, N = shifted_f_transform.shape

            # Tạo bộ lọc Butterworth Lowpass
            H = np.zeros((M, N), dtype=np.float32)
            center_x, center_y = M / 2, N / 2
            d_0 = 30.0 # Bán kính cắt (cut-off radius) - điều chỉnh giá trị này
            n_order = 2 # Bậc của bộ lọc Butterworth (thường là 1, 2, 3,...)

            for i in range(M):
                for j in range(N):
                    # Tính khoảng cách Euclidean từ tâm
                    r = math.sqrt((i - center_x)**2 + (j - center_y)**2)
                    # Áp dụng công thức Butterworth Lowpass Filter
                    H[i, j] = 1 / (1 + (r / d_0)**(2 * n_order))

            # Thực hiện phép nhân trong miền tần số (convolution)
            convoluted_transform = shifted_f_transform * H

            # Dịch chuyển tần số trở lại vị trí ban đầu trước khi IFFT
            inverse_shifted_transform = scipy.fftpack.ifftshift(convoluted_transform)

            # Thực hiện IFFT để chuyển về miền không gian
            filtered_image_complex = scipy.fftpack.ifft2(inverse_shifted_transform)

            # Lấy phần thực và giá trị tuyệt đối (có thể có giá trị âm nhỏ do sai số tính toán)
            filtered_image = np.abs(filtered_image_complex)

            # Chuẩn hóa về khoảng 0-255 và chuyển về kiểu uint8 để hiển thị
            filtered_image_normalized = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min()) * 255
            im3 = Image.fromarray(filtered_image_normalized.astype(np.uint8))

            # Hiển thị ảnh gốc và ảnh đã lọc
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f"Ảnh gốc - {filename}")
            axes[0].axis('off')

            axes[1].imshow(im3, cmap='gray')
            axes[1].set_title(f"BLPF (D0={d_0}, n={n_order}) - {filename}")
            axes[1].axis('off')

            plt.suptitle("Lọc ảnh trong miền tần suất (Butterworth Lowpass Filter)")
            plt.show()

        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {e}")

print("--- Hoàn thành xử lý Lọc ảnh với Butterworth Lowpass Filter ---")