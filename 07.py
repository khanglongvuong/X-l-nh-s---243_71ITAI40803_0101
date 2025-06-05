import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
from skimage import filters, feature # Import feature cho Canny
import matplotlib.pylab as plt
import os

# Đường dẫn đến thư mục chứa ảnh bài tập
exercise_folder = 'Exercise'
output_folder_bai7 = 'Output_Bai7'

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_folder_bai7, exist_ok=True)

print(f"\n--- Bắt đầu xử lý Bài tập 7: Xác định biên ảnh từ thư mục '{exercise_folder}' ---")

# Lặp qua tất cả các tệp trong thư mục Exercise
for filename in os.listdir(exercise_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        file_path = os.path.join(exercise_folder, filename)
        base_name = os.path.splitext(filename)[0]

        print(f"\nĐang xử lý ảnh: {filename}")

        try:
            # 1. Mở ảnh và chuyển sang grayscale (sử dụng mode='L' thay cho as_gray) [cite: 5, 6]
            a = iio.imread(file_path, mode='L').astype(np.uint8) 

            # Hiển thị ảnh gốc
            plt.figure(figsize=(6, 6))
            plt.imshow(a, cmap='gray')
            plt.title(f'Ảnh Gốc ({filename})')
            plt.axis('off')
            plt.show()

            # 2. Khử nhiễu trước khi xác định biên (Quan trọng!) [cite: 46]
            # Gaussian blur là lựa chọn tốt để làm mịn ảnh và giảm nhiễu trước khi phát hiện biên.
            a_denoised = sn.gaussian_filter(a, sigma=1.5) # sigma có thể điều chỉnh để đạt hiệu quả tốt nhất
            print(f"  Đã khử nhiễu cho {filename} bằng Gaussian Filter.")

            # 3. Áp dụng các bộ lọc xác định biên
            
            # Sobel filter
            # Sobel filter sử dụng 2 kênh để tìm cạnh ngang và dọc của biên ảnh. [cite: 39]
            edges_sobel = filters.sobel(a_denoised) 
            # Sobel có thể trả về giá trị float, cần chuẩn hóa về 0-255 và uint8 để lưu và hiển thị
            edges_sobel_uint8 = np.clip(edges_sobel * 255 / edges_sobel.max(), 0, 255).astype(np.uint8) 
            iio.imsave(os.path.join(output_folder_bai7, f'{base_name}_sobel_edge.png'), edges_sobel_uint8)
            print(f"  Đã áp dụng Sobel Filter và lưu: {base_name}_sobel_edge.png")
            plt.imshow(edges_sobel_uint8, cmap='gray') 
            plt.title('Sobel Edge Detection') 
            plt.axis('off') 
            plt.show() 

            # Prewitt filter
            edges_prewitt = filters.prewitt(a_denoised) 
            edges_prewitt_uint8 = np.clip(edges_prewitt * 255 / edges_prewitt.max(), 0, 255).astype(np.uint8) 
            iio.imsave(os.path.join(output_folder_bai7, f'{base_name}_prewitt_edge.png'), edges_prewitt_uint8)
            print(f"  Đã áp dụng Prewitt Filter và lưu: {base_name}_prewitt_edge.png")
            plt.imshow(edges_prewitt_uint8, cmap='gray') 
            plt.title('Prewitt Edge Detection') 
            plt.axis('off') 
            plt.show() 

            # Canny filter
            edges_canny = feature.canny(a_denoised, sigma=3).astype(np.uint8) * 255 
            iio.imsave(os.path.join(output_folder_bai7, f'{base_name}_canny_edge.png'), edges_canny)
            print(f"  Đã áp dụng Canny Filter và lưu: {base_name}_canny_edge.png")
            plt.imshow(edges_canny, cmap='gray') 
            plt.title('Canny Edge Detection') 
            plt.axis('off') 
            plt.show() 
            
            # Laplacian filter
            # Laplacian detection chỉ sử dụng 1 kênh để xác định biên của đối tượng. [cite: 42]
            # Laplacian rất nhạy cảm với nhiễu, nên việc khử nhiễu trước là rất cần thiết.
            edges_laplacian = sn.laplace(a_denoised) 
            edges_laplacian_uint8 = np.clip(edges_laplacian + 128, 0, 255).astype(np.uint8) # Dịch để đưa giá trị về 0-255 để hiển thị tốt hơn
            iio.imsave(os.path.join(output_folder_bai7, f'{base_name}_laplacian_edge.png'), edges_laplacian_uint8)
            print(f"  Đã áp dụng Laplacian Filter và lưu: {base_name}_laplacian_edge.png")
            plt.imshow(edges_laplacian_uint8, cmap='gray') 
            plt.title('Laplacian Edge Detection') 
            plt.axis('off') 
            plt.show() 

        except Exception as e: # Bắt lỗi để chương trình không dừng đột ngột
            print(f"  Lỗi khi xử lý {filename}: {e}")

print(f"\n--- Hoàn thành xử lý Bài tập 7. Kiểm tra kết quả trong thư mục '{output_folder_bai7}' ---")