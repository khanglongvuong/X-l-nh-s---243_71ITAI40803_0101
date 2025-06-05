import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
import matplotlib.pylab as plt
import os

# Đường dẫn đến thư mục chứa ảnh bài tập
exercise_folder = 'Exercise'
output_folder_bai6 = 'Output_Bai6'

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_folder_bai6, exist_ok=True)

print(f"--- Bắt đầu xử lý Bài tập 6: Khử nhiễu ảnh từ thư mục '{exercise_folder}' ---")

# Lặp qua tất cả các tệp trong thư mục Exercise
for filename in os.listdir(exercise_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        file_path = os.path.join(exercise_folder, filename)
        # base_name được định nghĩa ở đây để đảm bảo nó có sẵn khi sử dụng trong try block
        base_name = os.path.splitext(filename)[0]

        print(f"\nĐang xử lý ảnh: {filename}")

        try: # Khối try bắt đầu từ đây
            # 1. Mở ảnh và chuyển sang grayscale (ĐÃ SỬA LỖI as_gray)
            a = iio.imread(file_path, mode='L').astype(np.uint8) 
            
            # --- Tùy chọn: Thêm nhiễu vào ảnh để dễ đánh giá hiệu quả khử nhiễu ---
            row, col = a.shape
            mean = 0
            var = 400 # Độ lớn nhiễu, có thể điều chỉnh
            sigma = var**0.5
            gaussian_noise = np.random.normal(mean, sigma, (row, col))
            a_noisy = a + gaussian_noise
            a_noisy = np.clip(a_noisy, 0, 255).astype(np.uint8)
            current_image_to_process = a_noisy # Xử lý ảnh có nhiễu
            
            # Lưu ảnh có nhiễu để so sánh
            iio.imsave(os.path.join(output_folder_bai6, f'{base_name}_original_noisy.png'), current_image_to_process)
            print(f"  Đã tạo và lưu ảnh có nhiễu: {base_name}_original_noisy.png")

            # Hiển thị ảnh gốc và ảnh có nhiễu
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(a, cmap='gray')
            plt.title(f'Ảnh Gốc ({filename})')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(current_image_to_process, cmap='gray')
            plt.title(f'Ảnh Có Nhiễu')
            plt.axis('off')
            plt.show()
            # ----------------------------------------------------------------------

            # 2. Áp dụng Mean Filter
            k_size = 5 # Kích thước kernel 5x5 như ví dụ trong tài liệu
            kernel_mean = np.ones((k_size, k_size)) / (k_size * k_size) 
            img_mean = sn.convolve(current_image_to_process, kernel_mean).astype(np.uint8) 
            iio.imsave(os.path.join(output_folder_bai6, f'{base_name}_mean_filter.png'), img_mean)
            print(f"  Đã áp dụng Mean Filter và lưu: {base_name}_mean_filter.png")
            plt.imshow(img_mean, cmap='gray') 
            plt.title(f'Mean Filter ({k_size}x{k_size})') 
            plt.axis('off') 
            plt.show() 

            # 3. Áp dụng Median Filter
            img_median = sn.median_filter(current_image_to_process, size=5) 
            iio.imsave(os.path.join(output_folder_bai6, f'{base_name}_median_filter.png'), img_median)
            print(f"  Đã áp dụng Median Filter và lưu: {base_name}_median_filter.png")
            plt.imshow(img_median, cmap='gray') 
            plt.title(f'Median Filter (size=5)') 
            plt.axis('off') 
            plt.show() 

            # 4. Áp dụng Max Filter
            img_max = sn.maximum_filter(current_image_to_process, size=5) 
            iio.imsave(os.path.join(output_folder_bai6, f'{base_name}_max_filter.png'), img_max)
            print(f"  Đã áp dụng Max Filter và lưu: {base_name}_max_filter.png")
            plt.imshow(img_max, cmap='gray') 
            plt.title(f'Max Filter (size=5)') 
            plt.axis('off') 
            plt.show() 

            # 5. Áp dụng Min Filter
            img_min = sn.minimum_filter(current_image_to_process, size=5) 
            iio.imsave(os.path.join(output_folder_bai6, f'{base_name}_min_filter.png'), img_min)
            print(f"  Đã áp dụng Min Filter và lưu: {base_name}_min_filter.png")
            plt.imshow(img_min, cmap='gray') 
            plt.title(f'Min Filter (size=5)') 
            plt.axis('off') 
            plt.show() 

        except Exception as e: # Khối except để bắt lỗi và in ra thông báo
            print(f"  Lỗi khi xử lý {filename}: {e}")

print(f"\n--- Hoàn thành xử lý Bài tập 6. Kiểm tra kết quả trong thư mục '{output_folder_bai6}' ---")
print("Để biết filter nào khử nhiễu tốt nhất, hãy mở các ảnh đã lưu và so sánh trực quan.")
print("Thông thường, Median Filter hiệu quả với nhiễu muối tiêu, Gaussian Filter tốt với nhiễu Gaussian.")