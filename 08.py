import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
import matplotlib.pylab as plt
import os

# Đường dẫn đến thư mục chứa ảnh bài tập
exercise_folder = 'Exercise'
output_folder_bai8 = 'Output_Bai8'

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_folder_bai8, exist_ok=True)

print(f"\n--- Bắt đầu xử lý Bài tập 8: Đổi màu RGB ngẫu nhiên từ thư mục '{exercise_folder}' ---")

# Lặp qua tất cả các tệp trong thư mục Exercise
for filename in os.listdir(exercise_folder):
    # Kiểm tra xem tệp có phải là ảnh không
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        file_path = os.path.join(exercise_folder, filename)
        base_name = os.path.splitext(filename)[0]

        print(f"\nĐang xử lý ảnh: {filename}")

        try:
            # 1. Mở ảnh màu (imageio tự động đọc ảnh màu với 3 kênh nếu có)
            img_rgb = iio.imread(file_path)
            
            # Đảm bảo ảnh là 3 kênh màu (loại bỏ kênh alpha nếu có, ví dụ ảnh PNG)
            if img_rgb.shape[2] == 4: # Nếu là RGBA
                img_rgb = img_rgb[:, :, :3] # Chỉ lấy 3 kênh RGB
            
            # Chuyển đổi kiểu dữ liệu sang int32 để tránh tràn số khi cộng/trừ các giá trị dịch chuyển
            # Giá trị pixel của ảnh thường là uint8 (0-255). Nếu cộng thêm một số lớn, có thể vượt quá 255
            # và gây tràn số nếu vẫn ở uint8.
            img_rgb_int32 = img_rgb.astype(np.int32)

            # Hiển thị ảnh gốc
            plt.figure(figsize=(6, 6))
            plt.imshow(img_rgb)
            plt.title(f'Ảnh Gốc ({filename})')
            plt.axis('off') # Tắt trục tọa độ
            plt.show() # Hiển thị ảnh

            # 2. Khử nhiễu (áp dụng cho từng kênh màu)
            # Sử dụng Median filter vì nó hiệu quả với nhiễu "muối tiêu" và giữ cạnh tốt.
            # Áp dụng cho từng kênh R, G, B
            img_denoised_r = sn.median_filter(img_rgb_int32[:,:,0], size=3)
            img_denoised_g = sn.median_filter(img_rgb_int32[:,:,1], size=3)
            img_denoised_b = sn.median_filter(img_rgb_int32[:,:,2], size=3)
            
            # Stack lại các kênh đã khử nhiễu thành ảnh màu
            img_denoised = np.stack([img_denoised_r, img_denoised_g, img_denoised_b], axis=-1)
            print(f"  Đã khử nhiễu cho {filename} bằng Median Filter.")

            # 3. Thay đổi màu RGB ngẫu nhiên
            # Tạo các giá trị dịch chuyển ngẫu nhiên cho R, G, B
            # np.random.randint(-70, 70, size=3) sẽ tạo 3 số nguyên ngẫu nhiên từ -70 đến 69.
            # Bạn có thể điều chỉnh khoảng này để thay đổi mức độ "ngẫu nhiên" của màu.
            r_shift, g_shift, b_shift = np.random.randint(-70, 70, size=3) 

            # Áp dụng dịch chuyển cho từng kênh
            img_shifted_r = img_denoised[:,:,0] + r_shift
            img_shifted_g = img_denoised[:,:,1] + g_shift
            img_shifted_b = img_denoised[:,:,2] + b_shift

            # Giới hạn giá trị pixel trong khoảng [0, 255] và chuyển về kiểu dữ liệu uint8
            # np.clip(array, min_val, max_val) đảm bảo các giá trị nằm trong khoảng mong muốn.
            img_changed_rgb = np.stack([
                np.clip(img_shifted_r, 0, 255),
                np.clip(img_shifted_g, 0, 255),
                np.clip(img_shifted_b, 0, 255)
            ], axis=-1).astype(np.uint8)

            # 4. Lưu ảnh mới
            iio.imsave(os.path.join(output_folder_bai8, f'{base_name}_random_rgb.png'), img_changed_rgb)
            print(f"  Đã xử lý đổi màu RGB ngẫu nhiên và lưu: {base_name}_random_rgb.png")

            # Hiển thị ảnh đã đổi màu
            plt.figure(figsize=(6, 6))
            plt.imshow(img_changed_rgb)
            plt.title(f'Ảnh Đổi Màu RGB (dịch R:{r_shift}, G:{g_shift}, B:{b_shift})')
            plt.axis('off')
            plt.show()

        except Exception as e:
            # Bắt và in ra lỗi nếu có bất kỳ vấn đề nào xảy ra khi xử lý một ảnh cụ thể
            print(f"  Lỗi khi xử lý {filename}: {e}")

print(f"\n--- Hoàn thành xử lý Bài tập 8. Kiểm tra kết quả trong thư mục '{output_folder_bai8}' ---")