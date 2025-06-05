import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
import matplotlib.pylab as plt
import colorsys # Để chuyển đổi RGB <-> HSV
import os

# Đường dẫn đến thư mục chứa ảnh bài tập
exercise_folder = 'Exercise'
output_folder_bai9 = 'Output_Bai9'

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_folder_bai9, exist_ok=True)

print(f"\n--- Bắt đầu xử lý Bài tập 9: Đổi màu HSV ngẫu nhiên từ thư mục '{exercise_folder}' ---") # [cite: 48]

# Vectorize các hàm chuyển đổi màu của colorsys để áp dụng cho toàn bộ mảng
# colorsys.rgb_to_hsv và colorsys.hsv_to_rgb chỉ làm việc trên 1 pixel.
# np.vectorize giúp áp dụng hàm này cho toàn bộ mảng NumPy một cách hiệu quả. [cite: 16]
rgb2hsv_vec = np.vectorize(colorsys.rgb_to_hsv) 
hsv2rgb_vec = np.vectorize(colorsys.hsv_to_rgb) 

# Lặp qua tất cả các tệp trong thư mục Exercise
for filename in os.listdir(exercise_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        file_path = os.path.join(exercise_folder, filename)
        base_name = os.path.splitext(filename)[0]

        print(f"\nĐang xử lý ảnh: {filename}")

        try:
            # 1. Mở ảnh màu (imageio đọc ảnh với kênh màu từ 0-255)
            img_rgb = iio.imread(file_path)
            
            # Đảm bảo ảnh là 3 kênh màu (loại bỏ kênh alpha nếu có, ví dụ ảnh PNG)
            if img_rgb.shape[2] == 4: # Nếu là RGBA
                img_rgb = img_rgb[:, :, :3] # Chỉ lấy 3 kênh RGB
            
            # Chuẩn hóa ảnh về dải [0.0, 1.0] (kiểu float32) để colorsys có thể xử lý [cite: 13]
            img_rgb_normalized = img_rgb.astype(np.float32) / 255.0

            # Hiển thị ảnh gốc
            plt.figure(figsize=(6, 6))
            plt.imshow(img_rgb)
            plt.title(f'Ảnh Gốc ({filename})')
            plt.axis('off') 
            plt.show() 

            # 2. Khử nhiễu (áp dụng cho từng kênh màu RGB sau khi đã normalize)
            # Sử dụng Median filter vì nó hiệu quả với nhiễu "muối tiêu" và giữ cạnh tốt. [cite: 19]
            img_denoised_r = sn.median_filter(img_rgb_normalized[:,:,0], size=3)
            img_denoised_g = sn.median_filter(img_rgb_normalized[:,:,1], size=3)
            img_denoised_b = sn.median_filter(img_rgb_normalized[:,:,2], size=3)
            
            # Stack lại các kênh đã khử nhiễu thành ảnh màu (vẫn ở dải [0.0, 1.0])
            img_denoised_normalized = np.stack([img_denoised_r, img_denoised_g, img_denoised_b], axis=-1)
            print(f"  Đã khử nhiễu cho {filename} bằng Median Filter.")

            # 3. Chuyển đổi từ RGB sang HSV [cite: 12]
            # rgb2hsv_vec sẽ trả về 3 mảng NumPy tương ứng với H, S, V.
            h_old, s_old, v_old = rgb2hsv_vec( 
                img_denoised_normalized[:,:,0], 
                img_denoised_normalized[:,:,1], 
                img_denoised_normalized[:,:,2]
            )

            # 4. Thay đổi HSV ngẫu nhiên nhưng không trùng (mỗi ảnh một bộ dịch chuyển ngẫu nhiên) [cite: 48]
            # Tạo các giá trị dịch chuyển ngẫu nhiên cho H, S, V. 
            # Dải giá trị trong colorsys là [0.0, 1.0].
            h_shift = np.random.uniform(-0.5, 0.5) # Dịch chuyển sắc độ (Hue)
            s_shift = np.random.uniform(-0.3, 0.3) # Dịch chuyển độ bão hòa (Saturation)
            v_shift = np.random.uniform(-0.3, 0.3) # Dịch chuyển giá trị/độ sáng (Value)
            
            # Áp dụng dịch chuyển và giới hạn giá trị
            # Hue là vòng tròn, nên dùng modulo 1.0 để đảm bảo giá trị vẫn trong [0, 1) [cite: 10]
            h_new = (h_old + h_shift) % 1.0 
            # Saturation và Value giới hạn trong khoảng [0.0, 1.0] [cite: 11]
            s_new = np.clip(s_old + s_shift, 0.0, 1.0) 
            v_new = np.clip(v_old + v_shift, 0.0, 1.0) 
            print(f"  Đã áp dụng dịch chuyển H={h_shift:.2f}, S={s_shift:.2f}, V={v_shift:.2f}")

            # 5. Chuyển đổi lại từ HSV sang RGB [cite: 12]
            # hsv2rgb_vec sẽ trả về một tuple chứa 3 mảng NumPy cho R, G, B
            rgb_converted_tuple = hsv2rgb_vec(h_new, s_new, v_new) 
            
            # Stack lại các kênh R, G, B để tạo thành ảnh màu
            img_changed_hsv = np.stack([
                rgb_converted_tuple[0], # Kênh R
                rgb_converted_tuple[1], # Kênh G
                rgb_converted_tuple[2]  # Kênh B
            ], axis=-1)

            # Chuyển lại về dải 0-255 và kiểu uint8 để lưu ảnh (imageio yêu cầu)
            img_changed_hsv_uint8 = np.clip(img_changed_hsv * 255, 0, 255).astype(np.uint8)

            # 6. Lưu ảnh mới [cite: 48]
            iio.imsave(os.path.join(output_folder_bai9, f'{base_name}_random_hsv.png'), img_changed_hsv_uint8)
            print(f"  Đã xử lý đổi màu HSV ngẫu nhiên và lưu: {base_name}_random_hsv.png")

            # Hiển thị ảnh đã đổi màu
            plt.figure(figsize=(6, 6))
            plt.imshow(img_changed_hsv_uint8)
            plt.title(f'Ảnh Đổi Màu HSV (dịch H:{h_shift:.2f}, S:{s_shift:.2f}, V:{v_shift:.2f})')
            plt.axis('off')
            plt.show()

        except Exception as e:
            # Bắt và in ra lỗi nếu có bất kỳ vấn đề nào xảy ra khi xử lý một ảnh cụ thể
            print(f"  Lỗi khi xử lý {filename}: {e}")

print(f"\n--- Hoàn thành xử lý Bài tập 9. Kiểm tra kết quả trong thư mục '{output_folder_bai9}' ---")