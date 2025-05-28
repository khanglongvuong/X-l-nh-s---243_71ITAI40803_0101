import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
import os
import scipy.ndimage as sn 

# Hàm hỗ trợ để chuyển đổi ảnh màu về uint8 an toàn (không cần thiết trực tiếp cho bài này vì đã đọc grayscale)
def convert_to_uint8(image_array):
    """
    Chuyển đổi mảng numpy thành uint8 (0-255) và đảm bảo các giá trị nằm trong khoảng hợp lệ.
    """
    if image_array.dtype != np.float32 and image_array.dtype != np.float64:
        image_array = image_array.astype(np.float32)
    image_array = np.clip(image_array, 0, 255)
    return image_array.astype(np.uint8)

def bai5_mean_filter(img_gray, base_name, ext, output_dir):
    """
    Áp dụng mean filter (bộ lọc trung bình) cho ảnh grayscale và hiển thị.
    """
    print(f"  - Đang xử lý Bài 5: Mean Filter cho {base_name}{ext}")
    # Bộ lọc 5x5, chuẩn hóa để tổng = 1
    k = np.ones((5,5)) / 25 
    mean_filtered_img = sn.convolve(img_gray, k).astype(np.uint8) #

    # Lưu ảnh
    iio.imsave(os.path.join(output_dir, f'{base_name}_mean_filtered{ext}'), mean_filtered_img)
    print(f"    - Đã lưu: {base_name}_mean_filtered{ext}")

    # Hiển thị ảnh
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Ảnh gốc (Grayscale)')
    axes[0].axis('off')

    axes[1].imshow(mean_filtered_img, cmap='gray')
    axes[1].set_title('Ảnh sau Mean Filter')
    axes[1].axis('off')
    
    fig.suptitle(f'Bài 5: Áp dụng Mean Filter cho {filename}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    input_dir = 'Exercise'
    output_dir = 'OutputImages'

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Thư mục '{input_dir}' chưa tồn tại. Vui lòng đặt các file ảnh vào thư mục này.")
        try:
            sample_img_path = os.path.join(input_dir, 'sample_image.png')
            # Tạo một ảnh trắng đơn giản 100x100 RGB để làm mẫu nếu thư mục trống
            sample_img = np.full((100, 100, 3), 255, dtype=np.uint8)
            iio.imsave(sample_img_path, sample_img)
            print(f"Đã tạo ảnh mẫu '{sample_img_path}'. Vui lòng thêm ảnh thật của bạn.")
        except Exception as e:
            print(f"Không thể tạo ảnh mẫu: {e}")
        exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("--- Bắt đầu Bài tập 5 ---")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_dir, filename)
            base_name, ext = os.path.splitext(filename)
            
            try:
                # SỬA LỖI: Thay as_gray=True bằng mode='L'
                img_gray = iio.imread(image_path, mode='L').astype(np.uint8) 
                bai5_mean_filter(img_gray, base_name, ext, output_dir)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {filename}: {e}")
                print("Đảm bảo file ảnh không bị hỏng, là định dạng hợp lệ, và các thư viện đã được cài đặt.")
    print("--- Kết thúc Bài tập 5. Kết quả trong thư mục 'OutputImages' và hiển thị. ---")