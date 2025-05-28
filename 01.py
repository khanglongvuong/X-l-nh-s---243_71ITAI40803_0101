import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
import os

# Hàm hỗ trợ để chuyển đổi ảnh màu về uint8 an toàn
def convert_to_uint8(image_array):
    """
    Chuyển đổi mảng numpy thành uint8 (0-255) và đảm bảo các giá trị nằm trong khoảng hợp lệ.
    """
    if image_array.dtype != np.float32 and image_array.dtype != np.float64:
        image_array = image_array.astype(np.float32)
    image_array = np.clip(image_array, 0, 255)
    return image_array.astype(np.uint8)

def bai1_color_channels(img_original, base_name, ext, output_dir):
    """
    Tách ảnh RGB thành 3 ảnh riêng biệt cho từng kênh màu Đỏ, Xanh Lá, Xanh Dương và hiển thị.
    """
    print(f"  - Đang xử lý Bài 1: Kênh màu cho {base_name}{ext}")

    # Đảm bảo ảnh có 3 kênh màu, nếu là grayscale thì chuyển sang RGB giả
    if img_original.ndim == 2:
        img_original = np.stack([img_original, img_original, img_original], axis=-1)
    
    red_channel_img = np.zeros_like(img_original)
    red_channel_img[:,:,0] = img_original[:,:,0] # Giữ kênh đỏ
    
    green_channel_img = np.zeros_like(img_original)
    green_channel_img[:,:,1] = img_original[:,:,1] # Giữ kênh xanh lá
    
    blue_channel_img = np.zeros_like(img_original)
    blue_channel_img[:,:,2] = img_original[:,:,2] # Giữ kênh xanh dương

    # Lưu ảnh
    iio.imsave(os.path.join(output_dir, f'{base_name}_red_channel{ext}'), red_channel_img)
    print(f"    - Đã lưu: {base_name}_red_channel{ext}")
    iio.imsave(os.path.join(output_dir, f'{base_name}_green_channel{ext}'), green_channel_img)
    print(f"    - Đã lưu: {base_name}_green_channel{ext}")
    iio.imsave(os.path.join(output_dir, f'{base_name}_blue_channel{ext}'), blue_channel_img)
    print(f"    - Đã lưu: {base_name}_blue_channel{ext}")

    # Hiển thị ảnh
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    axes[0].imshow(img_original)
    axes[0].set_title('Ảnh gốc')
    axes[0].axis('off')

    axes[1].imshow(red_channel_img)
    axes[1].set_title('Kênh Đỏ')
    axes[1].axis('off')

    axes[2].imshow(green_channel_img)
    axes[2].set_title('Kênh Xanh Lá')
    axes[2].axis('off')

    axes[3].imshow(blue_channel_img)
    axes[3].set_title('Kênh Xanh Dương')
    axes[3].axis('off')
    
    fig.suptitle(f'Bài 1: Kênh màu của {filename}', fontsize=16)
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
            sample_img = np.full((100, 100, 3), 255, dtype=np.uint8)
            iio.imsave(sample_img_path, sample_img)
            print(f"Đã tạo ảnh mẫu '{sample_img_path}'. Vui lòng thêm ảnh thật của bạn.")
        except Exception as e:
            print(f"Không thể tạo ảnh mẫu: {e}")
        exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("--- Bắt đầu Bài tập 1 ---")
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_dir, filename)
            base_name, ext = os.path.splitext(filename)
            
            try:
                img_original = iio.imread(image_path)
                bai1_color_channels(img_original, base_name, ext, output_dir)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {filename}: {e}")
                print("Đảm bảo file ảnh không bị hỏng và là định dạng hợp lệ.")

    print("--- Kết thúc Bài tập 1. Kết quả trong thư mục 'OutputImages' và hiển thị. ---")