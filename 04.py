import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
import os
import colorsys

# Hàm hỗ trợ để chuyển đổi ảnh màu về uint8 an toàn
def convert_to_uint8(image_array):
    if image_array.dtype != np.float32 and image_array.dtype != np.float64:
        image_array = image_array.astype(np.float32)
    image_array = np.clip(image_array, 0, 255)
    return image_array.astype(np.uint8)

def bai4_hsv_transform(img_original, base_name, ext, output_dir):
    """
    Chuyển ảnh sang HSV, thay đổi Hue và Value theo công thức, sau đó chuyển lại RGB và hiển thị.
    """
    print(f"  - Đang xử lý Bài 4: Biến đổi HSV cho {base_name}{ext}")

    # Đảm bảo ảnh có 3 kênh màu, nếu là grayscale thì chuyển sang RGB giả
    if img_original.ndim == 2:
        img_original = np.stack([img_original, img_original, img_original], axis=-1)

    img_normalized = img_original / 255.0

    rgb2hsv_func = np.vectorize(colorsys.rgb_to_hsv) #
    h_old, s_old, v_old = rgb2hsv_func(img_normalized[:,:,0], img_normalized[:,:,1], img_normalized[:,:,2])

    h_new = (1/3) * h_old #
    v_new = (3/4) * v_old #

    # Đảm bảo các giá trị vẫn nằm trong khoảng hợp lệ của HSV
    h_new = np.clip(h_new, 0, 1)
    s_new = s_old # Giữ nguyên Saturation
    v_new = np.clip(v_new, 0, 1)

    hsv2rgb_func = np.vectorize(colorsys.hsv_to_rgb) #
    rgb_new_normalized = hsv2rgb_func(h_new, s_new, v_new)

    rgb_final = np.array(rgb_new_normalized).transpose((1,2,0)) * 255
    rgb_final = convert_to_uint8(rgb_final)

    # Lưu ảnh
    iio.imsave(os.path.join(output_dir, f'{base_name}_hsv_modified{ext}'), rgb_final)
    print(f"    - Đã lưu: {base_name}_hsv_modified{ext}")

    # Hiển thị ảnh
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_original)
    axes[0].set_title('Ảnh gốc')
    axes[0].axis('off')

    axes[1].imshow(rgb_final)
    axes[1].set_title(r'$H_{new}=1/3 H_{old}, V_{new}=3/4 V_{old}$') #
    axes[1].axis('off')
    
    fig.suptitle(f'Bài 4: Biến đổi HSV của {filename}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    input_dir = 'Exercise'
    output_dir = 'OutputImages'

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Thư mục '{input_dir}' chưa tồn tại. Vui lòng đặt các file ảnh vào thư mục này.")
        exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("--- Bắt đầu Bài tập 4 ---")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_dir, filename)
            base_name, ext = os.path.splitext(filename)
            
            try:
                img_original = iio.imread(image_path)
                bai4_hsv_transform(img_original, base_name, ext, output_dir)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {filename}: {e}")
    print("--- Kết thúc Bài tập 4. Kết quả trong thư mục 'OutputImages' và hiển thị. ---")