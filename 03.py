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

def bai3_hsv_channels(img_original, base_name, ext, output_dir):
    """
    Chuyển ảnh RGB sang HSV và lưu từng kênh Hue, Saturation, Value riêng biệt, sau đó hiển thị.
    """
    print(f"  - Đang xử lý Bài 3: Kênh HSV cho {base_name}{ext}")

    # Đảm bảo ảnh có 3 kênh màu, nếu là grayscale thì chuyển sang RGB giả
    if img_original.ndim == 2:
        img_original = np.stack([img_original, img_original, img_original], axis=-1)

    img_normalized = img_original / 255.0

    # Chuyển đổi từ RGB sang HSV
    rgb2hsv_func = np.vectorize(colorsys.rgb_to_hsv) #
    h, s, v = rgb2hsv_func(img_normalized[:,:,0], img_normalized[:,:,1], img_normalized[:,:,2])

    # Kênh Hue (hiển thị màu để dễ hình dung dải màu)
    h_colored = np.zeros_like(img_normalized)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            # Để thấy rõ Hue, đặt Saturation và Value tối đa (1.0)
            h_colored[i,j,:] = colorsys.hsv_to_rgb(h[i,j], 1.0, 1.0)
    h_colored_uint8 = convert_to_uint8(h_colored * 255)

    # Kênh Saturation (hiển thị grayscale)
    s_grayscale = convert_to_uint8(s * 255)

    # Kênh Value (hiển thị grayscale)
    v_grayscale = convert_to_uint8(v * 255)

    # Lưu ảnh
    iio.imsave(os.path.join(output_dir, f'{base_name}_h_channel_colored{ext}'), h_colored_uint8)
    print(f"    - Đã lưu: {base_name}_h_channel_colored{ext}")
    iio.imsave(os.path.join(output_dir, f'{base_name}_s_channel_gray{ext}'), s_grayscale)
    print(f"    - Đã lưu: {base_name}_s_channel_gray{ext}")
    iio.imsave(os.path.join(output_dir, f'{base_name}_v_channel_gray{ext}'), v_grayscale)
    print(f"    - Đã lưu: {base_name}_v_channel_gray{ext}")

    # Hiển thị ảnh
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    axes[0].imshow(img_original)
    axes[0].set_title('Ảnh gốc (RGB)')
    axes[0].axis('off')

    axes[1].imshow(h_colored_uint8)
    axes[1].set_title('Kênh Hue (màu)')
    axes[1].axis('off')

    axes[2].imshow(s_grayscale, cmap='gray')
    axes[2].set_title('Kênh Saturation (xám)')
    axes[2].axis('off')

    axes[3].imshow(v_grayscale, cmap='gray')
    axes[3].set_title('Kênh Value (xám)')
    axes[3].axis('off')
    
    fig.suptitle(f'Bài 3: Kênh HSV của {filename}', fontsize=16)
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

    print("--- Bắt đầu Bài tập 3 ---")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_dir, filename)
            base_name, ext = os.path.splitext(filename)
            
            try:
                img_original = iio.imread(image_path)
                bai3_hsv_channels(img_original, base_name, ext, output_dir)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {filename}: {e}")
    print("--- Kết thúc Bài tập 3. Kết quả trong thư mục 'OutputImages' và hiển thị. ---")