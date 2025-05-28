import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
import os

def bai2_swap_colors(img_original, base_name, ext, output_dir):
    """
    Hoán đổi vị trí các kênh màu (ví dụ: RGB -> BGR hoặc GRB) và hiển thị.
    """
    print(f"  - Đang xử lý Bài 2: Hoán đổi màu cho {base_name}{ext}")

    # Đảm bảo ảnh có 3 kênh màu, nếu là grayscale thì chuyển sang RGB giả
    if img_original.ndim == 2:
        img_original = np.stack([img_original, img_original, img_original], axis=-1)

    # Hoán đổi R và B (RGB -> BGR)
    img_bgr = img_original[:,:,[2,1,0]] # Lấy kênh Blue (index 2), Green (index 1), Red (index 0)
    
    # Hoán đổi R và G (RGB -> GRB)
    img_grb = img_original[:,:,[1,0,2]] # Lấy kênh Green (index 1), Red (index 0), Blue (index 2)

    # Lưu ảnh
    iio.imsave(os.path.join(output_dir, f'{base_name}_bgr_swapped{ext}'), img_bgr)
    print(f"    - Đã lưu: {base_name}_bgr_swapped{ext}")
    iio.imsave(os.path.join(output_dir, f'{base_name}_grb_swapped{ext}'), img_grb)
    print(f"    - Đã lưu: {base_name}_grb_swapped{ext}")

    # Hiển thị ảnh
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(img_original)
    axes[0].set_title('Ảnh gốc')
    axes[0].axis('off')

    axes[1].imshow(img_bgr)
    axes[1].set_title('Hoán đổi (BGR)')
    axes[1].axis('off')

    axes[2].imshow(img_grb)
    axes[2].set_title('Hoán đổi (GRB)')
    axes[2].axis('off')
    
    fig.suptitle(f'Bài 2: Hoán đổi màu của {filename}', fontsize=16)
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

    print("--- Bắt đầu Bài tập 2 ---")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_dir, filename)
            base_name, ext = os.path.splitext(filename)
            
            try:
                img_original = iio.imread(image_path)
                bai2_swap_colors(img_original, base_name, ext, output_dir)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {filename}: {e}")
    print("--- Kết thúc Bài tập 2. Kết quả trong thư mục 'OutputImages' và hiển thị. ---")