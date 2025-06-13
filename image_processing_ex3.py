import cv2
import numpy as np
import os
import random # Cần thư viện random cho việc chọn ngẫu nhiên
from matplotlib import pyplot as plt

# --- Cấu hình ---
EXERCISE_FOLDER = 'exercise' # Thư mục chứa ảnh đầu vào
OUTPUT_FOLDER = 'output_ex3' # Thư mục lưu ảnh đã xử lý

# Đảm bảo thư mục đầu ra tồn tại
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Đã tạo thư mục đầu ra: '{OUTPUT_FOLDER}'")

# --- Hàm hỗ trợ tải ảnh ---
def load_images_from_folder(folder):
    """Tải tất cả các file ảnh được hỗ trợ từ một thư mục."""
    images = []
    filenames = []
    print(f"Đang tải ảnh từ: {os.path.abspath(folder)}")
    if not os.path.exists(folder):
        print(f"Lỗi: Thư mục '{folder}' không tồn tại. Vui lòng tạo thư mục và đặt ảnh vào đó.")
        return [], []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) # Tải ảnh màu (OpenCV mặc định BGR)
            if img is not None:
                images.append(img)
                filenames.append(filename)
                print(f"  Đã tải: {filename}")
            else:
                print(f"  Cảnh báo: Không thể tải ảnh {filename}. Có thể file bị lỗi hoặc định dạng không được hỗ trợ.")
    if not images:
        print(f"Không tìm thấy ảnh nào trong '{folder}'. Vui lòng đảm bảo thư mục tồn tại và chứa các file ảnh hợp lệ.")
    return images, filenames

# --- Hàm hỗ trợ hiển thị ảnh ---
def display_images(original, processed, title_original="Ảnh Gốc", title_processed="Ảnh Đã Xử Lý"):
    """Hiển thị ảnh gốc và ảnh đã xử lý cạnh nhau."""
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title(title_original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if len(processed.shape) == 3:
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(processed, cmap='gray')
    plt.title(title_processed)
    plt.axis('off')
    plt.show()

# --- Hàm hỗ trợ lưu ảnh ---
def save_image(image, original_filename, suffix):
    """Lưu ảnh đã xử lý với một hậu tố."""
    base_name, ext = os.path.splitext(original_filename)
    new_filename = f"{base_name}_{suffix}{ext}"
    save_path = os.path.join(OUTPUT_FOLDER, new_filename)
    cv2.imwrite(save_path, image)
    print(f"Đã lưu: {new_filename} vào '{OUTPUT_FOLDER}'")

# --- Các hàm biến đổi ảnh từ Bài tập 1 (được tái sử dụng) ---

def image_inverse_transformation(image):
    """Thực hiện biến đổi nghịch đảo ảnh (ảnh âm bản)."""
    return 255 - image

def gamma_correction(image, gamma=0.5):
    """Thực hiện hiệu chỉnh Gamma."""
    if gamma <= 0:
        raise ValueError("Giá trị Gamma phải lớn hơn 0.")
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def log_transformation(image, c=1.0):
    """Thực hiện biến đổi Log."""
    normalized_image = image.astype(np.float32) / 255.0
    transformed_image = c * np.log(1 + normalized_image + 1e-8)
    transformed_image = cv2.normalize(transformed_image, None, 0, 255, cv2.NORM_MINMAX)
    return transformed_image.astype(np.uint8)

def histogram_equalization(image):
    """Thực hiện cân bằng lược đồ xám."""
    if len(image.shape) == 3: # Ảnh màu
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        equalized_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else: # Ảnh xám
        equalized_image = cv2.equalizeHist(image)
    return equalized_image

def contrast_stretching(image):
    """Thực hiện kéo giãn độ tương phản."""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return image.copy()
    lookup_table = np.zeros((256,), dtype='uint8')
    for pixel_value in range(256):
        lookup_table[pixel_value] = int(255 * (pixel_value - min_val) / (max_val - min_val))
    return cv2.LUT(image, lookup_table)

# --- Hàm xáo trộn kênh RGB ---
def shuffle_rgb_channels(image):
    """Thay đổi thứ tự kênh BGR (OpenCV) một cách ngẫu nhiên."""
    if len(image.shape) < 3: # Không phải ảnh màu, không có kênh để xáo trộn
        print("Cảnh báo: Ảnh không phải ảnh màu, không thể xáo trộn kênh RGB.")
        return image.copy()

    # OpenCV đọc ảnh dưới dạng BGR, nên chúng ta sẽ xáo trộn các kênh B, G, R
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    random.shuffle(channels) # Xáo trộn thứ tự các mảng kênh
    shuffled_image = cv2.merge(channels)
    return shuffled_image

# --- Hàm chạy chính cho Bài tập 3 ---
def run_exercise_3():
    print("\n--- Đang chạy Bài tập 3: Xáo trộn kênh RGB ngẫu nhiên + Biến đổi ảnh cơ bản ngẫu nhiên ---")
    images, filenames = load_images_from_folder(EXERCISE_FOLDER)

    if not images:
        print("Không tìm thấy ảnh nào để xử lý cho Bài tập 3. Vui lòng kiểm tra thư mục 'exercise'.")
        return

    # Danh sách các hàm biến đổi ảnh từ Bài tập 1 và hậu tố tương ứng
    # Lưu ý: lambda được dùng để truyền tham số mặc định cho gamma_correction
    basic_transform_funcs = [
        (image_inverse_transformation, 'inverse', 'Ảnh Nghịch Đảo'),
        (lambda img: gamma_correction(img, 0.5), 'gamma0.5', 'Hiệu Chỉnh Gamma'),
        (log_transformation, 'log', 'Biến Đổi Log'),
        (histogram_equalization, 'hist_eq', 'Cân Bằng Lược Đồ Xám'),
        (contrast_stretching, 'contrast_stretch', 'Kéo Giãn Tương Phản')
    ]

    for i, img in enumerate(images):
        original_filename = filenames[i]
        print(f"\nĐang xử lý {original_filename}...")

        # Bước 1: Xáo trộn kênh RGB ngẫu nhiên
        shuffled_img = shuffle_rgb_channels(img.copy())
        print(f"  Đã xáo trộn kênh RGB.")

        # Bước 2: Chọn ngẫu nhiên một phép biến đổi từ Bài tập 1
        chosen_transform_func, transform_suffix, transform_name_display = random.choice(basic_transform_funcs)
        print(f"  Đang áp dụng biến đổi ngẫu nhiên: {transform_name_display}")

        # Áp dụng phép biến đổi đã chọn lên ảnh đã xáo trộn kênh
        processed_img = chosen_transform_func(shuffled_img.copy())

        # Hiển thị và lưu ảnh
        display_images(img, processed_img,
                       f"Gốc ({original_filename})",
                       f"Xáo Trộn RGB + {transform_name_display} ({original_filename})")
        save_image(processed_img, original_filename, f"rgb_shuffled_{transform_suffix}")

    print("\nĐã hoàn tất Bài tập 3 cho tất cả các ảnh.")

# --- Điểm bắt đầu của chương trình ---
if __name__ == "__main__":
    run_exercise_3()
    print("\nChương trình đã kết thúc.")