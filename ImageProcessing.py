import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# --- Cấu hình ---
EXERCISE_FOLDER = 'exercise' # Thư mục chứa ảnh đầu vào
OUTPUT_FOLDER = 'output_ex1' # Thư mục lưu ảnh đã xử lý

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
        # Kiểm tra xem có phải là file ảnh không (bỏ qua thư mục con hoặc file không phải ảnh)
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
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)) # OpenCV đọc BGR, Matplotlib cần RGB
    plt.title(title_original)
    plt.axis('off') # Tắt trục tọa độ

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
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

# --- Các hàm biến đổi ảnh theo yêu cầu ---

def image_inverse_transformation(image):
    """Thực hiện biến đổi nghịch đảo ảnh (ảnh âm bản)."""
    # Đối với ảnh 8-bit (0-255), giá trị nghịch đảo là 255 - giá trị pixel
    return 255 - image

def gamma_correction(image, gamma=0.5):
    """Thực hiện hiệu chỉnh Gamma."""
    # Đảm bảo gamma khác 0 để tránh lỗi chia cho 0
    if gamma <= 0:
        raise ValueError("Giá trị Gamma phải lớn hơn 0.")
    inv_gamma = 1.0 / gamma
    # Tạo bảng tra cứu (Look-up Table - LUT) để tính toán nhanh hơn
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table) # Áp dụng bảng tra cứu

def log_transformation(image, c=1.0):
    """Thực hiện biến đổi Log. s = c * log(1 + r)."""
    # Chuyển ảnh sang kiểu float32 để tính toán, chuẩn hóa về khoảng [0, 1]
    normalized_image = image.astype(np.float32) / 255.0
    # Áp dụng biến đổi log. Thêm một giá trị epsilon nhỏ để tránh log(0).
    transformed_image = c * np.log(1 + normalized_image + 1e-8)
    # Chuẩn hóa lại về khoảng 0-255 và chuyển về kiểu uint8
    transformed_image = cv2.normalize(transformed_image, None, 0, 255, cv2.NORM_MINMAX)
    return transformed_image.astype(np.uint8)

def histogram_equalization(image):
    """Thực hiện cân bằng lược đồ xám."""
    # Đối với ảnh màu, thông thường sẽ cân bằng trên kênh cường độ (Y trong YUV hoặc V trong HSV)
    if len(image.shape) == 3: # Nếu là ảnh màu (BGR)
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) # Cân bằng kênh Y (cường độ sáng)
        equalized_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else: # Nếu là ảnh xám
        equalized_image = cv2.equalizeHist(image)
    return equalized_image

def contrast_stretching(image):
    """Thực hiện kéo giãn độ tương phản bằng cách ánh xạ min pixel về 0 và max pixel về 255."""
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val: # Tránh lỗi chia cho 0 nếu ảnh có độ tương phản bằng 0 (ảnh phẳng)
        return image.copy()

    # Tạo bảng tra cứu
    lookup_table = np.zeros((256,), dtype='uint8')
    for pixel_value in range(256):
        # Công thức kéo giãn tuyến tính: output = (input - min_input) * (max_output - min_output) / (max_input - min_input) + min_output
        # Ở đây, max_output = 255, min_output = 0
        lookup_table[pixel_value] = int(255 * (pixel_value - min_val) / (max_val - min_val))

    return cv2.LUT(image, lookup_table)

# --- Hàm chạy chính cho Bài tập 1 ---
def run_exercise_1():
    print("\n--- Đang chạy Bài tập 1: Các phép biến đổi ảnh cơ bản ---")
    images, filenames = load_images_from_folder(EXERCISE_FOLDER)

    if not images:
        print("Không tìm thấy ảnh nào để xử lý cho Bài tập 1. Vui lòng kiểm tra thư mục 'exercise'.")
        return

    # Định nghĩa các tùy chọn biến đổi: Phím tắt, Tên hiển thị, Hàm xử lý, Hậu tố file lưu
    transform_options = {
        'I': ('Biến đổi ảnh nghịch đảo (Negative)', image_inverse_transformation, 'inverse'),
        'G': ('Hiệu chỉnh Gamma (gamma=0.5)', lambda img: gamma_correction(img, 0.5), 'gamma0.5'), # Sử dụng lambda để truyền tham số gamma
        'L': ('Biến đổi Log (c=1.0)', log_transformation, 'log'),
        'H': ('Cân bằng lược đồ xám (Histogram Equalization)', histogram_equalization, 'hist_eq'),
        'C': ('Kéo giãn độ tương phản (Contrast Stretching)', contrast_stretching, 'contrast_stretch')
    }

    while True:
        print("\nChọn một phép biến đổi (hoặc 'Q' để thoát):")
        for key, (desc, _, _) in transform_options.items():
            print(f"  [{key}] {desc}")
        choice = input("Nhập lựa chọn của bạn: ").upper() # Chuyển đổi sang chữ hoa để dễ so sánh

        if choice == 'Q':
            print("Đang thoát Bài tập 1.")
            break

        if choice in transform_options:
            desc, func, suffix = transform_options[choice]
            print(f"\nĐang áp dụng '{desc}' cho tất cả các ảnh...")
            for i, img in enumerate(images):
                original_filename = filenames[i]
                print(f"  Đang xử lý {original_filename}...")
                processed_img = func(img.copy()) # Truyền bản sao của ảnh để tránh sửa đổi ảnh gốc

                display_images(img, processed_img,
                               f"Gốc ({original_filename})",
                               f"{desc} ({original_filename})")
                save_image(processed_img, original_filename, suffix)
            print(f"Đã hoàn tất áp dụng '{desc}'.")
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

# --- Điểm bắt đầu của chương trình ---
if __name__ == "__main__":
    run_exercise_1()
    print("\nChương trình đã kết thúc.")