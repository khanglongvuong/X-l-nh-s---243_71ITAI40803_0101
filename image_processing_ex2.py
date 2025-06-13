import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# --- Cấu hình ---
EXERCISE_FOLDER = 'exercise' # Thư mục chứa ảnh đầu vào
OUTPUT_FOLDER = 'output_ex2' # Thư mục lưu ảnh đã xử lý

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
    # Kiểm tra nếu ảnh gốc là ảnh màu thì chuyển đổi sang RGB để hiển thị đúng
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else: # Nếu là ảnh xám
        plt.imshow(original, cmap='gray')
    plt.title(title_original)
    plt.axis('off') # Tắt trục tọa độ

    plt.subplot(1, 2, 2)
    # Ảnh đã xử lý từ miền tần số thường là ảnh xám
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

# --- Các hàm biến đổi ảnh trong miền tần số ---

def fast_fourier_transform(image):
    """
    Thực hiện Biến đổi Fourier nhanh (FFT) và trả về phổ biên độ (magnitude spectrum)
    để trực quan hóa.
    """
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu, vì FFT thường áp dụng cho ảnh xám
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Chuyển sang kiểu float32 cho FFT
    f = np.fft.fft2(gray_image.astype(np.float32))
    # Dịch chuyển thành phần tần số 0 về trung tâm (để dễ nhìn)
    fshift = np.fft.fftshift(f)

    # Tính toán phổ biên độ (magnitude spectrum) và chuyển sang thang log
    # Thêm epsilon nhỏ để tránh log(0)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    # Chuẩn hóa về khoảng 0-255 và chuyển sang uint8 để hiển thị dưới dạng ảnh
    normalized_magnitude = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_magnitude.astype(np.uint8)

def create_butterworth_filter(shape, cutoff, order, filter_type='lowpass'):
    """
    Tạo mặt nạ bộ lọc Butterworth.
    shape: Kích thước của ảnh (rows, cols)
    cutoff: Tần số cắt D0
    order: Bậc của bộ lọc n
    filter_type: 'lowpass' (thông thấp) hoặc 'highpass' (thông cao)
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    D = np.zeros((rows, cols), dtype=np.float32) # Ma trận khoảng cách

    # Tính khoảng cách Euclidean từ trung tâm cho mỗi điểm
    for u in range(rows):
        for v in range(cols):
            D[u, v] = np.sqrt((u - center_row)**2 + (v - center_col)**2)

    # Công thức bộ lọc Butterworth
    if filter_type == 'lowpass':
        # H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
        H = 1 / (1 + (D / cutoff)**(2 * order))
    elif filter_type == 'highpass':
        # H(u,v) = 1 / (1 + (D0/D(u,v))^(2n))
        # Thêm epsilon vào D để tránh chia cho 0 tại trung tâm (D=0)
        H = 1 / (1 + (cutoff / (D + 1e-8))**(2 * order))
    else:
        raise ValueError("filter_type phải là 'lowpass' hoặc 'highpass'")
    return H

def apply_frequency_filter(image, filter_mask):
    """Áp dụng mặt nạ bộ lọc trong miền tần số lên một ảnh."""
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Thực hiện FFT
    f = np.fft.fft2(gray_image.astype(np.float32))
    # Dịch chuyển tần số 0 về trung tâm
    fshift = np.fft.fftshift(f)

    # Áp dụng mặt nạ bộ lọc trong miền tần số
    filtered_fshift = fshift * filter_mask

    # Thực hiện Inverse FFT để lấy lại ảnh
    f_ishift = np.fft.ifftshift(filtered_fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back) # Lấy giá trị tuyệt đối vì kết quả có thể là số phức

    # Chuẩn hóa về khoảng 0-255 và chuyển sang uint8
    filtered_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return filtered_image.astype(np.uint8)

def butterworth_lowpass_filter(image, cutoff=30, order=2):
    """Áp dụng Bộ lọc thông thấp Butterworth."""
    rows, cols = image.shape[:2] # Lấy kích thước ảnh (chiều cao, chiều rộng)
    filter_mask = create_butterworth_filter((rows, cols), cutoff, order, 'lowpass')
    return apply_frequency_filter(image, filter_mask)

def butterworth_highpass_filter(image, cutoff=30, order=2):
    """Áp dụng Bộ lọc thông cao Butterworth."""
    rows, cols = image.shape[:2]
    filter_mask = create_butterworth_filter((rows, cols), cutoff, order, 'highpass')
    return apply_frequency_filter(image, filter_mask)

# --- Hàm chạy chính cho Bài tập 2 ---
def run_exercise_2():
    print("\n--- Đang chạy Bài tập 2: Các bộ lọc trong miền tần số ---")
    images, filenames = load_images_from_folder(EXERCISE_FOLDER)

    if not images:
        print("Không tìm thấy ảnh nào để xử lý cho Bài tập 2. Vui lòng kiểm tra thư mục 'exercise'.")
        return

    # Định nghĩa các tùy chọn bộ lọc: Phím tắt, Tên hiển thị, Hàm xử lý, Hậu tố file lưu
    filter_options = {
        'F': ('Biến đổi Fourier nhanh (Phổ biên độ)', fast_fourier_transform, 'fft_mag'),
        'L': ('Bộ lọc thông thấp Butterworth (D0=30, Bậc=2)', butterworth_lowpass_filter, 'b_lowpass'),
        'H': ('Bộ lọc thông cao Butterworth (D0=30, Bậc=2)', butterworth_highpass_filter, 'b_highpass')
    }

    while True:
        print("\nChọn một bộ lọc (hoặc 'Q' để thoát):")
        for key, (desc, _, _) in filter_options.items():
            print(f"  [{key}] {desc}")
        choice = input("Nhập lựa chọn của bạn: ").upper() # Chuyển đổi sang chữ hoa để dễ so sánh

        if choice == 'Q':
            print("Đang thoát Bài tập 2.")
            break

        if choice in filter_options:
            desc, func, suffix = filter_options[choice]
            print(f"\nĐang áp dụng '{desc}' cho tất cả các ảnh...")
            for i, img in enumerate(images):
                original_filename = filenames[i]
                print(f"  Đang xử lý {original_filename}...")
                processed_img = func(img.copy()) # Truyền bản sao của ảnh để tránh sửa đổi ảnh gốc

                # Đối với FFT magnitude, ảnh gốc là màu, ảnh xử lý là ảnh xám (phổ biên độ)
                # Nên cần hiển thị đặc biệt một chút cho đẹp
                if choice == 'F':
                     plt.figure(figsize=(14, 7))
                     plt.subplot(1, 2, 1)
                     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                     plt.title(f"Gốc ({original_filename})")
                     plt.axis('off')
                     plt.subplot(1, 2, 2)
                     plt.imshow(processed_img, cmap='gray') # Phổ biên độ là ảnh xám
                     plt.title(f"{desc} ({original_filename})")
                     plt.axis('off')
                     plt.show()
                else:
                    # Các bộ lọc lowpass/highpass trả về ảnh đã lọc, vẫn có thể hiển thị như ảnh xám
                    display_images(img, processed_img,
                                   f"Gốc ({original_filename})",
                                   f"{desc} ({original_filename})")
                save_image(processed_img, original_filename, suffix)
            print(f"Đã hoàn tất áp dụng '{desc}'.")
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

# --- Điểm bắt đầu của chương trình ---
if __name__ == "__main__":
    run_exercise_2()
    print("\nChương trình đã kết thúc.")