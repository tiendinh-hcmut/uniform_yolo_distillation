import os
import random
import shutil

def create_golden_dataset(raw_dir="./acb_raw_dataset", golden_dir="./golden_dataset", num_samples=100):
    os.makedirs(golden_dir, exist_ok=True)
    
    # Lọc danh sách file ảnh
    all_images = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(all_images) < num_samples:
        print(f"Cảnh báo: Chỉ có {len(all_images)} ảnh, không đủ {num_samples} ảnh yêu cầu.")
        return

    # Chọn ngẫu nhiên 100 ảnh (sử dụng seed để có thể tái tạo lại kết quả random nếu cần)
    random.seed(42) 
    golden_images = random.sample(all_images, num_samples)
    
    print(f"Đang di chuyển {num_samples} ảnh ngẫu nhiên sang {golden_dir}...")
    for img in golden_images:
        src_path = os.path.join(raw_dir, img)
        dst_path = os.path.join(golden_dir, img)
        # Sử dụng shutil.move để chuyển hẳn ảnh sang thư mục mới, tách biệt khỏi tập train
        shutil.move(src_path, dst_path) 
        
    print(f"Hoàn tất! Thư mục gốc còn lại {len(all_images) - num_samples} ảnh để chạy inference.")

create_golden_dataset()