import os
import shutil
import random
from tqdm import tqdm  
from ultralytics import YOLOWorld

src_dir = '../acb_raw_dataset'
base_out_dir = 'dataset'
debug_dir = 'debug_visualize'  

# Làm sạch thư mục debug_visualize cũ (nếu có)
if os.path.exists(debug_dir):
    shutil.rmtree(debug_dir)
os.makedirs(debug_dir)

# 1. Tạo cây thư mục chuẩn YOLO và thư mục debug
for split in ['train', 'val']:
    os.makedirs(os.path.join(base_out_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(base_out_dir, 'labels', split), exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# 2. Lấy danh sách ảnh và chia tỷ lệ 80% Train, 20% Val
all_images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.seed(42)
random.shuffle(all_images)

split_idx = int(len(all_images) * 0.8)
splits = {
    'train': all_images[:split_idx],
    'val': all_images[split_idx:]
}

# 3. Khởi tạo mô hình Teacher
model = YOLOWorld('yolov8x-worldv2.pt')

# Danh sách từ khóa (Thứ tự: 0, 1, 2, 3, 4)
classes = ["green shirt", "blue shirt", "green uniform", "blue uniform", "shirt"]
model.set_classes(classes)

# TỪ ĐIỂN GỘP NHÃN (LABEL MAPPING)
# Chuyển đổi ID mà YOLO-World dự đoán sang ID chuẩn của Student (0 hoặc 1)
CLASS_MAP = {
    0: 0,   # "green shirt"   -> 0
    1: 1,   # "blue shirt"    -> 1
    2: 0,   # "green uniform" -> 0
    3: 1,   # "blue uniform"  -> 1
    4: -1   # "shirt"         -> -1 (Bỏ qua)
}

stats = {
    'train': {'total': 0, 'labeled': 0, 'empty': 0},
    'val': {'total': 0, 'labeled': 0, 'empty': 0}
}

MAX_DEBUG_IMAGES = 20 

# 4. Chạy inference, copy ảnh và sinh nhãn
print("Bắt đầu xử lý, sinh nhãn và chia thư mục YOLO...\n")

for split_name, images in splits.items():
    print(f"--- Đang xử lý tập {split_name} ({len(images)} ảnh) ---")
    debug_saved = 0  
    
    for i, img_name in enumerate(tqdm(images, desc=f"Tiến trình {split_name}", unit="ảnh")):
        src_img_path = os.path.join(src_dir, img_name)
        dst_img_path = os.path.join(base_out_dir, 'images', split_name, img_name)
        
        shutil.copy(src_img_path, dst_img_path)
        
        results = model.predict(src_img_path, conf=0.05, verbose=False)
        
        label_path = os.path.join(base_out_dir, 'labels', split_name, img_name.rsplit('.', 1)[0] + '.txt')
        has_valid_box = False  # Đổi thành has_valid_box để chỉ tính các ảnh có nhãn 0 hoặc 1
        
        with open(label_path, 'w') as f:
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        raw_cls_id = int(box.cls[0].item())
                        
                        # Sử dụng từ điển để gộp ID
                        mapped_cls_id = CLASS_MAP.get(raw_cls_id, -1)
                        
                        # Nếu rơi vào "-1" (ví dụ: "shirt"), bỏ qua bounding box này
                        if mapped_cls_id == -1:
                            continue
                        
                        has_valid_box = True
                        x_c, y_c, w, h = box.xywhn[0].tolist() 
                        f.write(f"{mapped_cls_id} {x_c} {y_c} {w} {h}\n")
                    
                    # Chỉ lưu ảnh debug nếu thực sự tìm được "green" hoặc "blue"
                    if has_valid_box and debug_saved < MAX_DEBUG_IMAGES:
                        debug_path = os.path.join(debug_dir, f"{split_name}_debug_{img_name}")
                        result.save(filename=debug_path)
                        debug_saved += 1
        
        # Cập nhật bộ đếm
        stats[split_name]['total'] += 1
        if has_valid_box:
            stats[split_name]['labeled'] += 1
        else:
            stats[split_name]['empty'] += 1

# 5. In báo cáo tổng kết
print("\n==================================================")
print("📊 BÁO CÁO KẾT QUẢ GÁN NHÃN TỰ ĐỘNG (AUTO-LABELING)")
print("==================================================")
total_labeled = 0
total_empty = 0

for split_name in ['train', 'val']:
    s = stats[split_name]
    total_labeled += s['labeled']
    total_empty += s['empty']
    print(f"Tập {split_name.upper()}:")
    print(f"  - Tổng số ảnh: {s['total']}")
    print(f"  - Ảnh vẽ được khung (Labeled): {s['labeled']}")
    print(f"  - Ảnh bị bỏ trống (Empty): {s['empty']}")
    print("-" * 50)

print(f"🎯 TỔNG CỘNG: {total_labeled} ảnh có nhãn | {total_empty} ảnh trống.")
print(f"📂 Đã xuất tối đa {MAX_DEBUG_IMAGES} ảnh mẫu CÓ CHỨA NHÃN tại thư mục: {debug_dir}/")
print("==================================================")