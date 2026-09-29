import os
import shutil
import random
import torch
import cv2
import torchvision.ops as ops
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ==========================================
# CẤU HÌNH THƯ MỤC VÀ NHÃN
# ==========================================
src_dir = '../acb_raw_dataset'
base_out_dir = 'dataset'
debug_dir = 'debug_visualize'

# Mở rộng từ khóa: Thêm các "chim mồi" (Negative Prompts) để hút các nhận diện sai
custom_classes = [
    "blue shirt", "green shirt", # Target chính
    "white shirt", "suit",                                # Lọc áo trắng/vest
    "blue tie", "blue lanyard",           # Lọc phụ kiện xanh                                                # Lọc nền nhiễu
]
text_prompt = ". ".join(custom_classes) + "."

# Ánh xạ nhãn: Chỉ lấy áo xanh, vứt bỏ toàn bộ áo trắng và phụ kiện
CLASS_MAP = {
    "blue shirt": 1,
    "green shirt": 0,
    # Các nhãn -1 dưới đây sẽ bị loại bỏ hoàn toàn khỏi file txt
    "white shirt": -1,
    "suit": -1,
    "blue tie": -1,
    "blue lanyard": -1,
}

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH VÀ THƯ MỤC
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang tải mô hình Grounding DINO trên: {device}...")
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

if os.path.exists(debug_dir):
    shutil.rmtree(debug_dir)
os.makedirs(debug_dir)

for split in ['train', 'val']:
    os.makedirs(os.path.join(base_out_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(base_out_dir, 'labels', split), exist_ok=True)

all_images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.seed(42)
random.shuffle(all_images)
split_idx = int(len(all_images) * 0.8)
splits = {'train': all_images[:split_idx], 'val': all_images[split_idx:]}

stats = {
    'train': {'total': 0, 'labeled': 0, 'empty': 0}, 
    'val': {'total': 0, 'labeled': 0, 'empty': 0}
}
MAX_DEBUG = 20

# ==========================================
# 2. XỬ LÝ ẢNH, NMS & XUẤT NHÃN YOLO
# ==========================================
print("\nBắt đầu Auto-Labeling bằng Grounding DINO (Có NMS & Lọc nhiễu)...")

for split_name, images in splits.items():
    print(f"\n--- Xử lý tập {split_name} ({len(images)} ảnh) ---")
    debug_count = 0
    
    for img_name in tqdm(images, desc=f"Tiến trình {split_name}"):
        src_img_path = os.path.join(src_dir, img_name)
        dst_img_path = os.path.join(base_out_dir, 'images', split_name, img_name)
        label_path = os.path.join(base_out_dir, 'labels', split_name, img_name.rsplit('.', 1)[0] + '.txt')
        
        shutil.copy(src_img_path, dst_img_path)
        
        image = Image.open(src_img_path).convert("RGB")
        img_w, img_h = image.size
        
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,      
            text_threshold=0.3, 
            target_sizes=[image.size[::-1]]
        )[0]
        
        valid_boxes = []
        valid_scores = []
        valid_labels = []
        valid_label_texts = []
        
        # Bước 1: Lọc qua bộ CLASS_MAP
        for score, label_text, box in zip(results["scores"], results["labels"], results["boxes"]):
            matched_class_id = -1
            for class_key, class_id in CLASS_MAP.items():
                if class_key in label_text:
                    matched_class_id = class_id
                    break
            
            # Chỉ lấy ID 0 hoặc 1, bỏ qua các nhãn rác (-1)
            if matched_class_id != -1:
                valid_boxes.append(box)
                valid_scores.append(score)
                valid_labels.append(matched_class_id)
                valid_label_texts.append(label_text)
                
        has_box = False
        yolo_annotations = []
        
        # Bước 2: Chạy thuật toán NMS để xóa khung trùng lặp
        if len(valid_boxes) > 0:
            valid_boxes_tensor = torch.stack(valid_boxes)
            valid_scores_tensor = torch.stack(valid_scores)
            
            # iou_threshold=0.5: Trùng nhau trên 50% diện tích thì xóa khung có score thấp hơn
            keep_indices = ops.nms(valid_boxes_tensor, valid_scores_tensor, iou_threshold=0.5)
            
            cv2_img = cv2.imread(src_img_path) if debug_count < MAX_DEBUG else None
            
            for idx in keep_indices:
                has_box = True
                box = valid_boxes_tensor[idx]
                cls_id = valid_labels[idx]
                score = valid_scores_tensor[idx].item()
                label_text = valid_label_texts[idx]
                
                # Chuyển hệ tọa độ
                x_min, y_min, x_max, y_max = [float(i) for i in box.tolist()]
                x_c = (x_min + x_max) / 2.0 / img_w
                y_c = (y_min + y_max) / 2.0 / img_h
                w = (x_max - x_min) / img_w
                h = (y_max - y_min) / img_h
                
                yolo_annotations.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                
                # Vẽ ảnh Debug
                if cv2_img is not None:
                    color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                    cv2.rectangle(cv2_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                    cv2.putText(cv2_img, f"{label_text} {score:.2f}", (int(x_min), int(y_min) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if cv2_img is not None and has_box:
                debug_file_path = os.path.join(debug_dir, f"{split_name}_{img_name}")
                cv2.imwrite(debug_file_path, cv2_img)
                debug_count += 1
                
        # Ghi file
        with open(label_path, 'w') as f:
            f.writelines(yolo_annotations)
            
        stats[split_name]['total'] += 1
        if has_box:
            stats[split_name]['labeled'] += 1
        else:
            stats[split_name]['empty'] += 1

# ==========================================
# 3. IN BÁO CÁO TỔNG KẾT
# ==========================================
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
print(f"📂 Đã xuất tối đa {MAX_DEBUG} ảnh mẫu CÓ CHỨA NHÃN tại thư mục: {debug_dir}/")
print("==================================================")