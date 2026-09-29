import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 1. Khởi tạo model và tự động đẩy lên GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang tải Grounding DINO trên thiết bị: {device}...")

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

# 2. Đọc ảnh và thiết lập Prompt
# Thay tên file tương ứng với bức ảnh đám đông bạn vừa test
image_path = "./acb_raw_dataset/acb_frame_99bd85b6.jpg" 
image = Image.open(image_path).convert("RGB")

# # LƯU Ý QUAN TRỌNG: Text prompt BẮT BUỘC phải cách nhau bằng dấu chấm (.) 
# text_prompt = "person in blue polo shirt. person in green shirt."

# Tách riêng các class để chuẩn bị cho việc đánh ID (0, 1, 2) khi train YOLO
custom_classes = ["blue polo shirt", "green shirt"]
# Nối các class bằng dấu chấm để tạo prompt cho DINO
text_prompt = ". ".join(custom_classes) + "."

# 3. Tiền xử lý và Inference
inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# 4. Hậu xử lý kết quả
# Grounding DINO chia làm 2 loại ngưỡng: 
# - box_threshold: Độ tự tin để xác định đó là một vật thể
# - text_threshold: Độ tự tin để gắn đoạn text vào vật thể đó
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.25,
    text_threshold=0.25,
    target_sizes=[image.size[::-1]]
)[0]

# 5. Vẽ Bounding Box lên ảnh bằng OpenCV
cv2_image = cv2.imread(image_path)

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [int(i) for i in box.tolist()]
#     score_val = round(score.item(), 2)
    
#     # Vẽ khung (Màu đỏ tươi)
#     cv2.rectangle(cv2_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    
#     # Ghi nhãn và độ tự tin
#     label_text = f"{label}: {score_val}"
#     cv2.putText(cv2_image, label_text, (box[0], box[1] - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [int(i) for i in box.tolist()]
    score_val = round(score.item(), 2)
    
    # Thiết lập màu sắc khác nhau cho từng loại áo để dễ phân biệt bằng mắt
    if "blue" in label:
        color = (255, 0, 0) # Xanh dương (BGR)
    elif "green" in label:
        color = (0, 255, 0) # Xanh lá
    else:
        color = (0, 0, 255) # Đỏ (cho khăn choàng)
        
    cv2.rectangle(cv2_image, (box[0], box[1]), (box[2], box[3]), color, 2)
    label_text = f"{label}: {score_val}"
    cv2.putText(cv2_image, label_text, (box[0], box[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# Hiển thị và lưu kết quả
cv2.imshow("Grounding DINO Test", cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("test_groundingdino_result.jpg", cv2_image)
print("Đã lưu kết quả vào file test_groundingdino_result.jpg")