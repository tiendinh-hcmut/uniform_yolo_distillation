from ultralytics import YOLOWorld
import cv2

# 1. Khởi tạo mô hình (tự động tải pre-trained weights yolov8s-world.pt)
model = YOLOWorld('yolov8x-worldv2.pt')

# 2. Thiết lập Text Prompt (Classes)
# Thay vì ghi chung chung "bank employee", ta chia nhỏ các đặc điểm cốt lõi
classes = [
    "person wearing green t-shirt", 
    "person wearing blue t-shirt", 
    # "skirt", 
    # "scarf",
]
model.set_classes(classes)

# 3. Lấy 1 ảnh bất kỳ từ tập raw để test (Nhớ đổi tên file cho khớp ảnh thực tế)
image_path = "./acb_raw_dataset/acb_frame_99bd85b6.jpg" 

# 4. Chạy inference
# YOLO-World là mô hình Zero-shot nên ban đầu set confidence thấp (conf=0.1 hoặc 0.05) để xem nó bắt được gì
results = model.predict(image_path, conf=0.1)

# 5. Hiển thị và lưu kết quả
annotated_frame = results[0].plot()
cv2.imshow("YOLO-World Test", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("test_yoloworld_result.jpg", annotated_frame)