import cv2
from ultralytics import YOLO

# Load file weights best.pt vừa được sinh ra sau khi train xong
model_path = 'runs/detect/acb_shirt_detect/student_model/weights/best.pt'
model = YOLO(model_path)

# Đường dẫn video test (bạn nhớ đổi tên file video tương ứng)
video_path = r"D:\acb_yolo_distillation\downloaded_videos\ACB - Ngày Hội Gia Đình 2018 - Sánh Bước Tương Lai_[qEQTFEw62XI]_clip.mp4" 
#video_path = r"D:\acb_yolo_distillation\downloaded_videos\1790577720408_2063630036087264699_6732529560504490358.mp4" 

cap = cv2.VideoCapture(video_path)

print("Bắt đầu mở Video Demo. Bấm phím 'q' trên cửa sổ video để thoát.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Inference trên từng frame
    results = model.predict(frame, conf=0.3, verbose=False)
    
    # Lấy frame đã vẽ sẵn bounding box
    annotated_frame = results[0].plot()

    # Hiển thị lên màn hình
    cv2.imshow("ACB Demo - Shirt Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()