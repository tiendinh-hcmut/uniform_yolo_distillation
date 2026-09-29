from ultralytics import YOLO

def main():
    # Khởi tạo mô hình Student (YOLOv8 Nano)
    model = YOLO('yolo26n.pt') 

    # Train model
    results = model.train(
        data='data.yml',
        epochs=10,
        imgsz=640,
        device='cuda',
        project='acb_shirt_detect',
        name='student_model',
        workers=4,
        exist_ok=True
    )
    print("Quá trình train hoàn tất! Trọng số đã được lưu.")

if __name__ == '__main__':
    main()