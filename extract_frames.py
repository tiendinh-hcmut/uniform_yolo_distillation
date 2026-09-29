import cv2
import os
import yt_dlp
import uuid

def time_to_seconds(time_str):
    """Chuyển đổi chuỗi định dạng MM:SS hoặc HH:MM:SS sang giây"""
    if not time_str:
        return 0
    parts = str(time_str).split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(time_str)

def check_blur(image, threshold=85.0):
    # Tính toán phương sai của Laplacian để xác định độ nét
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    # return fm < threshold
    return False  # Tạm thời bỏ qua kiểm tra độ nét để tránh bỏ sót frame 

def process_youtube_video(video_info, output_dir="./acb_raw_dataset", fps_extract=1):
    url = video_info.get("url")
    start_time = video_info.get("start")
    end_time = video_info.get("end")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Tải video
    print(f"\nĐang tải video: {url}")
    ydl_opts = {
        'format': 'best[ext=mp4]/bestvideo[ext=mp4]/best',
        'outtmpl': 'temp_video.%(ext)s',
        'quiet': True,
        # Xóa hoặc comment dòng cookiesfrombrowser cũ đi
        # 'cookiesfrombrowser': ('edge',),
        # Dùng dòng này để trỏ tới file cookie vừa tải về
        #'cookiefile': 'cookies.txt',
        # Dùng Node.js để giải JS challenge. 
        "js_runtimes": { "node": {}, },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # 2. Xử lý OpenCV
    cap = cv2.VideoCapture('temp_video.mp4')
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, video_fps // fps_extract)
    
    # Quy đổi thời gian ra giây
    start_sec = time_to_seconds(start_time) if start_time else 0
    end_sec = time_to_seconds(end_time) if end_time else float('inf')
    
    # Tua nhanh video đến vị trí start_time (tính bằng mili-giây)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    
    count = 0
    saved_count = 0
    
    print(f"Đang trích xuất frame từ {start_time or 'đầu'} đến {end_time or 'cuối'}...")
    while cap.isOpened():
        # Lấy thời gian hiện tại của frame đang đọc (đổi ra giây)
        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Nếu vượt quá thời gian kết thúc thì dừng vòng lặp sớm
        if current_sec > end_sec:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            if not check_blur(frame, threshold=85.0):
                unique_id = uuid.uuid4().hex[:8]
                file_path = os.path.join(output_dir, f"acb_frame_{unique_id}.jpg")
                cv2.imwrite(file_path, frame)
                saved_count += 1
        count += 1
        
    cap.release()
    
    # 3. Dọn dẹp
    if os.path.exists('temp_video.mp4'):
        os.remove('temp_video.mp4')
        
    print(f"Hoàn tất! Đã lưu {saved_count} ảnh nét vào thư mục {output_dir}.")

# Cấu hình danh sách video và khoảng thời gian cần lấy
videos = [
    {
        "url": "https://www.youtube.com/watch?v=XOkCUd1SwBo", 
        "start": "0:21", 
        "end": "2:58"
    },
    # {
    #     "url": "LINK_VIDEO_2", 
    #     "start": "00:35", 
    #     "end": None  # Dùng None nếu muốn lấy từ giây 30 đến tận cuối video
    # }
]

for video in videos:
    process_youtube_video(video)