import os
import cv2
import yt_dlp
from yt_dlp.utils import download_range_func

def time_to_seconds(time_str):
    """
    Chuyển đổi chuỗi thời gian 'MM:SS' hoặc 'HH:MM:SS' sang số giây (float).
    Ví dụ: '0:21' -> 21.0 giây, '2:58' -> 178.0 giây
    """
    if not time_str:
        return None
    try:
        parts = [float(p) for p in str(time_str).strip().split(":")]
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception as e:
        print(f"⚠️ Lỗi định dạng thời gian '{time_str}': {e}")
    return None

def process_youtube_video(video_info, output_dir="downloaded_videos"):
    url = video_info.get("url")
    start_time = video_info.get("start")
    end_time = video_info.get("end")
    
    start_sec = time_to_seconds(start_time)
    end_sec = time_to_seconds(end_time)

    # Tạo thư mục chứa video tải về nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n==================================================")
    print(f"Bắt đầu xử lý: {url}")
    if start_sec is not None and end_sec is not None:
        print(f"Cắt đoạn từ: {start_time} ({start_sec}s) đến {end_time} ({end_sec}s)")
    else:
        print("Tải toàn bộ video (không cắt đoạn).")

    # Đặt template tên file theo: Tiêu đề_ID_đoạn_thời_gian
    # Giới hạn tiêu đề 50 ký tự để không bị lỗi đường dẫn quá dài trên Windows
    outtmpl = os.path.join(output_dir, "%(title).50s_[%(id)s]_clip.%(ext)s")

    ydl_opts = {
        # Ưu tiên lấy định dạng MP4 có hình ảnh + âm thanh tốt nhất
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': outtmpl,
        'quiet': False,
        'no_warnings': True,
        # Sử dụng Node.js để giải quyết JS Challenge tránh bị chặn bot
        'js_runtimes': {"node": {}},
    }

    # Kích hoạt tính năng cắt đoạn trực tiếp khi tải
    if start_sec is not None and end_sec is not None:
        ydl_opts['download_ranges'] = download_range_func(None, [(start_sec, end_sec)])
        ydl_opts['force_keyframes_at_cuts'] = True  # Đảm bảo cắt chính xác theo khung hình I-frame

    downloaded_file = None
    try:
        # 1. Tải đoạn video về máy
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            
            # Xử lý trường hợp yt-dlp sau khi ghép chuyển đuôi thành .mp4
            if not os.path.exists(downloaded_file):
                base, _ = os.path.splitext(downloaded_file)
                if os.path.exists(f"{base}.mp4"):
                    downloaded_file = f"{base}.mp4"

        print(f"✅ Đã tải xong video: {downloaded_file}")

        # 2. Mở video bằng OpenCV để kiểm tra thông số
        if downloaded_file and os.path.exists(downloaded_file):
            cap = cv2.VideoCapture(downloaded_file)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0

                print(f"📊 Thông tin video nhận được từ OpenCV:")
                print(f"   - Độ phân giải: {width}x{height}")
                print(f"   - FPS: {fps:.2f}")
                print(f"   - Tổng frame: {total_frames}")
                print(f"   - Thời lượng đoạn cắt: {duration:.2f} giây")
                
                cap.release()
            else:
                print("⚠️ Không thể mở video bằng OpenCV.")

    except Exception as e:
        print(f"❌ Xảy ra lỗi trong quá trình tải/xử lý: {e}")

if __name__ == "__main__":
    # Cấu hình danh sách các video và khoảng thời gian cần lấy
    videos = [
        {
            "url": "https://www.youtube.com/watch?v=qEQTFEw62XI", 
            "start": "0:00", 
            "end": "0:30"
        },
        # Bạn có thể thêm nhiều video khác vào đây:
        # {
        #     "url": "https://www.youtube.com/watch?v=...",
        #     "start": "1:15",
        #     "end": "3:40"
        # }
    ]

    for video in videos:
        process_youtube_video(video)