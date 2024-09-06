import os
import cv2
import json

def extract_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def create_fps_json(root_dir, dest_dir, fps_file_path):
    """Tạo file JSON chứa thông tin FPS của các video trong thư mục

    Args:
        root_dir (str): Đường dẫn đến thư mục gốc chứa video.
        dest_dir (str): Đường dẫn đến thư mục lưu file JSON.
        fps_file_path (str): Đường dẫn đến file JSON đầu ra.
    """
    # Tạo một dictionary để lưu trữ thông tin FPS
    fps_dict = {}

    # Duyệt qua tất cả các file trong thư mục gốc
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp4'):  # Thay đổi phần mở rộng nếu cần
                file_name = os.path.splitext(file)[0]
                video_path = os.path.join(root, file)
                # Trích xuất FPS và thêm vào dictionary
                fps = extract_fps(video_path)
                fps_dict[file_name] = fps

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(dest_dir, exist_ok=True)

    # Ghi dữ liệu vào file JSON
    with open(fps_file_path, 'w') as f:
        json.dump(fps_dict, f, indent=4)

# Thay thế các đường dẫn bằng đường dẫn thực tế của bạn
root = '/media/dattruong/568836F88836D669/Users/DAT/Hackathon/HCMAI24/Data'
dest_path = root + '/dict'
fps_file_path = root + '/map-keyframes-b1/fps.json'

create_fps_json(root, dest_path, fps_file_path)