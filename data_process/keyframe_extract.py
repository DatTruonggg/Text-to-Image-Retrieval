import sys
import numpy as np
import glob
import os
import cv2
import ffmpeg

sys.path.append('/home/dattruong/dat/AI/Competition/HCMAI/src/process_data/TransNetV2/inference')  ##lưu ý đổi path đến TransNetv2 (sẽ xuất hiện khi git clone transnetv2)
from transnetv2 import TransNetV2

class KeyframeExtractor:
    def __init__(self, video_path, des_path):
        self.model = TransNetV2()
        self.video_path = video_path
        self.des_path = des_path

    def get_frames(self, fn, width=48, height=27):
        video_stream, _ = (
            ffmpeg
            .input(fn)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])
        return video
    
    def process_video(self):
        folder_name = os.path.basename(self.video_path).replace('.mp4', '')
        folder_path = os.path.join(self.des_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        video = self.get_frames(self.video_path, width=48, height=27)
        single_frame_predictions, _ = self.model.predict_frames(video)
        scenes = self.model.predictions_to_scenes(single_frame_predictions)

        cam = cv2.VideoCapture(self.video_path)
        currentframe = 0
        index = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            currentframe += 1
            if index < len(scenes):
                idx_first, idx_end = scenes[index]
                idx_025 = int(idx_first + (idx_end - idx_first) / 4)
                idx_05 = int(idx_first + (idx_end - idx_first) / 2)
                idx_075 = int(idx_first + 3 * (idx_end - idx_first) / 4)

                if currentframe - 1 == idx_first:
                    filename_first = f"{folder_path}/{idx_first:05d}.jpg"
                    cv2.imwrite(filename_first, frame)

                # if currentframe - 1 == idx_025:
                #     filename_025 = f"{folder_path}/{idx_025:05d}.jpg"
                #     cv2.imwrite(filename_025, frame)

                if currentframe - 1 == idx_05:
                    filename_05 = f"{folder_path}/{idx_05:05d}.jpg"
                    cv2.imwrite(filename_05, frame)

                # if currentframe - 1 == idx_075:
                #     filename_075 = f"{folder_path}/{idx_075:05d}.jpg"
                #     cv2.imwrite(filename_075, frame)

                if currentframe - 1 == idx_end:
                    filename_end = f"{folder_path}/{idx_end:05d}.jpg"
                    cv2.imwrite(filename_end, frame)
                    index += 1

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    vids_paths = sorted(glob.glob('/home/dattruong/dat/AI/Competition/HCMAI/Data/Videos_L12/*.mp4')) #Đổi tên Video file (ví dụ chạy Video_L12 thì đổi thành Video_L12 (tên folder))  
    frames_path = '/home/dattruong/dat/AI/Competition/HCMAI/Data/Keyframe/L12' # Đổi tên output folder cho giống với tên Video
    for video_path in vids_paths:
        extractor = KeyframeExtractor(video_path, frames_path)
        extractor.process_video()
