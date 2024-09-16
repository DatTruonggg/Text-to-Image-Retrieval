# Extract Keyframe with TransNetV2

## Set up environment

``` cmd
conda create --name aic24 python=3.11
```
## Install package 

```
!pip install ffmpeg-python pillow
!git clone https://github.com/soCzech/TransNetV2.git
%cd TransNetV2/inference
```
## Run code 
```
python keyframe_extract.py
```
## Note
```
if __name__ == "__main__":
    vids_paths = sorted(glob.glob('/home/dattruong/dat/AI/Competition/HCMAI/Data/Videos_L12/*.mp4')) #Đổi tên Video file (ví dụ chạy Video_L12 thì đổi thành Video_L12 (tên folder))  
    frames_path = '/home/dattruong/dat/AI/Competition/HCMAI/Data/Keyframe/L12' # Đổi tên output folder cho giống với tên Video
    for video_path in vids_paths:
        extractor = KeyframeExtractor(video_path, frames_path)
        extractor.process_video()
```

** Nhớ đổi tên các folder Video cho khớp với Video mình đang chạy. **

Ví dụ: Đạt extract frame từ Video_L20 chẳng hạn, thì sẽ đổi đường dẫn là `'/home/dattruong/dat/AI/Competition/HCMAI/Data/Videos_L20/*.mp4'` và `'/home/dattruong/dat/AI/Competition/HCMAI/Data/Keyframe/L20'`
