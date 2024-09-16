# Extract Keyframe with TransNetV2

**Chỉ cần 1 file code duy nhất thôi nhé !!!!**

Main path: `data_process/keyframe_extract.py`

## Set up environment

``` cmd
conda create --name aic24 python=3.11
```
## Install package 

```
!pip install ffmpeg-python pillow
!git clone https://github.com/soCzech/TransNetV2.git
```
**Lưu ý**: Sau khi install TransNetV2 bằng `git clone` thì chỉnh lại code chỗ này :

```python
sys.path.append('/home/dattruong/dat/AI/Competition/HCMAI/src/process_data/TransNetV2/inference')  ##lưu ý đổi path đến TransNetv2 (sẽ xuất hiện khi git clone transnetv2)
```

Thay bằng đường dẫn đến mục inference trong folder `TransNetV2`

## Run code 
```
python keyframe_extract.py
```
## Note
**Lưu ý code chỗ này nhe**
``` python
if __name__ == "__main__":
    vids_paths = sorted(glob.glob('/home/dattruong/dat/AI/Competition/HCMAI/Data/Videos_L12/*.mp4')) #Đổi tên Video file (ví dụ chạy Video_L12 thì đổi thành Video_L12 (tên folder))  
    frames_path = '/home/dattruong/dat/AI/Competition/HCMAI/Data/Keyframe/L12' # Đổi tên output folder cho giống với tên Video
    for video_path in vids_paths:
        extractor = KeyframeExtractor(video_path, frames_path)
        extractor.process_video()
```

**Nhớ đổi tên các folder Video cho khớp với Video mình đang chạy.**

Ví dụ: Đạt extract frame từ Video_L20 chẳng hạn, thì sẽ đổi đường dẫn là `'/home/dattruong/dat/AI/Competition/HCMAI/Data/Videos_L20/*.mp4'` và `'/home/dattruong/dat/AI/Competition/HCMAI/Data/Keyframe/L20'`
