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
