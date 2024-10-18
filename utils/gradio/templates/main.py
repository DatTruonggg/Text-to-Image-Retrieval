import gradio as gr
import sys
import os
from dotenv import load_dotenv
import numpy as np 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Update sys path to import TextSearch
from utils.text_search import TextSearch
from utils.context_encoding import VisualEncoding

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
VisualEncoder = VisualEncoding()
json_path = os.getenv("ID2IMG") 
json_path_cloud = os.getenv("ID2IMG_CLOUD")
clipb16_bin = os.getenv("FAISS_CLIP_B16")
clipv2_l14_bin = os.getenv("FAISS_CLIPV2_L14")
clipv2_h14_bin = os.getenv("FAISS_CLIPV2_H14")
audio_json_path = os.getenv("AUDO_ID2IMG_FPS")
scene_path = os.getenv("SCENE_ID2INFO")
img2audio_json_path = os.getenv("IMG_ID2AUDIO_ID")
video_division_path = os.getenv("VIDEO_DIVSION_TAG")
map_keyframes = os.getenv("MAP_KEYFRAME")
video_id2img_id = os.getenv("VIDEO_ID2IMG_ID")

search = TextSearch(json_path, json_path_cloud, clipb16_bin, clipv2_l14_bin, clipv2_h14_bin, audio_json_path, img2audio_json_path)
DictImagePath = search.load_json_file(json_path)
DictKeyframe2Id = search.load_json_file("/home/dattruong/dat/AI/Competition/Text-to-Image-Retrieval/data/id2img_fps_laptop.json")

# Function to apply filters (OCR, Tags Detection, Color Detection, Object Detection)
def apply_filter(ocr, tags, color, object_detection):
    filters = []
    if ocr:
        filters.append("OCR")
    if tags:
        filters.append("Tags Detection")
    if color:
        filters.append("Color Detection")
    if object_detection:
        filters.append("Object Detection")
    return f"Applied filters: {', '.join(filters)}"

# Function for text-to-image search and result display
def text_to_image(text: str, top_k: int, model_type: str, storage: str):
    global search  # Use the globally loaded model
    _, _, _, image_paths = search.text_search(text=text, top_k=top_k, model_type=model_type, storage=storage)

    # Create titles for each image
    titles = []
    for img_path in image_paths:
        folder = img_path.split("/")[-2]
        frame_idx = os.path.basename(img_path).split(".")[0].strip()  # Frame index
        
        # Giữ nguyên các số không nếu frame_idx là '000', '0000', hoặc '00000'
        if frame_idx in ["000", "0000", "00000"]:
            frame_idx_display = "0000"  # Giữ nguyên frame_idx
        else: 
            frame_idx_display = frame_idx.lstrip("0")  # Loại bỏ các số không ở đầu
        
        title = f"{folder}, {frame_idx_display}"
        titles.append(title)

    return [(img, title) for img, title in zip(image_paths, titles)]

# Function to get `idx_image` based on selected image path
def get_idx_from_image(folder_name, image_name):
    short_folder = folder_name[:3]
    selected_image = os.path.join("/media/dattruong/568836F88836D669/Users/DAT/Hackathon/HCMAI24/Data/Keyframe/", short_folder, folder_name, image_name + '.jpg')  # Fixed path construction
    for idx, info in json_path.items():
        img_path = info['image_path']  # Get image path from value
        if img_path == selected_image:  
            return idx  # Return idx if found
    return None  

# Function to search neighboring images
# Function to search neighboring images
def search_neighbor(folder_name, image_name):
    idx_image = get_idx_from_image(folder_name, image_name)

    images = []
    
    # Find start and end indices
    start_idx = max(0, idx_image - 10)  # Do not go below 0
    end_idx = min(idx_image + 189, len(json_path) - 1)  # Ensure we don't go out of bounds

    # Iterate through indices from `start_idx` to `end_idx`
    for idx in range(start_idx, end_idx + 1):
        if idx in json_path.keys():  # Check if `idx` is valid in string format
            image_path = json_path[idx]['image_path']  # Access using string keys
            
            folder = os.path.basename(os.path.dirname(image_path))  # Extract folder name from path
            frame_idx = os.path.basename(image_path).split(".")[0]  # Frame index
            
            # Giữ nguyên các số không nếu frame_idx là '000', '0000', hoặc '00000'
            if frame_idx in ["000", "0000", "00000"]:
                frame_idx_display = "0000"  # Giữ nguyên frame_idx
            else: 
                frame_idx_display = frame_idx.lstrip("0")  # Loại bỏ các số không ở đầu
            
            title = f"{folder}, {frame_idx_display}"
            images.append((image_path, title))        
    return images

def download_csv(folder_name, image_name, download_name):
    # Lấy danh sách hình ảnh bao gồm hình ảnh hiện tại và 99 hình ảnh tiếp theo
    images = []
    idx_image = get_idx_from_image(folder_name, image_name)
    start_idx = max(0, idx_image - 2)  # Do not go below 0
    end_idx = min(idx_image + 97, len(json_path) - 1)  # Ensure we don't go out of bounds

    # Iterate through indices from `start_idx` to `end_idx`
    for idx in range(start_idx, end_idx + 1):
        if idx in json_path.keys():  # Check if `idx` is valid in string format
            image_path = json_path[idx]['image_path']  # Access using string keys
            
            folder = os.path.basename(os.path.dirname(image_path))  # Extract folder name from path
            frame_idx = os.path.basename(image_path).split(".")[0]  # Frame index
            
            # Giữ nguyên các số không nếu frame_idx là '000', '0000', hoặc '00000'
            if frame_idx in ["000", "0000", "00000"]:
                frame_idx_display = "0000"  # Giữ nguyên frame_idx
            else: 
                frame_idx_display = frame_idx.lstrip("0")  # Loại bỏ các số không ở đầu
            
            title = f"{folder}, {frame_idx_display}"  # Sử dụng frame_idx_display để định dạng tiêu đề
            images.append((image_path, title)) 

    # Tạo danh sách để lưu đường dẫn hình ảnh và tiêu đề mà không có header
    csv_data = []
    
    for img_path, title in images:
        csv_data.append(title)  # Chỉ lấy tiêu đề (đã định dạng)

    # Tạo tên file CSV
    csv_file_name = f"{download_name}.csv"

    # Lưu danh sách vào file CSV mà không có header
    with open(csv_file_name, 'w') as f:
        for item in csv_data:
            f.write(f"{item}\n")  # Ghi từng tiêu đề vào file

    return csv_file_name  # Trả về tên file CSV


def get_initial_visual_encoding():
    """
    Trả về output ban đầu của VisualEncoder.

    Returns:
        List of tuples: Danh sách các tuple (hình ảnh, tiêu đề) để hiển thị ban đầu trên gr.Gallery.
    """
    icons_dir = "/home/dattruong/dat/AI/Competition/Text-to-Image-Retrieval/utils/gradio/icons"
    
    # Tách icon và color ra riêng
    initial_icons = ["airplane", "apple", "backpack", "banana", "baseball bat", "baseball glove",
                    "bear", "bed", "bench", "bicycle", "bird", "boat", "book", "bottle",
                    "bowl", "broccoli", "bus", "cake", "car", "carrot", "cat", "cell phone",
                    "chair", "clock", "couch", "cow", "cup", "dining table", "dog",
                    "donut", "elephant", "fire hydrant", "fork", "frisbee",
                    "giraffe", "hair drier", "horse", "hot dog ", "keyboard",
                    "kite", "knife", "laptop", "microwave", "motorcycle",
                    "mouse", "orange", "oven", "parking meter", "person", "pizza", 
                    "potted plant", "refrigerator", "remote", "sandwich", "scissors", "sheep",
                    "sink", "skis", "skateboard", "snowboard", "spoon", "sports ball", "stop sign",
                    "suitcase", "surfboard", "teddy bear", "tennis racket", "tie", "toaster", "toilet",
                    "toothbrush", "traffic light", "truck", "tv", "umbrella", "vase", "wine glass", "zebra"],  # Danh sách các icon ban đầu
    initial_colors = ['black', 'blue', 'brown', 'green', 'grey', 'orange_', 'pink', 'purple',
                          'red', 'white', 'yellow', 'gold','silver', 'teal', 'magenta','olive-green', 'sky-blue', 
                          'forest-green', 'sand', 'sunset', 'ice-blue', 'cool-gray', 'cyan', 'dark-blue',
                          'beige', 'rose', 'lavender','charcoal']  # Danh sách các màu sắc ban đầu

    images = []
    for icon_name, color_name in zip(initial_icons, initial_colors):
        icon_path = os.path.join(icons_dir, icon_name + ".png")
        color_path = os.path.join(icons_dir, color_name + ".png")
        icon_image = gr.Image.update(value=icon_path)
        color_image = gr.Image.update(value=color_path)
        title = f"Icon: {icon_name}, Color: {color_name}"
        images.extend([(icon_image, title), (color_image, title)])
    return images


def object_search():
    pass

# Gradio Interface with custom CSS for the search box and button
with gr.Blocks(css="""
    .input-button-container {
        display: flex;
        align-items: center;
    }
    .input-button-container input[type="text"] {
        flex-grow: 1;
    }
    .input-button-container button {
        width: 20%; /* Set button width to 1/5th of the input box */
    }
""") as demo:
    
    with gr.Row():
        
        # Left side (3 parts): Text-to-Image search
        with gr.Column(scale=3):
            gr.Markdown("## BMEazy")
            
            # First row: Text Input with Search Button inside
            with gr.Row(elem_classes="input-button-container"):
                text_input = gr.Textbox(show_label=False, placeholder="Enter your query...", scale=5)
                search_button = gr.Button("Search", scale=1)
                next_search_button = gr.Button("Search neighbor", scale=1)
                download_button = gr.Button("Download", scale=1)
            
            # Second row: Top K Results and Model Type underneath
            with gr.Row():
                top_k_input = gr.Number(label="Top K Results", value=300, scale=1)
                model_type_input = gr.Dropdown(choices=["clip", "clipv2_l14", "clipv2_h14"], label="Model Type", value="clipv2_l14", scale=1)
                storage = gr.Textbox(label="cloud or local", text_input="Cloud", scale=1)
                download_name = gr.Textbox(label="CSV name", placeholder="query_v3_*_kis", scale=2)
                folder_name = gr.Textbox(label="Folder name", placeholder="L00_V000....", scale=2)
                image_name = gr.Textbox(label="Image name", placeholder="0000....", scale=2)

            # Third row: Gallery for search results
            image_output = gr.Gallery(label="Search Results", elem_id="image_gallery", columns=5)
            csv_output = gr.File(label="Download CSV")

            # Set search button click action
            search_button.click(fn=text_to_image, 
                                inputs=[text_input, top_k_input, model_type_input, storage], 
                                outputs=image_output)

            # Correctly passing the function without calling it
            next_search_button.click(fn=search_neighbor,
                                     inputs=[folder_name, image_name],
                                     outputs=image_output)

            
            download_button.click(fn=download_csv, 
                                  inputs=[folder_name, image_name, download_name], 
                                  outputs=csv_output)  # Output to a file
            
        # Right side (1 part): Filters (OCR, Tag Detection, etc.)
        with gr.Column(scale=2):
            gr.Markdown("### Search Filters")
            #visual_output = gr.Gallery(
            #     label="Visual Encoding Output",
            #     columns=2,
            #     value=get_initial_visual_encoding(),  # Cung cấp giá trị ban đầu
            # )
            gird = gr.Image(
                label="grid",
                value= VisualEncoder.visualize_grid(grid_vis = (np.ones((250, 250, 3), dtype=np.uint8) * 255))
            )
            


# Launch the app
demo.launch()