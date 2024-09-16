import open_clip
import torch
import json 
import faiss
import numpy as np 
from nlp_processing import Translation
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.widgets import Slider
import clip
from sentence_transformers import SentenceTransformer, util
import os

class TextSearch:
    def __init__(self,
                 json_path: str,
                 clip_bin: str,
                 clipv2_l14_bin: str,
                 clipv2_h14_bin:str):
        self.json_path = self.load_json_file(json_path)
        self.index_clip = self.load_bin_file(clip_bin)    
        self.index_clipv2 = self.load_bin_file(clipv2_h14_bin)    
        self.index_clipv2_l14 = self.load_bin_file(clipv2_l14_bin)    

        self.__rootdb = "/media/dattruong/568836F88836D669/Users/DAT/Hackathon/HCMAI24/Data/Keyframe"
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

        self.features= np.load("/home/dattruong/dat/AI/Competition/Text-to-Image-Retrieval/data/concate.npy")
        self.clip, _ = clip.load("ViT-B/32", device=self.__device)

        self.clipv2, _, _ = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', device=self.__device, pretrained='dfn5b')
        self.clipv2_tokenizer = open_clip.get_tokenizer('ViT-H-14-quickgelu')
        
        self.clipv2_L14, _, _ = open_clip.create_model_and_transforms('ViT-L-14', device=self.__device, pretrained='datacomp_xl_s13b_b90k')
        self.clipv2_L14_tokenizer = open_clip.get_tokenizer('ViT-L-14')

        self.translater = Translation()

    def load_json_file(self, json_path: str,):
        with open(json_path, 'r') as f: 
            json_file = json.load(f)
        return {int(k):v for k,v in json_file.items()}    
    
    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)

    def text_search(self, text, top_k, model_type):
        text = self.translater(text)

        ###### TEXT FEATURES EXTRACTING ######
        if model_type == 'clip':
            text = clip.tokenize([text]).to(self.__device)  
            text_features = self.clip.encode_text(text)
        elif model_type == 'clipv2_l14':
            text = self.clipv2_L14_tokenizer([text]).to(self.__device)  
            text_features = self.clipv2_L14.encode_text(text)
        else:
            text = self.clipv2_tokenizer([text]).to(self.__device)  
            text_features = self.clipv2.encode_text(text)
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().detach().numpy().astype(np.float32)

        ###### SEARCHING #####
        if model_type == 'clip':
            index_choosed = self.index_clip
        elif model_type == 'clipv2_l14':
            index_choosed = self.index_clipv2_l14
        else:
            index_choosed = self.index_clipv2
        
        
        scores, idx_image = index_choosed.search(text_features, k=top_k)
        idx_image = idx_image.flatten()
        ###### GET INFOS KEYFRAMES_ID ######
        infos_query = list(map(self.json_path.get, list(idx_image)))
        image_paths = [info['image_path'] for info in infos_query]
        return scores.flatten(), idx_image, infos_query, image_paths
    
    def show_images(self, image_paths):
        fig = plt.figure(figsize=(25, 20))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))

        for i in range(1, columns*rows + 1):
            if i - 1 < len(image_paths):
                img = plt.imread(image_paths[i - 1])
                ax = fig.add_subplot(rows, columns, i)
                
                title_parts = image_paths[i - 1].split("/")[-2:]
                title = " - ".join(title_parts).rsplit('.', 1)[0] 
                
                ax.set_title(title)
                plt.imshow(img)
                plt.axis("off")

        plt.show()


    def show_segment(self, id_query_path):
        #id_query_path = id_query_path.replace('/', '\\')
        stt = None  # Khởi tạo stt bằng None trước vòng lặp
        for i, image_path in self.id2img_fps.items():
            if image_path[1] == id_query_path:
                stt = int(i)
                break  # Tìm thấy id_query_path, thoát khỏi vòng lặp

        if stt is not None:
            start = int(stt - 3)
            end = int(stt + 196)
            path_imgs = []
            infos = []
            for i in range(start, end + 1):
                path_imgs.append(self.get_path_frame(i))
                infos.append(self.get_frame_info(i))
            return path_imgs, infos

# if __name__  == "__main__":
#     json_path = "/home/dattruong/dat/AI/Competition/Text-to-Image-Retrieval/data/keyframe_id_continuous.json"
#     clipv2_l14_bin = "/home/dattruong/dat/AI/Competition/Text-to-Image-Retrieval/data/faiss_clipv2_l14.bin"
#     clipv2_h14_bin = "/home/dattruong/dat/AI/Competition/Text-to-Image-Retrieval/data/faiss_clipv2_cosine.bin"
#     clip_bin = "/home/dattruong/dat/AI/Competition/Text-to-Image-Retrieval/data/faiss_clip.bin"

#     text_search = TextSearch(json_path, clip_bin, clipv2_l14_bin, clipv2_h14_bin)

#     while True:
#         query = str(input("Query (type 'Stop' to exit): "))
#         if query.lower() == "stop":  # Kiểm tra nếu người dùng nhập "Stop" (hoặc "stop")
#             print("Stopping the search...")
#             break  # Thoát khỏi vòng lặp để dừng chương trình
        
#         top_k = 100
#         scores, idx_image, infos_query, image_paths = text_search.text_search(query, top_k, model_type="clipv2_l14")
#         text_search.show_images(image_paths)