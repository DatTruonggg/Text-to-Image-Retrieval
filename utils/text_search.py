import torch
import json 
import os
import math

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dotenv import load_dotenv
from PIL import Image
import io

import open_clip
import clip
import faiss

import boto3

from deep_translator import GoogleTranslator

load_dotenv()
# from utils.nlp_processing import Translation
from utils.combine_utils import merge_searching_results_by_addition
from utils.ocr_retrieval_engine.ocr_retrieval import ocr_retrieval
from utils.semantic_embed.speech_retrieval import speech_retrieval
from utils.object_retrieval_engine.object_retrieval import object_retrieval
# from huggingface_hub import login

# login(token="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")

class TextSearch:
    def __init__(self,
                 json_path: str,
                 json_path_cloud: str,
                 clip_bin: str,
                 clipv2_l14_bin: str,
                 clipv2_h14_bin:str,
                 audio_json_path: str, 
                 img2audio_json_path: str):
        self.json_path = self.load_json_file(json_path)
        self.json_path_cloud = self.load_json_file(json_path_cloud)
        self.audio_id2img_id = self.load_json_file(audio_json_path)
        self.img_id2audio_id = self.load_json_file(img2audio_json_path)

        self.object_retrieval = object_retrieval()
        self.ocr_retrieval = ocr_retrieval()
        self.asr_retrieval = speech_retrieval()

        self.index_clip = self.load_bin_file(clip_bin)    
        self.index_clipv2_h14 = self.load_bin_file(clipv2_h14_bin)    
        self.index_clipv2_l14 = self.load_bin_file(clipv2_l14_bin)    

        self.__rootdb = os.getenv("ROOT_DB")
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_B16, _ = clip.load("ViT-B/16", device=self.__device)

        self.clipv2_H14, _ = open_clip.create_model_from_pretrained('hf-hub:UCSC-VLAA/ViT-H-14-CLIPA-datacomp1B', device=self.__device)
        self.clipv2_H14_tokenizer = open_clip.get_tokenizer('hf-hub:UCSC-VLAA/ViT-H-14-CLIPA-datacomp1B')
        
        self.clipv2_L14, _, _ = open_clip.create_model_and_transforms('ViT-L-14', device=self.__device, pretrained='datacomp_xl_s13b_b90k')
        self.clipv2_L14_tokenizer = open_clip.get_tokenizer('ViT-L-14')

        #self.translater = GoogleTranslator()

        self.s3 = boto3.client('s3', region_name='ap-southeast-1')
        self.bucket_name = "bmeazy"

    def load_json_file(self, json_path: str,):
        with open(json_path, 'r') as f: 
            json_file = json.load(f)
        return {int(k):v for k,v in json_file.items()}    
    
    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)

    # def get_s3_image_url(self, image_path: str):
    #     # Tạo URL đầy đủ cho hình ảnh trong S3
    #     return f'https://{self.bucket_name}.s3.ap-southeast-1.amazonaws.com/{image_path}'
    
    def text_search(self, text:str , index,  top_k:int, model_type:str, storage: str):
        #text = self.translater(text)
        text = GoogleTranslator(source='auto', target='en').translate(text)
        ###### TEXT FEATURES EXTRACTING ######
        if model_type == 'clip':
            text = clip.tokenize([text]).to(self.__device)  
            text_features = self.clip_B16.encode_text(text)
        elif model_type == 'clipv2_l14':
            text = self.clipv2_L14_tokenizer([text]).to(self.__device)  
            text_features = self.clipv2_L14.encode_text(text)
        else:
            text = self.clipv2_H14_tokenizer([text]).to(self.__device)  
            text_features = self.clipv2_H14.encode_text(text)
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().detach().numpy().astype(np.float32)

        ###### SEARCHING #####
        if model_type == 'clip':
            index_choosed = self.index_clip
        elif model_type == 'clipv2_l14':
            index_choosed = self.index_clipv2_l14
        else:
            index_choosed = self.index_clipv2_h14
        
        if index is None:
          scores, idx_image = index_choosed.search(text_features, k=top_k)
        else:
          id_selector = faiss.IDSelectorArray(index)
          scores, idx_image = index_choosed.search(text_features, k=top_k, 
                                                   params=faiss.SearchParametersIVF(sel=id_selector))
          
        # scores, idx_image = index_choosed.search(text_features, k=top_k)
        idx_image = idx_image.flatten()

        ##### CHECK IDX ##### 
        ###### GET INFOS KEYFRAMES_ID ######
        if storage == "cloud": 
            infos_query = list(map(self.json_path_cloud.get, list(idx_image)))
        else: 
            infos_query = list(map(self.json_path.get, list(idx_image)))

        ##image_paths = [info['image_path'] for info in infos_query] ##TODO: Bug here 
        image_paths = [(info['image_path']) for info in infos_query]

        return scores.flatten(), idx_image, infos_query, image_paths
     
    def image_search(self, id_query, k, storage):
        query_feats = self.index_clip.reconstruct(id_query).reshape(1,-1)

        scores, idx_image = self.index_clip.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        if storage == "cloud": 
            infos_query = list(map(self.json_path_cloud.get, list(idx_image)))
        else: 
            infos_query = list(map(self.json_path.get, list(idx_image)))        

        image_paths = [info['image_path'] for info in infos_query]
        return scores.flatten(), idx_image, infos_query, image_paths
    
    def show_images(self, image_paths):  # Hiển thị demo trong localhost
        fig = plt.figure(figsize=(25, 20))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths) / columns))

        for i in range(1, columns * rows + 1):
            if i - 1 < len(image_paths):
                # Xây dựng đường dẫn hoàn chỉnh cho hình ảnh
                full_image_path = os.path.join(self.__rootdb, image_paths[i - 1])
                img = plt.imread(full_image_path)  # Đọc hình ảnh từ đường dẫn hoàn chỉnh
                
                ax = fig.add_subplot(rows, columns, i)
                
                title_parts = image_paths[i - 1].split("/")[-2:]  # Lấy phần cuối của đường dẫn
                title = " - ".join(title_parts).rsplit('.', 1)[0]  # Tạo tiêu đề từ đường dẫn
                
                ax.set_title(title)
                plt.imshow(img)
                plt.axis("off")

        plt.show()

    def show_images_cloud(self, image_paths): 
        fig = plt.figure(figsize=(25, 20))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths) / columns))

        for i in range(1, columns * rows + 1):
            if i - 1 < len(image_paths):
                image_key = image_paths[i - 1].lstrip('/')  # Remove leading slash if needed
                try:
                    # Retrieve image from S3
                    response = self.s3.get_object(Bucket=self.bucket_name, Key=image_key)
                    img_data = response['Body'].read()  # Get the image data
                    img = Image.open(io.BytesIO(img_data))  # Open the image with Pillow

                    ax = fig.add_subplot(rows, columns, i)
                    title_parts = image_key.split("/")[-2:]  # Extract parts for the title
                    title = " - ".join(title_parts).rsplit('.', 1)[0]  # Title without file extension
                    
                    ax.set_title(title)
                    ax.imshow(img)  # Show the image
                    plt.axis("off")  # Turn off axis
                except Exception as e:
                    print(f"Error accessing image: {str(e)}")  # Log any errors

        plt.show() 
    # def show_segment(self, id_query_path):
    #     stt = next((i for i, info in self.json_path.items() if info['image_path'] == id_query_path), None)
    #     if stt is None:
    #         return None, None  # Hoặc xử lý lỗi theo cách khác

    #     start = max(0, stt)
    #     end = min(stt + 99, len(self.json_path) - 1)

    #     return [(self.get_path_frame(i), f"{self.get_path_frame(i).split('/')[-2]}, {int(self.get_frame_info(i)['list_shot_id'][0])}") 
    #             for i in range(start, end + 1)]

    # def get_path_frame(self, index):
    #     # Trả về đường dẫn hình ảnh dựa vào index
    #     return self.json_path[index]['image_path']

    # def get_frame_info(self, index):
    #     # Trả về thông tin hình ảnh dựa vào index
    #     return self.json_path[index]

    def write_csv(self, infos_query, des_path):
        check_files = []
    
        ### GET INFOS SUBMIT ###
        for info in infos_query:
            video_name = info['image_path'].split('/')[-2]
            lst_frames = info['list_shot_id']

            for id_frame in lst_frames:
                check_files.append(os.path.join(video_name, id_frame))
            
        check_files = set(check_files)

        if os.path.exists(des_path):
            df_exist = pd.read_csv(des_path, header=None)
            lst_check_exist = df_exist.values.tolist()      
            check_exist = [info[0].replace('.mp4','/') + f"{info[1]:0>6d}" for info in lst_check_exist]

            ##### FILTER EXIST LINES FROM SUBMIT.CSV FILE #####
            check_files = [info for info in check_files if info not in check_exist]
        else:
            check_exist = []

        video_names = [i.split('/')[0] + '.mp4' for i in check_files]
        frame_ids = [i.split('/')[-1] for i in check_files]

        dct = {'video_names': video_names, 'frame_ids': frame_ids}
        df = pd.DataFrame(dct)

        if len(check_files) + len(check_exist) < 99:
            df.to_csv(des_path, mode='a', header=False, index=False)
            print(f"Save submit file to {des_path}")
        else:
            print('Exceed the allowed number of lines')




    def asr_post_processing(self, tmp_asr_scores, tmp_asr_idx_image, k):
        result = dict()
        for asr_idx, asr_score in zip(tmp_asr_idx_image, tmp_asr_scores):
            lst_ids = self.audio_id2img_id[asr_idx]
            for idx in lst_ids: 
                if result.get(idx, None) is None:
                    result[idx] = asr_score
                else:
                    result[idx] += asr_score

        result = sorted(result.items(), key=lambda x:x[1], reverse=True)
        asr_idx_image = [item[0] for item in result]
        asr_scores = [item[1] for item in result]
        return np.array(asr_scores)[:k], np.array(asr_idx_image)[:k]
    
    def asr_retrieval_helper(self, asr_input, k, index, semantic, keyword):
        # Map img_id to audio_id
        if index is not None:
            audio_temp = dict()
            for idx in index:
                audio_idxes = self.img_id2audio_id[idx]
                for audio_idx in audio_idxes:
                    if audio_temp.get(audio_idx, None) is None:
                        audio_temp[audio_idx] = [idx]
                    else:
                        audio_temp[audio_idx].append(idx)

            audio_index = np.array(list(audio_temp.keys())).astype('int64')
            tmp_asr_scores, tmp_asr_idx_image = self.asr_retrieval(asr_input, k=len(audio_index), index=audio_index, semantic=semantic, keyword=keyword)
            
            result = dict()
            for asr_idx, asr_score in zip(tmp_asr_idx_image, tmp_asr_scores):
                for idx in audio_temp[asr_idx]:
                    if result.get(idx, None) is None:
                        result[idx] = asr_score
                    else:
                        result[idx] += asr_score
            result = sorted(result.items(), key=lambda x:x[1], reverse=True)
            asr_idx_image = np.array([item[0] for item in result])[:k]
            asr_scores = np.array([item[1] for item in result])[:k]
        else:
            tmp_asr_scores, tmp_asr_idx_image = self.asr_retrieval(asr_input, k=k, index=None, semantic=semantic, keyword=keyword)
            asr_scores, asr_idx_image = self.asr_post_processing(tmp_asr_scores, tmp_asr_idx_image, k)
        return asr_scores, asr_idx_image

    def context_search(self, object_input, ocr_input, asr_input, k, semantic=False, keyword=True, index=None, useid=False):
        '''
        Example:
        inputs = {
            'bbox': "a0person",
            'class': "person0, person1",
            'color':None,
            'tag':None
        }
        '''
        scores, idx_image = [], []
        ###### SEARCHING BY OBJECT #####
        if object_input is not None:
            object_scores, object_idx_image = self.object_retrieval(object_input, k=k, index=index)
            scores.append(object_scores)
            idx_image.append(object_idx_image)

        ###### SEARCHING BY OCR #####
        if ocr_input is not None:
            ocr_scores, ocr_idx_image = self.ocr_retrieval(ocr_input, k=k, index=index)
            scores.append(ocr_scores)
            idx_image.append(ocr_idx_image)

        ###### SEARCHING BY ASR #####
        if asr_input is not None:
            if not useid:
                asr_scores, asr_idx_image = self.asr_retrieval_helper(asr_input, k, None, semantic, keyword)
            else:
                asr_scores, asr_idx_image = self.asr_retrieval_helper(asr_input, k, index, semantic, keyword)
            scores.append(asr_scores)
            idx_image.append(asr_idx_image)
        
        scores, idx_image = merge_searching_results_by_addition(scores, idx_image)

        ###### GET INFOS KEYFRAMES_ID ######
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info['image_path'] for info in infos_query]
        return scores, idx_image, infos_query, image_paths
    
    def image_search(self, id_query, k):
        query_feats = self.index_clip.reconstruct(id_query).reshape(1,-1)

        scores, idx_image = self.index_clip.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        
        image_paths = [info['image_path'] for info in infos_query]
        return scores.flatten(), idx_image, infos_query, image_paths
    
    def reranking(self, prev_result, lst_pos_vote_idxs, lst_neg_vote_idxs, k):
        '''
        Perform reranking using user feedback
        '''
        lst_vote_idxs = []
        lst_vote_idxs.extend(lst_pos_vote_idxs)
        lst_vote_idxs.extend(lst_neg_vote_idxs)
        lst_vote_idxs = np.array(lst_vote_idxs).astype('int64')        
        len_pos = len(lst_pos_vote_idxs)

        result = dict()
        for item in prev_result:
            for id, score in zip(item['video_info']['lst_idxs'], item['video_info']['lst_scores']):
                result[id] = score

        for key in lst_neg_vote_idxs:
            result.pop(key)

        id_selector = faiss.IDSelectorArray(np.array(list(result.keys())).astype('int64'))
        query_feats = self.index_clip.reconstruct_batch(lst_vote_idxs)
        lst_scores, lst_idx_images = self.index_clip.search(query_feats, k=min(k, len(result)),
                                                            params=faiss.SearchParametersIVF(sel=id_selector))

        for i, (scores, idx_images) in enumerate(zip(lst_scores, lst_idx_images)):
            for score, idx_image in zip(scores, idx_images):
                if 0 <= i < len_pos:
                    result[idx_image] += score
                else:
                    result[idx_image] -= score

        result = sorted(result.items(), key=lambda x:x[1], reverse=True)
        list_ids = [item[0] for item in result]
        lst_scores = [item[1] for item in result]
        infos_query = list(map(self.id2img_fps.get, list(list_ids)))
        list_image_paths = [info['image_path'] for info in infos_query]

        return lst_scores, list_ids, infos_query, list_image_paths


if __name__ == "__main__":
    query = "Một người mặc áo khoác có mũ màu trắng hai lần ném vật gì đó lên cao. \
            Tiếp theo là cảnh quay ba lá cờ treo trên một tòa nhà, mỗi lá cờ đều có những ngôi sao màu vàng. \
            Sau đó, một người mặc áo trắng ném một vật gì đó lên trời. Cuối cùng, cảnh quay cho thấy một quả pháo được bắn lên."
    json_path = "data/id2img_fps.json"
    json_path_cloud = "data/id2fps_cloud_v2.json"
    clipb16_bin = "data/faiss_clip_b16_test.bin"
    clipv2_l14_bin = "data/clipv2_l14_cosine.bin"
    clipv2_h14_bin = "data/clipv2_h14_cosine.bin"
    audio_json_path = "data/audio_id2img_id.json"
    scene_path = "data/scene_id2info.json"
    img2audio_json_path = "data/img_id2audio_id.json"
    video_division_path = "data/video_division_tag.json"
    map_keyframes = "data/map_keyframes.json"
    video_id2img_id = "data/video_id2img_id.json"
    search = TextSearch(json_path, json_path_cloud, clipb16_bin, clipv2_l14_bin, clipv2_h14_bin, audio_json_path, img2audio_json_path)
 
    _, _, _, image_paths = search.text_search(text=query, index=None, top_k=100, model_type="clipv2_h14", storage="local")
    search.show_images(image_paths)
