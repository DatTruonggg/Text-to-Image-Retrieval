import os
import sys
import json
import copy
import numpy as np

from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

# Thêm thư mục utils vào hệ thống đường dẫn
from utils.parse_frontend import parse_data
from utils.text_search import TextSearch
from utils.context_encoding import VisualEncoding
from utils.semantic_embed.tag_retrieval import tag_retrieval
from utils.combine_utils import merge_searching_results_by_addition
from utils.search_utils import group_result_by_video, search_by_filter


#### Get variable from .env ####
load_dotenv()
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


VisualEncoder = VisualEncoding()
CosineFaiss = TextSearch(json_path, json_path_cloud, clipb16_bin, clipv2_l14_bin, clipv2_h14_bin, audio_json_path, img2audio_json_path)
TagRecommendation = tag_retrieval()
DictImagePath = CosineFaiss.json_path
TotalIndexList = np.array(list(range(len(DictImagePath)))).astype('int64')

with open(scene_path, 'r') as f:
  Sceneid2info = json.load(f)

with open(map_keyframes, 'r') as f:
  KeyframesMapper = json.load(f)

with open(video_division_path, 'r') as f:
  VideoDivision = json.load(f)

with open(video_id2img_id, 'r') as f:
  Videoid2imgid = json.load(f)


def get_search_space(id):
  # id starting from 1 to 4
  search_space = []
  video_space = VideoDivision[f'list_{id}']
  for video_id in video_space:
    search_space.extend(Videoid2imgid[video_id])
  return search_space

SearchSpace = dict()
for i in range(1, 5):
  SearchSpace[i] = np.array(get_search_space(i)).astype('int64')
SearchSpace[0] = TotalIndexList

def get_near_frame(idx):
  image_info = DictImagePath[idx]
  scene_idx = image_info['scene_idx'].split('/')
  near_keyframes_idx = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]][scene_idx[2]][scene_idx[3]]['lst_keyframe_idxs'])
  return near_keyframes_idx

def get_related_ignore(ignore_index):
  total_ignore_index = []
  for idx in ignore_index:
    total_ignore_index.extend(get_near_frame(idx))
  return total_ignore_index

# Run Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)


@app.route('/data')
def index():
    pagefile = []
    for id, value in DictImagePath.items():
        if int(id) > 500:
          break
        pagefile.append({'imgpath': value['image_path'], 'id': id})
    data = {'pagefile': pagefile}
    return jsonify(data)

@app.route('/imgsearch')
def image_search():
    print("image search")
    k = int(request.args.get('k'))
    id_query = int(request.args.get('imgid'))
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.image_search(id_query, k=k)

    data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)

    return jsonify(data)

@app.route('/textsearch', methods=['POST'], strict_slashes=False)
def text_search():
    print("text search")
    data = request.json

    search_space_index = int(data['search_space'])
    k = int(data['k'])
    clip = data['clip']
    clipv2_l14 = data['clipv2_l14']
    clipv2_h14 = data['clipv2_h14']

    text_query = data['textquery']
    range_filter = int(data['range_filter'])
    storage = data["storage"]

    index = None
    if data['filter']:
      index = np.array(data['id']).astype('int64')
      k = min(k, len(index))
      print("using index")

    keep_index = None
    ignore_index = None
    if data['ignore']:
      ignore_index = get_related_ignore(np.array(data['ignore_idxs']).astype('int64'))
      keep_index = np.delete(TotalIndexList, ignore_index)
      print("using ignore")

    if keep_index is not None:
      if index is not None:
        index = np.intersect1d(index, keep_index)
      else:
        index = keep_index

    if index is None:
      index = SearchSpace[search_space_index]
    else:
      index = np.intersect1d(index, SearchSpace[search_space_index])
    k = min(k, len(index))

    if clip and clipv2_l14 and clipv2_h14:
      model_type = 'both'
    elif clip:
       model_type = 'clip'
    elif clipv2_l14:
       model_type = 'clipv2_l14'
    else: 
       model_type = 'clipv2_h14'

    if data['filtervideo'] != 0:
      print('filter video')
      mode = data['filtervideo']
      prev_result = data['videos']
      data = search_by_filter(prev_result, text_query, k, mode, model_type, range_filter, ignore_index, keep_index, Sceneid2info, DictImagePath, CosineFaiss, KeyframesMapper)
    else:
      if model_type == 'both':
        scores_clip, list_clip_ids, _, _ = CosineFaiss.text_search(text = text_query, index=index, top_k=k, model_type='clip', storage = storage)
        scores_clipv2_l14, list_clipv2_l14_ids, _, _ = CosineFaiss.text_search(text = text_query, index=index, top_k=k, model_type='clipv2_l14', storage = storage)
        scores_clipv2_h14, list_clipv2_h14_ids, _, _ = CosineFaiss.text_search(text = text_query, index=index, top_k=k, model_type='clipv2_h14', storage = storage)
        lst_scores, list_ids = merge_searching_results_by_addition([scores_clip, scores_clipv2_l14, scores_clipv2_h14],
                                                                  [list_clip_ids, list_clipv2_l14_ids, list_clipv2_h14_ids])
        infos_query = list(map(CosineFaiss.id2img_fps.get, list(list_ids)))
        list_image_paths = [info['image_path'] for info in infos_query]
      else:
        lst_scores, list_ids, _, list_image_paths = CosineFaiss.text_search(text = text_query, index=index, top_k=k, model_type=model_type, storage = storage)
      data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)

    return jsonify(data)

@app.route('/panel', methods=['POST'], strict_slashes=False)
def panel():
    print("panel search")
    search_items = request.json
    k = int(search_items['k'])
    search_space_index = int(search_items['search_space'])

    index = None
    if search_items['useid']:
      index = np.array(search_items['id']).astype('int64')
      k = min(k, len(index))

    keep_index = None
    if search_items['ignore']:
      ignore_index = get_related_ignore(np.array(search_items['ignore_idxs']).astype('int64'))
      keep_index = np.delete(TotalIndexList, ignore_index)
      print("using ignore")

    if keep_index is not None:
      if index is not None:
        index = np.intersect1d(index, keep_index)
      else:
        index = keep_index

    if index is None:
      index = SearchSpace[search_space_index]
    else:
      index = np.intersect1d(index, SearchSpace[search_space_index])
    k = min(k, len(index))

    # Parse json input
    object_input = parse_data(search_items, VisualEncoder)
    if search_items['ocr'] == "":
      ocr_input = None
    else:
      ocr_input = search_items['ocr']

    if search_items['asr'] == "":
      asr_input = None
    else:
      asr_input = search_items['asr']

    semantic = False
    keyword = True
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.context_search(object_input=object_input, ocr_input=ocr_input, asr_input=asr_input,
                                                                           k=k, semantic=semantic, keyword=keyword, index=index, useid=search_items['useid'])

    data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)
    return jsonify(data)

@app.route('/getrec', methods=['POST'], strict_slashes=False)
def getrec():
    print("get tag recommendation")
    k = 50
    text_query = request.json
    tag_outputs = TagRecommendation(text_query, k)
    return jsonify(tag_outputs)

@app.route('/relatedimg')
def related_img():
    print("related image")
    id_query = int(request.args.get('imgid'))
    image_info = DictImagePath[id_query]
    image_path = image_info['image_path']
    scene_idx = image_info['scene_idx'].split('/')

    video_info = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]])
    video_url = video_info['video_metadata']['watch_url']
    video_range = video_info[scene_idx[2]][scene_idx[3]]['shot_time']

    near_keyframes = video_info[scene_idx[2]][scene_idx[3]]['lst_keyframe_paths']
    near_keyframes.remove(image_path)

    data = {'video_url': video_url, 'video_range': video_range, 'near_keyframes': near_keyframes}
    return jsonify(data)

@app.route('/getvideoshot')
def get_video_shot():
    print("get video shot")

    if request.args.get('imgid') == 'undefined':
      return jsonify(dict())

    id_query = int(request.args.get('imgid'))
    image_info = DictImagePath[id_query]
    scene_idx = image_info['scene_idx'].split('/')
    shots = copy.deepcopy(Sceneid2info[scene_idx[0]][scene_idx[1]][scene_idx[2]])

    selected_shot = int(scene_idx[3])
    total_n_shots = len(shots)
    new_shots = dict()
    for select_id in range(max(0, selected_shot-5), min(selected_shot+6, total_n_shots)):
      new_shots[str(select_id)] = shots[str(select_id)]
    shots = new_shots

    for shot_key in shots.keys():
      lst_keyframe_idxs = []
      for img_path in shots[shot_key]['lst_keyframe_paths']:
        data_part, video_id, frame_id = img_path.replace('/data/KeyFrames/', '').replace('.webp', '').split('/')
        key = f'{data_part}_{video_id}'.replace('_extra', '')
        if 'extra' not in data_part:
          frame_id = KeyframesMapper[key][str(int(frame_id))]
        frame_id = int(frame_id)
        lst_keyframe_idxs.append(frame_id)
      shots[shot_key]['lst_idxs'] = shots[shot_key]['lst_keyframe_idxs']
      shots[shot_key]['lst_keyframe_idxs'] = lst_keyframe_idxs

    data = {
        'collection': scene_idx[0],
        'video_id': scene_idx[1],
        'shots': shots,
        'selected_shot': scene_idx[3]
    }
    return jsonify(data)

@app.route('/feedback', methods=['POST'], strict_slashes=False)
def feed_back():
    data = request.json
    k = int(data['k'])
    prev_result = data['videos']
    lst_pos_vote_idxs = data['lst_pos_idxs']
    lst_neg_vote_idxs = data['lst_neg_idxs']
    lst_scores, list_ids, _, list_image_paths = CosineFaiss.reranking(prev_result, lst_pos_vote_idxs, lst_neg_vote_idxs, k)
    data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)
    return jsonify(data)

@app.route('/translate', methods=['POST'], strict_slashes=False)
def translate():
  data = request.json
  text_query = data['textquery']
  text_query_translated = CosineFaiss.translater(text_query)
  return jsonify(text_query_translated)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8386)