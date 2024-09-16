import os
import glob
import faiss
import numpy as np
from tqdm import tqdm



feature_shape = 768
features_dir = '/home/dattruong/dat/AI/Competition/Text-to-Image-Retrieval/data/CLIPv2_L14'
index = faiss.IndexFlatL2(feature_shape)

for data_part in tqdm(sorted(os.listdir(features_dir))):
    for feature_path in tqdm(sorted(glob.glob(os.path.join(features_dir, data_part) +'/*.npy'))):
        feats = np.load(feature_path)
        for feat in feats:
            feat = feat.astype(np.float32).reshape(1,-1)
            index.add(feat)


faiss.write_index(index, f"faiss_clipv2_l14.bin")
