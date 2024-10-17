import os
# import glob
# import torch
# import numpy as np
# from typing import List
# import torch.nn.functional as F
import faiss
# from transformers import AutoTokenizer, AutoModel
from utils.semantic_extract import semantic_extract

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

class tag_retrieval(semantic_extract):
    def __init__(
            self,
            model = 'sentence-transformers/stsb-xlm-r-multilingual',
            context_path = os.path.join(PROJECT_ROOT, "data/tag/tag_corpus.txt"),
            context_vector_path = os.path.join(PROJECT_ROOT, "data/bin/tag_bin/tag_embedding.bin"),
            input_datatype='txt',
            output_datatype = 'bin',
    ):
        if not os.path.exists(os.path.join(PROJECT_ROOT, 'data/bin')):
            os.mkdir(os.path.join(PROJECT_ROOT, 'data/bin'))
        
        if not os.path.exists(os.path.join(PROJECT_ROOT, "data/bin/tag_bin")):
            os.mkdir(os.path.join(PROJECT_ROOT, "data/bin/tag_bin"))

        super().__init__(
            model,
            context_path,
            context_vector_path,
            input_datatype,
            output_datatype,
        )
        self.index = faiss.read_index(context_vector_path)

    def __call__(
            self,
            query:str,
            k:int=3,
    ):
        query_embed = self.get_embedding([query]).to('cpu').numpy()
        _, index = self.index.search(query_embed, k)
        result = [self.raw_data[idx] for idx in index[0]]
        return result

if __name__ == '__main__':
    obj = tag_retrieval()
    print(obj("một người đàn ông đang đi bộ trên cầu", 3))
    pass