import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch

#download the clip model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
#download the sbert model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")