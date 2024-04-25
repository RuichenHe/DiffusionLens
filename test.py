from utils import find_json, delete_collection,set_text_index, random_pick
from PIL import Image
import requests
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch

from tqdm import tqdm
import plotly.express as px
# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)

def sbert_embedding(model, sentence):
  embeddings = model.encode(sentence, convert_to_tensor=True)
  return embeddings

#download the clip model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
#download the sbert model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
# Print the selected documents
print("Model init finish...")
clip_text_features_list = []
clip_image_features_list = []
sbert_features_list = []
prompt_list = []
image_url_list = []
category_list = []
for doc in tqdm(random_pick("test")):
    print(doc)



sample = find_json("cat")
i = 0
for doc in tqdm(sample):
    i = i + 1
    if i > 50:
        break
    try:
        prompt = doc["p"]
        sbert_feature = sbert_embedding(sbert_model, prompt)
        sbert_feature = sbert_feature / sbert_feature.norm(p=2, dim=-1, keepdim=True)

        clip_text_inputs = clip_tokenizer([prompt], padding=True, return_tensors="pt")
        clip_text_features = clip_model.get_text_features(**clip_text_inputs)
        #print(doc)
        # Get image and get image features
        image_url = doc["imageURL"]
        image_url_list.append(image_url)
        image = Image.open(requests.get(image_url, stream=True).raw)
        # image.show()
        clip_inputs = clip_processor(images=image, return_tensors="pt")

        clip_image_features = clip_model.get_image_features(**clip_inputs)

        clip_text_features = clip_text_features / clip_text_features.norm(p=2, dim=-1, keepdim=True)
        clip_image_features = clip_image_features / clip_image_features.norm(p=2, dim=-1, keepdim=True)
    except Exception as e:
        print(e)
        continue
    category_list.append(0)
    clip_text_features_list.append(clip_text_features.cpu().detach().numpy())
    clip_image_features_list.append(clip_image_features.cpu().detach().numpy())
    sbert_features_list.append(sbert_feature.cpu().detach().numpy())
    prompt_list.append(prompt)

sample = find_json("dog")
i = 0
for doc in tqdm(sample):
    i = i + 1
    if i > 50:
        break
    try:
        prompt = doc["p"]

        sbert_feature = sbert_embedding(sbert_model, prompt)
        sbert_feature = sbert_feature / sbert_feature.norm(p=2, dim=-1, keepdim=True)

        clip_text_inputs = clip_tokenizer([prompt], padding=True, return_tensors="pt")
        clip_text_features = clip_model.get_text_features(**clip_text_inputs)
        #print(doc)
        # Get image and get image features
        image_url = doc["imageURL"]
        image_url_list.append(image_url)
        image = Image.open(requests.get(image_url, stream=True).raw)
        # image.show()
        clip_inputs = clip_processor(images=image, return_tensors="pt")

        clip_image_features = clip_model.get_image_features(**clip_inputs)

        clip_text_features = clip_text_features / clip_text_features.norm(p=2, dim=-1, keepdim=True)
        clip_image_features = clip_image_features / clip_image_features.norm(p=2, dim=-1, keepdim=True)
    except Exception as e:
        print(e)
        continue
    category_list.append(1)
    clip_text_features_list.append(clip_text_features.cpu().detach().numpy())
    clip_image_features_list.append(clip_image_features.cpu().detach().numpy())
    sbert_features_list.append(sbert_feature.cpu().detach().numpy())
    prompt_list.append(prompt)

# for i in range(len(prompt_list)):
#     print("\n")
#     print(prompt_list[i])
#     print(image_url_list[i])
#     print(torch.nn.functional.cosine_similarity(clip_text_features_list[i], clip_image_features_list[i]))
#     print("\n")

#X_pca = pca.fit_transform(np.vstack(clip_image_features_list))
X_tsne = tsne.fit_transform(np.vstack(clip_image_features_list))
print("tsne.kl_divergence_: ", tsne.kl_divergence_)
fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=np.vstack(category_list).flatten())
fig.update_layout(
    title="tSNE visualization",
    xaxis_title="First Principal Component",
    yaxis_title="Second Principal Component",
)
fig.show()