import gradio as gr
import numpy as np
import pickle
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

embeddings = embeddings.astype(np.float32)

embedding_size = embeddings.shape[1]

n_clusters = 3
num_results = 5

quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)

index.train(embeddings)

index.add(embeddings)

def search(query):
    query_embedding = model.encode(query)
    query_embedding = query_embedding.astype(np.float32)
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, num_results)
    images = [f"images/{idx}.jpg" for idx in indices[0]]
    return images

with gr.Blocks() as search_block:
    query = gr.Textbox(lines=1, label="search query")
    outputs = gr.Gallery(preview=True)
    submit = gr.Button(value="search")
    submit.click(search, inputs=query, outputs=outputs)

search_block.launch(share=True)