# Content_Based_Image_Search
This project implements a content-based image search system using DL models for generating rich textual descriptions of images and embedding them for similarity-based search. The system processes images from multiple datasets, generates descriptions, encodes them into embeddings, and allows users to perform searches based on textual queries. The system is deployed using a web interface built with Gradio.

## descriptions.py image description generation
Purpose: Generates detailed textual descriptions of images.
Model: Uses the InstructBlipForConditionalGeneration model from Salesforce instructblip-vicuna-7b to create descriptive captions for each image
Datasets: The script loads various image datasets including Fashionpedia, NFL Object Detection, Plane Detection, Snacks, Mini Pets, and Pokemon Classification.
Prompts: Two different prompts are used to extract detailed descriptions, which are concatenated for each image.
Output: Saves images and their respective descriptions to a CSV file (description.csv), where each row contains an image index and its corresponding description.

## embedding.py Text Embedding generation
Purpose: Converts the image descriptions into vector embeddings.
Model: Uses the SentenceTransformer model (all-mpnet-base-v2) to encode the descriptions into dense embeddings.
Input: Reads descriptions from the description.csv file.
Output: Saves the generated embeddings into a pickle file (embedding.pkl) for later retrieval.

## app.py Web Based Search interface
Purpose: Provides a user interface to search for images based on textual queries.
Model: Uses the SentenceTransformer model to encode the user’s search query into a vector.
Search Index: The script leverages Facebook AI Similarity Search to create a vector-based search index.
Clustering: Images are indexed into 3 clusters, and the system retrieves the top 5 closest matches.
Gradio Interface:
A Gradio-based web app allows users to input search queries.
Based on the query, the system retrieves and displays the most similar images using the precomputed embeddings.
Output: Displays the top 5 closest matching images from the dataset in response to the user's query.
