import pandas as pd
import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizer, BertModel
import nltk
from itertools import combinations
from sentence_transformers import SentenceTransformer
import jellyfish
from sklearn.preprocessing import MinMaxScaler
import torch

def read_csv():
    df = pd.read_csv("../../blocking/data/output/clean-with-blocks.csv")
    df.shape
    return df 

def read_model():
    with open('../../ts-train-model/data/output/trained_lr_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)
    return trained_model

nltk.download('punkt')  

stemmer = PorterStemmer()

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def tokenize_and_stem(context):
    tokens = word_tokenize(context)  
    stemmed_tokens = stem_tokens(tokens) 
    return stemmed_tokens

def calculate_string_similarity(s1, s2):
    return jellyfish.jaro_winkler_similarity(s1, s2)

def calculate_context_similarity(contexts1, contexts2):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    embeddings1 = model.encode(contexts1, convert_to_tensor=True)
    embeddings2 = model.encode(contexts2, convert_to_tensor=True)
    
    embeddings1_np = embeddings1.cpu().numpy()
    embeddings2_np = embeddings2.cpu().numpy()
    
    cosine_similarities = np.diag(cosine_similarity(embeddings1_np, embeddings2_np)).tolist()
    
    return cosine_similarities

def generate_pairwise_comparisons(df, model):
    df = df.fillna("")
    comparison_data = []

    # Flatten the list of all blocking keys to find unique keys across the dataset
    all_keys = set([key for sublist in df['blocking_keys'] for key in sublist])

    for key in all_keys:
        # Find all records that have this blocking key
        filtered_df = df[df['blocking_keys'].apply(lambda x: key in x)]
        unique_entities = filtered_df['person_uid'].unique()

        # Generate all combinations of unique entities within this filtered group
        for entity_1, entity_2 in combinations(unique_entities, 2):
            entity_1_row = filtered_df[filtered_df['person_uid'] == entity_1].iloc[0]
            entity_2_row = filtered_df[filtered_df['person_uid'] == entity_2].iloc[0]

            # Check if the first names and last names have a minimum level of similarity
            first_name_similarity = calculate_string_similarity(entity_1_row['first_name'], entity_2_row['first_name'])
            last_name_similarity = calculate_string_similarity(entity_1_row['last_name'], entity_2_row['last_name'])

            if first_name_similarity < 0.6 or last_name_similarity < 0.6:
                continue

            # Compute similarities and differences between entities
            features = {
                'entity_1_uid': entity_1,
                'entity_1_first_name': entity_1_row['first_name'],
                'entity_1_last_name': entity_1_row['last_name'],
                'entity_1_role': entity_1_row.get('officer_role', ''),
                'entity_1_context': entity_1_row.get('officer_context', ''),
                'entity_2_uid': entity_2,
                'entity_2_first_name': entity_2_row['first_name'],
                'entity_2_last_name': entity_2_row['last_name'],
                'entity_2_role': entity_2_row.get('officer_role', ''),
                'entity_2_context': entity_2_row.get('officer_context', ''),
                'first_name_similarity': first_name_similarity,
                'last_name_similarity': last_name_similarity,
                'role_similarity': calculate_string_similarity(entity_1_row.get('officer_role', ''), entity_2_row.get('officer_role', '')),
                'first_name_length_diff': abs(len(entity_1_row['first_name']) - len(entity_2_row['first_name'])),
                'last_name_length_diff': abs(len(entity_1_row['last_name']) - len(entity_2_row['last_name'])),
                'context_similarity': calculate_context_similarity([entity_1_row.get('officer_context', '')], [entity_2_row.get('officer_context', '')])[0],
            }

            # Apply the trained model to make predictions
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform([list(features.values())[-6:]])
            prediction = model.predict_proba(scaled_features)[0][0]
            features['prediction'] = prediction

            comparison_data.append(features)

    # Convert comparison data to DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


if __name__ == "__main__":
    df = read_csv()

    df = df.drop_duplicates(subset=["first_name","last_name","officer_context"])
    
    # df = df.iloc[:5]
    
    model = read_model()
    
    output = generate_pairwise_comparisons(df, model)
    output.to_csv("../data/output/output.csv", index=False)
    