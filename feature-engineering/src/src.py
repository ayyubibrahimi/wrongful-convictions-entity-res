import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import multiprocessing as mp
from functools import partial
import pickle
import torch
from itertools import combinations

nltk.download('punkt', quiet=True)
stemmer = PorterStemmer()

def read_csv(file_path):
    df = pd.read_csv(file_path)
    print(f"DataFrame shape: {df.shape}")
    return df

def read_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def tokenize_and_stem(context):
    tokens = word_tokenize(context)
    return stem_tokens(tokens)

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

def process_block(block_data, model):
    comparison_data = []
    
    unique_entities = block_data['person_uid'].unique()
    for entity_1, entity_2 in combinations(unique_entities, 2):
        entity_1_row = block_data[block_data['person_uid'] == entity_1].iloc[0]
        entity_2_row = block_data[block_data['person_uid'] == entity_2].iloc[0]
        
        first_name_similarity = calculate_string_similarity(entity_1_row['first_name'], entity_2_row['first_name'])
        last_name_similarity = calculate_string_similarity(entity_1_row['last_name'], entity_2_row['last_name'])
        
        if first_name_similarity < 0.6 or last_name_similarity < 0.6:
            continue
        
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
    
    return pd.DataFrame(comparison_data)

def generate_pairwise_comparisons(df, model):
    df = df.fillna("")
    
    # Convert string representation of list to actual list
    df['blocking_keys'] = df['blocking_keys'].apply(eval)
    
    # Flatten the list of all blocking keys to find unique keys across the dataset
    all_keys = set([key for sublist in df['blocking_keys'] for key in sublist])
    
    # Process blocks in parallel
    pool = mp.Pool(processes=mp.cpu_count())
    process_block_partial = partial(process_block, model=model)
    results = pool.map(process_block_partial, [df[df['blocking_keys'].apply(lambda x: key in x)] for key in all_keys])
    pool.close()
    pool.join()
    
    # Combine results
    comparison_df = pd.concat(results, ignore_index=True)
    return comparison_df

def main():
    df = read_csv("../../blocking/data/output/clean-with-blocks.csv")
    df = df.drop_duplicates()
    df = df.iloc[:5]
    model = read_model('../../ts-train-model/data/output/trained_lr_model.pkl')
    
    comparison_df = generate_pairwise_comparisons(df, model)
    
    print("Comparison DataFrame shape:", comparison_df.shape)
    comparison_df.to_csv("../data/output/pairwise_comparisons.csv", index=False)

if __name__ == "__main__":
    main()