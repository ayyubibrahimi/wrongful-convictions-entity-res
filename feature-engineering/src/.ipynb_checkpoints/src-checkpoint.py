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
import multiprocessing as mp

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
    # Ensure the inputs are strings
    s1 = str(s1)
    s2 = str(s2)
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


def compare_entities(entity_1_row, entity_2_row, model):
    def safe_str(value):
        return str(value) if pd.notna(value) else ''

    # Ensure the names are strings and handle NaN values
    entity_1_first_name = safe_str(entity_1_row['first_name'])
    entity_2_first_name = safe_str(entity_2_row['first_name'])
    entity_1_last_name = safe_str(entity_1_row['last_name'])
    entity_2_last_name = safe_str(entity_2_row['last_name'])
    
    first_name_similarity = calculate_string_similarity(entity_1_first_name, entity_2_first_name)
    last_name_similarity = calculate_string_similarity(entity_1_last_name, entity_2_last_name)
    
    if first_name_similarity < 0.6 or last_name_similarity < 0.6:
        return None
    
    features = {
        'entity_1_uid': safe_str(entity_1_row['person_uid']),
        'entity_1_first_name': entity_1_first_name,
        'entity_1_last_name': entity_1_last_name,
        'entity_1_role': safe_str(entity_1_row.get('officer_role', '')),
        'entity_1_context': safe_str(entity_1_row.get('officer_context', '')),
        'entity_1_fn': safe_str(entity_1_row['fn']),
        'entity_1_page_number': safe_str(entity_1_row['page_number']),
        'entity_2_uid': safe_str(entity_2_row['person_uid']),
        'entity_2_first_name': entity_2_first_name,
        'entity_2_last_name': entity_2_last_name,
        'entity_2_role': safe_str(entity_2_row.get('officer_role', '')),
        'entity_2_context': safe_str(entity_2_row.get('officer_context', '')),
        'entity_2_fn': safe_str(entity_2_row['fn']),
        'entity_2_page_number': safe_str(entity_2_row['page_number']),
        'first_name_similarity': first_name_similarity,
        'last_name_similarity': last_name_similarity,
        'role_similarity': calculate_string_similarity(
            safe_str(entity_1_row.get('officer_role', '')),
            safe_str(entity_2_row.get('officer_role', ''))
        ),
        'first_name_length_diff': abs(len(entity_1_first_name) - len(entity_2_first_name)),
        'last_name_length_diff': abs(len(entity_1_last_name) - len(entity_2_last_name)),
        'context_similarity': calculate_context_similarity(
            [safe_str(entity_1_row.get('officer_context', ''))],
            [safe_str(entity_2_row.get('officer_context', ''))]
        )[0],
    }
    
    # Apply the trained model to make predictions
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform([list(features.values())[-6:]])
    prediction = model.predict_proba(scaled_features)[0][0]
    features['prediction'] = prediction
    
    return features


def merge_rows(row1, row2):
    def safe_concat(val1, val2, separator):
        val1 = str(val1) if pd.notna(val1) else ''
        val2 = str(val2) if pd.notna(val2) else ''
        return (val1 + separator + val2).strip(separator)

    merged_row = row1.copy()
    
    # Safely concatenate 'entity_1_role'
    merged_row['entity_1_role'] = safe_concat(row1.get('entity_1_role', ''), 
                                              row2.get('entity_1_role', ''), 
                                              '; ')
    
    # Safely concatenate 'entity_1_context'
    merged_row['entity_1_context'] = safe_concat(row1.get('entity_1_context', ''), 
                                                 row2.get('entity_1_context', ''), 
                                                 ' ')
    
    # Safely concatenate 'entity_1_page_number'
    merged_row['entity_1_page_number'] = safe_concat(row1.get('entity_1_page_number', ''), 
                                                     row2.get('entity_1_page_number', ''), 
                                                     ', ')
    
    return merged_row


def iterative_merge(df, model):
    merged = False
    while not merged:
        merged = True
        to_drop = []
        new_rows = []
        for i, row1 in df.iterrows():
            for j, row2 in df.iterrows():
                if i >= j:
                    continue
                comparison = compare_entities(row1, row2, model)
                if comparison and comparison['prediction'] > 0.5:
                    new_row = merge_rows(row1, row2)
                    new_rows.append(new_row)
                    to_drop.extend([i, j])
                    merged = False
                    break
            if not merged:
                break
        
        # Drop the merged rows
        df = df.drop(to_drop).reset_index(drop=True)
        
        # Add the new merged rows
        if new_rows:
            new_rows_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_rows_df], ignore_index=True)
    
    return df


def process_chunk(chunk, df, model):
    to_drop = []
    new_rows = []
    for i, row1 in chunk.iterrows():
        for j, row2 in df.iterrows():
            if i >= j:
                continue
            comparison = compare_entities(row1, row2, model)
            if comparison and comparison['prediction'] > 0.5:
                new_row = merge_rows(row1, row2)
                new_rows.append(new_row)
                to_drop.extend([i, j])
                break
    return to_drop, new_rows

def parallel_iterative_merge(df, model, num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()

    pool = mp.Pool(processes=num_processes)
    
    # Group by filename
    grouped = df.groupby('fn')
    
    # First, deduplicate within each file
    deduped_groups = []
    for _, group in grouped:
        while True:
            chunk_size = max(1, len(group) // num_processes)
            chunks = [group.iloc[i:i+chunk_size] for i in range(0, len(group), chunk_size)]
            
            process_chunk_partial = partial(process_chunk, df=group, model=model)
            results = pool.map(process_chunk_partial, chunks)
            
            all_to_drop = set()
            all_new_rows = []
            for to_drop, new_rows in results:
                all_to_drop.update(to_drop)
                all_new_rows.extend(new_rows)
            
            if not all_to_drop:
                break
            
            group = group.drop(list(all_to_drop)).reset_index(drop=True)
            
            if all_new_rows:
                new_rows_df = pd.DataFrame(all_new_rows)
                group = pd.concat([group, new_rows_df], ignore_index=True)
        
        deduped_groups.append(group)
    
    # Combine all deduped groups
    df = pd.concat(deduped_groups, ignore_index=True)
    
    # Now deduplicate between files
    while True:
        chunk_size = max(1, len(df) // num_processes)
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        process_chunk_partial = partial(process_chunk, df=df, model=model)
        results = pool.map(process_chunk_partial, chunks)
        
        all_to_drop = set()
        all_new_rows = []
        for to_drop, new_rows in results:
            all_to_drop.update(to_drop)
            all_new_rows.extend(new_rows)
        
        if not all_to_drop:
            break
        
        df = df.drop(list(all_to_drop)).reset_index(drop=True)
        
        if all_new_rows:
            new_rows_df = pd.DataFrame(all_new_rows)
            df = pd.concat([df, new_rows_df], ignore_index=True)
    
    pool.close()
    pool.join()
    
    return df

def main():
    df = read_csv("../../blocking/data/output/clean-with-blocks.csv")
    df = df.drop_duplicates()
    df = df.sample(n=5000, random_state=1) 
    model = read_model('../../ts-train-model/data/output/trained_lr_model.pkl')
    
    merged_df = parallel_iterative_merge(df, model, num_processes=20)
    
    print("Merged DataFrame shape:", merged_df.shape)
    merged_df.to_csv("../data/output/merged_profiles.csv", index=False)

if __name__ == "__main__":
    main()
