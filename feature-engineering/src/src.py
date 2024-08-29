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
from itertools import combinations, chain
import networkx as nx

nltk.download('punkt', quiet=True)
stemmer = PorterStemmer()

def read_csv(file_path):
    df = pd.read_csv(file_path)
    print(f"DataFrame shape: {df.shape}")
    return df

def read_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_string_similarity(s1, s2):
    s1, s2 = str(s1), str(s2)
    return jellyfish.jaro_winkler_similarity(s1, s2)

def calculate_context_similarity(contexts1, contexts2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    embeddings1 = model.encode(contexts1, convert_to_tensor=True)
    embeddings2 = model.encode(contexts2, convert_to_tensor=True)
    
    cosine_similarities = np.diag(cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())).tolist()
    
    return cosine_similarities

def safe_str(value):
    return str(value) if pd.notna(value) else ''

def compare_entities(entity_1_row, entity_2_row, model):
    entity_1_first_name = safe_str(entity_1_row['first_name']).lower()
    entity_2_first_name = safe_str(entity_2_row['first_name']).lower()
    entity_1_last_name = safe_str(entity_1_row['last_name']).lower()
    entity_2_last_name = safe_str(entity_2_row['last_name']).lower()
    
    # Check for perfect match first
    if entity_1_first_name == entity_2_first_name and entity_1_last_name == entity_2_last_name:
        return {'prediction': 1.0}
    
    first_name_similarity = calculate_string_similarity(entity_1_first_name, entity_2_first_name)
    last_name_similarity = calculate_string_similarity(entity_1_last_name, entity_2_last_name)
    
    if first_name_similarity < 0.6 or last_name_similarity < 0.6:
        return None
    
    features = {
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
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform([list(features.values())])
    prediction = model.predict_proba(scaled_features)[0][1]
    return {'prediction': prediction}

def process_within_doc_comparisons(df, model, num_processes):
    grouped = df.groupby('fn')
    all_matches = []
    for _, group in grouped:
        comparisons = list(combinations(group.index, 2))
        chunk_size = max(1, len(comparisons) // num_processes)
        chunks = [comparisons[i:i + chunk_size] for i in range(0, len(comparisons), chunk_size)]
        
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(
                partial(process_comparison_chunk, df=df, model=model, within_doc=True),
                chunks
            )
        all_matches.extend(chain.from_iterable(results))
    return all_matches

def process_between_doc_comparisons(df, model, num_processes):
    comparisons = list(combinations(df.index, 2))
    comparisons = [pair for pair in comparisons if df.loc[pair[0], 'fn'] != df.loc[pair[1], 'fn']]
    
    chunk_size = max(1, len(comparisons) // num_processes)
    chunks = [comparisons[i:i + chunk_size] for i in range(0, len(comparisons), chunk_size)]
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(
            partial(process_comparison_chunk, df=df, model=model, within_doc=False),
            chunks
        )
    return list(chain.from_iterable(results))

def batch_parallel_iterative_merge(df, model, batch_size=10, num_processes=20):
    all_documents = df['fn'].unique()
    final_merged_df = pd.DataFrame()

    for i in range(0, len(all_documents), batch_size):
        batch_docs = all_documents[i:i+batch_size]
        batch_df = df[df['fn'].isin(batch_docs)]

        # Within-document deduplication for this batch
        within_doc_matches = process_within_doc_comparisons(batch_df, model, num_processes)
        batch_merged_within = merge_matches(batch_df, within_doc_matches)

        # Between-document comparisons for this batch
        between_doc_matches = process_between_doc_comparisons(batch_merged_within, model, num_processes)
        batch_final_merged = merge_matches(batch_merged_within, between_doc_matches)

        # Append to final results
        final_merged_df = pd.concat([final_merged_df, batch_final_merged])

    # Final between-document comparisons across all batches
    final_between_doc_matches = process_between_doc_comparisons(final_merged_df, model, num_processes)
    final_merged_df = merge_matches(final_merged_df, final_between_doc_matches)

    return final_merged_df

def process_comparison_chunk(chunk, df, model, within_doc):
    matches = []
    for i, j in chunk:
        if within_doc and are_perfect_match(df.loc[i], df.loc[j]):
            matches.append((i, j, {'prediction': 1.0}))
        else:
            comparison = compare_entities(df.loc[i], df.loc[j], model)
            if comparison and comparison['prediction'] > 0.5:
                matches.append((i, j, comparison))
    return matches

def are_perfect_match(row1, row2):
    return (safe_str(row1['first_name']).lower() == safe_str(row2['first_name']).lower() and
            safe_str(row1['last_name']).lower() == safe_str(row2['last_name']).lower())

def merge_matches(df, matches):
    G = nx.Graph()
    G.add_edges_from([(m[0], m[1]) for m in matches])
    
    merged_groups = list(nx.connected_components(G))
    
    merged_rows = []
    for group in merged_groups:
        merged_row = df.loc[list(group)[0]].copy()
        for idx in list(group)[1:]:
            merged_row = merge_rows(merged_row, df.loc[idx])
        merged_rows.append(merged_row)
    
    # Add unmatched rows
    unmatched = set(df.index) - set(chain.from_iterable(merged_groups))
    for idx in unmatched:
        merged_rows.append(df.loc[idx])
    
    return pd.DataFrame(merged_rows)
    
def merge_rows(row1, row2):
    merged_row = row1.copy()
    
    # Ensure merged_data is a dictionary
    if not isinstance(merged_row.get('merged_data'), dict):
        merged_row['merged_data'] = {}
    
    for row in [row1, row2]:
        fn = safe_str(row['fn'])
        if fn:
            if fn not in merged_row['merged_data']:
                merged_row['merged_data'][fn] = []
            merged_row['merged_data'][fn].append({
                'person_uid': safe_str(row['person_uid']),
                'officer_context': safe_str(row.get('officer_context', '')),
                'officer_role': safe_str(row.get('officer_role', ''))
            })
    
    merged_row['officer_context'] = ' '.join(filter(None, [safe_str(row1.get('officer_context', '')), safe_str(row2.get('officer_context', ''))]))
    merged_row['officer_role'] = '; '.join(filter(None, [safe_str(row1.get('officer_role', '')), safe_str(row2.get('officer_role', ''))]))
    merged_row['fn'] = ', '.join(filter(None, [safe_str(row1['fn']), safe_str(row2['fn'])]))
    
    return merged_row

def main():
    df = read_csv("../../blocking/data/output/clean-index-with-blocks.csv")
    print(df.shape)
    df = df.drop_duplicates()
    print(df.shape)
    # df = df.sample(n=500, random_state=1) 
    model = read_model('../../ts-train-model/data/output/trained_lr_model.pkl')
    
    merged_df = batch_parallel_iterative_merge(df, model, batch_size=10, num_processes=20)
    
    print("Merged DataFrame shape:", merged_df.shape)
    merged_df.to_csv("../data/output/merged_officer_profiles.csv", index=False)

if __name__ == "__main__":
    main()