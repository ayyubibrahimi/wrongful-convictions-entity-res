import pandas as pd
import numpy as np
from collections import defaultdict
import multiprocessing as mp
from functools import partial

def default_profile():
    return {
        'entity_uids': set(),
        'first_name': '',
        'last_name': '',
        'roles': set(),
        'contexts': set()
    }

def process_chunk(chunk, threshold=0.5):
    merged_profiles = defaultdict(default_profile)
    entity_to_profile_map = {}

    for _, row in chunk.iterrows():
        entity_1_uid = row['entity_1_uid']
        entity_2_uid = row['entity_2_uid']
        prediction = row['prediction']
        
        if prediction > threshold:
            profile_key_1 = entity_to_profile_map.get(entity_1_uid)
            profile_key_2 = entity_to_profile_map.get(entity_2_uid)
            
            if profile_key_1 is None and profile_key_2 is None:
                new_key = len(merged_profiles)
                profile_key_1 = profile_key_2 = new_key
            elif profile_key_1 is None:
                profile_key_1 = profile_key_2
            elif profile_key_2 is None:
                profile_key_2 = profile_key_1
            
            profile = merged_profiles[profile_key_1]
            other_profile = merged_profiles[profile_key_2]
            
            # Merge profiles
            profile['entity_uids'].update(other_profile['entity_uids'])
            profile['entity_uids'].update([entity_1_uid, entity_2_uid])
            profile['roles'].update(other_profile['roles'])
            profile['roles'].update([row['entity_1_role'], row['entity_2_role']])
            profile['contexts'].update(other_profile['contexts'])
            profile['contexts'].update([row['entity_1_context'], row['entity_2_context']])
            
            # Update name if necessary
            if not profile['first_name']:
                profile['first_name'] = row['entity_1_first_name']
            if not profile['last_name']:
                profile['last_name'] = row['entity_1_last_name']
            
            # Update entity_to_profile_map
            for uid in profile['entity_uids']:
                entity_to_profile_map[uid] = profile_key_1
            
            # Remove other profile if it was different
            if profile_key_2 != profile_key_1:
                del merged_profiles[profile_key_2]

    return dict(merged_profiles), entity_to_profile_map

def merge_chunk_results(results):
    global_merged_profiles = defaultdict(default_profile)
    global_entity_to_profile_map = {}
    
    for local_merged_profiles, local_entity_to_profile_map in results:
        for profile_key, profile in local_merged_profiles.items():
            global_key = len(global_merged_profiles)
            global_profile = global_merged_profiles[global_key]
            global_profile['entity_uids'].update(profile['entity_uids'])
            global_profile['roles'].update(profile['roles'])
            global_profile['contexts'].update(profile['contexts'])
            if not global_profile['first_name']:
                global_profile['first_name'] = profile['first_name']
            if not global_profile['last_name']:
                global_profile['last_name'] = profile['last_name']
            
            for uid in profile['entity_uids']:
                global_entity_to_profile_map[uid] = global_key

    return dict(global_merged_profiles), global_entity_to_profile_map

def merge_profiles_parallel(df, threshold=0.5, chunk_size=100000):
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)

    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    process_chunk_partial = partial(process_chunk, threshold=threshold)
    
    results = pool.map(process_chunk_partial, chunks)
    
    pool.close()
    pool.join()

    return merge_chunk_results(results)

def main():
    # Read the CSV file in chunks
    chunk_size = 100000  # Adjust this based on your available memory
    chunks = pd.read_csv('../../feature-engineering/data/output/pairwise_comparisons.csv', chunksize=chunk_size)

    # Process chunks in parallel
    merged_profiles, entity_to_profile_map = merge_profiles_parallel(pd.concat(chunks))

    print(f"Number of merged profiles: {len(merged_profiles)}")
    print(f"Number of mapped entities: {len(entity_to_profile_map)}")

    # Create a new DataFrame for the merged profiles
    merged_df = pd.DataFrame([
        {
            'profile_id': key,
            'entity_uids': ','.join(map(str, profile['entity_uids'])),
            'first_name': profile['first_name'],
            'last_name': profile['last_name'],
            'roles': ','.join(profile['roles']),
            'contexts': ' '.join(profile['contexts'])
        }
        for key, profile in merged_profiles.items()
    ])

    print("Merged DataFrame shape:", merged_df.shape)
    merged_df.to_csv("../data/output/merged_df.csv", index=False)

if __name__ == "__main__":
    main()