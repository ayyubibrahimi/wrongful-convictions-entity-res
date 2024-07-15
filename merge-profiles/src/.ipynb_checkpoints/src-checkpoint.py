import pandas as pd
from collections import defaultdict

def default_profile():
    return {
        'entity_uids': set(),
        'first_name': '',
        'last_name': '',
        'roles': set(),
        'contexts': set()
    }

def process_data(df, threshold=0.5):
    merged_profiles = defaultdict(default_profile)
    entity_to_profile_map = {}

    for _, row in df.iterrows():
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

def main():
    # Read the CSV file
    df = pd.read_csv('../../feature-engineering/data/output/pairwise_comparisons.csv')

    # Process data
    merged_profiles, entity_to_profile_map = process_data(df)

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
