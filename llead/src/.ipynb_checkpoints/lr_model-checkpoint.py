import pandas as pd
from collections import defaultdict
import hashlib
from sklearn.preprocessing import MinMaxScaler
import pickle
import multiprocessing as mp
from functools import partial

def read():
    dfa = pd.read_csv("../../feature-engineering/data/output/merged_profiles.csv")
    dfb = pd.read_csv("../data/input/personnel.csv")
    return dfa, dfb

def read_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_hash_uid(row):
    unique_string = f"{row['first_name1']}|{row['last_name1']}|{row['first_name2']}|{row['last_name2']}|{row['source1']}|{row['source2']}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def calculate_similarity(row1, row2, model, scaler):
    features = [
        abs(len(row1['first_name']) - len(row2['first_name'])),
        abs(len(row1['last_name']) - len(row2['last_name'])),
        int(row1['first_name'][0] == row2['first_name'][0]),
        int(row1['last_name'][0] == row2['last_name'][0]),
        len(set(row1['first_name']) & set(row2['first_name'])) / max(len(row1['first_name']), len(row2['first_name'])),
        len(set(row1['last_name']) & set(row2['last_name'])) / max(len(row1['last_name']), len(row2['last_name']))
    ]
    
    scaled_features = scaler.transform([features])
    similarity_score = model.predict_proba(scaled_features)[0][1]  # Probability of positive class
    
    return similarity_score

def process_wci_row(wci_row, llead_df, model, scaler):
    best_match = None
    best_score = -1
    
    for _, llead_row in llead_df.iterrows():
        if wci_row['fc'] == llead_row['fc'] and wci_row['lc'] == llead_row['lc']:
            sim_score = calculate_similarity(wci_row, llead_row, model, scaler)
            
            if sim_score > best_score:
                best_score = sim_score
                best_match = llead_row
    
    if best_match is not None:
        return {
            'sim_score': best_score,
            'first_name1': wci_row['first_name'],
            'last_name1': wci_row['last_name'],
            'fc1': wci_row['fc'],
            'source1': wci_row['source'],
            'agency1': wci_row['agency'],
            'wcoi_uid1': wci_row['wcoi_uid'],
            'llead_uid1': wci_row['llead_uid'],
            'first_name2': best_match['first_name'],
            'last_name2': best_match['last_name'],
            'fc2': best_match['fc'],
            'source2': best_match['source'],
            'agency2': best_match['agency'],
            'wcoi_uid2': best_match['wcoi_uid'],
            'llead_uid2': best_match['llead_uid']
        }
    return None

def custom_matcher_parallel(df, model, scaler, num_processes=25):
    wci_df = df[df['source'] == 'wci']
    llead_df = df[df['source'] == 'llead']
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(partial(process_wci_row, llead_df=llead_df, model=model, scaler=scaler), [row for _, row in wci_df.iterrows()])
    
    results = [r for r in results if r is not None]
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('sim_score', ascending=False)
    
    return result_df

if __name__ == "__main__":
    dfa, dfb = read()
    model = read_model('../../ts-train-model/data/output/trained_lr_model.pkl')
    scaler = MinMaxScaler()
    
    dfa["source"] = "wci"
    dfa["agency"] = "n/a"
    
    dfb["source"] = "llead"
    dfb["officer_role"] ="n/a"
    dfb["officer_context"] = "n/a"
    dfb = dfb[dfb.agency.str.contains("orleans-pd|orleans-so")]
    
    dfa = dfa.rename(columns={"person_uid":  "wcoi_uid"})
    dfb = dfb.rename(columns={"uid": "llead_uid"})
    
    df = pd.concat([dfa, dfb])
    
    df.loc[:, "first_name"] = df.first_name.str.lower().str.strip()
    df.loc[:, "last_name"] = df.last_name.str.lower().str.strip()
    
    df.loc[:, "fc"] = df.first_name.fillna("").map(lambda x: x[:1])
    df.loc[:, "lc"] = df.last_name.fillna("").map(lambda x: x[:1])
    
    df = df[["first_name", "last_name", "fc", "lc", "source", "wcoi_uid", "llead_uid", "agency"]]
    
    print(f"DF SHAPE BEFORE {df.shape}")
    df = df.drop_duplicates(subset=["wcoi_uid", "llead_uid"])
    print(f"DF SHAPE AFTER {df.shape}")
    
    df = df.reset_index(drop=True)
    df.loc[:, "full_name"] = df.first_name.str.cat(df.last_name, sep=" ")
    df = df[~((df.full_name.fillna("") == ""))]
    
    full_names = df.first_name.str.cat(df.last_name, sep=" ")
    
    df = custom_matcher_parallel(df, model, scaler, num_processes=25)
    df['person_uid'] = df.apply(create_hash_uid, axis=1)
    
    df.to_csv("../data/output/merged_officer_profiles_with_best_matches.csv", index=False)
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Number of unique WCI entities matched: {df['wcoi_uid1'].nunique()}")
    print(f"Number of unique LLEAD entities matched: {df['llead_uid2'].nunique()}")