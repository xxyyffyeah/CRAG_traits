"""
Add IMDb IDs to recommendation results using reverse lookup (Solution 1)
Fast and simple, no need to re-run evaluation
"""
import os
import re
import json
import pickle
import pandas as pd


def extract_movie_name(text):
    """Extract and normalize movie name (standalone version)"""
    text = text.split('/')[-1]
    text = text.replace('_', ' ').replace('-', ' ').replace('>', ' ')
    # Remove parentheses
    pattern = r"\([^()]*\)"
    text = re.sub(pattern, "", text)
    # Remove extra spaces
    pattern = r"\s+"
    text = re.sub(pattern, " ", text).strip()
    return text

# Configuration
data_root = "data/reddit"
tier = 30
K_list = list(range(5, 40, 5))

# Result files to process
result_files = [
    "results/trait_evaluation/test_with_traits_final.pkl",
    # "results/trait_evaluation/test_with_traits_retrieval.pkl",
    # "results/trait_evaluation/test_with_traits_rerank.pkl",
]


def load_name2id_mapping():
    """Load name -> IMDb ID mapping"""
    print("Loading IMDb ID mappings...")

    # Load entity2id.json
    old_meta_json = os.path.join(f'{data_root}/entity2id.json')
    old_name2id = json.load(open(old_meta_json))
    old_id2name = {v: extract_movie_name(k) for k, v in old_name2id.items()}

    # Load IMDb database
    database_root = "data/imdb_data"
    name_id_table = pd.read_csv(os.path.join(database_root, "imdb_titlenames_new.csv"))
    importance_table = pd.read_csv(os.path.join(database_root, "imdb_title_importance.csv"))
    importance_table = importance_table[importance_table["importance_tier"] < tier]
    name_id_table = pd.merge(name_id_table, importance_table, on='imdb_id', how='inner')
    name_id_table = name_id_table.sort_values(by='importance_rank')
    unique_titles = name_id_table.drop_duplicates(subset='title_name', keep='first')
    name2id = unique_titles.set_index('title_name')['imdb_id'].to_dict()
    id2name = {v: extract_movie_name(k) for k, v in name2id.items()}
    id2name.update(old_id2name)

    # Create reverse mapping (name -> id)
    name2id_reverse = {name: imdb_id for imdb_id, name in id2name.items()}

    print(f"  Loaded {len(name2id_reverse)} movie name to IMDb ID mappings")
    return name2id_reverse


def add_imdb_ids_for_titles(titles, name2id):
    """
    Convert list of movie titles to list of IMDb IDs

    Args:
        titles: List of movie titles
        name2id: Name to ID mapping dict

    Returns:
        imdb_ids: List of IMDb IDs (None for not found)
        not_found: List of titles not found
    """
    if not titles:
        return [], []

    imdb_ids = []
    not_found = []

    for title in titles:
        imdb_id = name2id.get(title)
        if imdb_id:
            imdb_ids.append(imdb_id)
        else:
            # Try normalized title
            normalized = extract_movie_name(title)
            imdb_id = name2id.get(normalized)
            if imdb_id:
                imdb_ids.append(imdb_id)
            else:
                imdb_ids.append(None)
                not_found.append(title)

    return imdb_ids, not_found


def process_file(file_path, name2id):
    """Process a single result file"""
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return

    print(f"\nProcessing: {file_path}")

    # Load data
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"  Loaded {len(data)} samples")

    # Statistics
    total_movies = 0
    found_ids = 0
    all_not_found = set()

    # Process each sample
    for item in data:
        # K=0 (zero-shot)
        if 'rec_list_raw_0' in item:
            ids, not_found = add_imdb_ids_for_titles(item['rec_list_raw_0'], name2id)
            item['rec_imdb_ids_raw_0'] = ids
            total_movies += len(ids)
            found_ids += sum(1 for id in ids if id)
            all_not_found.update(not_found)

        if 'rec_after_reflect_0' in item:
            ids, not_found = add_imdb_ids_for_titles(item['rec_after_reflect_0'], name2id)
            item['rec_imdb_ids_after_reflect_0'] = ids
            total_movies += len(ids)
            found_ids += sum(1 for id in ids if id)
            all_not_found.update(not_found)

        # K > 0
        for K in K_list:
            # Raw recommendation list
            if f'rec_list_raw_{K}' in item:
                ids, not_found = add_imdb_ids_for_titles(item[f'rec_list_raw_{K}'], name2id)
                item[f'rec_imdb_ids_raw_{K}'] = ids
                total_movies += len(ids)
                found_ids += sum(1 for id in ids if id)
                all_not_found.update(not_found)

            # After rerank
            if f'rec_after_reflect_{K}' in item:
                ids, not_found = add_imdb_ids_for_titles(item[f'rec_after_reflect_{K}'], name2id)
                item[f'rec_imdb_ids_after_reflect_{K}'] = ids
                total_movies += len(ids)
                found_ids += sum(1 for id in ids if id)
                all_not_found.update(not_found)

            # Retrieved movies
            if f'retrieval_after_reflect_{K}' in item:
                ids, not_found = add_imdb_ids_for_titles(item[f'retrieval_after_reflect_{K}'], name2id)
                item[f'retrieval_imdb_ids_{K}'] = ids
                total_movies += len(ids)
                found_ids += sum(1 for id in ids if id)
                all_not_found.update(not_found)

    # Save updated data
    output_path = file_path.replace('.pkl', '_with_imdb_ids.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    # Print statistics
    coverage = found_ids / total_movies * 100 if total_movies > 0 else 0
    print(f"  Saved to: {output_path}")
    print(f"  Statistics: {found_ids}/{total_movies} ({coverage:.1f}% found IMDb IDs)")

    if all_not_found:
        print(f"  Warning: {len(all_not_found)} movie titles not found in IMDb mapping")
        print(f"  Examples: {list(all_not_found)[:3]}")


def main():
    print("="*80)
    print("ADD IMDb IDs TO RECOMMENDATION RESULTS")
    print("="*80)

    # Load mapping
    name2id = load_name2id_mapping()

    # Process each file
    for file_path in result_files:
        process_file(file_path, name2id)

    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
    print("\nNew files created with '_with_imdb_ids.pkl' suffix")
    print("Each recommendation list now has corresponding IMDb ID list:")
    print("  - rec_imdb_ids_raw_{K}")
    print("  - rec_imdb_ids_after_reflect_{K}")
    print("  - retrieval_imdb_ids_{K}")


if __name__ == '__main__':
    main()
