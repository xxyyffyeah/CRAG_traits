"""
Create trait-labeled dataset for Reddit conversations.

This script:
1. Loads Reddit test data and runs original CRAG to get recommendations
2. Maps recommended movie titles to IMDB IDs
3. Queries sensitivity table for warning tags
4. Matches each conversation to the best trait based on warning statistics
5. Saves the dataset with trait labels (pkl and JSON formats)
"""

import os
import time
import json
import pickle
import random
import threading
import multiprocessing

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import openai
import numpy as np
import pandas as pd
from editdistance import eval as distance

from libs.utils import extract_movie_name, process_item_raw
from libs.utils import process_retrieval_reflect_raw
from libs.model import cf_retrieve, get_response

from dotenv import load_dotenv
load_dotenv()

# Configuration
external = True

if external:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if os.environ.get('OPENAI_ORG') is not None:
        openai.organization = os.environ.get('OPENAI_ORG')
    if openai.api_key is None:
        raise Exception('OPENAI_API_KEY is not set')

# Path configuration
model = "gpt-4o"
dataset = "reddit"
version = "_with_titles"
datafile = f"test_clean{version}"

data_root = f"data/{dataset}"
from_pkl = f"{data_root}/{datafile}.pkl"

cf_model = "large_pop_adj_07sym"
cf_root = f"{cf_model}"

# Hyperparameters
temperature = 0.0
max_tokens = 512
n_threads = 50  # Reduced from 500 to avoid rate limiting
tier = 30
K = 20  # Number of retrieved items for CF
n_samples = -1  # Number of samples to process (set to -1 for all)

# Prompt for zero-shot recommendation
prompt_no_title = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system). "
    "Based on the conversation, you need to reply with 20 recommendations without extra sentences. "
    "List the standardized title of each movie in each line.\n"
    "Here is the conversation: {context}\n"
    "System:"
)

# Prompt for recommendation with retrieved titles
prompt_with_retrieved_titles = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system). "
    "Based on the conversation, you need to reply with 20 movie recommendations without extra sentences. "
    "List the standardized title of each movie on a separate line.\n"
    "Here is the conversation: {context}\n"
    "Based on movies mentioned in the conversation, here are some movies that are usually liked by other users: {retrieved_titles}.\n"
    "Use the above information at your discretion (i.e., do not confine your recommendation to the above movies). "
    "System:"
)

# Prompt for retrieval reflection
prompt_reflect_titles = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system), "
    "as well as some movies retrieved from our movie database based on the similarity with the movies mentioned by the user in the context."
    "You need to judge whether each retrieved movie is a good recommendation based on the context.\n"
    "Here is the conversation: {context}\n"
    "Here are retrieved movies: {retrieved_titles}.\n"
    "You need to reply with the judgement of each movie in a line, in the form of movie_name####judgment, "
    "where judgement is a binary number 0, 1. Judgment 0 means the movie is a bad recommendation, whereas judgment 1 means the movie is a good recommendation. "
    "System:"
)


def load_and_process_cf_model():
    """
    Load and process the item-item similarity matrix for collaborative retrieval
    """
    sim_mat_pkl = os.path.join(cf_root, "BBsim.pickle")
    with open(sim_mat_pkl, "rb") as f:
        sim_mat = pickle.load(f)

    row2imdb_id_pkl = os.path.join(cf_root, "imdb_ids.pickle")
    with open(row2imdb_id_pkl, "rb") as f:
        raw_row2imdb_id = pickle.load(f)

    raw_imdb_id2row = {iid:i for i, iid in enumerate(raw_row2imdb_id)}
    raw_row2imdb_id = {i:iid for i, iid in enumerate(raw_imdb_id2row)}

    raw_imdb_id2col = deepcopy(raw_imdb_id2row)
    raw_col2imdb_id = deepcopy(raw_row2imdb_id)

    # Load the reddit test context movie database
    reddit_test_meta_pkl = os.path.join(f'{data_root}/entity2id{version}.pkl')
    with open(reddit_test_meta_pkl, "rb") as f:
        reddit_test_id_name_table = pickle.load(f)

    # Process the row of the sim matrix
    reddit_test_name2id = reddit_test_id_name_table.set_index('title')['imdb_id'].to_dict()
    reddit_test_id2name = {v: extract_movie_name(k) for k, v in reddit_test_name2id.items()}
    relevant_indices = [row for row, imdb_id in raw_row2imdb_id.items() if imdb_id in reddit_test_id2name]
    sim_mat = sim_mat[relevant_indices, :]
    row2imdb_id = {new_row: raw_row2imdb_id[old_row] for new_row, old_row in enumerate(relevant_indices)}
    imdb_id2row = {imdb_id: row for row, imdb_id in row2imdb_id.items()}

    # Process the column of the sim matrix
    reddit_test_resp_meta_pkl = os.path.join(f'{data_root}/entity2id_resp{version}.pkl')
    with open(reddit_test_resp_meta_pkl, "rb") as f:
        reddit_test_resp_id_name_table = pickle.load(f)

    reddit_test_resp_name2id = reddit_test_resp_id_name_table.set_index('title')['imdb_id'].to_dict()
    reddit_test_resp_id2name = {v: extract_movie_name(k) for k, v in reddit_test_resp_name2id.items()}
    relevant_indices = [col for col, imdb_id in raw_col2imdb_id.items() if imdb_id in reddit_test_resp_id2name]
    sim_mat = sim_mat[:, relevant_indices]
    col2imdb_id = {new_col: raw_col2imdb_id[old_col] for new_col, old_col in enumerate(relevant_indices)}
    imdb_id2col = {imdb_id: col for col, imdb_id in col2imdb_id.items()}

    catalog_imdb_ids = set(reddit_test_resp_id_name_table["imdb_id"]).intersection(set(raw_imdb_id2col.keys()))

    # Get the title information
    old_meta_json = os.path.join(f'{data_root}/entity2id.json')
    old_name2id = json.load(open(old_meta_json))
    old_id2name = {v: extract_movie_name(k) for k, v in old_name2id.items()}

    database_root = "data/imdb_data"
    name_id_table = pd.read_csv(os.path.join(database_root, "imdb_titlenames_new.csv"))
    importance_table = pd.read_csv(os.path.join(database_root, "imdb_title_importance.csv"))
    importance_table = importance_table[importance_table["importance_tier"]<tier]
    name_id_table = pd.merge(name_id_table, importance_table, on='imdb_id', how='inner')
    name_id_table = name_id_table.sort_values(by='importance_rank')
    unique_titles = name_id_table.drop_duplicates(subset='title_name', keep='first')
    name2id = unique_titles.set_index('title_name')['imdb_id'].to_dict()
    id2name = {v: extract_movie_name(k) for k, v in name2id.items()}
    id2name.update(old_id2name)

    return sim_mat, catalog_imdb_ids, imdb_id2row, imdb_id2col, col2imdb_id, id2name


def pre_process(test_data, catalog_imdb_ids):
    """
    Pre-process the data
    """
    test_data = [item for item in test_data if item["old"]["is_user"] == 0]
    test_data = [item for item in test_data if all(iid in catalog_imdb_ids for iid in item["clean_resp_imdb_ids"])]

    # Create a dictionary to keep track of seen turn_id
    seen_turn_ids = {}
    unique_test_data = []

    for item in test_data:
        turn_id = item["turn_id"]
        if turn_id not in seen_turn_ids:
            seen_turn_ids[turn_id] = True
            unique_test_data.append(item)

    test_data_with_rec = unique_test_data
    return test_data_with_rec


def load_title_to_imdb_mapping():
    """
    Load title to IMDB ID mapping from entity2id files
    """
    # Load response entity mapping
    resp_pkl = os.path.join(f'{data_root}/entity2id_resp{version}.pkl')
    with open(resp_pkl, "rb") as f:
        resp_table = pickle.load(f)

    # Create title to imdb_id mapping
    title_to_imdb = resp_table.set_index('title')['imdb_id'].to_dict()

    # Also load from context entity mapping
    context_pkl = os.path.join(f'{data_root}/entity2id{version}.pkl')
    with open(context_pkl, "rb") as f:
        context_table = pickle.load(f)

    context_title_to_imdb = context_table.set_index('title')['imdb_id'].to_dict()
    title_to_imdb.update(context_title_to_imdb)

    # Clean title names for better matching
    clean_title_to_imdb = {}
    for title, imdb_id in title_to_imdb.items():
        clean_title = extract_movie_name(title)
        clean_title_to_imdb[clean_title] = imdb_id
        clean_title_to_imdb[clean_title.lower()] = imdb_id

    title_to_imdb.update(clean_title_to_imdb)

    return title_to_imdb


def load_sensitivity_table():
    """
    Load the sensitivity table with warning tags
    """
    sensitivity_path = "data/movielens/ml-ddd_sensitivity_with_imdb.csv"
    sensitivity_df = pd.read_csv(sensitivity_path)

    # Get all warning tag columns (those starting with "Clear Yes:")
    warning_columns = [col for col in sensitivity_df.columns if col.startswith('Clear Yes:')]

    print(f"Loaded sensitivity table with {len(sensitivity_df)} movies and {len(warning_columns)} warning tags")

    return sensitivity_df, warning_columns


def load_traits_warnings():
    """
    Load traits with warning definitions
    """
    with open('traits_warnings.json', 'r') as f:
        traits_warnings = json.load(f)

    print(f"Loaded {len(traits_warnings)} traits from traits_warnings.json")

    return traits_warnings


def get_movie_warnings(imdb_id, sensitivity_df, warning_columns):
    """
    Query movie warning tags (columns starting with 'Clear Yes:')

    Args:
        imdb_id: IMDB ID of the movie
        sensitivity_df: Sensitivity DataFrame
        warning_columns: List of warning column names

    Returns:
        list: List of warning tags that apply to this movie
    """
    row = sensitivity_df[sensitivity_df['imdb_id'] == imdb_id]

    if len(row) == 0:
        return []

    warnings = []
    for col in warning_columns:
        tag = col.replace('Clear Yes: ', '')
        try:
            if row[col].values[0] == 1:
                warnings.append(tag)
        except:
            continue

    return warnings


def match_trait(rec_imdb_ids, sensitivity_df, warning_columns, traits_warnings, title_to_imdb):
    """
    Match the best trait based on recommended movies' warnings

    Args:
        rec_imdb_ids: List of recommended movie IMDB IDs
        sensitivity_df: Sensitivity DataFrame
        warning_columns: List of warning column names
        traits_warnings: List of trait definitions
        title_to_imdb: Title to IMDB ID mapping

    Returns:
        tuple: (best_trait, score, matched_warnings, all_warnings)
    """
    # Collect all warnings from recommended movies
    all_warnings = set()
    movie_warnings = {}

    for imdb_id in rec_imdb_ids:
        if imdb_id:
            warnings = get_movie_warnings(imdb_id, sensitivity_df, warning_columns)
            movie_warnings[imdb_id] = warnings
            all_warnings.update(warnings)

    # Count how many warnings from each trait's avoid list appear
    trait_scores = {}
    trait_matched_warnings = {}

    for trait_obj in traits_warnings:
        trait_name = trait_obj['trait']
        avoid_list = set(trait_obj['avoid'])
        matched = avoid_list & all_warnings
        score = len(matched)
        trait_scores[trait_name] = score
        trait_matched_warnings[trait_name] = list(matched)

    # Select trait with highest score
    if max(trait_scores.values()) == 0:
        # No warnings found, randomly select a trait
        best_trait = random.choice(traits_warnings)['trait']
        best_score = 0
        best_matched = []
    else:
        best_trait = max(trait_scores, key=trait_scores.get)
        best_score = trait_scores[best_trait]
        best_matched = trait_matched_warnings[best_trait]

    return best_trait, best_score, best_matched, list(all_warnings)


def context_aware_retrieval(test_data_with_rec, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name):
    """
    Retrieve collaborative knowledge and reflect on contextual relevancy
    """
    print(f"-----Context-aware Reflection-----")

    EXSTING = {}
    threads, results = [], []

    for i, item in enumerate(tqdm(test_data_with_rec,
                                total=len(test_data_with_rec),
                                desc=f"Reflecting on retrieved titles - {K} raw retrieval...")):
        context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])

        flattened_triples = [
            (iid, title, attitude)
            for iids, titles, attitudes in zip(item["clean_context_imdb_ids"], item["clean_context_titles"], item["clean_context_attitudes"])
            for iid, title, attitude in zip(iids, titles, attitudes)
        ]

        # Filter titles with non-negative attitudes and ensure uniqueness
        context_ids = list({
            iid for iid, title, attitude in flattened_triples if attitude in {"0", "1", "2"}
        })

        # Collaborative filtering augmented retrieval
        retrieved_titles, _ = cf_retrieve(context_ids, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name, K)
        retrieved_titles_str = ", ".join([f"{i+1}. {title}" for i, title in enumerate(retrieved_titles)])

        input_text = {
            "context" : context,
            "retrieved_titles" : retrieved_titles_str
        }
        prompt = prompt_reflect_titles

        execute_thread = threading.Thread(
            target=get_response,
            args=(i, input_text, prompt, model, temperature, max_tokens, results, EXSTING)
        )

        time.sleep(0.15)  # Increased delay to avoid rate limiting
        threads.append(execute_thread)
        execute_thread.start()
        if len(threads) == n_threads:
            for execute_thread in threads:
                execute_thread.join()

            for res in results:
                index = res["index"]
                test_data_with_rec[index][f"reflect_retrieval_from_llm_{K}"] = res

            threads = []
            results = []
            time.sleep(2)  # Additional delay between batches

    if len(threads) > 0:
        for execute_thread in threads:
            execute_thread.join()

    for res in results:
        index = res["index"]
        test_data_with_rec[index][f"reflect_retrieval_from_llm_{K}"] = res

    # Process retrieval reflection results - filter out failed requests
    processed_data = []
    error_count = 0
    for item in test_data_with_rec:
        try:
            err, processed_item = process_retrieval_reflect_raw(item, K)
            if not err:
                processed_data.append(processed_item)
            else:
                # Keep item but without reflection data
                item[f"retrieval_after_reflect_{K}"] = []
                processed_data.append(item)
                error_count += 1
        except Exception as e:
            print(f"  Warning: Failed to process item, keeping without reflection data")
            item[f"retrieval_after_reflect_{K}"] = []
            processed_data.append(item)
            error_count += 1
    test_data_with_rec = processed_data
    print(f"# errors for {K}: {error_count}")

    return test_data_with_rec


def recommend_with_retrieval(test_data_with_rec):
    """
    Generate CRAG recommendations with retrieved items as extra collaborative information
    """
    print(f"-----CF-Augmented Recommendation-----")

    EXSTING = {}
    threads, results = [], []

    for i, item in enumerate(tqdm(test_data_with_rec,
                                total=len(test_data_with_rec),
                                desc=f"Generating recommendations - {K} raw retrieval...")):
        context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])

        retrieved_titles = item[f"retrieval_after_reflect_{K}"]

        # Use retrieved titles after self-reflection
        if retrieved_titles:
            retrieved_titles_str = ", ".join([f"{i+1}. {title}" for i, title in enumerate(retrieved_titles)])
            input_text = {
                "context" : context,
                "retrieved_titles" : retrieved_titles_str
            }
            prompt = prompt_with_retrieved_titles

        # Deteriorate into zero-shot recommendation
        else:
            input_text = {
                "context" : context,
            }
            prompt = prompt_no_title

        execute_thread = threading.Thread(
            target=get_response,
            args=(i, input_text, prompt, model, temperature, max_tokens, results, EXSTING)
        )

        time.sleep(0.15)  # Increased delay to avoid rate limiting
        threads.append(execute_thread)
        execute_thread.start()
        if len(threads) == n_threads:
            for execute_thread in threads:
                execute_thread.join()

            for res in results:
                index = res["index"]
                test_data_with_rec[index][f"rec_from_llm_{K}"] = res

            threads = []
            results = []
            time.sleep(2)  # Additional delay between batches

    if len(threads) > 0:
        for execute_thread in threads:
            execute_thread.join()

    for res in results:
        index = res["index"]
        test_data_with_rec[index][f"rec_from_llm_{K}"] = res

    # Process recommendations with error handling
    processed_data = []
    for item in test_data_with_rec:
        try:
            processed_item = process_item_raw(item, K)
            processed_data.append(processed_item)
        except Exception as e:
            print(f"  Warning: Failed to process recommendation, keeping raw data")
            item[f"rec_list_raw_{K}"] = []
            processed_data.append(item)
    test_data_with_rec = processed_data

    return test_data_with_rec


def get_rec_imdb_ids(item, title_to_imdb):
    """
    Get IMDB IDs for recommended movies
    """
    rec_list = item.get(f"rec_list_raw_{K}", [])

    imdb_ids = []
    for title in rec_list:
        # Try exact match first
        if title in title_to_imdb:
            imdb_ids.append(title_to_imdb[title])
        # Try lowercase match
        elif title.lower() in title_to_imdb:
            imdb_ids.append(title_to_imdb[title.lower()])
        # Try cleaned title
        else:
            clean_title = extract_movie_name(title)
            if clean_title in title_to_imdb:
                imdb_ids.append(title_to_imdb[clean_title])
            elif clean_title.lower() in title_to_imdb:
                imdb_ids.append(title_to_imdb[clean_title.lower()])
            else:
                imdb_ids.append(None)

    return imdb_ids


def main():
    """
    Main pipeline for creating trait-labeled dataset
    """
    print("=" * 60)
    print("CREATING TRAIT-LABELED DATASET")
    print("=" * 60)

    # Step 1: Load all data
    print("\nStep 1: Loading data...")

    print("  Loading collaborative filtering model...")
    sim_mat, catalog_imdb_ids, imdb_id2row, imdb_id2col, col2imdb_id, id2name = load_and_process_cf_model()

    print("  Loading test data...")
    with open(from_pkl, "rb") as f:
        test_data = pickle.load(f)
    print(f"  Loaded {len(test_data)} test samples")

    print("  Loading title to IMDB mapping...")
    title_to_imdb = load_title_to_imdb_mapping()
    print(f"  Loaded {len(title_to_imdb)} title mappings")

    print("  Loading sensitivity table...")
    sensitivity_df, warning_columns = load_sensitivity_table()

    print("  Loading traits warnings...")
    traits_warnings = load_traits_warnings()

    # Step 2: Pre-process data
    print("\nStep 2: Pre-processing data...")
    test_data_with_rec = pre_process(test_data, catalog_imdb_ids)
    print(f"  After preprocessing: {len(test_data_with_rec)} samples")

    # Sample data if n_samples is set
    if n_samples > 0 and n_samples < len(test_data_with_rec):
        random.seed(42)
        test_data_with_rec = random.sample(test_data_with_rec, n_samples)
        print(f"  Sampled {n_samples} samples for testing")

    # Step 3: Run CRAG recommendations
    print("\nStep 3: Running CRAG recommendations...")
    test_data_with_rec = context_aware_retrieval(test_data_with_rec, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name)
    test_data_with_rec = recommend_with_retrieval(test_data_with_rec)

    # Step 4: Match traits based on warnings
    print("\nStep 4: Matching traits based on warning statistics...")

    trait_distribution = defaultdict(int)
    mapping_results = {}

    for item in tqdm(test_data_with_rec, desc="Matching traits..."):
        # Get IMDB IDs for recommended movies
        rec_imdb_ids = get_rec_imdb_ids(item, title_to_imdb)

        # Match trait
        best_trait, score, matched_warnings, all_warnings = match_trait(
            rec_imdb_ids, sensitivity_df, warning_columns, traits_warnings, title_to_imdb
        )

        # Store results in item
        item['assigned_trait'] = best_trait
        item['trait_score'] = score
        item['matched_warnings'] = matched_warnings
        item['rec_warnings'] = all_warnings
        item['rec_imdb_ids'] = rec_imdb_ids

        # Track distribution
        trait_distribution[best_trait] += 1

        # Store mapping
        turn_id = item.get('turn_id', str(id(item)))
        mapping_results[turn_id] = {
            'trait': best_trait,
            'score': score,
            'matched_warnings': matched_warnings
        }

    # Print trait distribution
    print("\nTrait Distribution:")
    for trait, count in sorted(trait_distribution.items(), key=lambda x: -x[1]):
        print(f"  {trait}: {count}")

    # Step 5: Save results
    print("\nStep 5: Saving results...")

    # Save pkl file
    pkl_path = f"{data_root}/test_with_trait_mapping.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(test_data_with_rec, f)
    print(f"  Saved pkl file to: {pkl_path}")

    # Save JSON mapping
    json_path = f"{data_root}/trait_mapping.json"
    with open(json_path, "w") as f:
        json.dump(mapping_results, f, indent=2)
    print(f"  Saved JSON mapping to: {json_path}")

    # Save trait distribution
    dist_path = f"{data_root}/trait_distribution.json"
    with open(dist_path, "w") as f:
        json.dump(dict(trait_distribution), f, indent=2)
    print(f"  Saved trait distribution to: {dist_path}")

    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE!")
    print("=" * 60)

    return test_data_with_rec


if __name__ == '__main__':
    main()
