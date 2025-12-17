"""
Evaluate CRAG with trait-based warning avoidance.

This script:
1. Loads the trait-labeled dataset
2. Runs CRAG with trait information in prompts
3. Evaluates safety metrics (warning rate, safety improvement)
4. Evaluates accuracy metrics (Recall@k, NDCG@k)
"""

import os
import time
import json
import pickle
import datetime
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
from libs.utils import process_rec_reflect_raw
from libs.model import cf_retrieve, get_response
from libs.metrics import evaluate_direct_match
from libs.metrics import evaluate_direct_match_reflect_rerank

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

data_root = f"data/{dataset}"
cf_model = "large_pop_adj_07sym"
cf_root = f"{cf_model}"

# Hyperparameters
temperature = 0.0
max_tokens = 512
n_threads = 50  # Reduced from 500 to avoid rate limiting
tier = 30
K_list = [20]  # Focus on K=20 for trait evaluation
k_list = [5, 10, 15, 20]  # top-k for evaluation
metrics = {}
avg_metrics = {}

# Prompt templates with trait information
prompt_reflect_titles_with_trait = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system), "
    "as well as some movies retrieved from our movie database based on the similarity with the movies mentioned by the user in the context.\n"
    "You need to judge whether each retrieved movie is a good recommendation based on the context.\n"
    "IMPORTANT: The user has sensitivity to: {trait}. "
    "Movies containing the following content should be avoided: {avoid_list}.\n"
    "Here is the conversation: {context}\n"
    "Here are retrieved movies: {retrieved_titles}.\n"
    "You need to reply with the judgement of each movie in a line, in the form of movie_name####judgment, "
    "where judgement is a binary number 0, 1. Judgment 0 means the movie is a bad recommendation, whereas judgment 1 means the movie is a good recommendation. "
    "System:"
)

prompt_with_retrieved_titles_with_trait = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system). "
    "Based on the conversation, you need to reply with 20 movie recommendations without extra sentences. "
    "List the standardized title of each movie on a separate line.\n"
    "IMPORTANT: The user has sensitivity to: {trait}. "
    "Movies containing the following content should be avoided: {avoid_list}.\n"
    "Here is the conversation: {context}\n"
    "Based on movies mentioned in the conversation, here are some movies that are usually liked by other users: {retrieved_titles}.\n"
    "Use the above information at your discretion (i.e., do not confine your recommendation to the above movies). "
    "System:"
)

prompt_no_title_with_trait = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system). "
    "Based on the conversation, you need to reply with 20 recommendations without extra sentences. "
    "List the standardized title of each movie in each line.\n"
    "IMPORTANT: The user has sensitivity to: {trait}. "
    "Movies containing the following content should be avoided: {avoid_list}.\n"
    "Here is the conversation: {context}\n"
    "System:"
)

prompt_reflect_rec_titles_with_trait = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system), "
    "as well as some movie candidates from our movie database. "
    "You need to rate each retrieved movie as recommendations into five levels based on the conversation: "
    "2 (great), 1 (good), 0 (normal), -1 (not good), -2 (bad).\n"
    "IMPORTANT: The user has sensitivity to: {trait}. "
    "Movies containing the following content should be rated lower or negative: {avoid_list}.\n"
    "Here is the conversation: {context}\n"
    "Here are the movie candidates: {rec_titles}.\n"
    "You need to reply with the rating of each movie in a line, in the form of movie_name####rating, "
    "where rating should be an Integer, and 2 means great, 1 means good, 0 means normal, -1 means not good, and -2 means bad.\n"
    "System:"
)


def load_traits_warnings():
    """
    Load traits with warning definitions
    """
    with open('traits_warnings.json', 'r') as f:
        traits_warnings = json.load(f)

    # Create lookup by trait name
    trait_lookup = {t['trait']: t for t in traits_warnings}

    return traits_warnings, trait_lookup


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


def load_title_to_imdb_mapping():
    """
    Load title to IMDB ID mapping from entity2id files
    """
    # Load response entity mapping
    resp_pkl = os.path.join(f'{data_root}/entity2id_resp{version}.pkl')
    with open(resp_pkl, "rb") as f:
        resp_table = pickle.load(f)

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

    return sensitivity_df, warning_columns


def get_movie_warnings(imdb_id, sensitivity_df, warning_columns):
    """
    Query movie warning tags
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


def get_rec_imdb_ids(rec_list, title_to_imdb):
    """
    Get IMDB IDs for recommended movies
    """
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


def context_aware_retrieval_with_trait(test_data_with_rec, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name, trait_lookup):
    """
    Retrieve collaborative knowledge and reflect with trait information
    """
    print(f"-----Context-aware Reflection (with Traits)-----")

    for K in K_list:
        EXSTING = {}
        threads, results = [], []

        for i, item in enumerate(tqdm(test_data_with_rec,
                                    total=len(test_data_with_rec),
                                    desc=f"Reflecting on retrieved titles - {K} raw retrieval...")):
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])

            # Get trait information
            trait_name = item.get('assigned_trait', '')
            trait_info = trait_lookup.get(trait_name, {})
            avoid_list = ", ".join(trait_info.get('avoid', [])[:5])  # Limit to first 5 for prompt length

            flattened_triples = [
                (iid, title, attitude)
                for iids, titles, attitudes in zip(item["clean_context_imdb_ids"], item["clean_context_titles"], item["clean_context_attitudes"])
                for iid, title, attitude in zip(iids, titles, attitudes)
            ]

            context_ids = list({
                iid for iid, title, attitude in flattened_triples if attitude in {"0", "1", "2"}
            })

            retrieved_titles, _ = cf_retrieve(context_ids, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name, K)
            retrieved_titles_str = ", ".join([f"{i+1}. {title}" for i, title in enumerate(retrieved_titles)])

            input_text = {
                "context": context,
                "retrieved_titles": retrieved_titles_str,
                "trait": trait_name,
                "avoid_list": avoid_list
            }
            prompt = prompt_reflect_titles_with_trait

            execute_thread = threading.Thread(
                target=get_response,
                args=(i, input_text, prompt, model, temperature, max_tokens, results, EXSTING)
            )

            time.sleep(0.02)
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

        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()

        for res in results:
            index = res["index"]
            test_data_with_rec[index][f"reflect_retrieval_from_llm_{K}"] = res

    for K in K_list:
        test_data_with_rec = [process_retrieval_reflect_raw(item, K) for item in test_data_with_rec]
        errors = [item[0] for item in test_data_with_rec if item[0]]
        test_data_with_rec = [item[1] for item in test_data_with_rec]
        print(f"# errors for {K}: {len(errors)}")

    return test_data_with_rec


def recommend_with_retrieval_with_trait(test_data_with_rec, trait_lookup):
    """
    Generate CRAG recommendations with trait information
    """
    for K in K_list:
        print(f"-----CF-Augmented Recommendation (with Traits)-----")

        EXSTING = {}
        threads, results = [], []

        for i, item in enumerate(tqdm(test_data_with_rec,
                                    total=len(test_data_with_rec),
                                    desc=f"Generating recommendations - {K} raw retrieval...")):
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])

            # Get trait information
            trait_name = item.get('assigned_trait', '')
            trait_info = trait_lookup.get(trait_name, {})
            avoid_list = ", ".join(trait_info.get('avoid', [])[:5])

            retrieved_titles = item[f"retrieval_after_reflect_{K}"]

            if retrieved_titles:
                retrieved_titles_str = ", ".join([f"{i+1}. {title}" for i, title in enumerate(retrieved_titles)])
                input_text = {
                    "context": context,
                    "retrieved_titles": retrieved_titles_str,
                    "trait": trait_name,
                    "avoid_list": avoid_list
                }
                prompt = prompt_with_retrieved_titles_with_trait

            else:
                input_text = {
                    "context": context,
                    "trait": trait_name,
                    "avoid_list": avoid_list
                }
                prompt = prompt_no_title_with_trait

            execute_thread = threading.Thread(
                target=get_response,
                args=(i, input_text, prompt, model, temperature, max_tokens, results, EXSTING)
            )

            time.sleep(0.025)
            threads.append(execute_thread)
            execute_thread.start()
            if len(threads) == n_threads:
                for execute_thread in threads:
                    execute_thread.join()

                for res in results:
                    index = res["index"]
                    test_data_with_rec[index][f"rec_from_llm_trait_{K}"] = res

                threads = []
                results = []

        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()

        for res in results:
            index = res["index"]
            test_data_with_rec[index][f"rec_from_llm_trait_{K}"] = res

    # Process recommendations
    for K in K_list:
        for item in test_data_with_rec:
            item[f"rec_from_llm_{K}"] = item.get(f"rec_from_llm_trait_{K}", {})
        test_data_with_rec = [process_item_raw(item, K) for item in test_data_with_rec]

    return test_data_with_rec


def reflect_and_rerank_with_trait(test_data_with_rec, trait_lookup):
    """
    Reflect and rerank with trait information
    """
    for K in K_list:
        print(f"-----Reflect and Rerank (with Traits)-----")

        EXSTING = {}
        threads, results = [], []

        for i, item in enumerate(tqdm(test_data_with_rec,
                                    total=len(test_data_with_rec),
                                    desc=f"Reflect and rerank - {K} raw retrieval...")):
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])

            # Get trait information
            trait_name = item.get('assigned_trait', '')
            trait_info = trait_lookup.get(trait_name, {})
            avoid_list = ", ".join(trait_info.get('avoid', [])[:5])

            rec_list_raw = item[f"rec_list_raw_{K}"]
            rec_titles = ", ".join([f"{i+1}. {title}" for i, title in enumerate(rec_list_raw)])

            input_text = {
                "context": context,
                "rec_titles": rec_titles,
                "trait": trait_name,
                "avoid_list": avoid_list
            }
            prompt = prompt_reflect_rec_titles_with_trait

            execute_thread = threading.Thread(
                target=get_response,
                args=(i, input_text, prompt, model, temperature, max_tokens, results, EXSTING)
            )

            time.sleep(0.02)
            threads.append(execute_thread)
            execute_thread.start()
            if len(threads) == n_threads:
                for execute_thread in threads:
                    execute_thread.join()

                for res in results:
                    index = res["index"]
                    test_data_with_rec[index][f"reflect_rec_from_llm_{K}"] = res

                threads = []
                results = []

        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()

        for res in results:
            index = res["index"]
            test_data_with_rec[index][f"reflect_rec_from_llm_{K}"] = res

    for K in K_list:
        test_data_with_rec = [process_rec_reflect_raw(item, K) for item in test_data_with_rec]
        errors = [item[0] for item in test_data_with_rec if item[0]]
        test_data_with_rec = [item[1] for item in test_data_with_rec]
        print(f"# errors for {K}: {len(errors)}")

    return test_data_with_rec


def post_processing(test_data_with_rec):
    """
    Post-process data to get groundtruth
    """
    for item in test_data_with_rec:
        clean_resp_titles = item["clean_resp_titles"]
        clean_resp_attitude = item["clean_resp_attitude"]
        clean_context_titles = item["clean_context_titles"]

        groundtruth = []

        for title, attitude in zip(clean_resp_titles, clean_resp_attitude):
            if attitude not in ("-2", "-1") and not any(title in context for context in clean_context_titles):
                groundtruth.append(title)

        item["groundtruth"] = groundtruth

    test_data_with_rec = [item for item in test_data_with_rec if (item["old"]["is_user"] == 0) and (item["groundtruth"])]

    return test_data_with_rec


def calculate_safety_metrics(test_data_with_rec, title_to_imdb, sensitivity_df, warning_columns, trait_lookup):
    """
    Calculate safety metrics for recommendations

    Returns:
        dict: Safety metrics including warning rate and comparison with baseline
    """
    print("\n-----Calculating Safety Metrics-----")

    safety_results = {
        'with_trait': {'warning_rates': [], 'warning_counts': []},
        'baseline': {'warning_rates': [], 'warning_counts': []}
    }

    for item in tqdm(test_data_with_rec, desc="Calculating safety metrics..."):
        trait_name = item.get('assigned_trait', '')
        trait_info = trait_lookup.get(trait_name, {})
        avoid_list = set(trait_info.get('avoid', []))

        for K in K_list:
            # Get recommendations after rerank (with trait)
            rec_list_with_trait = item.get(f"rec_after_reflect_{K}", [])[:20]
            rec_imdb_ids = get_rec_imdb_ids(rec_list_with_trait, title_to_imdb)

            # Count warnings in recommendations
            warning_count = 0
            for imdb_id in rec_imdb_ids:
                if imdb_id:
                    movie_warnings = set(get_movie_warnings(imdb_id, sensitivity_df, warning_columns))
                    if movie_warnings & avoid_list:
                        warning_count += 1

            warning_rate = warning_count / len(rec_list_with_trait) if rec_list_with_trait else 0
            safety_results['with_trait']['warning_rates'].append(warning_rate)
            safety_results['with_trait']['warning_counts'].append(warning_count)

            # Compare with baseline (original recommendations without trait)
            baseline_warning_count = 0
            baseline_rec_imdb_ids = item.get('rec_imdb_ids', [])[:20]
            for imdb_id in baseline_rec_imdb_ids:
                if imdb_id:
                    movie_warnings = set(get_movie_warnings(imdb_id, sensitivity_df, warning_columns))
                    if movie_warnings & avoid_list:
                        baseline_warning_count += 1

            baseline_rate = baseline_warning_count / len(baseline_rec_imdb_ids) if baseline_rec_imdb_ids else 0
            safety_results['baseline']['warning_rates'].append(baseline_rate)
            safety_results['baseline']['warning_counts'].append(baseline_warning_count)

    # Calculate average metrics
    avg_safety = {
        'with_trait': {
            'avg_warning_rate': np.mean(safety_results['with_trait']['warning_rates']),
            'avg_warning_count': np.mean(safety_results['with_trait']['warning_counts'])
        },
        'baseline': {
            'avg_warning_rate': np.mean(safety_results['baseline']['warning_rates']),
            'avg_warning_count': np.mean(safety_results['baseline']['warning_counts'])
        }
    }

    # Calculate improvement
    baseline_rate = avg_safety['baseline']['avg_warning_rate']
    trait_rate = avg_safety['with_trait']['avg_warning_rate']

    if baseline_rate > 0:
        improvement = (baseline_rate - trait_rate) / baseline_rate * 100
    else:
        improvement = 0

    avg_safety['improvement_percentage'] = improvement

    return avg_safety, safety_results


def evaluate_accuracy(test_data_with_rec_filtered):
    """
    Evaluate accuracy metrics (Recall@k, NDCG@k)
    """
    avg_metrics_filtered = {}
    metrics_filtered = {}

    for K in K_list:
        print(f"Processing {K} retrievals")

        errors = set()
        results = {k:[] for k in k_list}

        for k in k_list:
            for i, item in enumerate(tqdm(test_data_with_rec_filtered,
                                     total=len(test_data_with_rec_filtered),
                                     desc=f"Evaluating Recall@{k}...")):
                try:
                    results[k].append(evaluate_direct_match_reflect_rerank(item, k, K, gt_field="groundtruth"))
                except:
                    errors.add(i)

        recalls = {k:[res[0] for res in results[k]] for k in k_list}
        ndcgs = {k:[res[1] for res in results[k]] for k in k_list}

        metrics_filtered[K] = (recalls, ndcgs)

        avg_recalls_filtered = {k:np.mean(recalls[k]) for k in k_list}
        avg_ndcgs_filtered = {k:np.mean(ndcgs[k]) for k in k_list}
        avg_metrics_filtered[K] = (avg_recalls_filtered, avg_ndcgs_filtered)

        print(f"Number of errors: {len(errors)}")

    return avg_metrics_filtered


def main():
    """
    Main evaluation pipeline
    """
    print("=" * 60)
    print("TRAIT-AWARE CRAG EVALUATION")
    print("=" * 60)

    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/trait_warning_evaluation/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Load data
    print("\nStep 1: Loading data...")

    print("  Loading trait-labeled dataset...")
    dataset_path = f"{data_root}/test_with_trait_mapping.pkl"
    if not os.path.exists(dataset_path):
        print(f"  ERROR: Dataset not found at {dataset_path}")
        print("  Please run create_trait_dataset.py first!")
        return

    with open(dataset_path, "rb") as f:
        test_data_with_rec = pickle.load(f)
    print(f"  Loaded {len(test_data_with_rec)} samples with trait labels")

    print("  Loading traits warnings...")
    traits_warnings, trait_lookup = load_traits_warnings()

    print("  Loading collaborative filtering model...")
    sim_mat, catalog_imdb_ids, imdb_id2row, imdb_id2col, col2imdb_id, id2name = load_and_process_cf_model()

    print("  Loading title to IMDB mapping...")
    title_to_imdb = load_title_to_imdb_mapping()

    print("  Loading sensitivity table...")
    sensitivity_df, warning_columns = load_sensitivity_table()

    # Step 2: Run CRAG with trait information
    print("\nStep 2: Running CRAG with trait information...")
    test_data_with_rec = context_aware_retrieval_with_trait(
        test_data_with_rec, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name, trait_lookup
    )
    test_data_with_rec = recommend_with_retrieval_with_trait(test_data_with_rec, trait_lookup)
    test_data_with_rec = reflect_and_rerank_with_trait(test_data_with_rec, trait_lookup)

    # Step 3: Post-processing
    print("\nStep 3: Post-processing...")
    test_data_filtered = post_processing(test_data_with_rec)
    print(f"  After post-processing: {len(test_data_filtered)} samples with groundtruth")

    # Step 4: Calculate safety metrics
    print("\nStep 4: Calculating safety metrics...")
    avg_safety, safety_results = calculate_safety_metrics(
        test_data_filtered, title_to_imdb, sensitivity_df, warning_columns, trait_lookup
    )

    # Step 5: Evaluate accuracy
    print("\nStep 5: Evaluating accuracy metrics...")
    accuracy_metrics = evaluate_accuracy(test_data_filtered)

    # Step 6: Print and save results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Safety Metrics ---")
    print(f"Baseline Warning Rate: {avg_safety['baseline']['avg_warning_rate']:.4f}")
    print(f"With Trait Warning Rate: {avg_safety['with_trait']['avg_warning_rate']:.4f}")
    print(f"Safety Improvement: {avg_safety['improvement_percentage']:.2f}%")

    print("\n--- Accuracy Metrics ---")
    for K in K_list:
        recalls, ndcgs = accuracy_metrics[K]
        print(f"K={K}:")
        print(f"  Recall@k: {recalls}")
        print(f"  NDCG@k: {ndcgs}")

    # Save results
    results = {
        'safety': avg_safety,
        'accuracy': {K: {'recall': recalls, 'ndcg': ndcgs} for K, (recalls, ndcgs) in accuracy_metrics.items()},
        'config': {
            'model': model,
            'K_list': K_list,
            'k_list': k_list,
            'n_samples': len(test_data_filtered)
        }
    }

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    results = convert_to_serializable(results)

    with open(os.path.join(results_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(results_dir, "test_data_evaluated.pkl"), 'wb') as f:
        pickle.dump(test_data_filtered, f)

    print(f"\nResults saved to: {results_dir}")
    print("=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
