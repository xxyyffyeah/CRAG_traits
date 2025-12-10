import os
import time
import json
import pickle
import random
import threading
import datetime
import multiprocessing
import re

from tqdm import tqdm
from pprint import pprint
from copy import deepcopy
from functools import partial
from collections import defaultdict

import openai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from editdistance import eval as distance

from libs.utils import extract_movie_name, process_item_raw
from libs.utils import process_retrieval_reflect_raw
from libs.utils import process_rec_reflect_raw
from libs.model import cf_retrieve, get_response
from libs.metrics import evaluate_direct_match
from libs.metrics import evaluate_direct_match_reflect_rerank

import pdb
from dotenv import load_dotenv
load_dotenv()

external = True

if external:
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    if os.environ.get('OPENAI_ORG') is not None:
        openai.organization = os.environ.get('OPENAI_ORG')

    if openai.api_key is None:
        raise Exception('OPENAI_API_KEY is not set')
else:
    import nflx_copilot as ncp
    openai = ncp
    ncp.project_id = "reddit"

### Path and model configuration
model = "gpt-4o"
evaluation_trait_model = "gpt-5.1"
method = "CRAG_with_traits"
dataset = "reddit"
version = "_with_titles"
prompt_type = "rag"
datafile = f"test_clean{version}"

data_root = f"data/{dataset}"
from_pkl = f"{data_root}/{datafile}.pkl"

cf_model = "large_pop_adj_07sym"
cf_root = f"{cf_model}"

### Hyperparameters
temperature = 0.0
max_tokens = 512
n_threads = 500
n_print = 100
tier = 30
n_samples = 100  # Number of samples to test
random_seed = 42  # For reproducibility
K_list = list(range(5, 40, 5))
k_list = [5, 10, 15, 20]
metrics = {}
avg_metrics = {}

### Trait-enabled prompt templates
prompt_reflect_titles_with_trait = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system), "
    "as well as some movies retrieved from our movie database based on the similarity with the movies mentioned by the user in the context."
    "You need to judge whether each retrieved movie is a good recommendation based on the context.\n"
    "Here is the conversation: {context}\n"
    "Personal Trait: {trait}\n"
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
    "Here is the conversation: {context}\n"
    "Personal Trait: {trait}\n"
    "Based on movies mentioned in the conversation, here are some movies that are usually liked by other users: {retrieved_titles}.\n"
    "Use the above information at your discretion (i.e., do not confine your recommendation to the above movies). "
    "System:"
)

prompt_no_title_with_trait = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system). "
    "Based on the conversation, you need to reply with 20 recommendations without extra sentences. "
    "List the standardized title of each movie in each line.\n"
    "Here is the conversation: {context}\n"
    "Personal Trait: {trait}\n"
    "System:"
)

prompt_reflect_rec_titles_with_trait = (
    "Pretend you are a movie recommender system.\n"
    "I will give you a conversation between a user and you (a recommender system), "
    "as well as some movie candidates from our movie database. "
    "You need to rate each retrieved movie as recommendations into five levels based on the conversation: "
    "2 (great), 1 (good), 0 (normal), -1 (not good), -2 (bad).\n"
    "Here is the conversation: {context}\n"
    "Personal Trait: {trait}\n"
    "Here are the movie candidates: {rec_titles}.\n"
    "You need to reply with the rating of each movie in a line, in the form of movie_name####rating, "
    "where rating should be an Integer, and 2 means great, 1 means good, 0 means normal, -1 means not good, and -2 means bad.\n"
    "System:"
)

prompt_trait_alignment = (
    "You are an expert movie recommendation evaluator.\n"
    "Given a user's personal trait and a list of recommended movies, "
    "you need to judge whether the recommendations are appropriate and aligned with the user's trait.\n\n"
    "Personal Trait: {trait}\n\n"
    "Recommended Movies (in order):\n{movie_list}\n\n"
    "Please evaluate whether these recommendations are suitable for a user with the given trait. "
    "Consider whether the movies avoid content that might be inappropriate, triggering, or unsuitable for this user.\n\n"
    "Rate the overall alignment on a scale of 1-5:\n"
    "1 - Very Poor: Multiple recommendations are highly inappropriate\n"
    "2 - Poor: Several recommendations are unsuitable\n"
    "3 - Acceptable: Some issues but mostly appropriate\n"
    "4 - Good: Recommendations are well-aligned\n"
    "5 - Excellent: All recommendations perfectly respect the trait\n\n"
    "Reply with ONLY a single number (1-5) followed by a brief explanation (one sentence).\n"
    "Format: RATING####EXPLANATION\n"
    "System:"
)


def load_traits(traits_file='traits.json'):
    """
    Load and flatten traits.json into a single list

    Returns:
        list: List of trait dictionaries with category, trait, and description
    """
    with open(traits_file, 'r') as f:
        traits_data = json.load(f)

    all_traits = []
    for category, trait_list in traits_data.items():
        for trait_obj in trait_list:
            all_traits.append({
                'category': category,
                'trait': trait_obj['trait'],
                'description': trait_obj['description']
            })

    print(f"Loaded {len(all_traits)} traits from {len(traits_data)} categories")
    return all_traits


def sample_test_data_with_traits(test_data, all_traits, n_samples=100, random_seed=42):
    """
    Sample n_samples from test_data and assign a random trait to each

    Args:
        test_data: Full test dataset
        all_traits: List of trait dictionaries
        n_samples: Number of samples to select
        random_seed: Random seed for reproducibility

    Returns:
        list: Sampled data with assigned traits
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    total_samples = len(test_data)
    print(f"Sampling {n_samples} from {total_samples} total samples...")

    sampled_indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
    sampled_data = [deepcopy(test_data[i]) for i in sampled_indices]

    for item in sampled_data:
        random_trait = random.choice(all_traits)
        item['assigned_trait'] = random_trait['description']
        item['assigned_trait_name'] = random_trait['trait']
        item['assigned_trait_category'] = random_trait['category']

    trait_counts = defaultdict(int)
    category_counts = defaultdict(int)
    for item in sampled_data:
        trait_counts[item['assigned_trait_name']] += 1
        category_counts[item['assigned_trait_category']] += 1

    print(f"Sampled {len(sampled_data)} items")
    print(f"Unique traits assigned: {len(trait_counts)}")
    print(f"Categories represented: {len(category_counts)}")

    return sampled_data


def load_and_process_cf_model():
    """
    This function loads and process the item-item
    similarity matrix for collaborative retrieval
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

    reddit_test_meta_pkl = os.path.join(f'{data_root}/entity2id{version}.pkl')
    with open(reddit_test_meta_pkl, "rb") as f:
        reddit_test_id_name_table = pickle.load(f)

    reddit_test_name2id = reddit_test_id_name_table.set_index('title')['imdb_id'].to_dict()
    reddit_test_id2name = {v: extract_movie_name(k) for k, v in reddit_test_name2id.items()}
    relevant_indices = [row for row, imdb_id in raw_row2imdb_id.items() if imdb_id in reddit_test_id2name]
    sim_mat = sim_mat[relevant_indices, :]
    row2imdb_id = {new_row: raw_row2imdb_id[old_row] for new_row, old_row in enumerate(relevant_indices)}
    imdb_id2row = {imdb_id: row for row, imdb_id in row2imdb_id.items()}

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
    This function pre-processes the data
    """
    test_data = [item for item in test_data if item["old"]["is_user"] == 0]
    test_data = [item for item in test_data if all(iid in catalog_imdb_ids for iid in item["clean_resp_imdb_ids"])]

    seen_turn_ids = {}
    unique_test_data = []

    for item in test_data:
        turn_id = item["turn_id"]
        if turn_id not in seen_turn_ids:
            seen_turn_ids[turn_id] = True
            unique_test_data.append(item)

    test_data_with_rec = unique_test_data
    return test_data_with_rec


def context_aware_retrieval_with_trait(test_data_with_rec, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name):
    """
    This module retrieves the collaborative knowledge
    and reflects on its contextual relevancy (with trait information)
    """

    print(f"-----Context-aware Reflection (with Traits)-----")
    for K in K_list:
        EXSTING = {}
        threads, results = [], []

        for i, item in enumerate(tqdm(test_data_with_rec,
                                    total=len(test_data_with_rec),
                                    desc=f"reflecting on the retrieved titles - {K} raw retrieval...")):
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
            trait = item['assigned_trait']

            flattened_triples = [
                (iid, title, attitude)
                for iids, titles, attitudes in zip(item["clean_context_imdb_ids"], item["clean_context_titles"], item["clean_context_attitudes"])
                for iid, title, attitude in zip(iids, titles, attitudes)
            ]

            context_ids = list({
                iid for iid, title, attitude in flattened_triples if attitude in {"0", "1", "2"}
            })
            context_titles = list({
                title for iid, title, attitude in flattened_triples if attitude in {"0", "1", "2"}
            })

            retrieved_titles, _ = cf_retrieve(context_ids, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name, K)
            retrieved_titles = ", ".join([f"{i+1}. {title}" for i, title in enumerate(retrieved_titles)])

            input_text = {
                "context": context,
                "retrieved_titles": retrieved_titles,
                "trait": trait
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
                time.sleep(0)

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


def recommend_zero_shot_with_trait(test_data_with_rec):
    """
    This module conducts the zero-shot recommendation
    with LLM as the baseline (with trait information)
    """
    EXSTING = {}
    threads, results = [], []

    print(f"-----Zero-shot Recommendation (with Traits)-----")
    for i, item in enumerate(tqdm(test_data_with_rec,
                                total=len(test_data_with_rec),
                                desc="generating recommendations...")):
        context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
        trait = item['assigned_trait']

        input_text = {
            "context": context,
            "trait": trait
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
                test_data_with_rec[index][f"rec_from_llm_0"] = res

            threads = []
            results = []
            time.sleep(0)

    if len(threads) > 0:
        for execute_thread in threads:
            execute_thread.join()

    for res in results:
        index = res["index"]
        test_data_with_rec[index][f"rec_from_llm_0"] = res

    test_data_with_rec = [process_item_raw(item, 0) for item in test_data_with_rec]

    return test_data_with_rec


def recommend_with_retrieval_with_trait(test_data_with_rec):
    """
    This module conducts the CRAG recommendation with
    retrieved items as extra collaborative information (with trait)
    """
    for K in K_list:
        print(f"-----CF-Augmented Recommendation (with Traits)-----")

        EXSTING = {}
        threads, results = [], []

        for i, item in enumerate(tqdm(test_data_with_rec,
                                    total=len(test_data_with_rec),
                                    desc=f"generating recommendations - {K} raw retrieval...")):
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
            trait = item['assigned_trait']

            retrieved_titles = item[f"retrieval_after_reflect_{K}"]

            if retrieved_titles:
                retrieved_titles = ", ".join([f"{i+1}. {title}" for i, title in enumerate(retrieved_titles)])
                input_text = {
                    "context": context,
                    "retrieved_titles": retrieved_titles,
                    "trait": trait
                }
                prompt = prompt_with_retrieved_titles_with_trait

            else:
                input_text = {
                    "context": context,
                    "trait": trait
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
                    test_data_with_rec[index][f"rec_from_llm_{K}"] = res

                threads = []
                results = []
                time.sleep(0)

        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()

        for res in results:
            index = res["index"]
            test_data_with_rec[index][f"rec_from_llm_{K}"] = res

    for K in K_list:
        test_data_with_rec = [process_item_raw(item, K) for item in test_data_with_rec]

    save_dir = "results/trait_evaluation"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, "test_with_traits_retrieval.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(test_data_with_rec, f)

    print(f"results saved to: {save_file}!")

    return test_data_with_rec


def reflect_and_rerank_with_trait(test_data_with_rec):
    """
    This module reflects upon the final recommendation list
    and rerank them based on the relevance score (with trait).
    """
    K_list_extended = K_list.copy()
    if "rec_from_llm_0" in test_data_with_rec[0] and 0 not in K_list_extended:
        K_list_extended = [0] + K_list_extended

    for K in K_list_extended:
        print(f"-----Reflect and Rerank (with Traits)-----")

        EXSTING = {}
        threads, results = [], []

        for i, item in enumerate(tqdm(test_data_with_rec,
                                    total=len(test_data_with_rec),
                                    desc=f"reflect and rerank the recommended titles - {K} raw retrieval...")):
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
            trait = item['assigned_trait']

            rec_list_raw = item[f"rec_list_raw_{K}"]
            rec_titles = ", ".join([f"{i+1}. {title}" for i, title in enumerate(rec_list_raw)])

            input_text = {
                "context": context,
                "rec_titles": rec_titles,
                "trait": trait
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
                time.sleep(0)

        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()

        for res in results:
            index = res["index"]
            test_data_with_rec[index][f"reflect_rec_from_llm_{K}"] = res

    for K in K_list_extended:
        test_data_with_rec = [process_rec_reflect_raw(item, K) for item in test_data_with_rec]
        errors = [item[0] for item in test_data_with_rec if item[0]]
        test_data_with_rec = [item[1] for item in test_data_with_rec]
        print(f"# errors for {K}: {len(errors)}")

    save_dir = "results/trait_evaluation"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, "test_with_traits_rerank.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(test_data_with_rec, f)

    print(f"results saved to: {save_file}!")

    return test_data_with_rec


def evaluate_trait_alignment(test_data_with_rec):
    """
    Use GPT-4o to evaluate whether recommendations align with personal trait
    """
    K_list_extended = [0] + K_list

    for K in K_list_extended:
        print(f"-----Evaluating Trait Alignment (K={K})-----")

        EXSTING = {}
        threads, results = [], []

        for i, item in enumerate(tqdm(test_data_with_rec,
                                    total=len(test_data_with_rec),
                                    desc=f"Evaluating trait alignment - K={K}...")):
            trait = item['assigned_trait']

            if f"rec_after_reflect_{K}" in item and item[f"rec_after_reflect_{K}"]:
                rec_list = item[f"rec_after_reflect_{K}"][:20]
            elif f"rec_list_raw_{K}" in item and item[f"rec_list_raw_{K}"]:
                rec_list = item[f"rec_list_raw_{K}"][:20]
            else:
                continue

            movie_list = "\n".join([f"{idx+1}. {movie}" for idx, movie in enumerate(rec_list)])

            input_text = {
                "trait": trait,
                "movie_list": movie_list
            }

            execute_thread = threading.Thread(
                target=get_response,
                args=(i, input_text, prompt_trait_alignment, evaluation_trait_model, temperature, 128, results, EXSTING)
            )

            time.sleep(0.02)
            threads.append(execute_thread)
            execute_thread.start()

            if len(threads) == n_threads:
                for execute_thread in threads:
                    execute_thread.join()

                for res in results:
                    index = res["index"]
                    test_data_with_rec[index][f"trait_alignment_{K}"] = res

                threads = []
                results = []

        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()

            for res in results:
                index = res["index"]
                test_data_with_rec[index][f"trait_alignment_{K}"] = res

    for K in K_list_extended:
        for item in test_data_with_rec:
            if f"trait_alignment_{K}" not in item:
                item[f"trait_alignment_score_{K}"] = 3
                item[f"trait_alignment_explanation_{K}"] = "No evaluation"
                continue

            raw_response = item[f"trait_alignment_{K}"]['resp']['choices'][0]['message']['content']
            try:
                parts = raw_response.strip().split("####")
                if len(parts) >= 1:
                    rating = int(parts[0].strip())
                    rating = max(1, min(5, rating))
                    explanation = parts[1].strip() if len(parts) > 1 else ""
                    item[f"trait_alignment_score_{K}"] = rating
                    item[f"trait_alignment_explanation_{K}"] = explanation
                else:
                    match = re.search(r'(\d)', raw_response)
                    if match:
                        rating = int(match.group(1))
                        rating = max(1, min(5, rating))
                        item[f"trait_alignment_score_{K}"] = rating
                        item[f"trait_alignment_explanation_{K}"] = raw_response
                    else:
                        item[f"trait_alignment_score_{K}"] = 3
                        item[f"trait_alignment_explanation_{K}"] = "Parse failed"
            except:
                item[f"trait_alignment_score_{K}"] = 3
                item[f"trait_alignment_explanation_{K}"] = "Parse error"

    return test_data_with_rec


def compute_trait_metrics(test_data_with_rec):
    """
    Compute aggregated trait alignment metrics
    """
    K_list_extended = [0] + K_list
    trait_metrics = {}

    for K in K_list_extended:
        scores = [item.get(f"trait_alignment_score_{K}", 3) for item in test_data_with_rec]

        trait_metrics[K] = {
            'mean_alignment': np.mean(scores),
            'std_alignment': np.std(scores),
            'median_alignment': np.median(scores),
            'excellent_rate': sum(s == 5 for s in scores) / len(scores),
            'good_or_better_rate': sum(s >= 4 for s in scores) / len(scores),
            'poor_or_worse_rate': sum(s <= 2 for s in scores) / len(scores),
        }

    return trait_metrics


def post_processing(test_data_with_rec):
    """
    Obtain the groundtruth by filtering resp titles
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


def evaluate_no_rerank(test_data_with_rec_filtered):
    """
    Evaluate the performance when NO reflect and rerank
    is imposed on the final recommendation list
    """
    avg_metrics_filtered = {}
    metrics_filtered = {}

    for K in K_list:
        print(f"Processing {K} retrievals")

        errors = set()
        results = {k:[] for k in k_list}

        for k in k_list:
            print(f"Processing top {k}")
            for i, item in enumerate(tqdm(test_data_with_rec_filtered,
                                     total=len(test_data_with_rec_filtered),
                                     desc="Evaluating via direct match...")):
                try:
                    results[k].append(evaluate_direct_match(item, k, K, gt_field="groundtruth"))
                except:
                    errors.add(i)

        recalls = {k:[res[0] for res in results[k]] for k in k_list}
        ndcgs = {k:[res[1] for res in results[k]] for k in k_list}

        metrics_filtered[K] = (recalls, ndcgs)

        avg_recalls_filtered = {k:np.mean(recalls[k]) for k in k_list}
        avg_ndcgs_filtered = {k:np.mean(ndcgs[k]) for k in k_list}
        avg_metrics_filtered[K] = (avg_recalls_filtered, avg_ndcgs_filtered)

        print(f"number of errors: {len(errors)}")

    metrics["No Reflect and Rerank"] = metrics_filtered
    avg_metrics["No Reflect and Rerank"] = avg_metrics_filtered

    for K in K_list:
        print(f"{K}-retrieval results:")
        avg_recalls_filtered, avg_ndcgs_filtered = avg_metrics_filtered[K]
        print("Results on the filtered dataset:")
        print(f"top-k recalls: {avg_recalls_filtered}")
        print(f"top-k ndcgs: {avg_ndcgs_filtered}\n")


def evaluate_with_rerank(test_data_with_rec_filtered):
    """
    Evaluate the performance when reflect and rerank
    is imposed on the final recommendation list
    """
    metrics_reflect_rerank = {}
    avg_metrics_reflect_rerank = {}

    for K in K_list:
        print(f"Processing {K} retrievals")

        errors = set()
        results = {k:[] for k in k_list}

        for k in k_list:
            print(f"Processing top {k}")
            for i, item in enumerate(tqdm(test_data_with_rec_filtered,
                                     total=len(test_data_with_rec_filtered),
                                     desc="Evaluating via direct match...")):
                try:
                    results[k].append(evaluate_direct_match_reflect_rerank(item, k, K, gt_field="groundtruth"))
                except:
                    errors.add(i)

        recalls = {k:[res[0] for res in results[k]] for k in k_list}
        ndcgs = {k:[res[1] for res in results[k]] for k in k_list}

        metrics_reflect_rerank[K] = (recalls, ndcgs)

        avg_recalls_reflect_rerank = {k:np.mean(recalls[k]) for k in k_list}
        avg_ndcgs_reflect_rerank = {k:np.mean(ndcgs[k]) for k in k_list}
        avg_metrics_reflect_rerank[K]= (avg_recalls_reflect_rerank, avg_ndcgs_reflect_rerank)

        print(f"number of errors: {len(errors)}")

    metrics["With Reflect and Rerank"] = metrics_reflect_rerank
    avg_metrics["With Reflect and Rerank"] = avg_metrics_reflect_rerank

    for K in K_list:
        print(f"{K}-retrieval results:")
        avg_recalls_reflect_rerank, avg_ndcgs_reflect_rerank = avg_metrics_reflect_rerank[K]
        print("Results on the filtered dataset:")
        print(f"top-K recalls: {avg_recalls_reflect_rerank}")
        print(f"top-K ndcgs: {avg_ndcgs_reflect_rerank}\n")


def print_comprehensive_results(test_data_with_rec, trait_metrics):
    """
    Print comprehensive results including traditional metrics and trait alignment
    """
    K_list_extended = [0] + K_list

    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80 + "\n")

    for K in K_list_extended:
        print(f"\n{'='*60}")
        print(f"Results for K={K} retrieval")
        print(f"{'='*60}")

        if K in avg_metrics.get("No Reflect and Rerank", {}):
            recalls, ndcgs = avg_metrics["No Reflect and Rerank"][K]
            print(f"\nRecall@k (no rerank): {recalls}")
            print(f"NDCG@k (no rerank): {ndcgs}")

        if K in avg_metrics.get("With Reflect and Rerank", {}):
            recalls, ndcgs = avg_metrics["With Reflect and Rerank"][K]
            print(f"\nRecall@k (with rerank): {recalls}")
            print(f"NDCG@k (with rerank): {ndcgs}")

        if K in trait_metrics:
            print(f"\nTrait Alignment Metrics:")
            tm = trait_metrics[K]
            print(f"  Mean alignment score: {tm['mean_alignment']:.3f} ± {tm['std_alignment']:.3f}")
            print(f"  Median alignment score: {tm['median_alignment']:.1f}")
            print(f"  Excellent rate (score=5): {tm['excellent_rate']*100:.1f}%")
            print(f"  Good or better rate (score>=4): {tm['good_or_better_rate']*100:.1f}%")
            print(f"  Poor or worse rate (score<=2): {tm['poor_or_worse_rate']*100:.1f}%")


def plot_trait_comparison(trait_metrics, test_data_with_rec, save_dir="results/trait_evaluation"):
    """
    Plot comprehensive comparison including trait alignment
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    K_values = [0] + K_list

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.3)

    mean_scores = [trait_metrics[K]['mean_alignment'] for K in K_values]
    std_scores = [trait_metrics[K]['std_alignment'] for K in K_values]

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(K_values))
    ax1.bar(x, mean_scores, yerr=std_scores, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Number of Retrieved Items (K)', fontsize=12)
    ax1.set_ylabel('Mean Trait Alignment Score (1-5)', fontsize=12)
    ax1.set_title('Trait Alignment vs. Retrieval Count', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'K={k}' for k in K_values])
    ax1.set_ylim(1, 5)
    ax1.axhline(y=3, color='r', linestyle='--', linewidth=2, label='Acceptable threshold')
    ax1.axhline(y=4, color='g', linestyle='--', linewidth=2, label='Good threshold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    score_distributions = {K: [0]*5 for K in K_values}
    for K in K_values:
        for item in test_data_with_rec:
            score = item.get(f"trait_alignment_score_{K}", 3)
            score_distributions[K][score-1] += 1

    for K in K_values:
        total = sum(score_distributions[K])
        if total > 0:
            score_distributions[K] = [count/total*100 for count in score_distributions[K]]

    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']
    labels = ['Score 1 (Very Poor)', 'Score 2 (Poor)', 'Score 3 (Acceptable)',
              'Score 4 (Good)', 'Score 5 (Excellent)']

    bottom = np.zeros(len(K_values))
    for score_idx in range(5):
        values = [score_distributions[K][score_idx] for K in K_values]
        ax2.bar(x, values, bottom=bottom, label=labels[score_idx],
                color=colors[score_idx], alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += values

    ax2.set_xlabel('Number of Retrieved Items (K)', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Distribution of Trait Alignment Scores', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'K={k}' for k in K_values])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    ax3 = fig.add_subplot(gs[2, 0])
    good_rates = [trait_metrics[K]['good_or_better_rate']*100 for K in K_values]
    excellent_rates = [trait_metrics[K]['excellent_rate']*100 for K in K_values]
    poor_rates = [trait_metrics[K]['poor_or_worse_rate']*100 for K in K_values]

    width = 0.25
    ax3.bar(x - width, excellent_rates, width, label='Excellent (5)', color='#1f77b4', alpha=0.8, edgecolor='black')
    ax3.bar(x, good_rates, width, label='Good+ (≥4)', color='#2ca02c', alpha=0.8, edgecolor='black')
    ax3.bar(x + width, poor_rates, width, label='Poor- (≤2)', color='#d62728', alpha=0.8, edgecolor='black')

    ax3.set_xlabel('Number of Retrieved Items (K)', fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Trait Alignment Quality Rates', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'K={K}' for K in K_values])
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "trait_evaluation_results.jpg"),
                format='jpg', bbox_inches='tight', dpi=300)
    print(f"\nPlot saved to: {os.path.join(save_dir, 'trait_evaluation_results.jpg')}")
    plt.close()


def plot_trait_by_category(test_data_with_rec, save_dir="results/trait_evaluation"):
    """
    Plot trait alignment by category
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    category_scores = defaultdict(list)

    K_ref = 20 if 20 in K_list else K_list[len(K_list)//2]

    for item in test_data_with_rec:
        category = item.get('assigned_trait_category', 'unknown')
        score = item.get(f'trait_alignment_score_{K_ref}', 3)
        category_scores[category].append(score)

    category_means = {cat: np.mean(scores) for cat, scores in category_scores.items()}
    sorted_categories = sorted(category_means.items(), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_categories) * 0.4)))
    categories = [cat.replace('_', ' ').title() for cat, _ in sorted_categories]
    means = [score for _, score in sorted_categories]

    y_pos = np.arange(len(categories))
    colors = ['#d62728' if m < 3 else '#ff7f0e' if m < 4 else '#2ca02c' for m in means]

    ax.barh(y_pos, means, alpha=0.8, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel('Mean Trait Alignment Score', fontsize=12)
    ax.set_title(f'Trait Alignment by Category (K={K_ref})', fontsize=14, fontweight='bold')
    ax.axvline(x=3, color='orange', linestyle='--', linewidth=2, label='Acceptable threshold')
    ax.axvline(x=4, color='green', linestyle='--', linewidth=2, label='Good threshold')
    ax.legend(fontsize=10)
    ax.set_xlim(1, 5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "trait_alignment_by_category.jpg"),
                format='jpg', bbox_inches='tight', dpi=300)
    print(f"Category plot saved to: {os.path.join(save_dir, 'trait_alignment_by_category.jpg')}")
    plt.close()


def save_detailed_log(test_data_with_rec, trait_metrics, save_dir="results/trait_evaluation"):
    """
    Save detailed evaluation log
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"evaluation_log_{timestamp}.txt")

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TRAIT-AWARE MOVIE RECOMMENDATION EVALUATION\n")
        f.write("="*80 + "\n\n")

        f.write(f"Evaluation Date: {datetime.datetime.now()}\n")
        f.write(f"Number of samples: {len(test_data_with_rec)}\n")
        f.write(f"Trait Model: {evaluation_trait_model}\n")
        f.write(f"K values: {K_list}\n")
        f.write(f"Random seed: {random_seed}\n\n")

        f.write("="*80 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*80 + "\n\n")

        for K in [0] + K_list:
            f.write(f"\n--- K={K} ---\n")
            if K in trait_metrics:
                tm = trait_metrics[K]
                f.write(f"Mean alignment: {tm['mean_alignment']:.3f} ± {tm['std_alignment']:.3f}\n")
                f.write(f"Median alignment: {tm['median_alignment']:.1f}\n")
                f.write(f"Excellent rate: {tm['excellent_rate']*100:.1f}%\n")
                f.write(f"Good+ rate: {tm['good_or_better_rate']*100:.1f}%\n")
                f.write(f"Poor- rate: {tm['poor_or_worse_rate']*100:.1f}%\n")

        f.write("\n\n" + "="*80 + "\n")
        f.write("SAMPLE-LEVEL DETAILS\n")
        f.write("="*80 + "\n\n")

        K_ref = 20 if 20 in K_list else K_list[len(K_list)//2]

        for i, item in enumerate(test_data_with_rec):
            f.write(f"\n{'='*60}\n")
            f.write(f"Sample {i+1}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Turn ID: {item.get('turn_id', 'N/A')}\n")
            f.write(f"Assigned Trait: {item.get('assigned_trait', 'N/A')}\n")
            f.write(f"Trait Category: {item.get('assigned_trait_category', 'N/A')}\n\n")

            # Add conversation content
            if 'context_raw' in item:
                f.write("Conversation:\n")
                f.write("-" * 60 + "\n")
                for turn in item['context_raw']:
                    speaker, content = turn
                    f.write(f"{speaker}: {content}\n")
                f.write("-" * 60 + "\n\n")

            if f"rec_after_reflect_{K_ref}" in item:
                f.write(f"Recommendations (K={K_ref}, after rerank):\n")
                for idx, movie in enumerate(item[f"rec_after_reflect_{K_ref}"][:10]):
                    f.write(f"  {idx+1}. {movie}\n")

            f.write(f"\nTrait Alignment Score: {item.get(f'trait_alignment_score_{K_ref}', 'N/A')}/5\n")
            f.write(f"Explanation: {item.get(f'trait_alignment_explanation_{K_ref}', 'N/A')}\n")

            if 'groundtruth' in item:
                f.write(f"\nGroundtruth (first 5): {item['groundtruth'][:5]}\n")

    print(f"Detailed log saved to: {log_file}")


def save_comprehensive_metrics(trait_metrics, save_dir="results/trait_evaluation"):
    """
    Save all metrics to JSON files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "trait_metrics.json"), 'w') as f:
        json.dump(trait_metrics, f, indent=2)

    comprehensive = {
        'trait_metrics': trait_metrics,
        'traditional_metrics': {
            'no_rerank': avg_metrics.get("No Reflect and Rerank", {}),
            'with_rerank': avg_metrics.get("With Reflect and Rerank", {})
        },
        'config': {
            'model': model,
            'n_samples': n_samples,
            'random_seed': random_seed,
            'K_list': K_list,
            'k_list': k_list,
            'temperature': temperature
        }
    }

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

    comprehensive = convert_to_serializable(comprehensive)

    with open(os.path.join(save_dir, "comprehensive_metrics.json"), 'w') as f:
        json.dump(comprehensive, f, indent=2)

    print(f"Metrics saved to: {save_dir}")


def main():
    """
    Main evaluation pipeline with trait-aware recommendations
    """
    print("\n" + "="*80)
    print("TRAIT-AWARE MOVIE RECOMMENDATION EVALUATION")
    print("="*80 + "\n")

    print("Step 1: Loading collaborative filtering model...")
    sim_mat, catalog_imdb_ids, imdb_id2row, imdb_id2col, col2imdb_id, id2name = load_and_process_cf_model()

    print("\nStep 2: Loading test data...")
    with open(from_pkl, "rb") as f:
        test_data = pickle.load(f)
    print(f"Loaded {len(test_data)} test samples")

    print("\nStep 3: Pre-processing data...")
    test_data_with_rec = pre_process(test_data, catalog_imdb_ids)
    print(f"After preprocessing: {len(test_data_with_rec)} samples")

    print("\nStep 4: Loading traits and sampling...")
    all_traits = load_traits('traits.json')
    test_data_with_rec = sample_test_data_with_traits(test_data_with_rec, all_traits, n_samples, random_seed)

    save_dir = "results/trait_evaluation"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "sampled_data_seed42.pkl"), "wb") as f:
        pickle.dump(test_data_with_rec, f)
    print(f"Sampled data saved!")

    print("\nStep 5: Context-aware retrieval with traits...")
    test_data_with_rec = context_aware_retrieval_with_trait(test_data_with_rec, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name)

    print("\nStep 6: Zero-shot recommendation with traits...")
    test_data_with_rec = recommend_zero_shot_with_trait(test_data_with_rec)

    print("\nStep 7: Recommendation with retrieval and traits...")
    test_data_with_rec = recommend_with_retrieval_with_trait(test_data_with_rec)

    print("\nStep 8: Reflect and rerank with traits...")
    test_data_with_rec = reflect_and_rerank_with_trait(test_data_with_rec)

    print("\nStep 9: Evaluating trait alignment...")
    test_data_with_rec = evaluate_trait_alignment(test_data_with_rec)

    print("\nStep 10: Post-processing...")
    test_data_with_rec = post_processing(test_data_with_rec)
    print(f"After post-processing: {len(test_data_with_rec)} samples with groundtruth")

    print("\nStep 11: Computing trait metrics...")
    trait_metrics = compute_trait_metrics(test_data_with_rec)

    print("\nStep 12: Evaluating recommendations (no rerank)...")
    evaluate_no_rerank(test_data_with_rec)

    print("\nStep 13: Evaluating recommendations (with rerank)...")
    evaluate_with_rerank(test_data_with_rec)

    print("\nStep 14: Printing comprehensive results...")
    print_comprehensive_results(test_data_with_rec, trait_metrics)

    print("\nStep 15: Generating visualizations...")
    plot_trait_comparison(trait_metrics, test_data_with_rec, save_dir)
    plot_trait_by_category(test_data_with_rec, save_dir)

    print("\nStep 16: Saving results...")
    with open(os.path.join(save_dir, "test_with_traits_final.pkl"), "wb") as f:
        pickle.dump(test_data_with_rec, f)

    save_comprehensive_metrics(trait_metrics, save_dir)
    save_detailed_log(test_data_with_rec, trait_metrics, save_dir)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {save_dir}/")
    print("\nGenerated files:")
    print("  - sampled_data_seed42.pkl")
    print("  - test_with_traits_retrieval.pkl")
    print("  - test_with_traits_rerank.pkl")
    print("  - test_with_traits_final.pkl")
    print("  - trait_metrics.json")
    print("  - comprehensive_metrics.json")
    print("  - trait_evaluation_results.jpg")
    print("  - trait_alignment_by_category.jpg")
    print("  - evaluation_log_<timestamp>.txt")


if __name__ == '__main__':
    main()
