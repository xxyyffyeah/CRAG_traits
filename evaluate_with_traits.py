import os
import time
import json
import pickle
import random
import threading
import datetime
import multiprocessing
import re
import shutil

from tqdm import tqdm
from pprint import pprint
from copy import deepcopy
from functools import partial
from collections import defaultdict

import openai
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt  # Removed: No longer generating plots
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

prompt_semantic_trait_matching = (
    "You are an expert at matching user conversations to personal traits for movie recommendations.\n\n"
    "TASK: Analyze the following user conversation and determine which ONE trait from the provided list "
    "best matches the user's preferences. Identify which specific prefer/avoid themes match the conversation.\n\n"
    "USER CONVERSATION:\n{conversation}\n\n"
    "AVAILABLE TRAITS (130 total):\n{traits_list}\n\n"
    "INSTRUCTIONS:\n"
    "1. Read the user's conversation carefully to understand their movie preferences\n"
    "2. Choose the SINGLE best matching trait from the list\n"
    "3. Identify which specific 'Prefer' themes from that trait match what the user is looking for\n"
    "4. Identify which 'Avoid' themes the user mentions wanting to avoid (if any are mentioned in conversation)\n"
    "5. Provide detailed reasoning for each matched prefer/avoid item\n\n"
    "OUTPUT FORMAT (JSON only, no extra text):\n"
    "{{\n"
    '  "matched_trait_id": "trait_name",\n'
    '  "confidence": 0.85,\n'
    '  "matched_prefer_items": ["Prefer theme 1", "Prefer theme 2"],\n'
    '  "prefer_reasoning": {{\n'
    '    "Prefer theme 1": "User explicitly requested X which aligns with this theme",\n'
    '    "Prefer theme 2": "Conversation mentions Y which relates to this preference"\n'
    '  }},\n'
    '  "matched_avoid_items": ["Avoid theme 1"],\n'
    '  "avoid_reasoning": {{\n'
    '    "Avoid theme 1": "User wants to avoid Z"\n'
    '  }},\n'
    '  "overall_reasoning": "Summary of why this trait is the best match overall"\n'
    "}}\n\n"
    "IMPORTANT: \n"
    "- matched_trait_id must exactly match a trait ID from the list\n"
    "- matched_prefer_items should list prefer themes from the selected trait that match the conversation\n"
    "- matched_avoid_items should only include avoid themes if the user explicitly mentions avoiding them\n"
    "- If no avoid themes are mentioned, use an empty list: []\n"
    "- Return ONLY valid JSON\n"
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


def load_traits_full(traits_file='traits.json'):
    """
    Load traits.json with complete information including avoid/prefer lists

    Returns:
        list: List of trait dictionaries with all fields (category, trait, description, avoid, prefer)
    """
    with open(traits_file, 'r') as f:
        traits_data = json.load(f)

    all_traits = []
    for category, trait_list in traits_data.items():
        for trait_obj in trait_list:
            all_traits.append({
                'category': category,
                'trait': trait_obj['trait'],
                'description': trait_obj['description'],
                'avoid': trait_obj.get('avoid', []),
                'prefer': trait_obj.get('prefer', [])
            })

    print(f"Loaded {len(all_traits)} traits with full semantic information from {len(traits_data)} categories")
    return all_traits


def sample_test_data_with_traits(test_data, all_traits, n_samples=100, random_seed=42):
    """
    Sample n_samples from test_data and semantically match the best trait to each

    Args:
        test_data: Full test dataset
        all_traits: List of trait dictionaries (not used, kept for backward compatibility)
        n_samples: Number of samples to select
        random_seed: Random seed for reproducibility

    Returns:
        list: Sampled data with semantically matched traits
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    total_samples = len(test_data)
    print(f"Sampling {n_samples} from {total_samples} total samples...")

    # KEEP EXISTING: Random sampling of test items
    sampled_indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)
    sampled_data = [deepcopy(test_data[i]) for i in sampled_indices]

    print(f"Sampled {len(sampled_data)} items")

    # NEW: Load full trait information with avoid/prefer lists
    print("\nLoading complete trait information for semantic matching...")
    all_traits_full = load_traits_full('traits.json')

    # NEW: Semantic matching instead of random assignment
    sampled_data = semantic_match_traits(
        sampled_data,
        all_traits_full,
        model='gpt-5.1',
        temperature=0.3,  # Lower temp for more deterministic matching
        max_tokens=300,
        n_threads=500
    )

    return sampled_data


def format_traits_for_prompt(all_traits):
    """
    Format all traits into a compact, readable format for the GPT-5.1 prompt

    Args:
        all_traits: List of trait dictionaries with full information

    Returns:
        str: Formatted traits list
    """
    lines = []
    for i, trait in enumerate(all_traits, 1):
        # Take first 3 items from avoid/prefer to save tokens
        avoid_str = ", ".join(trait['avoid'][:3]) if trait['avoid'] else "N/A"
        prefer_str = ", ".join(trait['prefer'][:3]) if trait['prefer'] else "N/A"

        lines.append(
            f"{i}. [{trait['trait']}] {trait['description']} "
            f"| Avoid: {avoid_str}... | Prefer: {prefer_str}..."
        )

    return "\n".join(lines)


def semantic_match_traits(sampled_data, all_traits_full, model='gpt-5.1',
                         temperature=0.3, max_tokens=300, n_threads=500):
    """
    Use GPT-5.1 to semantically match each sampled conversation to the best trait

    Args:
        sampled_data: List of sampled test items
        all_traits_full: List of all traits with complete information
        model: Model to use for matching (default: gpt-5.1)
        temperature: Temperature for API calls (lower = more deterministic)
        max_tokens: Max tokens for response
        n_threads: Number of concurrent threads

    Returns:
        list: sampled_data with semantically matched traits assigned
    """
    print(f"\n{'='*60}")
    print("SEMANTIC TRAIT MATCHING WITH GPT-5.1")
    print(f"{'='*60}")
    print(f"Matching {len(sampled_data)} conversations to {len(all_traits_full)} traits...")
    print(f"Using model: {model}")
    print(f"Expected API calls: {len(sampled_data)}")

    # Format traits list once (same for all conversations)
    traits_formatted = format_traits_for_prompt(all_traits_full)

    # Create trait lookup by trait name for fast assignment
    trait_lookup = {t['trait']: t for t in all_traits_full}

    # Threading setup (following existing pattern from lines 332-350)
    EXSTING = {}
    threads, results = [], []

    for i, item in enumerate(tqdm(sampled_data,
                                  total=len(sampled_data),
                                  desc="Semantic trait matching...")):
        # Format conversation (following pattern from line 306)
        conversation = "\n".join([": ".join(rnd) for rnd in item['context_raw']])

        # Prepare input for prompt
        input_text = {
            "conversation": conversation,
            "traits_list": traits_formatted
        }

        # Create thread for API call (following pattern from lines 332-334)
        execute_thread = threading.Thread(
            target=get_response,
            args=(i, input_text, prompt_semantic_trait_matching,
                  model, temperature, max_tokens, results, EXSTING)
        )

        time.sleep(0.02)  # Rate limiting
        threads.append(execute_thread)
        execute_thread.start()

        # Batch processing (following pattern from lines 340-350)
        if len(threads) == n_threads:
            for execute_thread in threads:
                execute_thread.join()

            # Process results batch
            for res in results:
                index = res["index"]
                sampled_data[index]["semantic_trait_match_raw"] = res

            threads = []
            results = []
            time.sleep(0)

    # Process remaining threads (following pattern from lines 352-358)
    if len(threads) > 0:
        for execute_thread in threads:
            execute_thread.join()

        for res in results:
            index = res["index"]
            sampled_data[index]["semantic_trait_match_raw"] = res

    # Parse results and assign traits (following pattern from lines 639-668)
    print("\nParsing semantic matching results...")
    successful_matches = 0
    fallback_matches = 0
    random_fallbacks = 0

    for i, item in enumerate(sampled_data):
        try:
            # Extract response
            raw_response = item["semantic_trait_match_raw"]['resp']['choices'][0]['message']['content']

            # Parse JSON (primary method)
            try:
                # Clean response - remove markdown code blocks if present
                cleaned = raw_response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                match_data = json.loads(cleaned)
                matched_trait_id = match_data.get('matched_trait_id', '').strip()
                confidence = match_data.get('confidence', 0.0)

                # NEW: Extract prefer/avoid items and reasoning
                matched_prefer_items = match_data.get('matched_prefer_items', [])
                prefer_reasoning = match_data.get('prefer_reasoning', {})
                matched_avoid_items = match_data.get('matched_avoid_items', [])
                avoid_reasoning = match_data.get('avoid_reasoning', {})
                overall_reasoning = match_data.get('overall_reasoning', match_data.get('reasoning', ''))

                # Validate trait exists
                if matched_trait_id in trait_lookup:
                    matched_trait = trait_lookup[matched_trait_id]
                    item['assigned_trait'] = matched_trait['description']
                    item['assigned_trait_name'] = matched_trait['trait']
                    item['assigned_trait_category'] = matched_trait['category']
                    item['semantic_match_confidence'] = confidence
                    item['semantic_match_reasoning'] = overall_reasoning
                    item['semantic_match_method'] = 'json_parse'
                    # NEW: Store prefer/avoid details
                    item['matched_prefer_items'] = matched_prefer_items
                    item['prefer_reasoning'] = prefer_reasoning
                    item['matched_avoid_items'] = matched_avoid_items
                    item['avoid_reasoning'] = avoid_reasoning
                    successful_matches += 1
                else:
                    # Trait ID not found - use fallback
                    raise ValueError(f"Invalid trait ID: {matched_trait_id}")

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Fallback: try to extract trait name from text
                print(f"  Warning: JSON parse failed for item {i}, trying fallback...")

                # Search for any trait name in the response
                found_trait = None
                for trait_name in trait_lookup.keys():
                    if trait_name.lower() in raw_response.lower():
                        found_trait = trait_name
                        break

                if found_trait and found_trait in trait_lookup:
                    matched_trait = trait_lookup[found_trait]
                    item['assigned_trait'] = matched_trait['description']
                    item['assigned_trait_name'] = matched_trait['trait']
                    item['assigned_trait_category'] = matched_trait['category']
                    item['semantic_match_confidence'] = 0.5  # Lower confidence for fallback
                    item['semantic_match_reasoning'] = f"Fallback match from response text - detailed analysis unavailable"
                    item['semantic_match_method'] = 'text_search'
                    # NEW: Set empty lists for prefer/avoid when fallback is used
                    item['matched_prefer_items'] = []
                    item['prefer_reasoning'] = {}
                    item['matched_avoid_items'] = []
                    item['avoid_reasoning'] = {}
                    fallback_matches += 1
                else:
                    # Ultimate fallback: random assignment
                    raise ValueError("No trait found in response")

        except Exception as e:
            # Ultimate fallback: assign random trait (following existing pattern from line 193)
            print(f"  Error for item {i}: {e}. Using random fallback.")
            random_trait = random.choice(all_traits_full)
            item['assigned_trait'] = random_trait['description']
            item['assigned_trait_name'] = random_trait['trait']
            item['assigned_trait_category'] = random_trait['category']
            item['semantic_match_confidence'] = 0.0
            item['semantic_match_reasoning'] = f"Random fallback due to parsing error"
            item['semantic_match_method'] = 'random_fallback'
            # NEW: Set empty lists for prefer/avoid when random fallback is used
            item['matched_prefer_items'] = []
            item['prefer_reasoning'] = {}
            item['matched_avoid_items'] = []
            item['avoid_reasoning'] = {}
            random_fallbacks += 1

    print(f"\nMatching Results:")
    print(f"  Successful JSON matches: {successful_matches}/{len(sampled_data)}")
    print(f"  Fallback text matches: {fallback_matches}/{len(sampled_data)}")
    print(f"  Random fallbacks: {random_fallbacks}/{len(sampled_data)}")

    # Print trait distribution (following pattern from lines 198-206)
    trait_counts = defaultdict(int)
    category_counts = defaultdict(int)
    for item in sampled_data:
        trait_counts[item['assigned_trait_name']] += 1
        category_counts[item['assigned_trait_category']] += 1

    print(f"\nTrait Distribution After Semantic Matching:")
    print(f"  Unique traits used: {len(trait_counts)}/{len(all_traits_full)}")
    print(f"  Categories represented: {len(category_counts)}")
    print(f"  Most common traits: {sorted(trait_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")

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


def recommend_with_retrieval_with_trait(test_data_with_rec, save_dir):
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

    save_file = os.path.join(save_dir, "test_with_traits_retrieval.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(test_data_with_rec, f)

    print(f"results saved to: {save_file}!")

    return test_data_with_rec


def reflect_and_rerank_with_trait(test_data_with_rec, save_dir):
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

    save_file = os.path.join(save_dir, "test_with_traits_rerank.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(test_data_with_rec, f)

    print(f"results saved to: {save_file}!")

    return test_data_with_rec


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


def print_comprehensive_results(test_data_with_rec):
    """
    Print comprehensive results including traditional metrics
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


def save_detailed_log(test_data_with_rec, save_dir="results"):
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
        f.write(f"Model: {model}\n")
        f.write(f"K values: {K_list}\n")
        f.write(f"Random seed: {random_seed}\n\n")

        f.write("="*80 + "\n")
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

            # NEW: Add semantic matching details
            f.write("--- Semantic Matching Details ---\n")
            f.write(f"Matching Method: {item.get('semantic_match_method', 'N/A')}\n")
            f.write(f"Confidence Score: {item.get('semantic_match_confidence', 'N/A')}\n\n")

            # Matched prefer items
            if 'matched_prefer_items' in item and item['matched_prefer_items']:
                f.write(f"Matched PREFER Themes:\n")
                prefer_reasoning = item.get('prefer_reasoning', {})
                for pref_item in item['matched_prefer_items']:
                    reason = prefer_reasoning.get(pref_item, 'No reason provided')
                    f.write(f"  • {pref_item}\n")
                    f.write(f"    Reason: {reason}\n")
                f.write("\n")

            # Matched avoid items
            if 'matched_avoid_items' in item and item['matched_avoid_items']:
                f.write(f"Matched AVOID Themes:\n")
                avoid_reasoning = item.get('avoid_reasoning', {})
                for avoid_item in item['matched_avoid_items']:
                    reason = avoid_reasoning.get(avoid_item, 'No reason provided')
                    f.write(f"  • {avoid_item}\n")
                    f.write(f"    Reason: {reason}\n")
                f.write("\n")

            # Overall reasoning
            if 'semantic_match_reasoning' in item and item['semantic_match_reasoning']:
                f.write(f"Overall Matching Reasoning:\n")
                f.write(f"  {item['semantic_match_reasoning']}\n\n")

            if f"rec_after_reflect_{K_ref}" in item:
                f.write(f"Recommendations (K={K_ref}, after rerank):\n")
                for idx, movie in enumerate(item[f"rec_after_reflect_{K_ref}"][:10]):
                    f.write(f"  {idx+1}. {movie}\n")
                f.write("\n")

            if 'groundtruth' in item:
                f.write(f"\nGroundtruth (first 5): {item['groundtruth'][:5]}\n")

    print(f"Detailed log saved to: {log_file}")


def save_comprehensive_metrics(save_dir="results"):
    """
    Save all metrics to JSON files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    comprehensive = {
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

    # Create timestamped directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = "results"
    timestamped_dir = os.path.join(base_save_dir, timestamp)
    last_dir = os.path.join(base_save_dir, "last")

    # Create directories
    os.makedirs(timestamped_dir, exist_ok=True)

    # Update or create 'last' directory
    if os.path.islink(last_dir):
        os.unlink(last_dir)
    elif os.path.exists(last_dir):
        shutil.rmtree(last_dir)

    # Create symbolic link to timestamped directory
    try:
        os.symlink(timestamp, last_dir, target_is_directory=True)
        print(f"Created timestamped directory: {timestamped_dir}")
        print(f"'last' symlink points to: {timestamp}\n")
    except OSError:
        # If symlink creation fails (e.g., on Windows), copy instead
        shutil.copytree(timestamped_dir, last_dir)
        print(f"Created timestamped directory: {timestamped_dir}")
        print(f"'last' directory will be updated at the end\n")

    save_dir = timestamped_dir

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

    with open(os.path.join(save_dir, "sampled_data_seed42.pkl"), "wb") as f:
        pickle.dump(test_data_with_rec, f)
    print(f"Sampled data saved!")

    print("\nStep 5: Context-aware retrieval with traits...")
    test_data_with_rec = context_aware_retrieval_with_trait(test_data_with_rec, sim_mat, imdb_id2row, imdb_id2col, col2imdb_id, id2name)

    print("\nStep 6: Zero-shot recommendation with traits...")
    test_data_with_rec = recommend_zero_shot_with_trait(test_data_with_rec)

    print("\nStep 7: Recommendation with retrieval and traits...")
    test_data_with_rec = recommend_with_retrieval_with_trait(test_data_with_rec, save_dir)

    print("\nStep 8: Reflect and rerank with traits...")
    test_data_with_rec = reflect_and_rerank_with_trait(test_data_with_rec, save_dir)

    print("\nStep 9: Post-processing...")
    test_data_with_rec = post_processing(test_data_with_rec)
    print(f"After post-processing: {len(test_data_with_rec)} samples with groundtruth")

    print("\nStep 10: Evaluating recommendations (no rerank)...")
    evaluate_no_rerank(test_data_with_rec)

    print("\nStep 11: Evaluating recommendations (with rerank)...")
    evaluate_with_rerank(test_data_with_rec)

    print("\nStep 12: Printing comprehensive results...")
    print_comprehensive_results(test_data_with_rec)

    print("\nStep 13: Saving results...")
    with open(os.path.join(save_dir, "test_with_traits_final.pkl"), "wb") as f:
        pickle.dump(test_data_with_rec, f)

    save_comprehensive_metrics(save_dir)
    save_detailed_log(test_data_with_rec, save_dir)

    # Update 'last' directory if not using symlink
    if not os.path.islink(last_dir):
        if os.path.exists(last_dir):
            shutil.rmtree(last_dir)
        shutil.copytree(timestamped_dir, last_dir)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - Timestamped: {timestamped_dir}/")
    print(f"  - Latest: {last_dir}/")
    print("\nGenerated files:")
    print("  - sampled_data_seed42.pkl")
    print("  - test_with_traits_retrieval.pkl")
    print("  - test_with_traits_rerank.pkl")
    print("  - test_with_traits_final.pkl")
    print("  - comprehensive_metrics.json")
    print("  - evaluation_log_<timestamp>.txt")


if __name__ == '__main__':
    main()
