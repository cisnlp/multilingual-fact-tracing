import pickle
import statistics
import random
import numpy as np
import os
import json
import glob
from collections import defaultdict
import re
import requests
import time


languages_12 = [
    "en",  # English
    "fr",  # French
    "es",  # Spanish
    "ar",  # Arabic
    "zh",  # Chinese
    "ru",  # Russian
    "ja",  # Japanese
    "tr",  # Turkish
    "uk",  # Ukrainian
    "ca",  # Catalan
    "ko",  # Korean
    "el",  # Greek
]

relation_list_12 = [
    "capital_of",
    "continent",
    "country_of_citizenship",
    "headquarters_location",
    "instrument",
    "language_of_work_or_name",
    "languages_spoken",
    "manufacturer",
    "native_language",
    "place_of_birth",
    "place_of_death",
    "religion",
]


def search_ngram_frequency(ngram_list=None):
    assert len(ngram_list) == 2
    subject, object = ngram_list
    payload = {
        'index': 'v4_dolma-v1_7_llama',
        'query_type': 'count',
        'query': f"{subject} AND {object}",
        'max_diff_tokens': 1000,
        'max_clause_freq': 50000
    }

    attempts = 0
    dolma_document_num = None

    while attempts < 10:
        try:
            response = requests.post('https://api.infini-gram.io/', json=payload)
            result = response.json()

            if 'count' in result:
                dolma_document_num = result['count']
                break
            else:
                attempts += 1
                sleep_time = 1 + attempts * 0.5  # increasing sleep time after each failure
                time.sleep(sleep_time)
        except Exception as e:
            print(f"Error on attempt {attempts+1}: {e}")
            attempts += 1
            sleep_time = 1 + attempts * 0.5
            time.sleep(sleep_time)

    if dolma_document_num is None:
        raise Exception("Failed to retrieve count after 10 attempts.")

    return dolma_document_num



def main():
    
    # ====  Group file paths by relation ====
    json_paths = glob.glob("../klar_source/*/*.json")
    path_map = defaultdict(dict)  # (relation -> lang -> path)

    for path in json_paths:
        lang = os.path.basename(os.path.dirname(path))
        rel = os.path.splitext(os.path.basename(path))[0]
        if lang in languages_12 and rel in relation_list_12:
            path_map[rel][lang] = path
    
    # initialize or load existing samples
    output_path = "./lang_rel_frequencies_infini.json"
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f_in:
            samples = json.load(f_in)
        print("Loaded existing intermediate results.")
    else:
        # Initialize fresh
        samples = {lang: {rel: [] for rel in relation_list_12} for lang in languages_12}
        print("Starting fresh.")
        
    for rel, lang_paths in path_map.items():
        print(f"Processing relation {rel} .....")
        print()
        for lang in languages_12:
            print()
            print(f"Processing language {lang} ......")
            print()

            with open(lang_paths[lang], "r", encoding="utf-8") as f:
                content = json.load(f)
                loaded_samples = content["samples"]

                # get already processed indices to skip
                processed_indices = set(
                    sample["index"] for sample in samples.get(lang, {}).get(rel, [])
                )

                for sample in loaded_samples:
                    if sample["index"] in processed_indices:
                        continue  # skip if already processed

                    # check frequency
                    frequency = search_ngram_frequency([sample['subject'], sample['object']])
                    print(f"{sample['subject']} -- {sample['object']}:  {frequency}")

                    new_sample = {
                        "subject": sample["subject"],
                        "object": sample["object"],
                        "index": sample["index"],
                        "frequency": frequency
                    }

                    samples[lang][rel].append(new_sample)

                    # Save after every sample
                    with open(output_path, "w", encoding="utf-8") as f_out:
                        json.dump(samples, f_out, ensure_ascii=False, indent=2)

    print(f"All samples successfully saved to {output_path}")

if __name__ == "__main__":
    main()


# 22637 lines