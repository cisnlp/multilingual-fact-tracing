from wimbd.es import es_init, count_documents_containing_phrases
import pickle
import statistics
import random
import numpy as np
import os
import json
import glob
from collections import defaultdict
import re


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


def init_wimbd():
    path = '/mounts/data/proj/yihong/CIS-relation-neuron-zeroshot/analysis_scripts/es_config_8.yml'
    dolma_path = '/mounts/data/proj/yihong/CIS-relation-neuron-zeroshot/analysis_scripts/es_config_dolma_1_7_2.yml'

    es_config_8 = es_init(config=path, timeout=180)
    es_dolma = es_init(config=dolma_path, timeout=180)
    
    return es_config_8, es_dolma


def search_ngram_frequency(ngram_list, dataset=None):
    es_config_8, es_dolma = init_wimbd()
    
    dolma_document_num = count_documents_containing_phrases("docs_v1.7_2024-06-04", ngram_list, all_phrases=True, es=es_dolma)
    
    return dolma_document_num
    # c4_ngram_num = count_documents_containing_phrases("c4", ngram, es=es_config_8)
    # oscar_ngram_num = count_documents_containing_phrases("re_oscar", ngram, es=es_config_8)
    # openwebtext_ngram_num = count_documents_containing_phrases("openwebtext", ngram, es=es_config_8)
    
    # result = {'dolma': dolma_document_num, 'c4': c4_ngram_num,
    #         'oscar': oscar_ngram_num, 'openwebtext': openwebtext_ngram_num}
    # if dataset:
    #     return result[dataset]
    # return {'dolma': dolma_document_num, 'c4': c4_ngram_num,
    #         'oscar': oscar_ngram_num, 'openwebtext': openwebtext_ngram_num}

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
    output_path = "./lang_rel_frequencies.json"
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


# dolma_document_num = count_documents_containing_phrases("docs_v1.7_2024-06-04", "Artificial Intelligence", es=es_dolma)

# c4_ngram_num = count_documents_containing_phrases("c4", "Artificial Intelligence", es=es_config_8)
# oscar_ngram_num = count_documents_containing_phrases("re_oscar", "Artificial Intelligence", es=es_config_8)
# openwebtext_ngram_num = count_documents_containing_phrases("openwebtext", "Artificial Intelligence", es=es_config_8)

# print(f"dolma: {dolma_document_num}")
# print(f"c4: {c4_ngram_num}")
# print(f"oscar: {oscar_ngram_num}")
# print(f"openwebtext: {openwebtext_ngram_num}")
