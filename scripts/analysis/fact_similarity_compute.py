import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import re


# Extract step and associate with original line
def extract_step(line, step_interval=1000):
    match = re.search(r"step[_-]?(\d+)", line)
    # print(int(match.group(1)))
    if match:
        step = int(match.group(1))
        return step if step % step_interval == 0 else -1
    return -1


def get_checkpoint_lists(step1, step2, multiple_of=None):
    input_file = "/mounts/data/proj/yihong/alignment-consistency/olmo_revision.txt"
    with open(input_file, "r") as f:
        lines = f.readlines()

    parsed = [(extract_step(line), line.strip()) for line in lines if extract_step(line) != -1]
    
    # Sort by step
    sorted_lines = sorted(parsed, key=lambda x: x[0])
    
    checkpoints = []
    for step, line in sorted_lines:
        if step >= step1 and step <= step2:
            if multiple_of is None or step % multiple_of == 0:
                checkpoints.append(line)
    
    return checkpoints


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

language_info = {
    "en": {"iso639_3": "eng", "script": "Latn"},  # English
    "fr": {"iso639_3": "fra", "script": "Latn"},  # French
    "es": {"iso639_3": "spa", "script": "Latn"},  # Spanish
    "ar": {"iso639_3": "ara", "script": "Arab"},  # Arabic
    "zh": {"iso639_3": "zho", "script": "Hans"},  # Chinese (Simplified)
    "ru": {"iso639_3": "rus", "script": "Cyrl"},  # Russian
    "ja": {"iso639_3": "jpn", "script": "Jpan"},  # Japanese (mixed: Kanji+Kana)
    "tr": {"iso639_3": "tur", "script": "Latn"},  # Turkish
    "uk": {"iso639_3": "ukr", "script": "Cyrl"},  # Ukrainian
    "ca": {"iso639_3": "cat", "script": "Latn"},  # Catalan
    "ko": {"iso639_3": "kor", "script": "Kore"},  # Korean (Hangul + Hanja)
    "el": {"iso639_3": "ell", "script": "Grek"},  # Greek (Modern)
}

# Create the inverse dictionary
inverse_language_info = {
    f"{info['iso639_3']}_{info['script']}": lang_code
    for lang_code, info in language_info.items()
}

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

# load the indeces for the parallel data used to computing sentence representations
indeces_path = "../parallel_data_index.json"
with open(indeces_path, "r", encoding="utf-8") as f_in:
    data = json.load(f_in)

indeces_list = []
for i, (relation, indeces) in enumerate(data.items()):
    assert relation == relation_list_12[i]
    indeces_list += indeces

# load the indeces for fn facts for each language
data_path = "../fact_frequency/analysis_results/crosslingual_fn_freqs.json"
with open(data_path, "r", encoding="utf-8") as f_in:
    data = json.load(f_in)
fn_indeces_per_language = {l: set(data[l].keys()) for l in data.keys()}
    
# load the indeces for each language (all indeces, indeces of true negative)
data_path = "../fact_frequency/analysis_results/index_classification_by_language.json"
with open(data_path, "r", encoding="utf-8") as f_in:
    data = json.load(f_in)
all_indeces_per_language = {l: set(data[l]['all_indices']) for l in data.keys()}
tn_indeces_per_language = {l: set(data[l]['true_negative_indices']) for l in data.keys()}

# maybe also true positives and false positives
tp_indeces_per_language = {l: set(data[l]['true_positive_indices']) for l in data.keys()}
fp_indeces_per_language = {l: set(data[l]['false_positive_indices']) for l in data.keys()}

# load the indeces for facts that have exactly the same object
data_path = "../fact_frequency/analysis_results/identical_object_indices.json"
with open(data_path, "r", encoding="utf-8") as f_in:
    data = json.load(f_in)
identical_object_indeces_per_language = {l: set(data[l]['identical_object_indices']) for l in data.keys()}

# Map FN indices to their positions in indeces_list
# idx is the fact id while i is the absolute index in the list
index_mapping = {str(idx): i for i, idx in enumerate(indeces_list)}

# all
mapped_indices_per_language = {}
for lang, indices in all_indeces_per_language.items():
    if lang == 'en':
        continue
    mapped_indices = [index_mapping[str(idx)] for idx in indices if str(idx) in index_mapping]
    mapped_indices_per_language[lang] = mapped_indices

#fn
mapped_indices_per_language_fn = {}
for lang, fn_indices in fn_indeces_per_language.items():
    if lang == 'en':
        continue
    mapped_indices = [index_mapping[str(idx)] for idx in fn_indices if str(idx) in index_mapping]
    mapped_indices_per_language_fn[lang] = mapped_indices
    
#tn
mapped_indices_per_language_tn = {}
for lang, tn_indices in tn_indeces_per_language.items():
    if lang == 'en':
        continue
    mapped_indices = [index_mapping[str(idx)] for idx in tn_indices if str(idx) in index_mapping]
    mapped_indices_per_language_tn[lang] = mapped_indices
    
#fp
mapped_indices_per_language_fp = {}
for lang, fp_indices in fp_indeces_per_language.items():
    if lang == 'en':
        continue
    mapped_indices = [index_mapping[str(idx)] for idx in fp_indices if str(idx) in index_mapping]
    mapped_indices_per_language_fp[lang] = mapped_indices
    
#tp
mapped_indices_per_language_tp = {}
for lang, tp_indices in tp_indeces_per_language.items():
    if lang == 'en':
        continue
    mapped_indices = [index_mapping[str(idx)] for idx in tp_indices if str(idx) in index_mapping]
    mapped_indices_per_language_tp[lang] = mapped_indices
    
    
#same_object
mapped_indices_per_language_same_object = {}
for lang, same_object_indices in identical_object_indeces_per_language.items():
    if lang == 'en':
        continue
    mapped_indices = [index_mapping[str(idx)] for idx in same_object_indices if str(idx) in index_mapping]
    mapped_indices_per_language_same_object[lang] = mapped_indices


def compute_average_pairwise_cosine_similarity(lang_embd, embedding_type='embd_weighted', num_sents=1500, lang='None', compute_which="fn"):
    if lang is None:
        raise ValueError("You have to specify a language to compute similarity")
    similarities_dict = {}
    
    # if also consder the facts where there are the same objects
    # for layer in lang_embd.keys():
    #     # removing the indices of the same object
    #     if compute_which == "fn":
    #         # compute the similarity of FN facts
    #         pivot_vectors = [pivot_embd[layer][i][embedding_type] for i in mapped_indices_per_language_fn[lang]]
    #         lang_vectors = [lang_embd[layer][i][embedding_type] for i in mapped_indices_per_language_fn[lang]]
    #     elif compute_which == "tn":
    #         # compute the similarity of TN facts
    #         pivot_vectors = [pivot_embd[layer][i][embedding_type] for i in mapped_indices_per_language_tn[lang]]
    #         lang_vectors = [lang_embd[layer][i][embedding_type] for i in mapped_indices_per_language_tn[lang]]
    #     elif compute_which == "all":
    #         # compute the similarity of all facts
    #         pivot_vectors = [pivot_embd[layer][i][embedding_type] for i in mapped_indices_per_language[lang]]
    #         lang_vectors = [lang_embd[layer][i][embedding_type] for i in mapped_indices_per_language[lang]]
    #     else:
    #         raise ValueError("Compute similarity between other facts are not supported.")
    
    for layer in lang_embd.keys():
        # removing the indices of the same object
        if compute_which == "fn":
            # compute the similarity of FN facts
            pivot_vectors = [pivot_embd[layer][i][embedding_type] for i in mapped_indices_per_language_fn[lang] if i not in mapped_indices_per_language_same_object[lang]]
            lang_vectors = [lang_embd[layer][i][embedding_type] for i in mapped_indices_per_language_fn[lang] if i not in mapped_indices_per_language_same_object[lang]]
        elif compute_which == "tn":
            # compute the similarity of TN facts
            pivot_vectors = [pivot_embd[layer][i][embedding_type] for i in mapped_indices_per_language_tn[lang] if i not in mapped_indices_per_language_same_object[lang]]
            lang_vectors = [lang_embd[layer][i][embedding_type] for i in mapped_indices_per_language_tn[lang] if i not in mapped_indices_per_language_same_object[lang]]
        elif compute_which == "tp":
            # compute the similarity of TP facts
            pivot_vectors = [pivot_embd[layer][i][embedding_type] for i in mapped_indices_per_language_tp[lang] if i not in mapped_indices_per_language_same_object[lang]]
            lang_vectors = [lang_embd[layer][i][embedding_type] for i in mapped_indices_per_language_tp[lang] if i not in mapped_indices_per_language_same_object[lang]]
        elif compute_which == "fp":
            # compute the similarity of FP facts
            pivot_vectors = [pivot_embd[layer][i][embedding_type] for i in mapped_indices_per_language_fp[lang] if i not in mapped_indices_per_language_same_object[lang]]
            lang_vectors = [lang_embd[layer][i][embedding_type] for i in mapped_indices_per_language_fp[lang] if i not in mapped_indices_per_language_same_object[lang]]
        elif compute_which == "all":
            # compute the similarity of all facts
            pivot_vectors = [pivot_embd[layer][i][embedding_type] for i in mapped_indices_per_language[lang] if i not in mapped_indices_per_language_same_object[lang]]
            lang_vectors = [lang_embd[layer][i][embedding_type] for i in mapped_indices_per_language[lang] if i not in mapped_indices_per_language_same_object[lang]]
        elif compute_which == "other":
            # compute the similarity of facts other than FN facts
            # compute the similarity of non-FN facts
            total_indices = set(range(len(pivot_embd[layer])))
            fn_indices = set(mapped_indices_per_language_fn[lang])
            non_fn_indices = sorted(total_indices - fn_indices)
            pivot_vectors = [pivot_embd[layer][i][embedding_type] for i in non_fn_indices if i not in mapped_indices_per_language_same_object[lang]]
            lang_vectors = [lang_embd[layer][i][embedding_type] for i in non_fn_indices if i not in mapped_indices_per_language_same_object[lang]]
        else:
            raise ValueError("Compute similarity between other facts are not supported.")
            
        # Ensure length alignment
        min_len = min(len(pivot_vectors), len(lang_vectors))
        assert len(pivot_vectors) == len(lang_vectors)
        
        print(min_len)
        
        if len(pivot_vectors) == 0:
            for l in lang_embd.keys():
                similarities_dict[l] = 0.0
            return similarities_dict
        
        pivot_matrix = np.array(pivot_vectors[:min_len])
        lang_matrix = np.array(lang_vectors[:min_len])

        # Compute cosine similarity matrix
        sim_matrix = sklearn_cosine_similarity(pivot_matrix, lang_matrix)
        
        # Compute average of pairwise similarities
        diagonal = np.diag(sim_matrix)
        diagonal_clipped = np.clip(diagonal, 0.0, 1.0)
        similarities_dict[layer] = float(np.mean(diagonal_clipped))

    return similarities_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process embeddings and compute alignments.')
    
    parser.add_argument('--pivot', type=str, default='eng_Latn', help='Pivot language code (default: eng_Latn)')
    parser.add_argument('--file_ext', type=str, default='.pkl', help='File extension for embedding files (default: .pkl)')
    parser.add_argument('--embedding_path', type=str, required=True, help='Path to the directory containing embedding files.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the results.')
    parser.add_argument('--num_sents', type=int, default=2000, help='Maximum number of sentences to process (default: 100)')
    parser.add_argument('--embedding_type', type=str, choices=['embd_weighted', 'embd_lasttoken'], default='embd_weighted', help='Type of embedding to use (default: embd_weighted)')
    # parser.add_argument('--revision', type=str, default=None, help='The revision if there is multple checkpoints for the model, e.g., step1000-tokens4B')
    parser.add_argument('--step_start', type=int, required=True, help="The checkpint set you want to start with")
    parser.add_argument('--step_end', type=int, required=True, help="The checkpint set you want to end with")
    parser.add_argument('--dataset_names', type=str, required=True, help="the name of datasets being used, seperated by ',' ")
    parser.add_argument('--multiple_of', type=int, default=1000)
    
    args = parser.parse_args()

    # Set the global variables based on input arguments
    pivot = args.pivot
    file_ext = args.file_ext
    save_path = args.save_path
    num_sents = args.num_sents
    embedding_type = args.embedding_type

    dataset_names = args.dataset_names.split(',')
    
    checkpoints = get_checkpoint_lists(step1=args.step_start, step2=args.step_end, multiple_of=args.multiple_of)
    
    for revision in checkpoints:
        print(f"Processing {revision} ... ")
        for fact_type in ['fn', 'tn', 'fp', 'tp', 'all', 'other']:
            # Load the pivot embeddings
            print(fact_type)
            for dataset_name in dataset_names:
                embedding_path = os.path.join(args.embedding_path, dataset_name)
                embedding_path = os.path.join(embedding_path, revision)   

                with open(os.path.join(embedding_path, f"{pivot}{file_ext}"), "rb") as pickle_file:
                    pivot_embd = pickle.load(pickle_file)

                languages = sorted([filename[:-len(file_ext)] for filename in os.listdir(embedding_path) if filename.endswith(file_ext)])
                print(f"languages: {languages}")

                for lang in tqdm(languages):
                    if lang == 'eng_Latn':
                        continue
                    print(lang)
                    save_filepath = os.path.join(args.save_path, dataset_name + "_no_identical_object", revision)
                    
                    if not os.path.exists(save_filepath):
                        os.makedirs(save_filepath)
                    
                    if embedding_type == 'embd_lasttoken':
                        filepath = f"{save_filepath}/{lang}_cosine_lasttoken"
                    else:
                        filepath = f"{save_filepath}/{lang}_cosine"
                    
                    filepath += f"_{fact_type}.json"
                    
                    if os.path.exists(filepath):
                        continue
                    
                    with open(os.path.join(embedding_path, f"{lang}.pkl"), "rb") as pickle_file:
                            lang_embd = pickle.load(pickle_file)
                
                    similarity_lang = compute_average_pairwise_cosine_similarity(lang_embd, embedding_type=embedding_type, num_sents=num_sents, lang=inverse_language_info[lang], compute_which=fact_type)

                    with open(filepath, "w") as json_file:
                        json.dump(similarity_lang, json_file)
                    print()
                print()