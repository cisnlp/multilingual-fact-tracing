import json
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from collections import Counter
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


frequency_save_path = "./lang_rel_frequencies.json"
with open(frequency_save_path, "r", encoding="utf-8") as f_in:
    frequencies = json.load(f_in)

# record the resuls for the final checkpoint
checkpoint = get_checkpoint_lists(400000, 400000, multiple_of=1000)[0]
print(checkpoint)
correctness_save_path = f"../consistency_olmo_vllm_details/{checkpoint}/results_details.json"
with open(correctness_save_path, "r", encoding="utf-8") as f_co:
    correctnesses = json.load(f_co)
    
# consider per language, merge all relations

# Build {language: {index: (frequency, (subject, object))}}
language_facts = {}
for lang, relations in frequencies.items():
    facts = {}
    for rel, fact_list in relations.items():
        for fact in fact_list:
            idx = fact['index']
            freq = fact['frequency']
            subj = fact['subject']
            obj = fact['object']
            facts[idx] = (freq, (subj, obj))
    language_facts[lang] = facts


# Find global max frequency for consistent axis scaling
global_max_freq = 1
for facts in language_facts.values():
    if facts:
        max_freq = max(f[0] for f in facts.values())
        global_max_freq = max(global_max_freq, max_freq)

# Find best frequency threshold per language
results = {}


savepath = './figures/frequency_distrubution'
if not os.path.exists(savepath):
    os.makedirs(savepath)
        
# Suppose you have already per-language freqs, labels collected:
list_of_freq_arrays = []
list_of_label_arrays = []

for lang in language_facts:
    fact_info = language_facts[lang]
    correct_set = set(correctnesses.get(lang, {}).get('correct_indices', []))
    if not fact_info:
        raise ValueError(f"No data for {lang}")

    data = []
    for idx, (freq, _) in fact_info.items():
        is_correct = idx in correct_set
        data.append((idx, freq, is_correct))

    if not data:
        raise ValueError(f"No data for {lang}")

    data.sort(key=lambda x: x[1])  # sort by frequency
    indices = np.array([d[0] for d in data])
    freqs = np.array([d[1] for d in data])
    labels = np.array([d[2] for d in data])

    best_acc = 0
    best_thresh = None
    best_fp_indices = []
    best_fn_indices = []

    # Try thresholds between every unique frequency
    unique_freqs = np.unique(freqs)
    for thresh in unique_freqs:
        preds = freqs >= thresh  # predict correct if freq >= threshold
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            # Calculate false positives and false negatives
            fp_indices = indices[(preds == 1) & (labels == 0)].tolist()
            fn_indices = indices[(preds == 0) & (labels == 1)].tolist()
            
            # Calculate true postives and true negatives
            tp_indices = indices[(preds == 1) & (labels == 1)].tolist()
            tn_indices = indices[(preds == 0) & (labels == 0)].tolist()

            best_fp_indices = fp_indices
            best_fn_indices = fn_indices

    results[lang] = {
        'best_threshold': int(best_thresh) if best_thresh is not None else None,
        'best_accuracy': best_acc,
        'false_positives': len(best_fp_indices),
        'false_negatives': len(best_fn_indices),
        'false_positive_indices': best_fp_indices,
        'false_negative_indices': best_fn_indices,
        'true_positive_indices': tp_indices,
        'true_negative_indices': tn_indices,
        'all_indices': indices.tolist()
    }
    
    list_of_freq_arrays.append(freqs)
    list_of_label_arrays.append(labels)

all_freqs = np.concatenate(list_of_freq_arrays)
all_labels = np.concatenate(list_of_label_arrays)

# Output the results
for lang, res in results.items():
    print(f"Language: {lang}")
    print(f"  Best Threshold: {res['best_threshold']}")
    print(f"  Best Accuracy: {res['best_accuracy']:.4f}")
    print(f"  False Positives (incorrect but above threshold): {res['false_positives']}")
    print(f"  False Negatives (correct but below threshold): {res['false_negatives']}")
    print(f"  True Positives (correct and above threshold): {len(res['true_positive_indices'])}")
    print(f"  True Negatives (incorrect and below threshold): {len(res['true_negative_indices'])}")
    # print(f"  False Positive Indices: {res['false_positive_indices']}")
    # print(f"  False Negative Indices: {res['false_negative_indices']}")
    print()