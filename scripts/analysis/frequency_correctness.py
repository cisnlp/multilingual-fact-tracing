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


def build_fn_crosslingual_frequencies_full_with_correctness(freq_data, fn_indices_per_language, correct_indices_data):
    """
    For each FN fact, get its subject, object, relation, frequencies and correctness across all languages.

    Args:
        freq_data (dict): {lang: {relation: list of facts}}
        fn_indices_per_language (dict): {lang: list of FN indices}
        correct_indices_data (dict): {lang: {correct_indices: [...]}}  # original correctness file

    Returns:
        crosslingual_full (dict): {lang: {fact_idx: {subject, object, relation, frequencies: {}, correctness: {}}}}
    """
    # Step 1: Build global maps
    index_info = {}  # {index: {lang: {subject, object, relation}}}
    index_language_freq = {}  # {index: {lang: frequency}}

    for lang, relations in freq_data.items():
        for relation, facts in relations.items():
            for fact in facts:
                idx = fact['index']
                freq = fact['frequency']
                subj = fact['subject']
                obj = fact['object']
                
                # because the first language is English, 
                # so that index_info only contains mapping from idx to english subejct and object
                if idx not in index_info:
                    index_info[idx] = {}
                index_info[idx][lang] = {'subject': subj, 'object': obj, 'relation': relation}
                
                if idx not in index_language_freq:
                    index_language_freq[idx] = {}
                index_language_freq[idx][lang] = freq

    # Step 2: Build global correct_indices lookup
    lang_correct_sets = {}
    for lang, data in correct_indices_data.items():
        correct_set = set(data.get('correct_indices', []))
        lang_correct_sets[lang] = correct_set

    # Step 3: Build per-language FN dictionaries
    crosslingual_full = {}

    for lang, fn_indices in fn_indices_per_language.items():
        lang_result = {}
        for idx in fn_indices:
            fact_info = index_info.get(idx, {}).get(lang, {})
            
            freqs_in_other_langs = {}
            correctness_in_other_langs = {}
            
            for other_lang in freq_data.keys():
                freqs_in_other_langs[other_lang] = index_language_freq.get(idx, {}).get(other_lang, 0)
                correctness_in_other_langs[other_lang] = idx in lang_correct_sets.get(other_lang, set())
            
            lang_result[idx] = {
                'subject': fact_info.get('subject', ''),
                'object': fact_info.get('object', ''),
                'relation': fact_info.get('relation', ''),
                'frequencies': freqs_in_other_langs,
                'correctness': correctness_in_other_langs
            }
        
        crosslingual_full[lang] = lang_result

    return crosslingual_full


def analyze_fn_relations(freq_data, fn_indices_per_language):
    """
    Analyze the relation distribution of False Negatives for each language.

    Args:
        freq_data (dict): Original frequency dataset, structured by language -> relation -> list of facts.
        fn_indices_per_language (dict): {language: list of FN indices}

    Returns:
        fn_relations_per_language (dict): {language: Counter(relation: count)}
    """
    fn_relations_per_language = {}

    for lang, fn_indices in fn_indices_per_language.items():
        relation_counter = Counter()
        
        # Build a quick lookup: index -> relation
        index_to_relation = {}
        for relation, facts in freq_data.get(lang, {}).items():
            for fact in facts:
                idx = fact['index']
                index_to_relation[idx] = relation
        
        # Count relations for FN indices
        for idx in fn_indices:
            relation = index_to_relation.get(idx, None)
            if relation:
                relation_counter[relation] += 1
        
        fn_relations_per_language[lang] = relation_counter
    
    return fn_relations_per_language


def analyze_combined_frequency_correctness(all_freqs, all_labels, save_dir='./', checkpoint=None):
    """
    Analyzes the frequency-correctness relationship across all languages combined.
    """
    global_max_freq = np.max(all_freqs)
    
    bins = np.logspace(np.log10(1), np.log10(global_max_freq + 1), 30)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    
    correct_freqs = all_freqs[all_labels == 1]
    total_hist, _ = np.histogram(all_freqs, bins=bins)
    correct_hist, _ = np.histogram(correct_freqs, bins=bins)
    
    prob_correct = correct_hist / np.maximum(total_hist, 1)
    
    # Modern plotting settings
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14})
    
    # Plot 1: Probability(correctly predicted | frequency)
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.plot(bin_centers, prob_correct, marker='o', color='tab:blue', linewidth=2)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, global_max_freq + 10)
    ax.set_xlabel('Frequency (log scale)')
    ax.set_ylabel('Probability of Correct Fctual Recall')
    ax.set_title(f"{checkpoint}", fontsize=20)
    ax.grid(True, which='both', ls='--', linewidth=0.7)
    
    plt.tight_layout()
    save_dir += '/overall'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint = checkpoint.split('-')[0]
    plt.savefig(f'{save_dir}/all_languages_probability_correct_vs_frequency_{checkpoint}.pdf')
    plt.close()
    
    # # Plot 2: Distribution of Correctly Predicted Facts
    # fig, ax = plt.subplots(figsize=(9, 6))
    
    # ax.hist(correct_freqs, bins=bins, color='tab:green', alpha=0.7, edgecolor='black')
    # ax.set_xscale('log')
    # ax.set_xlabel('Frequency (log scale)')
    # ax.set_ylabel('Number of Correctly Predicted Facts')
    # ax.set_title('Distribution of Correctly Predicted Facts (All Languages)', pad=15)
    # ax.set_xlim(1, global_max_freq + 10)
    # ax.grid(True, which='both', ls='--', linewidth=0.7)
    
    # plt.tight_layout()
    # plt.savefig(f'{save_dir}/all_languages_correct_fact_distribution_{checkpoint}.pdf')
    # plt.close()


def analyze_frequency_correctness(freqs, labels, lang, save_dir='./', global_max_freq=None):
    if global_max_freq is None:
        global_max_freq = np.max(freqs)
    
    # Define log-spaced bins
    bins = np.logspace(np.log10(1), np.log10(global_max_freq + 1), 30)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    # Histograms
    total_hist, _ = np.histogram(freqs, bins=bins)
    correct_freqs = freqs[labels == 1]
    correct_hist, _ = np.histogram(correct_freqs, bins=bins)
    
    # Probability
    # prob_correct = correct_hist / np.maximum(total_hist, 1)
    prob_correct = np.where(total_hist > 0, correct_hist / total_hist, np.nan)

    # Create merged figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Line plot for probability of correctness
    ax1.plot(bin_centers, prob_correct, color='darkred', marker='o', label='P(Correct|Frequency)')
    ax1.set_xscale('log')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Frequency (log scale)')
    ax1.set_ylabel('Probability of Correct Factual Recall', color='darkred')
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax1.grid(True, which='both', ls='--', linewidth=0.5)

    # Bar plots for counts
    ax2 = ax1.twinx()
    width = np.diff(bins)
    ax2.bar(bin_centers, total_hist, width=width, align='center', color='lightgray', label='Total', alpha=0.7)
    ax2.bar(bin_centers, correct_hist, width=width, align='center', color='cornflowerblue', label='Correct', alpha=0.7)
    ax2.set_ylabel('Number of Facts', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Save
    lang_full = f"{language_info[lang]['iso639_3']}_{language_info[lang]['script']}"
    lang_full_escaped = lang_full.replace('_', '\\_')
    
    plt.title(f'Correct Recall vs. Fact Frequency & Distribution ($\\mathbf{{{lang_full_escaped}}}$)')
    plt.tight_layout()
    out_dir = f"{save_dir}/{lang}"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/merged_correctness_frequency_plot.pdf")
    plt.close()


frequency_save_path = "./lang_rel_frequencies.json"
with open(frequency_save_path, "r", encoding="utf-8") as f_in:
    frequencies = json.load(f_in)
    
checkpoints1 = get_checkpoint_lists(1000, 50000, multiple_of=5000)
checkpoints2 = get_checkpoint_lists(100000, 400000, multiple_of=100000)
checkpoints = checkpoints1 + checkpoints2
for checkpoint in checkpoints:

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
        
        list_of_freq_arrays.append(freqs)
        list_of_label_arrays.append(labels)

    all_freqs = np.concatenate(list_of_freq_arrays)
    all_labels = np.concatenate(list_of_label_arrays)
    analyze_combined_frequency_correctness(all_freqs, all_labels, save_dir=savepath, checkpoint=checkpoint)
    


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
    
    analyze_frequency_correctness(freqs, labels, lang=lang, save_dir=savepath, global_max_freq=global_max_freq)


all_freqs = np.concatenate(list_of_freq_arrays)
all_labels = np.concatenate(list_of_label_arrays)
analyze_combined_frequency_correctness(all_freqs, all_labels, save_dir=savepath, checkpoint=checkpoint.split('-')[0])

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


fn_indices_per_language = {lang: res['false_negative_indices'] for lang, res in results.items()}
fn_relations = analyze_fn_relations(frequencies, fn_indices_per_language)

# Example: print top relations for each language
print('----------------------------')
print("FN Fcats grouped in relations")
print('----------------------------')
print()
for lang, counter in fn_relations.items():
    print(f"Language: {lang}")
    for rel, count in counter.most_common():
        print(f"  {rel}: {count}")
    print()


# Assuming you already have
crosslingual_fn_freqs = build_fn_crosslingual_frequencies_full_with_correctness(frequencies, fn_indices_per_language, correctnesses)

# Make sure your save directory exists
save_dir = './analysis_results'
os.makedirs(save_dir, exist_ok=True)

# Save fn_relations
with open(os.path.join(save_dir, 'fn_relations.json'), 'w', encoding='utf-8') as f:
    json.dump({lang: dict(counter) for lang, counter in fn_relations.items()}, f, indent=2, ensure_ascii=False)

# Save crosslingual_fn_freqs
with open(os.path.join(save_dir, 'crosslingual_fn_freqs.json'), 'w', encoding='utf-8') as f:
    json.dump(crosslingual_fn_freqs, f, indent=2, ensure_ascii=False)
print(f"Saved fn_relations and crosslingual_fn_freqs into '{save_dir}' folder!")

# Save all index classification results per language
with open(os.path.join(save_dir, 'index_classification_by_language.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved detailed index classification data to 'index_classification_by_language.json'")
