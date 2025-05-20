import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import json
import glob
import random
import argparse
import numpy
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import time
import re
from checkpoint_helper import *
from vllm import LLM, SamplingParams


start = time.time()

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="allenai/OLMo-7B-0424-hf")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument('--cache_dir', type=str, default='/nfs/datz/olmo_models', 
                    help='Directory for caching the model (optional).')
parser.add_argument('--save_path', type=str, default='./consistency_olmo_vllm_details/', help='Path to save the results.')
parser.add_argument('--step_start', type=int, required=True, help="The checkpint set you want to start with")
parser.add_argument('--step_end', type=int, required=True, help="The checkpint set you want to end with")
parser.add_argument('--multiple_of', type=int, default=1000, help='how many steps two consecutive checkpoints should be')

args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    """Globally set random seed."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)


# languages_9 = [
#     "en",  # English
#     "fr",  # French
#     "es",  # Spanish
#     "ar",  # Arabic
#     "zh",  # Chinese
#     "ru",  # Russian
#     "ja",  # Japanese
#     "tr",  # Turkish
#     "uk",  # Ukrainian
# ]

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

valid_langs = languages_12
valid_rels = relation_list_12

# ====  Group file paths by relation ====
json_paths = glob.glob("klar_source/*/*.json")
path_map = defaultdict(dict)  # (relation -> lang -> path)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue

    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content = json.load(f)
            loaded_samples = content["samples"]
            template = content["prompt_templates"][0]

            for sample in loaded_samples:
                new_sample = {
                    "subject": sample["subject"],
                    "object": sample["object"],
                    "language": lang,
                    "relation": rel,
                    "template": template,
                    "index": sample["index"]
                }
                samples.append(new_sample)

# ==== Apply prompt formatting ====
def apply_prompt(example):
    prompt = re.sub(r"<mask>.*", "", example["template"])
    prompt = prompt.replace("<subject>", example["subject"])
    example["input"] = prompt.strip()
    example["target"] = " " + example["object"]
    return example

dataset = Dataset.from_list([apply_prompt(ex) for ex in samples])


# ==== Evaluation ====
def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def overlapping_ratio(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


checkpoints = get_checkpoint_lists(step1=args.step_start, step2=args.step_end, multiple_of=args.multiple_of)
    
    
for revision in checkpoints:
    
    print(f"Evaluating checkpoint {revision} ... ")
    # check if the reviison is already there:
    if os.path.exists(os.path.join(args.save_path, revision)):
        continue
    else:
        os.makedirs(os.path.join(args.save_path, revision))
    
    # ==== Tokenizer & Model ====
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = LLM(model=model_name, 
                tokenizer=model_name, 
                trust_remote_code=True, 
                revision=revision, 
                gpu_memory_utilization=0.6,
                download_dir=args.cache_dir)


    def evaluate(model, dataset, tokenizer, max_new_tokens=10, n_shot=3, save_path="./"):
        
        sampling_params = SamplingParams(
            temperature=0.0,     # ensures greedy decoding
            top_p=1.0,
            top_k=-1,
            max_tokens=max_new_tokens        # adjust as needed
        )
        
        # model.eval()
        test_data = list(dataset)
        correct_total = 0
        total_total = 0
        per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": [],
                                                "total_indices_per_relation":  {r: [] for r in sorted(list(valid_rels))},
                                                "correct_indices_per_relation": {r: [] for r in sorted(list(valid_rels))}})

        print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
        for idx, ex in enumerate(tqdm(test_data)):
            lang = ex.get("language", "unknown")
            relation = ex.get("relation", "unknown")
            index = ex.get("index", None)
            candidates = [c for c in test_data if c.get("index") != index and c.get("language") == lang and c.get("relation") == relation]
            demonstrations = random.sample(candidates, min(n_shot, len(candidates)))

            few_shot_prompt = "".join([f"{d['input']}{d['target']}\n" for d in demonstrations]) + ex["input"]
            target = ex["target"]
            
            outputs = model.generate([few_shot_prompt], sampling_params)
            prediction = outputs[0].outputs[0].text.strip()

            # match = is_nontrivial_prefix(prediction, target)
            match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
            correct_total += match
            total_total += 1
            per_lang_results[lang]["correct"] += match
            per_lang_results[lang]["total"] += 1
            
            per_lang_results[lang]["total_indices_per_relation"][relation].append(index)
            
            if match:
                per_lang_results[lang]["correct_indices"].append(index)
                per_lang_results[lang]["correct_indices_per_relation"][relation].append(index)
            
            # print(f"[{lang}] Q: {ex['input']} | Pred: {prediction} | Label: {target} | Match: {match}")

        overall_acc = correct_total / total_total if total_total > 0 else 0
        # print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")

        results = {"overall_acc": overall_acc, "overall_clc": None, 
                "per_language_acc": {}, 
                "per_language_clc": {},
                "per_language_per_relation_acc": {l: {} for l in sorted(list(valid_langs))},
                "per_language_per_relation_clc": {l: {} for l in sorted(list(valid_langs))},
                }

        for lang, res in sorted(per_lang_results.items()):
            lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
            results["per_language_acc"][lang] = lang_acc
            # print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")
            for r in sorted(list(valid_rels)):
                lang_rel_acc = len(res['correct_indices_per_relation'][r]) / len(res['total_indices_per_relation'][r]) \
                if len(res['total_indices_per_relation'][r]) > 0 else 0
                results['per_language_per_relation_acc'][lang][r] = lang_rel_acc
            #     print(f"    {r}: {lang_rel_acc:.2%} ({len(res['correct_indices_per_relation'][r])} / {len(res['total_indices_per_relation'][r])}")
            # print()
                
        langs = sorted(list(per_lang_results.keys()))
        for lang in langs:
            if lang == 'en':
                results["per_language_clc"]['en'] = 1.0
            scores = overlapping_ratio(per_lang_results[lang]["correct_indices"], per_lang_results['en']["correct_indices"])
            consistency = scores
            results["per_language_clc"][lang] = consistency
            for r in sorted(list(valid_rels)):
                if lang == 'en':
                    results['per_language_per_relation_clc'][lang][r] = 1.0
                    continue
                lang_rel_clc = overlapping_ratio(per_lang_results[lang]["correct_indices_per_relation"][r], 
                                                per_lang_results['en']["correct_indices_per_relation"][r])
                results['per_language_per_relation_clc'][lang][r] = lang_rel_clc
            
        results["overall_clc"] = sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())
        
        # print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')
        
        # for lang in langs:
            # print(f'  {lang} cross-lingual consistency: {results["per_language_clc"][lang]:.2%}')
            # for r in sorted(list(valid_rels)):
                # print(f'    {lang} -- {r} cross-lingual consistency: \
                    # {results["per_language_per_relation_clc"][lang][r]:.2%}')
        
        with open(f"{save_path}/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        with open(f"{save_path}/results_details.json", "w", encoding="utf-8") as f:
            json.dump(per_lang_results, f, indent=4, ensure_ascii=False)

        return results, per_lang_results

    save_path = args.save_path
    save_filepath = os.path.join(save_path, revision)

    if not os.path.exists(save_filepath):
        os.makedirs(save_filepath)
    
    evaluate(model, dataset, tokenizer, max_new_tokens=10, n_shot=3, save_path=save_filepath)

    end = time.time()

    print(f"Function took {end - start:.4f} seconds.")
    
    del model
    # gc.collect()
    torch.cuda.empty_cache()
