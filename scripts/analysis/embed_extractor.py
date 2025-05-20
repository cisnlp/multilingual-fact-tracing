import argparse
import os
import torch
import json
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import gc
from checkpoint_helper import *
import numpy as np


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


# Function to handle weighted embeddings
def weighted_embeddings(layer, attention_mask, device='cuda'):
    # Compute weights for non-padding tokens
    weights_for_non_padding = attention_mask * torch.arange(start=1, end=layer.shape[1] + 1, device=device).unsqueeze(0)
    sum_embeddings = torch.sum(layer * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_non_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_non_padding_tokens
    sentence_embeddings = sentence_embeddings.squeeze().to(torch.float32).cpu().numpy()
    return sentence_embeddings


def lasttoken_embeddings(layer, attention_mask, device='cuda'):
    # Compute the index of the last non-padding token
    idx_of_last_token = attention_mask.bool().sum().item() - 1  # scalar index
    # Extract the embedding from the layer
    embedding = layer[0, idx_of_last_token, :]  # shape: [hidden_dim]
    sentence_embedding = embedding.to(torch.float32).cpu().numpy()
    return sentence_embedding


# Function to extract embeddings
def get_embedding_layers(text, model, tokenizer, device='cuda'):
    
    # before this not added
    model.eval()
    
    tokens = tokenizer(text, return_tensors='pt', padding=True).to(device)
    attention_mask = tokens.attention_mask.to(device)

    sentence_embeddings_weighted = []
    sentence_embeddings_last_token = []
    
    with torch.no_grad():
        hidden_state_layers = model(**tokens, output_hidden_states=True)["hidden_states"]

        for layer in hidden_state_layers:
            embd_weighted = weighted_embeddings(layer, attention_mask, device)
            embd_last_token = lasttoken_embeddings(layer, attention_mask, device)

            sentence_embeddings_weighted.append(embd_weighted)
            sentence_embeddings_last_token.append(embd_last_token)

    return sentence_embeddings_weighted, sentence_embeddings_last_token


def get_embedding_layers_batch(texts, model, tokenizer, device='cuda', batch_size=64):
    
    model.eval()
    num_layers = None
    all_embeddings_weighted = None
    all_embeddings_last_token = None

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        attention_mask = tokens.attention_mask.to(device)

        # Manually fix position_ids
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        with torch.no_grad():
            hidden_state_layers = model(**tokens, output_hidden_states=True, position_ids=position_ids)["hidden_states"]

            if num_layers is None:
                num_layers = len(hidden_state_layers)
                all_embeddings_weighted = [[] for _ in range(num_layers)]
                all_embeddings_last_token = [[] for _ in range(num_layers)]

            for layer_idx, layer in enumerate(hidden_state_layers):
                
                weights_for_non_padding = attention_mask * torch.arange(start=1, end=layer.shape[1] + 1, device=device).unsqueeze(0)
                sum_embeddings = torch.sum(layer * weights_for_non_padding.unsqueeze(-1), dim=1)
                num_of_non_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
                batch_weighted_embeds = sum_embeddings / num_of_non_padding_tokens

                idx_of_last_token = (attention_mask != 0).sum(dim=1) - 1
                batch_indices = torch.arange(layer.shape[0], device=device)
                batch_last_token_embeds = layer[batch_indices, idx_of_last_token]

                all_embeddings_weighted[layer_idx].append(batch_weighted_embeds.to(torch.float32).cpu().numpy())
                all_embeddings_last_token[layer_idx].append(batch_last_token_embeds.to(torch.float32).cpu().numpy())

    for layer_idx in range(num_layers):
        all_embeddings_weighted[layer_idx] = np.concatenate(all_embeddings_weighted[layer_idx], axis=0)
        all_embeddings_last_token[layer_idx] = np.concatenate(all_embeddings_last_token[layer_idx], axis=0)

    return all_embeddings_weighted, all_embeddings_last_token



def verify_same(embeddings_dict_old, embeddings_dict_new, atol=5e-5):
    for layer in embeddings_dict_old.keys():
        entries_old = embeddings_dict_old[layer]
        entries_new = embeddings_dict_new[layer]

        if len(entries_old) != len(entries_new):
            print(f"Mismatch in number of sentences at layer {layer}")
            return False

        for idx in range(len(entries_old)):
            id_old = entries_old[idx]['id']
            id_new = entries_new[idx]['id']

            if id_old != id_new:
                print(f"Mismatch in IDs at layer {layer}, idx {idx}: old {id_old}, new {id_new}")
                return False

            weighted_old = entries_old[idx]['embd_weighted']
            weighted_new = entries_new[idx]['embd_weighted']
            lasttoken_old = entries_old[idx]['embd_lasttoken']
            lasttoken_new = entries_new[idx]['embd_lasttoken']

            if not np.allclose(weighted_old, weighted_new, atol=atol):
                print(f"Mismatch in weighted embeddings at layer {layer}, idx {idx}")
                print(weighted_old)
                print(weighted_new)
                return False

            if not np.allclose(lasttoken_old, lasttoken_new, atol=atol):
                print(f"Mismatch in last token embeddings at layer {layer}, idx {idx}")
                print(lasttoken_old)
                print(lasttoken_new)
                return False

    print("✅ Verification passed: old and new embeddings are identical (within tolerance)")
    return True


# Main function
def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from a model")

    # Add arguments for the parser
    parser.add_argument('--model_name', type=str, required=True, help='The model name from Hugging Face.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the parallel data directory.')
    parser.add_argument('--gpus', type=str, default='0', help='GPUs to use, e.g. "0".')
    parser.add_argument('--num_sents', type=int, default=100, help='Maximum number of sentences to process.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the embeddings.')
    parser.add_argument('--token', type=str, default=None, help='Hugging Face token (optional).')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Directory for caching the model (optional).')
    parser.add_argument('--file_ext', type=str, default='.txt', help='File extension for input files (optional, default: .txt).')
    parser.add_argument('--step_start', type=int, required=True, help="The checkpint set you want to start with")
    parser.add_argument('--step_end', type=int, required=True, help="The checkpint set you want to end with")
    parser.add_argument('--dataset_names', type=str, required=True, help="the name of datasets being used, seperated by ',' ")
    parser.add_argument('--multiple_of', type=int, default=1000, help='how many steps two consecutive checkpoints should be')
    # parser.add_argument('--revision', type=str, default=None, help='The revision if there is multple checkpoints for the model, e.g., step1000-tokens4B')
    
    # Parse the arguments
    args = parser.parse_args()

    # Set GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Define model name and token
    model_name = args.model_name
    token = args.token  # Optional token
    
     # Directory and number of sentences
    dataset_names = args.dataset_names.split(',')
    number_of_sents = args.num_sents
    
    checkpoints = get_checkpoint_lists(step1=args.step_start, step2=args.step_end, multiple_of=args.multiple_of)
    
    languages_to_consider = [f"{language_info[lang]['iso639_3']}_{language_info[lang]['script']}" for lang in languages_12]
    print(languages_to_consider)
    
    for revision in checkpoints:
        
        print(f"Processing {revision} ... ")
        
        # check if computation for this revision is done
        paths = [os.path.join(args.save_path, dataset_name, revision) for dataset_name in dataset_names]
        flag = False
        for path in paths:
            if not os.path.exists(path):
                flag = True
        
        if not flag:
            continue
        
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, device_map='auto', cache_dir=args.cache_dir, use_auth_token=token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        tokenizer.pad_token = tokenizer.eos_token

        for dataset_name in dataset_names:
            directory = os.path.join(args.data_path, dataset_name)
            
            # Initialize a dictionary to store embeddings
            result_dict = {}

            # Process the files in the directory
            for filename in os.listdir(directory):
                if filename.endswith(args.file_ext):
                    language = filename.split('.')[0]
                    if language not in languages_to_consider:
                        continue
                    filepath = os.path.join(directory, filename)
                    sentences = []
                    with open(filepath, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for idx, line in enumerate(lines):
                            # for klar dataset, we want all sentences
                            if 'klar' in dataset_name:
                                # in case it is empty, we use "None" as a placeholder to avoid error
                                sentence = line.strip()
                                if len(sentence) == 0:
                                    sentence= 'None'
                                sentences.append({'id': idx + 1, 'text': sentence})
                            else:
                                if idx < number_of_sents:
                                    sentence = line.strip()
                                    # in case it is empty, we use "None" as a placeholder to avoid error
                                    if len(sentence) == 0:
                                        sentence= 'None'
                                    sentences.append({'id': idx + 1, 'text': sentence})

                    result_dict[language] = sentences

            # Extract embeddings
            for language, texts in tqdm(result_dict.items()):
                
                print(f"{language}, {len(texts)}")
                if language not in languages_to_consider:
                    continue
                
                # Save the embeddings as pickle
                save_filepath = os.path.join(args.save_path, dataset_name, revision)
                
                if os.path.exists(f"{save_filepath}/{language}.pkl"):
                    continue

                if not os.path.exists(save_filepath):
                    os.makedirs(save_filepath)

                embeddings_dict = {}
                text_strings = [text['text'] for text in texts]
                text_ids = [text['id'] for text in texts]

                embds_weighted, embds_last_token = get_embedding_layers_batch(text_strings, model, tokenizer)
                for layer in range(len(embds_weighted)):
                    if layer not in embeddings_dict:
                        embeddings_dict[layer] = []

                    for idx in range(len(text_strings)):
                        embeddings_dict[layer].append({
                            'id': text_ids[idx],
                            'embd_weighted': embds_weighted[layer][idx],
                            'embd_lasttoken': embds_last_token[layer][idx]
                        })
                
                # print('doing verification ....')
                # verify_same(embeddings_dict_old, embeddings_dict)

                with open(f"{save_filepath}/{language}.pkl", "wb") as pickle_file:
                    pickle.dump(embeddings_dict, pickle_file)

        # Clean up previous model

        del model
        # gc.collect()
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()
