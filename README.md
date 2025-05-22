# Tracing Multilingual Factual Knowledge Acquisition in Pretraining

**Authors**: Yihong Liu, Mingyang Wang, Amir Hossein Kargaran, Felicia Körner, Ercong Nie, Barbara Plank, François Yvon, Hinrich Schütze  
**Preprint**: [arXiv](https://arxiv.org/abs/2505.14824)  

---

## Overview

Large Language Models (LLMs) are known to memorize and recall factual knowledge across multiple languages. 
However, the process through which this knowledge emerges during pretraining remains unclear.

In this work, we investigate how multilingual factual recall and crosslingual consistency evolve over the course of pretraining, using **[OLMo-7B](https://huggingface.co/allenai/OLMo-7B-0424-hf)** and its acompanied checkpoints as a case study. We present empirical results demonstrating two key mechanisms of factual knowledge acquisition:

1. **Frequency-driven learning** (dominant and language-agnostic)
2. **Crosslingual transfer** (notable in early pretraining, some non-English low-frequency facts benefit from it)

---

## Repository Structure

```
.
├── README.md
├── data/
│   ├── klar_variant_full/
│   │   ├── ara_Arab.txt
│   │   ├── cat_Latn.txt
│   │   ├── ell_Grek.txt
│   │   ├── eng_Latn.txt
│   │   ├── fra_Latn.txt
│   │   ├── jpn_Jpan.txt
│   │   ├── kor_Kore.txt
│   │   ├── rus_Cyrl.txt
│   │   ├── spa_Latn.txt
│   │   ├── tur_Latn.txt
│   │   ├── ukr_Cyrl.txt
│   │   └── zho_Hans.txt
│   ├── lang_rel_frequencies.json
│   └── lang_rel_frequencies_infini.json
└── scripts/
    ├── analysis/
    │   ├── embed_extractor.py
    │   ├── fact_similarity_compute.py
    │   ├── frequency_classifier.py
    │   ├── frequency_correctness.py
    │   ├── dolma-lang-stat.py
    ├── factual_recall_computation/
    │   └── factual_recall_vllm_with_details.py
    └── frequency_computation/
        ├── frequency_computation_infini.py
        └── frequency_computation_wimbd.py
```

---

## Data Provided

### Fact Frequencies

We provide the fact frequencies of facts in [KLAR](https://arxiv.org/pdf/2504.04264) computed by [WIMBD](https://github.com/allenai/wimbd/) and [infini gram](https://infini-gram.readthedocs.io/en/latest/api.html), respectively. 
However, we use [WIMBD](https://github.com/allenai/wimbd/) in our project since the statistics are much more reliable.

- [`lang_rel_frequencies.json`](data/lang_rel_frequencies.json): computed by [WIMBD](https://github.com/allenai/wimbd/)
- [`lang_rel_frequencies_infini.json`](data/lang_rel_frequencies_infini.json): computed by [infini gram](https://infini-gram.readthedocs.io/en/latest/api.html)

### KLAR Parallel Texts

The texts are used to compute sentence representations and cosine similarities among facts of different languages (12 in total)

- [`klar_variant_full`](data/klar_variant_full): each line in the file for a language is the translation of corresponding lines of other languages



### Requirements

- Python 3.9+
- [KLAR](https://arxiv.org/pdf/2504.04264) dataset for tracing multilingual factual recall (please refer to the corresponding github).
- [vLLM](https://docs.vllm.ai/en/latest/) is used to obtain the factual recall response.
- [WIMBD](https://github.com/allenai/wimbd/) is used to obtain the fact frequncies across languages.


### Run Factual Recall Evaluation

Example: Evaluate the `allenai/OLMo-7B-0424-hf` for each 1K step between checkpoint 0 and 50K.


```bash
python filter_knowns_vllm_details.py 
  --model_name allenai/OLMo-7B-0424-hf
  --step_start 0 
  --step_end 50000 
  --multiple_of 1000 
```

### Run Fact Frequency Computation

Using [WIMBD](https://github.com/allenai/wimbd/) (recommend, since it does not do approximation.)

```bash
python frequency_computation_wimbd.py
```


Using [infini gram](https://infini-gram.readthedocs.io/en/latest/api.html) (when it performs approximation, the results can be highly unreliable)

```bash
python frequency_computation_infini.py
```

### Analysis

To build a simple frequency-based classifier for each language, and obtain error breakdown:

```bash
python frequency_classifier.py
```


To generate some visualizations and save the indices for different type of facts and their associated statistics:

```bash
python frequency_correctness.py
```


To compute the sentence-level emebdding for each fact in each language 
across checkpoints, e.g., for each 1K step between checkpoint 0 and 50K.
(this script is adapted from [MEXA](https://github.com/cisnlp/MEXA/blob/main/embed_extractor.py)):

```bash
python embed_extractor.py 
  --model_name allenai/OLMo-7B-0424-hf 
  --step_start 0 
  --step_end 50000 
  --multiple_of 1000 
  --data_path ./datasets 
  --dataset_names klar_variant_full 
  --gpus '0' 
  --num_sents 1500 
  --save_path ./embd_olmo/ 
  --cache_dir /nfs/datz/olmo_models 
  --file_ext .txt
```

To compute the cosine similarity between facts in each language and their English counterparts 
across checkpoints, e.g., for each 1K step between checkpoint 0 and 50K.
(this script is adapted from [MEXA](https://github.com/cisnlp/MEXA/blob/main/compute_mexa.py)):

```bash
python fact_similarity_compute.py 
  --embedding_path ./embd_olmo/
  --dataset_names klar_variant_full 
  --step_start 0 
  --step_end 50000 
  --multiple_of 1000 
  --save_path ./results 
  --num_sents 1500 
  --embedding_type embd_lasttoken 
  --pivot eng_Latn 
  --file_ext .pkl
```


---


## Citation

If you find our method, code and scores useful for your research, please considering citing:  


KLAR dataset:

```bibtex
@misc{wang2025lost,
      title={Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models}, 
      author={Mingyang Wang and Heike Adel and Lukas Lange and Yihong Liu and Ercong Nie and Jannik Strötgen and Hinrich Schütze},
      year={2025},
      eprint={2504.04264},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.04264}
}
```

and this paper:


```bibtex
@misc{liu2025tracing,
      title={Tracing Multilingual Factual Knowledge Acquisition in Pretraining}, 
      author={Yihong Liu and Mingyang Wang and Amir Hossein Kargaran and Felicia Körner and Ercong Nie and Barbara Plank and François Yvon and Hinrich Schütze},
      year={2025},
      eprint={2505.14824},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14824}
}
```

