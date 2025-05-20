#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! pip install wimbd
# ! pip install elasticsearch==8.6.2


# In[2]:


language_wordlists = {
    'eng_Latn': ['the', 'that', 'and', 'with'], # correct
    'fra_Latn': ['dans', 'vous', 'sont', 'avec'], # correct
    'spa_Latn': ['las', 'sus', 'fue', 'él'], # correct
    'arb_Arab': ['في', 'على', 'إلى', 'أن'], # correct
    'cmn_Hani': ['的', '在', '是', '了'], # correct
    'rus_Cyrl': ['в', 'с', 'что', 'для'], # correct
    'jpn_Jpan': ['の', 'に', 'は', 'を'], # correct
    'tur_Latn': ['için', 'olarak', 'çok', 'gibi'], # correct
    'ukr_Cyrl': ['що', 'та', 'як', 'його'], # correct
    'cat_Latn': ['els', 'amb', 'més', 'Són'], # correct, ELs/els
    'kor_Hang': ['는', '을', '이', '한'], #correct
    'ell_Grek': ['και', 'της', 'του', 'την'] # correct
}


# In[3]:


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


def init_wimbd():
    path = '/mounts/data/proj/yihong/CIS-relation-neuron-zeroshot/analysis_scripts/es_config_8.yml'
    dolma_path = '/mounts/data/proj/yihong/CIS-relation-neuron-zeroshot/analysis_scripts/es_config_dolma_1_7_2.yml'

    es_config_8 = es_init(config=path, timeout=180)
    es_dolma = es_init(config=dolma_path, timeout=180)
    
    return es_config_8, es_dolma


def search_ngram_frequency(ngram_list, dataset=None, all_phrases=False):
    es_config_8, es_dolma = init_wimbd()
    
    dolma_document_num = count_documents_containing_phrases("docs_v1.7_2024-06-04", ngram_list, all_phrases=all_phrases, es=es_dolma)
    
    return dolma_document_num


# In[4]:


language_wordlists = {
    'eng_Latn': ['the', 'that', 'and', 'with'], # correct
    'fra_Latn': ['dans', 'vous', 'sont', 'avec'], # correct
    'spa_Latn': ['las', 'sus', 'fue', 'él'], # correct
    'ara_Arab': ['في', 'على', 'إلى', 'أن'], # correct
    'zho_Hani': ['的', '在', '是', '了'], # correct
    'rus_Cyrl': ['в', 'с', 'что', 'для'], # correct
    'jpn_Jpan': ['の', 'に', 'は', 'を'], # correct
    'tur_Latn': ['için', 'olarak', 'çok', 'gibi'], # correct
    'ukr_Cyrl': ['що', 'та', 'як', 'його'], # correct
    'cat_Latn': ['els', 'amb', 'més', 'Són'], # correct, Els/els
    'kor_Kore': ['는', '을', '이', '한'], #correct
    'ell_Grek': ['και', 'της', 'του', 'την'] # correct
}


# In[5]:


from itertools import combinations

res = {}
for lang in language_wordlists.keys():
    
    ngram_list = language_wordlists[lang]
    pairs = list(combinations(ngram_list, 2))

    frequencies = []
    for pair in pairs:
        frequency = search_ngram_frequency(list(pair), all_phrases=True)
        frequencies.append(frequency)
    
        res[lang] = frequencies


# In[6]:


print(res)


# In[7]:


import matplotlib.pyplot as plt
import pandas as pd

data = res

# Create DataFrame and sort by mean
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
mean_values = df.mean().sort_values(ascending=False)
df_sorted = df[mean_values.index]

# Plot
plt.figure(figsize=(12, 8))
box = plt.boxplot(df_sorted.values, labels=df_sorted.columns, patch_artist=True, showmeans=True)

# Style customization
colors = ['#76bdff'] * len(df_sorted.columns)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

for median in box['medians']:
    median.set(color='red', linewidth=1.5)

for mean in box['means']:
    mean.set(marker='^', markerfacecolor='purple', markeredgecolor='black', markersize=8)

# Log scale and layout
plt.yscale('log')
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Language", fontsize=20)
plt.ylabel("Pair Frequency (log scale)", fontsize=20)
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.grid(True)

plt.tight_layout()

# Show the plot
plt.savefig('dolma-lang-stat.pdf', dpi=300)
plt.show()


# In[ ]:




