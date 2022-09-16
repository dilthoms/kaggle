# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: kaggle_cenv
#     language: python
#     name: kaggle_cenv
# ---

# %%
import pandas as pd
import transformers
import torch
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# %%
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader

# %%
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertConfig
from transformers import AutoModelForSequenceClassification

# %%
labels = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar',
       'conventions']

# %%
chkpt = 'distilbert-base-uncased-finetuned-sst-2-english'

# %%
tokenizer = AutoTokenizer.from_pretrained(chkpt)
model = AutoModel.from_pretrained(chkpt)

# %%
device ='cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)


# %%
class ELLDS(Dataset):
    def __init__(self, df):
        self.df = df
    def __getitem__(self, idx):
        return dftr.iloc[idx]['full_text'], dftr.iloc[idx][labels]
    def __len__(self):
        return len(self.df)


# %%
dftr = pd.read_csv('data/train.csv')
dfte = pd.read_csv('data/test.csv')

# %%
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# %%
bsz = 64
feats = []
for i in range(0, len(dftr), bsz):
    inputs = tokenizer(dftr.iloc[i:i+bsz]['full_text'].tolist(),truncation=True, padding=True, return_tensors='pt')
    inputs = inputs.to(device)
    with torch.no_grad():
        out = model(**inputs)
        feats.append(out.last_hidden_state.mean(axis=1))

# %%
npfeats = []
for f in feats:
    npfeats.extend(f.detach().cpu().numpy())

# %%
np.array(npfeats)

# %%
np.array(npfeats).shape

# %%
for tr_idx, va_idx in mskf.split(dftr['full_text'], dftr[labels]):
    tr_idx = np.random.choice(tr_idx,len(tr_idx), replace=False)
    for i in range(0, len(tr_idx), 16):
        inputs = tokenizer(dftr.iloc[tr_idx[i:i+16]]['full_text'].tolist(),truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs)
            out.last_hidden_state.mean(axis=1)
        break
    break

# %%

# %%
dftr.iloc[tr_idx]['cohesion'].value_counts()

# %%
dftr.iloc[va_idx]['cohesion'].value_counts()

# %%
iter(tr_dl).next()

# %%
config

# %%
config.output_last_state=True

# %%
inputs = tokenizer(list(dftr.iloc[0:16]['full_text'].values), truncation=True, padding=True, return_tensors='pt')

# %%
inputs.input_ids.shape

# %%
with torch.no_grad():
    output = model(**inputs)

# %%
output.last_hidden_state.mean(axis=1).shape

# %%
inputs.input_ids.shape

# %%
output.last_hidden_state

# %%
dftr.head()

# %%
