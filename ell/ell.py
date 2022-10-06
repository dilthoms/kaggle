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
# %load_ext autoreload
# %autoreload 2

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
from sklearn.linear_model import LinearRegression

# %%
from collections import defaultdict

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
        return df.iloc[idx]['full_text'], df.iloc[idx][labels]
    def __len__(self):
        return len(self.df)


# %%
dftr = pd.read_csv('data/train.csv')
dfte = pd.read_csv('data/test.csv')

# %%
dftr.head()

# %%
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)


# %%
def get_feats(df, bsz=64):
    feats = []
    for i in range(0, len(df), bsz):
        inputs = tokenizer(df.iloc[i:i+bsz]['full_text'].tolist(),truncation=True, padding=True, return_tensors='pt')
        inputs = inputs.to(device)
        with torch.no_grad():
            out = model(**inputs)
            feats.append(out.last_hidden_state.mean(axis=1).detach().cpu().numpy())
    return feats


# %%
trfeats = np.concatenate(get_feats(dftr))

# %%
tefeats = np.concatenate(get_feats(dfte))

# %%
labmodels = defaultdict(list)
va_idxs = []
for tr_idx, va_idx in mskf.split(trfeats, dftr[labels]):
    va_idxs.append(va_idx)
    tr_idx = np.random.choice(tr_idx,len(tr_idx), replace=False)
    Xtr = trfeats[tr_idx]
    ytr = dftr.iloc[tr_idx][labels]
    for label in labels:
        lrm = LinearRegression()
        lrm.fit(Xtr,ytr[label])
        labmodels[label].append(lrm) 
        

# %%
from sklearn.metrics import mean_squared_error
import pdb
def RMSE(targ, pred):
    return mean_squared_error(targ,pred)
def MCRMSE(targ, pred):
    return [RMSE(targ[label],pred[label]) for label in labels]


# %%
pred = defaultdict(list)
targ = defaultdict(list)
rmse = defaultdict(list)
for i,va_idx in enumerate(va_idxs):
    Xte = trfeats[va_idx]
    yte = dftr.iloc[va_idx][labels]
    for label in labels:
        lrm = labmodels[label][i]
        pred[label].append(lrm.predict(Xte))
        targ[label].append(yte[label])
        rmse[label].append(RMSE(targ[label][-1], pred[label][-1]))

# %%
pd.DataFrame(rmse).mean().mean()
