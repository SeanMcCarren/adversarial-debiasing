# Code snippets from
# https://colab.research.google.com/notebooks/ml_fairness/adversarial_debiasing.ipynb
# The purpose of this code is to reproduce exactly the above buy using Fairlearn
# Recommended to run this in a notebook, because we do not cache data
# %%
import requests
url = 'http://download.tensorflow.org/data/questions-words.txt'
# all_categories = ['capital-common-countries','capital-world','currency','city-in-state','family','gram1-adjective-to-adverb','gram2-opposite','gram3-comparative','gram4-superlative','gram5-present-participle','gram6-nationality-adjective','gram7-past-tense','gram8-plural','gram9-plural-verbs']
# Choose categories by including them here:
# NOTE: all_categories contains all the possible categories in the dataset
categories = ['family']
r = requests.get(url, allow_redirects=False)
lines = r.text.split('\n')
analogies = []
valid_category = False
for line in lines:
    sp = line.split(' ')
    if len(sp) == 4 and valid_category:
        analogies.append(sp)
    elif len(sp) == 2:
        valid_category = sp[1] in categories

# assert len(analogies) == 19544
print(f"{len(analogies)} analogies!")

# %%
# This takes a long time :(

import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

# %%
# Make data
import numpy as np
X = []
y = []
for analogy in analogies:
    x = []
    for word in analogy[:-1]:
        x.append(wv[word])
    X.append(x)
    y.append(wv[analogy[-1]])
X = np.array(X)
y = np.array(y)

# Calculate "gender direction"
pairs = [
    ("woman", "man"),
    ("her", "his"),
    ("she", "he"),
    ("aunt", "uncle"),
    ("niece", "nephew"),
    ("daughters", "sons"),
    ("mother", "father"),
    ("daughter", "son"),
    ("granddaughter", "grandson"),
    ("girl", "boy"),
    ("stepdaughter", "stepson"),
    ("mom", "dad"),
]
m = np.array([wv[wf] - wv[wm] for wf, wm in pairs])
m = np.cov(np.array(m).T)
evals, evecs = np.linalg.eig(m)
dir = np.real(evecs[:, np.argmax(evals)])
gender_direction = dir / np.linalg.norm(dir) # normalized

# Calculate sensitive feature
a = np.array([np.dot(y_, gender_direction) for y_ in y])
# %%

import torch
torch.manual_seed(42)

class PredictorModel(torch.nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        init = torch.randn([300, 1])
        unit_init = init / (torch.norm(init))
        self.w = torch.nn.Parameter(unit_init, requires_grad=True)

    def forward(self, x):
        v = x[:, 1, :] + x[:, 2, :] - x[:, 0, :]
        y = v - torch.matmul(torch.matmul(v, self.w), self.w.T)
        return y

predictor_model = PredictorModel()

from fairlearn.adversarial import AdversarialFairness

def validate(model, step):
    if step % 100: return
    w = model.backendEngine_.predictor_model.w.detach().clone().numpy()
    proj = np.dot(w.T, gender_direction)
    size = np.linalg.norm(w)
    print(f"Learned w has |w|={size} and <w,g>={proj}.")

# scheduler = None
# def optim_constructor(model):
#     optim = torch.optim.Adam(model.parameters(), lr=2**(-14), weight_decay=0.01)
#     global scheduler
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9999)
#     return optim


model = AdversarialFairness(
    predictor_model=predictor_model,
    adversary_model=[],
    learning_rate=2**(-16),
    alpha=1.0,
    max_iter=10000,
    batch_size=1000,
    epochs=-1,
    callbacks=validate,
    random_state=42
)

model.fit(X, y, sensitive_features=a)
# %%

