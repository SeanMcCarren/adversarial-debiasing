# UCI adult example from Zhang et al, 2018, but with openml dataset 1590
# %%
from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_openml

X, y = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)
a = X['sex']

X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
    X, y, a, test_size=0.2, random_state=42
)


ct = make_column_transformer(
    (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("normalizer", StandardScaler()),
            ]
        ),
        make_column_selector(dtype_include="number"),
    ),
    (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="if_binary", sparse=False)),
            ]
        ),
        make_column_selector(dtype_include="category"),
    ),
)

pipeline = None # set later
def evaluate(*args):
    if len(args) == 2 and args[1] % 30 > 0:
        return
    global pipeline
    
    pred = pipeline.predict(X_test)

    pos_label = y_test[0:1].values[0]
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection rate": selection_rate},
        y_true=y_test == pos_label,
        y_pred=pred == pos_label,
        sensitive_features={"true label":y_test, "gender":a_test},
    )
    print(mf.overall)
    print(mf.by_group)

# %%

eps = 10e-6

import torch
class AdversaryModel(torch.nn.Module):
    def __init__(self):
        super(AdversaryModel, self).__init__()
        self.c = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.l = torch.nn.Linear(3, 1)

    def forward(self, x):
        y_hat = x[:,0]
        y = x[:,1]

        # ERROR INTERVALS

        sig_inv_y_hat = torch.log((y_hat + eps) / ((1+eps)-y_hat))
        c_plus_1 = 1+torch.abs(self.c)
        mul = torch.mul(c_plus_1, sig_inv_y_hat)
        s = torch.sigmoid(mul)
        
        s = s.view(-1, 1)
        y = y.view(-1, 1)
        concat = torch.hstack((s, torch.mul(s, y), torch.mul(s, 1-y)))

        z_hat = torch.sigmoid(self.l(concat))

        return z_hat

adversary_model = AdversaryModel()

def optimizer_constructor(model):
    global schedulers
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    schedulers.append(torch.optim.lr_scheduler.ExponentialLR(optim, 0.99))
    return optim

schedulers = []
def update_schedulers(*args):
    global schedulers
    for scheduler in schedulers: scheduler.step()

from math import sqrt
def update_alpha(model, step):
    model.alpha = sqrt(step)

model = AdversarialFairnessClassifier(
    predictor_model=[],
    adversary_model=adversary_model,
    predictor_optimizer=optimizer_constructor,
    constraints="equalized_odds",
    epochs=2,
    batch_size=2 ** 7,
    learning_rate=0.1,
    callbacks=[update_schedulers, update_alpha, evaluate],
    random_state=123,
)


pipeline = Pipeline(
    [
        ("preprocessor", ct),
        ("classifier", model),
    ]
)
# with torch.autograd.detect_anomaly():
pipeline.fit(X_train, y_train, classifier__sensitive_features=a_train)

evaluate()

# %%
# ON TRAINING SET
pred = pipeline.predict(X_train)
pos_label = y_test[0:1].values[0]
mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection rate": selection_rate},
    y_true=y_train == pos_label,
    y_pred=pred == pos_label,
    sensitive_features={"true label":y_train, "gender":a_train},
)
print(mf.overall)
print(mf.by_group)
# %%
