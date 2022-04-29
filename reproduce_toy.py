# %%
from numpy import random
from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from numpy import column_stack, random

# %% Toy example
rng = random.default_rng(seed=123)
n = 100000
r = rng.choice([0, 1], size=n, replace=True)  # sensitive feature
v = rng.normal(loc=r, scale=1, size=n)
u = rng.normal(loc=v, scale=1, size=n)
w = rng.normal(loc=v, scale=1, size=n)

X = column_stack((r, u))
y = 1.0 * (w > 0)

X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
    X, y, r, test_size=0.1, random_state=42
)


def evaluate(pred):
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection rate": selection_rate},
        y_true=y_test,
        y_pred=pred,
        sensitive_features=a_test,
    )
    print(mf.overall)
    print(mf.by_group)
    return mf

import torch
def get_opt(model):
    return torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

model = AdversarialFairnessClassifier(
    predictor_model=[],
    adversary_model=[],
    predictor_optimizer=get_opt,
    adversary_optimizer=get_opt,
    epochs=-1,
    max_iter=5000,
    alpha=1,
    batch_size=2 ** 5,
    learning_rate=0.01,
    random_state=123,
)

model.fit(X_train, y_train, sensitive_features=a_train)

pred = model.predict(X_test)

mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection rate": selection_rate},
    y_true=y_test,
    y_pred=pred,
    sensitive_features=a_test,
)
print(mf.overall)
print(mf.by_group)

# %%
