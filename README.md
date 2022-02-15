# adversarial-debiasing
This repository contains a experimentations using the Adversarial Debiasing [1] technique to debias the predictions of machine learning models. I implemented this technique in the [Fairlearn](https://github.com/SeanMcCarren/fairlearn) package (but it is currently still under review). The experimentation procedure is replicated from [1], and validates these findings.

### Reproduce

To reproduce these experiments:
- Install python and pip, preferably Python 3.8 or higher.
- Clone this repository.
- Install dependencies using `pip install -r requirements.txt` from the root of this repository.
- Install Fairlearn by cloning [this repo](https://github.com/SeanMcCarren/fairlearn) and calling `pip install .` at the root.
- Run the experiment files in this repository.

### Discussion
An advantage of Adversarial Fairness is that we can train both accurate and fair models, and we can even balance these objectives. In fact, in Section \ref{sec:bench}, we have seen that we can train a classifier that is closer to satisfying a fairness constraint than a classifier trained to maximize predictive accuracy. Note that the predictive accuracy of our classifier is lower though, but this is a trade-off one would expect. 

A disadvantage of this method, however, is that training may be difficult to converge properly. This difficulty may deter less experienced users. Similar to other adversarial learning techniques, we occasionally encounter mode collapse, which is the situation where all predictions collapse to predicting one constant value. Especially in deeper models, mode collapse seems unavoidable. In the original paper, there was no documentation of experiments using neural networks consisting of multiple hidden layers, so perhaps we may assume the authors had similar difficulties. In fact, the authors state "the adversarial training method is hard to get right and often touchy, in that getting the hyperparameters wrong
results in quick divergence of the algorithm." [1, Sec. 8]

In shallow neural networks, I did not encounter this problem as often. Moreover, I tried typical remedies for mode collapse problems such as training the adversary more frequently than the predictor, varying the learning hyperparameters while training, and balancing the loss per classification target. Only varying the learning rate and $\alpha$ hyperparameters seemed to improve the likelihood of training convergence. Nonetheless, the problem of occasional mode collapse remains, which is a disadvantage. Other than this, adversarial debiasing may work really well and can be used to improve the fairness of arbitrary neural networks.

### References

[1] Zhang, B. H., Lemoine, B., & Mitchell, M. (2018, December). Mitigating unwanted biases with adversarial learning. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society (pp. 335-340).

