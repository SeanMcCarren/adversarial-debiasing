# adversarial-debiasing
This repository contains a experimentations using the Adversarial Debiasing [1] technique to debias the predictions of machine learning models. I implemented this technique in the [Fairlearn](https://github.com/SeanMcCarren/fairlearn) package (but it is currently still under review). The experimentation procedure is replicated from [1], and validates these findings.

### Reproduce

To reproduce these experiments:
- Install python and pip, preferably Python 3.8 or higher.
- Clone this repository.
- Install dependencies using `pip install -r requirements.txt` from the root of this repository.
- Install Fairlearn by cloning [this repo](https://github.com/SeanMcCarren/fairlearn) and calling `pip install .` at the root.
- Run the experiment files in this repository.

### References

[1] Zhang, B. H., Lemoine, B., & Mitchell, M. (2018, December). Mitigating unwanted biases with adversarial learning. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society (pp. 335-340).

