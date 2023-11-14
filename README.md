# Data imputation using Determinantal Point Process (DPP) - based methods

This work presents an implementation of the models presented in the "[Improved clinical data imputation via classical and quantum determinantal point processes](https://arxiv.org/abs/2303.17893)" paper

## Prerequisites

Python 3.9

## Usage

```python

from models.imputers import DPPMissForest

ddpp_mf = DPPMissForest(batch_size=100, max_iter=5, n_estimators=10)

X_imputed = ddpp_mf.fit_transform(X_missing)

```

## License

[MIT](https://choosealicense.com/licenses/mit/)
