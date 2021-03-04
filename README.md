[![PyPI version](https://img.shields.io/pypi/v/waveml.svg?colorB=4cc61e)](https://pypi.org/project/waveml/) 
[![PyPI license](https://img.shields.io/pypi/l/waveml.svg)](https://github.com/leffff/waveml/blob/main/LICENSE)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/waveml.svg)](https://pypi.python.org/pypi/waveml/)

# waveml
Open source machine learning library for performance of a weighted average  and linear transformations over stacked predictions

### Pip
```
pip install waveml
```

## Overview

waveml features four models: </br>
> [WaveStackingTransformer](https://github.com/leffff/waveml#WaveStackingTransformer)</br>
> [WaveRegressor](https://github.com/leffff/waveml#WaveRegressor)</br>
> [WaveTransformer](https://github.com/leffff/waveml#WaveTransformer)</br>
> [WaveEncoder](https://github.com/leffff/waveml#WaveEncoder)</br>


## WaveStackingTransformer
Performs Classical Stacking

Can be used for following objectives:</br>
> Regression</br>
> Classification</br>
> Probability Prediction</br>

### Usage example

```python
from waveml import WaveStackingTransformer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

wst = WaveStackingTransformer(
    models=[
      ("CBR", CatBoostRegressor()),
      ("XGBR", XGBRegressor()),
      ("LGBMR", LGBMRegressor())
    ],
    n_folds=5,
    verbose=True,
    regression=True,
    random_state=42,
    shuffle=True
)

from sklearn.datasets import load_boston
form sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True)

SX_train = wst.fit_transform(X_train, y_train, prettified=True)
SX_test = wst.transform(X_test, prettified=True)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(SX_train, y_train)
lr.predict(SX_test)
```

### Sklear compatebility

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    steps=[
        ("Stack_L1", wst),
        ("Final Estimator", lr)
    ]
)

pipeline.fit(X_train, y_train)
pipeline.predict(X_test)
```

## WaveRegressor
Performs weighted average over stacked predictions

## WaveTransformer
Performs cross validated linear transformations over stacked predictions

## WaveEncoder
Performs encoding of categorical features in the initial dataset
