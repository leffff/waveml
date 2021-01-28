# waveml
Open source machine learning library for performance of a weighted average  and linear transformations over stacked predictions

### Pip
```
pip install waveml
```

## Usage Example:
```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from vecstack import StackingTransformer
from sklearn.metrics import mean_squared_error
from waveml import WaveRegressor, WaveTransformer
from waveml.metrics import SAE
```
Loss function
```python
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())    
```
Stacking ensemble
```python
stack = StackingTransformer(
    estimators=[
        ["GBR", GradientBoostingRegressor()],
        ["RFR", RandomForestRegressor()],
        ["ETR", ExtraTreesRegressor()]
    ],
    n_folds=5,
    shuffle=True,
    random_state=42,
    metric=rmse,
    variant="A",
    verbose=0
)
```
Data
```python
X, y = load_boston(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
```
Training a stacking ensemble
```python
stack.fit(X_train, y_train)
print("Individual scores:", np.mean(stack.scores_, axis=1))
```
Output:
```
Individual scores: [3.54600289 3.7031519  3.31942812]
```

Stacked predictions
```python
SX_train = stack.transform(X_train)
SX_test = stack.transform(X_test)
```

LinearRegression
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(SX_train, y_train)
print("LinearRegression:", rmse(y_test, lr.predict(SX_test)))
```
Output
```
LinearRegression: 3.064970532826568
```

## What is WaveRegressor?
WaveRegressor is a model that performs a weighted average over stacked predictions

```python
wr = WaveRegressor(verbose=0, n_opt_rounds=1000, loss_function=SAE)
wr.fit(SX_train, y_train)
print("WaveRegressor:", rmse(y_test, wr.predict(SX_test)))
```
Output:
```
WaveRegressor: 3.026784272554217
```

## Why is it better than Linear Regression?
The three main differance between WaveRegressor and linear regression: </br>
>1) WaveRegressor does not fit an intercept. Only coefficients </br>
>2) It can optimize several metrics that are present in ```metrics.py``` </br>
>3) To achieve a higher performce you should experiment with a ```loss_function``` parameter </br>

## What is WaveTransformer?
WaveTransformer is a model that performs linear transformations on each feature in a way that minimizes an error betbeen a feature and a target value </br>
WaveTransformer does a cross validation process therefore it does not overfit and can be used to transform training data

## Why to combine the two?
Combining the two models increases prediction quality

## Combining example
Tune stacked predictions
```python
wt = WaveTransformer(verbose=0, n_opt_rounds=1000, learning_rate=0.0001, loss_function=SAE)
wt.fit(SX_train, y_train, n_folds=5)
TSX_train = wt.transform(SX_train)
TSX_test = wt.transform(SX_test)
```
Perform weighted average over transformed stacked predictions
```python
wr.fit(TSX_train, y_train)
print("WaveTransformer + WaveRegressor:", rmse(y_test, wr.predict(SX_test)))
```
Output:
```
WaveTransformer + WaveRegressor: 3.0190282172825995
```
