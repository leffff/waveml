# wave
Open source machine learning library for performance of a weighted average over stacked predictions

## Installation
```
git clone https://github.com/leffff/waveml.git
```
### Pip
```
pip install -r requirements.txt
```
### Conda
```
conda install --file requirements.txt
```

## Usage Example:
```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from vecstack import StackingTransformer
from sklearn.metrics import mean_squared_error
from waveml import WaveRegressor, WavePredictionTuner
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
    metric=mean_squared_error,
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
Individual scores: [13.214764   14.77008154 11.36905454]
```

Stacked predictions
```python
SX_train = stack.transform(X_train)
SX_test = stack.transform(X_test)
```
Perform a weighted average
```python
wr = WaveRegressor(verbose=0)
wr.fit(SX_train, y_train)
print("WaveRegressor:", mean_squared_error(y_test, wr.predict(SX_test)))
```
Output:
```
WaveRegressor: 9.730383467673033
```
Tune stacked predictions
```python
wpt = WavePredictionTuner(verbose=0)
wpt.fit(SX_train, y_train)
TSX_train = wpt.transform(SX_train)
TSX_test = wpt.transform(SX_test)
```
Perform weighted average over tuned stacked predictions
```python
wr.fit(TSX_train, y_train)
print("WavePredictionTuner + WaveRegressor:", mean_squared_error(y_test, wr.predict(SX_test)))
```
Output:
```
WavePredictionTuner + WaveRegressor: 9.68138105847055
```
