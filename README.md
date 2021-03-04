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
> * [WaveStackingTransformer](https://github.com/leffff/waveml#WaveStackingTransformer)</br>
> * [WaveRegressor](https://github.com/leffff/waveml#WaveRegressor)</br>
> * [WaveTransformer](https://github.com/leffff/waveml#WaveTransformer)</br>
> * [WaveEncoder](https://github.com/leffff/waveml#WaveEncoder)</br>


### WaveStackingTransformer
Performs Classical Stacking

### WaveRegressor
Performs weighted average over stacked predictions

### WaveTransformer
Performs cross validated linear transformations over stacked predictions

### WaveEncoder
Performs encoding of categorical features in the initial dataset
