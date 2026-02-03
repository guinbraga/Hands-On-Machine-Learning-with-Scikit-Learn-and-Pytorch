---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
---

# 1-machine_learning_landscape

### First example

The book starts with an first example of using linear
regression to make a prediction of life satisfaction 
according to GDP per capita.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
```

First We'll download and prepare the data:

```python
data_root = 'https://github.com/ageron/data/raw/main/'
lifesat=pd.read_csv(data_root + 'lifesat/lifesat.csv')
lifesat.head()
```

We'll then **visualize the data**

```python
x = lifesat[['GDP per capita (USD)']].values
y = lifesat[['Life satisfaction']].values

lifesat.plot.scatter(x='GDP per capita (USD)', y='Life satisfaction')
plt.axis([23_500, 62_500, 4, 9])
plt.show()
```
Then **select the model**

```python
model = LinearRegression()
```

**train the model**

```python
model.fit(x, y)
```

And make a prediction for puerto rico

```python
x_new = [[33_442.8]]
print(model.predict(x_new))
```

Funnily enough, replacing Linear Regresion with `knn` regression
is as simple as importing KNeighborsRegressor and using 
`model = KNeighborsRegressor(n_neighbors=3)`
