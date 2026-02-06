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

# 2 - End to end machine learning project

In this chapter, we will work through an end-to-end machine
learning project to get the feel for the field. We will work
with real-estate data, pretending to work for a real-estate 
company.

```python
import pandas as pd
import numpy as np
```

## Look at the Big Picture

Our first task will be to create a model to find the median
house price for each district in the data.

### Framing the problem

First, we need to understand what is the objective of building
this model. We ask our boss, and he answers that this model
will be used to decide whether or not it is worth it to invest
in a given area. Moreover, our model results will be fed to 
another ML system to make this decision, so it is important
our results are as accurate as possible.

The next question is **what are the current solutions** to this
objective. This will help us find a metric for our performance,
as well as put our solution in perspective. Our boss answers
that currently there is a team of experts that do this estimation,
and that sometimes their estimates are off by up to 30%.

Let's begin designing our system by answering some questions.
The most basic one - what type of system is ours? Supervised,
for a regression task, I believe. As for batch or online learning,
batch will suffice, as housing sells are usually a slow process,
and accuracy is key in this context.

The authors says this is, more specifically a **multiple regression**
problem, and an **univariate** one, because we are using multiple 
features to predict a single variable.

### Select a Performance Measure

A typical regression measure for regression problems is the 
root mean squared error (RMSE), in which for each instance, we
calculate the error, square this value, sum all square errors
and take the mean, and than root this value. This is more sensible
to high errors.

### Check the Assumptions

It is important to check if what we're assuming is actually right
about this system. For example, maybe the team that will use our
output data won't use the numeric data, but will categorize it 
in 'Cheap', 'Expensive', 'Medium', etc. In that case, our task 
would not be a regression task, but a classification one! We're 
better finding this out now, instead of months down the line.

In this case, after talking to the downstream team, they reassure
us that they'll need the numeric data.

## Get the Data

Time to get our hands dirty!

### Download the Data

We'll do that by code:

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
  tarball_path = Path('datasets/housing.tgz')
  if not tarball_path.is_file():
    Path('datasets').mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
      housing_tarball.extractall(path='datasets', filter='data')
  return pd.read_csv('datasets/housing/housing.csv')

housing_full = load_housing_data()
```

```python
housing_full.head()
```

```python
housing_full.info()
```

```python
housing_full['ocean_proximity'].value_counts()
```

```python
housing_full.describe()
```

```python
import matplotlib.pyplot as plt
housing_full.hist(bins=50, figsize=(12, 8))
```
We notice a few things:

- Median income is not expressed in USD. We talk to the
team that collected the data and discover the data has been
capped and scaled. The numbers roughly mean tens of thousands
of dollars.``

- House median value and age have also been capped, which 
might be a problem as median value is our target feature.
We might have to look for labels for these capped values or 
remove them entirely.

- The attributes are at very different scales;

- Many histograms are skewed right, which might need
some data transforming for the ML algorithms to properly
detect patterns

### Create a Test Set

We need to create a test set now, before cleaning 
the data and making decisions about which model to use.
This is because we might end up in the *data snooping bias*
situation. sklearn has a fuction for this, `train_test_split`:

```python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing_full, test_size=0.2,
                                       random_state=42)
```

This would be a purely random sampling method. It is fine if
the data is large enough, but if it is not, we run at the risk
of introducing sampling bias. 

Suppose we talked to experts who told us *median_income* is an
important predictor for our target measure. If we wanted to make
sure that our data has representatives of each income category,
we might want to make an *stratified split*.
We notice most values of median income are between 1.5 and 6.
Thus, we can use `pd.cut()` to make categories for that:

```python
housing_full['income_cat'] = pd.cut(housing_full['median_income'],
                                    bins=[0, 1.5, 3, 4.5, 6, np.inf],
                                    labels=[1, 2, 3, 4, 5])
```

```python
housing_full['income_cat'].value_counts().sort_index().plot.bar(rot=0)
```

Now we can use sklearn.model_selection's StratifiedShuffleSplit,
which returns a n lists of training and test indices. The n part
is good for later cross-validation. 

More precisely, StratifiedShuffleSplit has a `split()` method that
returns an iterator with the train and test indices.

```python
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

strat_splits = []
for train_idx, test_idx in splitter.split(housing_full,
                                          housing_full['income_cat']):
  train_set = housing_full.iloc[train_idx]
  test_set = housing_full.iloc[test_idx]
  strat_splits.append([train_set, test_set])
```

For now, we'll use the first split:

```python
strat_train_set, strat_test_set = strat_splits[0]
```

There's a shorter way to do this, although it will
require looping for using 10 strats:

```python
# strat_train_set, strat_test_set = train_test_split(
#   housing_full, test_size=0.2, stratify=housing_full['income_cat'],
#   random_state=42
# )
```

Lets check if the income categories proportions are 
equal to those in the original dataset:

```python
print(housing_full['income_cat'].value_counts(normalize=True))
print(strat_test_set['income_cat'].value_counts(normalize=True))
```

Pretty accurate. Now that we won't use `income_cat`
again, we might as well just drop it:

```python
for train_set, test_set in strat_splits:
  train_set = train_set.drop('income_cat', axis=1)
  train_set = test_set.drop('income_cat', axis=1)
```

```python
strat_train_set.head()
```

We took quite some time preparing this test set.
But it is our goal, what we will fit our model to 
excel at. It is a often neglected, but certainly important 
part of the machine learning process. Now, it is time to...

## Explore and Visualize the Data to Gain Insights

We will explore the training set. If it was enormous,
we would make an exploring sample. For now, we will
just make a copy of the training set so that we can
transform it freely:

```python
housing = strat_train_set.copy()
```

### Visualizing Geografical Data

Since we have Latitude and Logitude values, we can 
plot them in a scatter plot to get a good idea of 
where our data is coming from:

```python
import seaborn as sns
sns.scatterplot(x='longitude', y='latitude', data=housing, alpha=0.2)
```

```python
housing.plot.scatter(x='longitude', y='latitude', alpha=0.2)
```

Two ways of plotting: seaborn and native pandas plot

Now we'll add more elements to the visualization: higher
circle radius for bigger population, and redder colors
for more median house value:

```python
housing.plot.scatter(x='longitude', y='latitude', s=housing['population']/100,
                     c='median_house_value', cmap='jet', legend=True)
```

Here we can see that in general districts closer
to the ocean are more expensive, although there
seems to be more factors involved.

### Look for correlations

We can compute a correlation matrix with the 
`.corr` method with `numeric_only=True` argument:

```python
corr_matrix = housing.corr(numeric_only=True)
```

```python
corr_matrix['median_house_value'].sort_values(ascending=False)
```

Another way to check for correlations is to compute a
correlation matrix. Since we have 9 numerical attributes,
we'll do this with only the most promising ones:

```python
from pandas.plotting import scatter_matrix
atributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[atributes], alpha=0.2)
```

The most promising one seems indeed to be `median_income`, so 
we'll zoom in on that:

```python
housing.plot.scatter(x='median_income', y='median_house_value', alpha=0.2,
                     grid=True)
```

Although the data is noisy, we can clearly see an upward
trend. We can also notice a few more subtle problems:
There's a horizontal line around 450000, and also 35000
and maybe 280000. We may consider removing these districts
to avoid our algorithm to learning these data quirks.

### Exploring attribute combinations

Another thing we can do is create attribute combinations
that might be correlated to our target. Consider our
current columns:

```python
housing.columns
```

Perhaps better than total_rooms in a district, rooms per 
household might make more sense. Also, population per
household could be interesting, as well as a ration between
bedrooms and total rooms. Let's create these columns
and then check their correlation to the target feature:

```python
housing['relative_population'] = housing['population'] / housing['households']
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedroom_ratio'] = housing['total_bedrooms'].div(housing['total_rooms'],
                                                        fill_value=housing['total_bedrooms'].median())
```

```python
corr_matrix = housing.corr(numeric_only=True)
corr_matrix['median_house_value'].sort_values(ascending=False)
```

We found that our `rooms_per_household` and `bedroom_ratio`
have stronger correlations than the previous attributes. That
might be nice!

We will halt exploration for now, but we can always come back 
this later. Now, we'll...

## Prepare the Data for Machine Learning Algorithms!

The author recommends creating functions for feature
transformation instead of transforming manually. The
advantages are clear: reproducibility, ease to try
different transformations on the dataset, among others.

First, we'll revert to using the `strat_train_set`,
but without our target label, as we may want to transform
it in a different way than the predicting features.

```python
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
```

### Clean the Data

Imputing values for the NA values can easily be done
with pandas `fill_na` method, but the authors presents
a new way of doing this: With the `sklearn.impute` class
`SimpleImputer`. SimpleImputer takes an `strategy parameter`,
in this case we'll use the median. 

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
```

Since the median can only be calculated on numerical
values, we'll subselect those of our dataset:

```python
housing_num = housing.select_dtypes(include=[np.number])
```

Now we can calculate the attribute from the imputer:

```python
imputer.fit(housing_num)
```

The imputer has stored the result in it's `statistics_` 
instance variable. Let's check it out:

```python
imputer.statistics_
```

```python
housing_num.median().values
```

Now we can use the imputer to `transform` the training
dataset:

```python
X = imputer.transform(housing_num)
```

Note that the imputer also support other strategies,
such as `mean`, `most_frequent`, `constant, fill_value=`.

There are also more powerful imputer classes, such as
the `KNNImputer`, which uses the mean of the k-nearest
neighbours of that instance. There's also `IterativeImputer`,
which trains a regression model based on the non-missing
values to predict the missing ones. It then trains the
model again with the updated data and improves this
with each iteration.

The output of sklearn's transforms is by default a numpy
array, and thus has no index nor columns. If we set
`sklearn.set_config(transform_output='pandas')`, we can 
get pandas in, pandas out. Here, we'll just reconstruct 
the DataFrame:

```python
housing_tr = pd.DataFrame(X, index=housing_num.index,
                          columns=housing_num.columns)
```

### Scikit-Learn Design Principles

Scikit-Learn has a simple and consistent API, so building
a mental model of how things work is useful. We have:

#### Estimators

There are objects that estimate some values based on the
data given, such as our `SimpleImputer` with its `fit()`
method and `statistics_` result. All estimators use the
same `fit()` method. Some use two datasets as input: one
for the features, another for the labels. Any extra specification
is considered a hyperparameter and must be set in the estimator 
constructor, such as the `strategy` of the SimpleImputer.

#### Transformers

Some estimator also transform a dataset, such as `SimpleImputer`.
The API is again simple: use the `transform()` method with the 
dataset as input to transform it. These objects also have a 
`fit_transform()` method which is more optimized and does both
steps in a single method.

#### Predictors

Some objects are able to predict data based on features received
from a dataset. These are so called predictors, and an example
is the `LinearRegression` model used previously. A predictor
has a `predict()` method and also a `score()` method for evaluating
the results on a test set (and the corresponding labels, in the
case of supervised learning algorithms).

#### Inspection

All hyperparameters are accessible via instance attributes,
and calculated parameters are accessible with an underscore
suffix (such as `statistics_`)

### Handling Text and Categorical Attributes



