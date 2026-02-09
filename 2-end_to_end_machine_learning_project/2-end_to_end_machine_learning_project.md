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

from sklearn import preprocessing

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
"""
strat_train_set, strat_test_set = train_test_split(
  housing_full, test_size=0.2, stratify=housing_full['income_cat'],
  random_state=42
)
""" 
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

Now it's time to handle text in our dataset. We only
have one text column, `ocean_proximity`. Let's look
at it's values:

```python
housing_cat = housing[['ocean_proximity']]
housing_cat['ocean_proximity'].unique()
```

There are two ways to encode this: One is with sequential
integers, another with one-hot encoding, which in pandas we
had as `get_dummy()` attribute. Sequential integers can be
obtained with the `OrdinalEncoder` class, from `sklearn.preprocessing`:

```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8]
```

We can check the categories with `categories_` attribute:

```python
ordinal_encoder.categories_
```

The problem here is that ML algorithms tend to consider
closer integers... closer. So we can use dummies with
one-hot encoding instead:

```python
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
housing_cat_encoded = one_hot_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8]
```

What is this output? It's a sparse matrix. Sparse matrices
are a more efficient way of storing matrices composed of 
only zeroes and ones. Instead of storing the whole matrix,
they store only the 1 values and their position. We can
convert to a dense numpy array with `.toarray()`, or
use `sparse_output=False` in the construction of the 
`OneHotEncoder()`.

Why use sklearn's encoder instead of pandas? sklearn remembers
the features it was trained on, and if you feed it a category
that wasn't previously informed, it will Raise or ignore,
depending on the set up when constructing the encoder.

### Feature Scaling and Transformation
 
Most ML algorithms perform better when features are
in a same scale. Then, it is a matter of transforming
and inverse transforming any set of features. This can
be done with the labels, too.

#### Min-Max scaling

Scales features down to a certain range, default 0-1:

```python
from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler(feature_range=(-1, 1))
housing_num = housing.select_dtypes(np.number)
housin_num_mm_scale = mmscaler.fit_transform(housing_num)
housin_num_mm_scale[:3, :]
```

#### Standard Scaling

Creates a z-value by subtracting the mean and dividing
by the standard deviation

```python
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std = std_scaler.fit_transform(housing_num)
housing_num_std[:3]
```

#### Bucketizing as numerical

This method can be used (for example, with `pd.qcut`)
for dealing with heavy-tailed features, and then using
the buckets as numerical values.

#### Bucketizing as categories

This strategy is useful for multimodal distributions
(with two or more peaks). We again bucketize, but then
treat the buckets id's as categories, meaning we encode
them again with perhaps `OneHotEncoder`, for example.
This ensures that the ML algorithm can learn different
patterns for different ranges of a value, such as of
`housing_median_age`

#### Radial Basis Function

Another way of dealing with multimodal distributions
is to create a feature for each mode representing the 
similarity between each instance's value and the
mode. This is usually done with a distance function,
such as the Gaussian Radial Basis Function, which
decays exponentially the further from the mode. 
The equation is `e^{-y(x-35)^2}`. It is implemented
in sklearn in `sklearn.metrics.pairwise.rbf_ernel`,
and we'll demonstrate it with the `housing_median_age`, 
which has a one of it's modes around 35:

```python
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
age_simil_35 = rbf_kernel(housing[['housing_median_age']].sort_values('housing_median_age'),
                          [[35]], gamma=0.1)
fig, ax = plt.subplots()
ax.hist(x = housing[['housing_median_age']].sort_values('housing_median_age'), bins=70)
ax.set_xlim((0, 51))
ax2 = ax.twinx()
ax2.set_xlim((0, 51))
ax2.plot(housing[['housing_median_age']].sort_values('housing_median_age'), age_simil_35,
         color='red')
```

#### Transforming the target feature

Transforming the label is also interesting
when it has a heavy tail. But since we want to 
predict the feature itself, not it's transformed
form, we need to reverse the transformation after
the prediction is made. Sklearn implements this 
in the `sklearn.compose` `TransformedTargetRegressor`
class. It has hyperparameters for the model and
`transformer` class instance to use, and returns
the reversed-transformed value with it's `predict`
method:

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[['median_income']], housing_labels)
some_new_data = housing[['median_income']].iloc[:5] # pretend this is new data
predictions = model.predict(some_new_data)
```

```python
predictions
```

### Transformation Pipelines

Transformation pipelines are a way of automating a 
sequence of steps in data transformation, to lastly
make a transformation or another estimator action,
such as a prediction. We make a pipeline with
`sklearn.pipeline`'s `Pipeline` object. In the following
example, we create a numerical attribute pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('stardardize', StandardScaler()),
])
```

The pipeline takes a list of name-estimator tuples. 
The names will be useful for hyperparameter tuning.
If you're lazy, you can use the `make_pipeline`
function instead of the `Pipeline` class, which
will accept transformer classes as positional
arguments, and automatically set names equal to the
classes names in lowercase and no underscores:

```python
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy='median'),
                             StandardScaler())
```
The pipeline itself has the `fit()`, `transform()`, or
`predict()` methods, depending on if the last estimator
has any of those. When called, the pipeline will `fit_transform()`
every single estimator and pass the result to the next one,
until the last estimator, where it will fit, transform, or predict
according to the appropriate method.

Let's look at the pipeline in action with our `housing_num`
variable:

```python
housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:2].round(2)
```

If we want to construct this as a DataFrame, we can use
the `get_feature_names_out()` method:

```python
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, 
    columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index
)
df_housing_num_prepared.iloc[:2].round(2)
```

We can access each estimator from the pipeline with by indexing
with ints or with the name of the estimator, such as a dictionary
object. 

With `ColumnTransformer` we're able to apply different pipelines
to different columns. We import it from `sklearn.compose`, and
the class takes a list of 3-tuples with the pipeline name,
the pipeline itself, and a list of columns to apply it to:

```python
from sklearn.compose import ColumnTransformer

cat_columns = ['ocean_proximity']
num_columns = ["longitude", "latitude", "housing_median_age",
               "total_rooms",  "total_bedrooms", "population",
               "households", "median_income"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

preprocessing = ColumnTransformer([
    ('cat', cat_pipeline, cat_columns),
    ('num', num_pipeline, num_columns)
])
```

However, we can get lazier than that. Listing all column
names is obviously not that convenient, and perhaps we don't
really care about naming each pipeline step. So instead
of using the class itself, we use `make_column_transformer()` and
`make_column_selector`:

```python
from sklearn.compose import make_column_transformer, make_column_selector

preprocessing = make_column_transformer(
    (cat_pipeline, make_column_selector(dtype_include=object)),
    (num_pipeline, make_column_selector(dtype_include=np.number))
)
```

Now we can just apply this column transformer to our housing data:

```python
housing_prepared = preprocessing.fit_transform(housing)
```

### Clusters

Advanced section, but used in the final pipeline:

```python
from sklearn.cluster import KMeans 
from sklearn.base import BaseEstimator, TransformerMixin

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters  
        self.gamma = gamma  
        self.random_state = random_state  

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)  
        self.kmeans_.fit(X, sample_weight=sample_weight)  
        return self # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)  

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

```

### Recap and wrap the pipeline

We've made some decisions about handling the data that
should be recapped before we make the final code that will
transform and prepare all the data:

- NA values in numerical attributes will be replaced by the median.
- NA values in categorical attributes will be replaced by the most frequent
- Categorical attributes will be mapped with One Hot Encoding
- Ratio columns will be added to hopefully better correlate with `median_house_value`: `bedroom_ratio`, `rooms_per_house` and `people_per_house`.
- Clusters and proximity to clusters will be added as they could
be more useful than longitude and latitude;
- Long-tailed features will be replaced with their log;
- All numerical features will be standardized;

The following code will build the pipeline to all of that:

```python
from sklearn.preprocessing import FunctionTransformer

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ['ratio'] # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_ratio,
                            feature_names_out=ratio_name),
        StandardScaler()
    )

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.,
                                  random_state=42)

default_num_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
)

preprocessing = ColumnTransformer([
    ('bedrooms', ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
    ('rooms_per_house', ratio_pipeline(), ['total_rooms', 'households']),
    ('people_per_house', ratio_pipeline(), ['population', 'households']),
    ('log', log_pipeline, ['total_bedrooms', 'total_rooms', 'population',
                           'households', 'median_income']),
    ('geo', cluster_simil, ['latitude', 'longitude']),
    ('cat', cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline # housing_median_age
)
```

Running this ColumnTransformer returns a np array with
24 features:

```python
housing.drop('income_cat', axis=1, inplace=True)
housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape
```

```python
preprocessing.get_feature_names_out()
```

## Select and Train a Model

After we've framed the problem, gathered the data,
explored and visualized it, and set a transformation
pipeline to automatically clean it up and prepare it
for the ML algorithms, it is finally time to select
and train an ML model!

### Train and Evaluate on the Training Set

Thanks to the fact that we did all the previous steps,
this should now be easy! Let's begin by training a simple
linear regression model:

```python
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
housing_predictions = lin_reg.predict(housing)
```

Since we chose root mean squared error as our
performance measure, let's measure it in the 
whole training set: 

```python
from sklearn.metrics import root_mean_squared_error
lin_rmse = root_mean_squared_error(housing_predictions,
                                   housing_labels)
lin_rmse
```

This is quite the error! It is explainable by the model
underfitting the training data. We cannot reduce constraints
in the model as it was not regularized, so perhaps we could
feed it different features or choose a more powerful model.
Let's go with the latter, using a `DecisionTreeRegressor`:

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_predictions,
                                    housing_labels)
tree_rmse
```

What's that, a perfect model? Such a thing doesn't exist, 
so the model is probably overfitting the data. To be sure,
we can use cross-validation to better evaluate it:

### Better Evaluation with Cross-Validation

We'll split the training set into 10 pieces. Each 
iteration, we'll train with 9/10 of the pieces and
evaluate with the 1/10 left out. This ensures the new
piece is not part of the learnt data, and thus cannot
have been overfit. `sklearn.model_selection` implements
this with `cross_val_score`:

```python
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                             scoring="neg_root_mean_squared_error",
                             cv=10)
```

cross_val_score expects an utility function instead of
a cost function, so we use a negative sign and negative
root mean squared error to evaluate. Let's check the
results:

```python
pd.Series(tree_rmses).describe()
```

Our tree is performing almost quite as poorly as the
linear regression model. We'll now try an even more
powerful model, an ensemble model: `RandomForestRegressor`,
imported from `sklearn.ensemble`:

```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring='neg_root_mean_squared_error',
                                cv=10)
pd.Series(forest_rmses).describe()
```
