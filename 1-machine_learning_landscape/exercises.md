# Chapter 1 Exercises

1. How would you define machine learning?.

I would define machine learning as writing a program that learns to do a task
from the data provided, instead of by following a hard coded set of rules.

2. Can you name four types of applications where it shines?.

- Applications where the goals are easily defined, but the algorithms to perform the task are not or are too complex;
- Applications where the environment is constantly evolving and thus can be updated with new data;
- Applications where exploring the data and gaining new insights is just as valuable as the classification or regression task at hand;
- Problems with no solution, so perhaps Machine Learning can find one

3. What is a labeled training set?.

A labeled training set is a set of data used for training a machine learning algorithm where the data is labeled - that is, for each sample, you (and the algorithm) know the value of the target feature.

4. What are the two most common supervised tasks?.

Classification is one for sure, but regression can be considered too, I think.

5. Can you name four common unsupervised tasks?.

Clustering is one of them, to find clusters within data of similar samples

Visualization is also one of them, where the algorithm creates a visualization of the unlabeled data so that we may find hidden patterns or just best comprehend the data at hand.

Dimensionality Reduction can be considered an unsupervised task, where we try to reduce the data features without losing information.

Lastly, anomaly detection also needs not to be done with labeled data.

Association Rule learning is also a common use - the goal is to find patterns within data

6. What type of algorithm would you use to allow a robot to walk in various.
unknown terrains?

An online learning algorithm, as the information about the terrain is being updated constantly and it has no previous accurate representation of such terrain. Moreover, reinforcement learning might also be interesting.

7. What type of algorithm would you use to segment your customers into multiple.
groups?

A clustering algorithm is the ask here.

8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?

Perhaps it is a semi-supervised problem. The data is labeled for starters, but the algorithm will then work and label emails on it's on. It is certainly more supervised than not, though.

9. What is an online learning system?.

It is a system that's constantly being fed new data and evolving, as contrary to batch systems which are trained with a huge batch of data

10. What is out-of-core learning?.

It's a strategy involving online learning where the training set would overflow the model's memory, so we feed the data to it in smaller batches and train it like so until it has completed the training.

11. What type of algorithm relies on a similarity measure to make predictions?.

Instance-based algorithms evaluate how similar the new instance is compared to it's training data.

12. What is the difference between a model parameter and a model hyperparameter?.

While a model parameter refer to the parameters of the model itself, such as the coefficients of a linear regression, the hyperparameter is a meta-parameter. In the linear regression example, it refers for example to how freely the model can vary these coefficient parameters

13. What do model-based algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?

Model based algorithms search for the best model for making predictions on the data they are using. They use the training data to fit their parameters, and then use their fine-tuned model to predict new data.

14. Can you name four of the main challenges in machine learning?.

Most come from data. We need a great amount of data, but also data that represents our problem and that does not contain many errors, missing data or outliers. We need relevant features and feature engineering to define so. We need neither to underfit nor to overfit the data, and there are deployment issues about maintenance of our system.

15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?

The model is overfitting. We may need a validation set, a train-dev set, or simply to choose a simpler model or to regulate it's hyperparameters appropriately.

16. What is a test set, and why would you want to use it?.

A test set is necessary to evaluate the model before it is deployed, as a bad model could cost a lot.

17. What is the purpose of a validation set?.

A validation set has the purpose of avoiding overfitting

18. What is the train-dev set, when do you need it, and how do you use it?.

The train-dev set has the purpose of helping with the evaluation of the model, specifically to know if poor performance is coming from the model itself (if it performs well in the validation set but poorly in the train-dev set) or from your data not being being representative of the data in which your model will be used, in the case of it performing will in the train-dev set but poorly on the test set (or dev-set)

19. What can go wrong if you tune hyperparameters using the test set?.

The model may become overfitted for that data, and thus generalize poorly.
