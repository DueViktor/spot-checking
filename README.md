# spot-checking

A repo build for the purpose of spot-checking (proof-of-concept) for classification and regression problems.

Should take a datasat, method and the y-column and then do the following:

1. Preprocess with appropiate values. Scaling
2. One-hot all categorical values.
3. Scale values of type int and float.

Perform 5 fold crossvalidation of the data splitted into a dataset for cross-validation aswell as one for the final testing.

Inspiration:
- https://machinelearningmastery.com/spot-check-regression-machine-learning-algorithms-python-scikit-learn/
- https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-python/
- https://thatascience.com/learn-machine-learning/spot-checking/
- https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
- https://www.reddit.com/r/MachineLearning/comments/kqm5pn/p_which_machine_learning_classifiers_are_best_for/
- http://www.davidsbatista.net/blog/2018/02/23/model_optimization/