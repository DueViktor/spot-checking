from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from pathlib import Path

RESULTS_DIRECTORY = Path('results')


CLASSIFIER_SPACE = {
    'classification': [
        {
            'clf__estimator': [RandomForestClassifier()],
            'clf__estimator__criterion': ['gini', 'entropy']
        },
        {
            'clf__estimator': [SVC()]
        },
        {
            'clf__estimator': [KNeighborsClassifier()]
        },
        {
            'clf__estimator': [DecisionTreeClassifier()]
        }
    ],
    'regression': [
        {
            'clf__estimator': [RandomForestRegressor()]
        }
    ]
}
