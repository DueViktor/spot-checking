from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluator import ClassificationEvaluator, RegressionEvaluator
from src.switcher import ClfSwitcher
from config import CLASSIFIER_SPACE, RESULTS_DIRECTORY
import pandas as pd
from src.utils import misc


class Pipe:
    def __init__(self, method: str) -> None:
        self.method = method
        self.steps = []
        # self.pipe.append(self.classification_pipe(
        # ) if self.method == 'classification' else self.regression_pipe())

    def get_pipeline(self):
        return Pipeline(steps=[
            ('StandardScaler', StandardScaler()),
            ('clf', ClfSwitcher())
        ])

    def evaluate(self, X_train, y_train) -> None:
        pipeline = self.get_pipeline()
        search = GridSearchCV(
            pipeline, CLASSIFIER_SPACE[self.method], n_jobs=-1, cv=3)
        search.fit(X_train, y_train)
        misc.save_df(
            df=pd.DataFrame(search.cv_results_).sort_values(
                by=['rank_test_score']).reset_index(drop='index'),
            path=f'{RESULTS_DIRECTORY}/scores.csv')
