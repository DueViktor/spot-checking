from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from typing import Optional
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluator import ClassificationEvaluator, RegressionEvaluator
from src.switcher import ClfSwitcher
from config import CLASSIFIER_SPACE, RESULTS_DIRECTORY
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from src.utils import misc


class Pipe:
    def __init__(self, method: str, verbose: Optional[bool] = False) -> None:
        self.method = method
        self.verbose = True
        self.evaluator = ClassificationEvaluator(
        ) if method == 'classification' else RegressionEvaluator()

    def get_pipeline(self):
        return Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('standardscaler', StandardScaler()),
            ('clf', ClfSwitcher())
        ])

    def evaluate(self, X, y) -> None:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42)

        pipeline = self.get_pipeline()
        search = GridSearchCV(
            pipeline, CLASSIFIER_SPACE[self.method], n_jobs=-1, cv=3, verbose=self.verbose)
        search.fit(X_train, y_train)

        output_path = f'{RESULTS_DIRECTORY}/scores.csv'
        if self.verbose:
            print(f'Outputting cross validation scores to {output_path}')

        misc.save_df(
            df=pd.DataFrame(search.cv_results_).sort_values(
                by=['rank_test_score']).reset_index(drop='index'),
            path=output_path)

        self.evaluator.evaluate(search, X_test, y_test)
