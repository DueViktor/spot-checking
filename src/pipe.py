from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.evaluator import ClassificationEvaluator, RegressionEvaluator


class Pipe:
    def __init__(self, method: str) -> None:
        self.method = method
        self.pipe = self.classification_pipe(
        ) if self.method == 'classification' else self.regression_pipe()

    def classification_pipe(self) -> None:
        return make_pipeline(RandomForestClassifier())

    def regression_pipe(self) -> None:
        return make_pipeline(RandomForestRegressor())

    def evaluate(self, X_train, y_train, X_test, y_test) -> None:
        self.pipe.fit(X_train, y_train)
        evaluator = ClassificationEvaluator(
        ) if self.method == 'classification' else RegressionEvaluator()
        evaluator.evaluate(X_test, y_test, self.pipe)
