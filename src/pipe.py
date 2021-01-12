from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class Pipe:
    def init(self, method: str) -> None:
        self.method = method
        self.pipe = None

    def create_pipe(self):
        self.classification_pipe() if self.method == 'classification' else self.regression_pipe()

    def classification_pipe(self) -> None:
        self.pipe = make_pipeline(RandomForestClassifier())

    def regression_pipe(self) -> None:
        self.pipe = make_pipeline(RandomForestRegressor())

    def evaluate(self, X_train, y_train, X_test, y_test) -> None:
        self.create_pipe()
        self.pipe.fit(X_train, y_train)
        print(self.pipe.score(X_test, y_test))
