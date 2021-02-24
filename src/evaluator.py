from sklearn.metrics import classification_report


class Evaluator:
    def __init__(self, pipe) -> None:
        self.pipe = pipe

    def predictor(self, X_test):
        preds = self.pipe.predict(X_test)
        return preds


class ClassificationEvaluator:
    def __init__(self) -> None:
        pass

    def evaluate(self, pipe, X_test, y_test):
        preds = Evaluator(pipe).predictor(X_test)
        print(classification_report(y_test, preds))


class RegressionEvaluator:
    def __init__(self) -> None:
        pass

    def evaluate(self, pipe, X_test, y_test):
        preds = Evaluator(pipe).predictor(X_test)
        print(3)
