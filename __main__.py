from src.pipe import Pipe
from sklearn.model_selection import train_test_split
from pandas import read_csv


def main(method: str):  # , yTarget: str, filePath: str) -> None:

    df = read_csv('data/classification/tmp.csv', header=None)
    X, y = df[df.columns.values[:-1]], df[8]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    pipeline = Pipe(method)
    pipeline.evaluate(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='describe the parser')

    parser.add_argument('--method', type=str, required=True, choices=[
                        'classification', 'regression'], help='specify whether the model should spot-check for classification or regression problems')
    # parser.add_argument('--file', type=str, required=True,
    #                     help='specify the file containing you data')
    # parser.add_argument('--target', type=str, required=True,
    #                     help='specify the name of the y-target')

    args = parser.parse_args()

    main(method=args.method)  # , yTarget=args.target, filePath=args.file)
