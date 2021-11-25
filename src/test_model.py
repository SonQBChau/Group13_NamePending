"""
This script test the model
Usage: test_model.py --model_path=<arg1> --out_path=<arg2>

Options:
--model_path=<arg1>              Path (including filename) of where to locate the model file
--out_path=<arg2>           Path (not including filename) of where to locally write the file

Example:
python test_model.py --model_path=../data/raw/ --out_path=../data/raw/
"""



from docopt import docopt
from pandas.io.parsers import read_csv
from pathlib import Path
import ml_models as ml
import pandas as pd



def main():

    # parse arguments
    args = docopt(__doc__)

    # assign args to variables
    model_path = args['--model_path']
    out_path = args['--out_path']

    # load model
    model = ml.load_model(model_path)
    path = '../data/processed/'
    X_test = read_csv(path + 'X_test.csv')
    y_test = read_csv(path + 'y_test.csv')

    # test model
    results = model.score(X_test, y_test)

    # path = Path(path)
    # path.mkdir(parents=True, exist_ok=True)

    # results = pd.DataFrame(results)
    # results.to_csv(f"{path}/{path.name}.csv")




if __name__ == "__main__":
    main()
