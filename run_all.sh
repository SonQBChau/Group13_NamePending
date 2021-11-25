# run_all.sh
# Son Chau, November 2021

# download data
python download_data.py --url=http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip --path=../data/raw/

# run eda report
python papermill

# pre-process data 
python src/preprocess.py data/raw/winequality/???? results/processed/????

# create exploratory data analysis figure and write to file 
python src/eda.py --train=data/processed/training.csv --out_dir=results

# tune model
python src/ml_models.py --train=data/processed/training.csv --out_dir=results

# test model
python src/test_results.r --test=data/processed/test.csv --out_dir=results

# render final report
jupyter nbconvert --to <output format> <input notebook>
# or even better
jupyter-book build my_book/
