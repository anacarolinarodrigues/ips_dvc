# train.py
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
from pathlib import Path

import yaml
params = yaml.safe_load(open("./params.yaml"))["modelling"]


def load_data(data_path):
	train_data = pd.read_csv(data_path)
	labels = train_data['Labels']
	data = train_data.drop('Labels', axis=1)
	return data, labels

def main(repo_path):
	train_csv_path = repo_path / "data/processed/training_data.csv"
	X, y = load_data(train_csv_path)
	neigh = KNeighborsClassifier(n_neighbors=params['n_neighbors'], 
								 weights=params['weights'], 
								 algorithm=params['algorithm'])
	trained_model = neigh.fit(X, y)
	dump(trained_model, repo_path / "model/model.joblib")

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)