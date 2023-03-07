# evaluate.py
from pathlib import Path
import pandas as pd
import os
import joblib
import json
from sklearn.model_selection import cross_validate
from sklearn.metrics import cohen_kappa_score, roc_auc_score, fbeta_score, make_scorer, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve
from modelling import load_data
from dvclive import Live

def calculate_metrics(repo_path, model, train_csv_path, test_csv_path, scoring, metrics_path):
	#load train data
	train_X, train_y = load_data(train_csv_path)
	#calculate cv performance
	cv_performance = cross_validate(model, train_X, train_y, cv=2, scoring=scoring, return_estimator=False, error_score="raise")
	#load test data
	test_X, test_y = load_data(test_csv_path)
	#calculate test performance
	y_pred = model.predict(test_X)
	y_prob = model.predict_proba(test_X)[:, 1]
	metrics = {'ts_F2':fbeta_score(test_y, y_pred, beta=2), 
			   'ts_CohensKappa':cohen_kappa_score(test_y, y_pred), 
			   'ts_AUC':roc_auc_score(test_y, y_prob)}
	#add train performance to dict
	for k in list(cv_performance.keys())[2:]:
		metrics[k.replace('test', 'cv')] = cv_performance[k][0]
	metrics_path.write_text(json.dumps(metrics))

def plot(repo_path, model, data, out_path):
	#load test data
	test_X, test_y = load_data(data)
	#calculate test performance
	y_pred = model.predict(test_X)
	y_prob = model.predict_proba(test_X)[:, 1]
	#roc plot
	fpr, tpr, thresholds = roc_curve(test_y, y_prob)
	roc = []
	for i in range(len(fpr)):
		roc.append({'fpr':fpr[i], 'tpr':tpr[i], 'thresholds':thresholds[i]})
	out_path.write_text(json.dumps({"roc": roc}))
	
	#live = Live()
	#live.log_sklearn_plot("roc", test_y, y_prob, out_path)

def main(repo_path):
	#load model
	model = joblib.load(repo_path / "model/model.joblib")
	
	#define scoring object
	scoring = {'F2': make_scorer(fbeta_score, beta=2), 
			   'Cohens Kappa': make_scorer(cohen_kappa_score),
			   'AUC': make_scorer(roc_auc_score)}
	
	train_csv_path = repo_path / "data/processed/training_data.csv"
	test_csv_path = repo_path / "data/processed/testing_data.csv"
	metrics_path = repo_path / "metrics/performance.json"
	calculate_metrics(repo_path, model, train_csv_path, test_csv_path, scoring, metrics_path)
	plot(repo_path, model, train_csv_path, repo_path / "metrics/train_roc.json")
	plot(repo_path, model, test_csv_path, repo_path / "metrics/test_roc.json")
	
if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)