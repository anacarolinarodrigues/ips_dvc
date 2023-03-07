import itertools
import subprocess

n_neighbors = [1, 2, 3, 4, 5]
weights = ['uniform', 'distance']
algorithm = ['ball_tree', 'kd_tree', 'brute']

for neigh, wei, alg in itertools.product(n_neighbors, weights, algorithm):
	subprocess.run(['dvc', 'exp', 'run', '--queue', 
					'--set-param', f'modelling.n_neighbors={neigh}',
					'--set-param', f'modelling.weights={wei}',
					'--set-param', f'modelling.algorithm={alg}'])