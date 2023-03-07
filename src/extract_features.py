# prepare_data.py
from pathlib import Path
import pandas as pd
import os
import re
import radiomics
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

def extractor(images_files, masks_files, settings_path = None, bfcorrection = False):
	features={}
	for image_file, mask_file in zip(images_files, masks_files):
		inputImage = sitk.ReadImage(image_file, sitk.sitkFloat32)
		inputMask = sitk.ReadImage(mask_file, sitk.sitkUInt8)
		extr = featureextractor.RadiomicsFeatureExtractor(normalize = True, imageType = 'original')
		res = extr.execute(inputImage, inputMask)
		features[os.path.basename(os.path.normpath(image_file)).split('_')[0]] = res
	
	df = pd.DataFrame(features, columns = features.keys()).transpose()
	for col in df.columns:
		if re.search('diagnostics', col):
			del df[col]
	return df


def save_as_csv(features, labels, destination):
	features['Labels'] = labels
	features.to_csv(destination, index=False)

def main(repo_path):
	data_path = repo_path / "data"
	train_path = data_path / "raw/train"
	test_path = data_path / "raw/val"
	prepared = data_path / "processed"
	train_csv = pd.read_csv(os.path.join(prepared, 'train.csv'))
	test_csv = pd.read_csv(os.path.join(prepared, 'test.csv'))
	
	image_files = train_csv['filename']
	mask_files = train_csv['masks']
	labels = list(train_csv['label'])
	features = extractor(image_files, mask_files)
	save_as_csv(features, labels, os.path.join(prepared, 'training_data.csv'))
	
	image_files = test_csv['filename']
	mask_files = test_csv['masks']
	labels = list(test_csv['label'])
	features = extractor(image_files, mask_files)
	save_as_csv(features, labels, os.path.join(prepared, 'testing_data.csv'))

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)