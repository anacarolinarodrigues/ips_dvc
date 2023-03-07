# prepare_data.py
from pathlib import Path
import pandas as pd
import os

def get_files_and_labels(source_path):
	target = pd.read_csv(os.path.join(source_path, 'labels.csv'))
	images = []
	masks = []
	labels = []
	for image_path in source_path.rglob("*T2WAx.nii.gz"):
		filename = image_path.absolute()
		images.append(filename)
		P_ID = os.path.basename(os.path.normpath(image_path)).split('_')[0]
		label = target.loc[target['ID']==P_ID, 'Target'].values[0]
		labels.append(label)
		mask = os.path.join(source_path, P_ID+'_t2wax_gland.nii.gz')
		masks.append(mask)
	return images, masks, labels

def save_as_csv(filenames, masks, labels, destination):
    data_dictionary = {"filename": filenames, "masks": masks, "label": labels}
    data_frame = pd.DataFrame(data_dictionary)
    data_frame.to_csv(destination, index=False)

def main(repo_path):
    data_path = repo_path / "data"
    train_path = data_path / "raw/train"
    test_path = data_path / "raw/val"
    train_files, train_masks, train_labels = get_files_and_labels(train_path)
    test_files, test_masks, test_labels = get_files_and_labels(test_path)
    prepared = data_path / "processed"
    save_as_csv(train_files, train_masks, train_labels, prepared / "train.csv")
    save_as_csv(test_files, test_masks, test_labels, prepared / "test.csv")

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)