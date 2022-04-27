import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from show_weights import show_chan_weights, show_data


def read_data(filename):
	df = pd.read_csv(filename, sep=",", header=None)
	return df.to_numpy().T

@ignore_warnings(category=ConvergenceWarning)
def first_level_cross_val(data, type="none"):
	kf = KFold(n_splits=6, random_state = None, shuffle=False)
	kf.get_n_splits(data)
	fprs = {}
	tprs = {}
	decision_function = []
	labels = np.empty(data.shape[0]*data.shape[1])
	y_fin = []
	total_acc = 0

	i = 1
	for train_index, test_index in kf.split(data):
		X, y = get_all_splits(data, train_index, test_index)
		reg_coef = second_level_cross_val(data[train_index], i==1)
		acc, weights, model = run_svc(X["train"], X["test"], y["train"], y["test"], reg_coef)
		df = model.predict_proba(X["test"])[:, 1]

		decision_function.extend(df)
		y_fin.extend(y["test"])
		total_acc+=acc
		fprs[f"Fold {i}"], tprs[f"Fold {i}"], thresholds = roc_curve(y["test"], df, drop_intermediate=False)

		print(f"Accuracy for fold {i} with reg coef {reg_coef}: {acc}")
		if i == 1:
			# Visualize weights for brain surface 1
			show_chan_weights(np.abs(weights[0]), type)
			# Plot weight for each channel
			plot_weights_by_channel(weights[0], type)
			# Provide list of 5 dominant channels
			dominant = sorted(range(len(weights[0])), key = lambda sub: np.abs(weights[0][sub]))[-5:]
			print(f"Dominant channels for Fold 1 {type}")
			for index in dominant:
				print(f"{index} & ${weights[0][index]}$ \\\\")
		i+=1

	print(f"Total Cross-Validated Accuracy: {total_acc/6}")

	plot(fprs, tprs, type)
	fpr, tpr, thresh = roc_curve(y_fin, decision_function, drop_intermediate=False)
	plt.plot(fpr, tpr, label="Total")
	plt.legend()
	plt.show()

def plot_weights_by_channel(weights, type):
	plt.figure()
	plt.plot(weights)
	plt.title(f"Fold 1 Channel Weights for {type} Dataset")
	plt.xlabel("Channel")
	plt.ylabel("Weight", labelpad=15)
	plt.tight_layout()

def second_level_cross_val(data, plot):
	kf = KFold(n_splits=5, random_state = None, shuffle=False)
	kf.get_n_splits(data)
	n = 100
	lambdas = np.logspace(-10, 10, n)
	best_lambda = 0
	max_acc = 0
	accs = []
	for reg_coef in lambdas:
		mean_acc = 0
		for train_index, test_index in kf.split(data):
			X, y = get_all_splits(data, train_index, test_index)
			acc, weights, model = run_svc(X["train"], X["test"], y["train"], y["test"], reg_coef)
			mean_acc += acc
		accs.append(mean_acc/5)
		if mean_acc > max_acc:
			max_acc = mean_acc
			best_lambda = reg_coef
	if plot:
		plt.figure()
		plt.title("Fold 1 Accuracy per Regularization Parameter - Overt")
		plt.xlabel("Regularization Parameter")
		plt.ylabel("Classification Accuracy")
		plt.plot(lambdas, accs)
		plt.xscale('log')

	return best_lambda

def run_svc(train_X, test_X, train_y, test_y, reg_coef):
	# TODO: Determine if this is actually right
	svc = SVC(kernel='linear', C=1/reg_coef, probability=True)

	svc.fit(train_X, train_y)
	return svc.score(test_X, test_y), svc.coef_, svc

def get_all_splits(data, train_index, test_index):
	X_dict = {}
	y_dict = {}
	train_data = data[train_index]
	test_data = data[test_index]
	X, y = reshape_for_model(data)
	train_X, train_y = reshape_for_model(train_data)
	test_X, test_y = reshape_for_model(test_data)
	X_dict = {"train": train_X, "test": test_X, "all": X}
	y_dict = {"train": train_y, "test": test_y, "all": y}
	return X_dict, y_dict

def reshape_for_model(data):
	X = np.empty((data.shape[0] * 2, data.shape[2]))
	y = np.empty((data.shape[0] * 2))
	for i, trial in enumerate(data):
		X[i]=trial[0]
		y[i]=0
		X[i+data.shape[0]]=trial[1]
		y[i+data.shape[0]]=1
	return X, y

def concatenate_data(class_one_data, class_two_data):
	data = np.zeros((class_one_data.shape[0], 2, class_one_data.shape[1]))
	for i, trial in enumerate(class_one_data):
		data[i][0] = trial
		data[i][1] = class_two_data[i]
	return data

def plot(fprs, tprs, type):
	plt.figure()
	plt.grid(alpha=.4,linestyle='--')
	for name, fpr in fprs.items():
		plt.plot(fpr, tprs[name], label=name)
	plt.title(f"Fold and Total ROCs for {type} Dataset")
	plt.xlabel("P_FA")
	plt.ylabel("P_D")
	plt.legend(loc='lower right')

def main():
	CLASS_1_IMG_FILENAME = "data/feaSubEImg_1.csv"
	CLASS_2_IMG_FILENAME = "data/feaSubEImg_2.csv"
	CLASS_1_OVERT_FILENAME = "data/feaSubEOvert_1.csv"
	CLASS_2_OVERT_FILENAME = "data/feaSubEOvert_2.csv"

	data_class_one_img = read_data(CLASS_1_IMG_FILENAME)
	data_class_two_img = read_data(CLASS_2_IMG_FILENAME)
	data_class_one_overt = read_data(CLASS_1_OVERT_FILENAME)
	data_class_two_overt = read_data(CLASS_2_OVERT_FILENAME)

	data_img = concatenate_data(data_class_one_img, data_class_two_img)
	data_overt = concatenate_data(data_class_one_overt, data_class_two_overt)

	"""
	show_data(data_img[0][0], type="Class 1 Imagined Data")
	show_data(data_img[0][1], type="Class 2 Imagined Data")
	show_data(data_overt[0][0], type="Class 1 Overt Data")
	show_data(data_overt[0][1], type="Class 2 Overt Data")
	plt.show()
	"""

	# first_level_cross_val(data_img, type="Imagined")
	first_level_cross_val(data_overt, type="Overt")

if __name__ == "__main__":
	main()