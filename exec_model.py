#!/usr/bin/env python3
import numpy as np #Matrix / Arrays
import pandas as pd #Data Analysis
from tensorflow import keras

CHERRY_TEST = "./data/CH_TEST.csv"

def main():
	dataset = pd.read_csv(CHERRY_TEST)

	def normalize(dataset):
		return (dataset-min(dataset))/(max(dataset)-min(dataset))

	def prepare_dataset():
		""" Extract labels from dataset """
		labels = dataset["Variety"].copy()
		dataset.drop(labels=["Variety"], axis=1, inplace=True)
		""" Normalize """
		""" durofel_inf_65	durofel_65_70	durofel_sup_75	durofel_70_75
		color_red	color_red_brown	color_brown	color_black	color_dark_brown
		brix_red	brix_dark_brown	brix_brown	brix_black
		Temperature_Outside	Temperature_Frut
		Variety """

		dataset['durofel'] = normalize(dataset['durofel'])
		dataset['color'] = normalize(dataset['color'])
		dataset['max_brix'] = normalize(dataset['max_brix'])
		return labels

	""" Load pre-trained model """
	model = keras.models.load_model('./models/cherry_class.ker')
	labels = prepare_dataset()
	""" Evalute model """
	scores = model.evaluate(x=dataset.values, y=labels.values, verbose=0)
	print('Model scores : ' + str(scores))

if __name__ == "__main__":
	main()
