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

		dataset['durofel_inf_65'] = normalize(dataset['durofel_inf_65'])
		dataset['durofel_65_70'] = normalize(dataset['durofel_65_70'])
		dataset['durofel_sup_75'] = normalize(dataset['durofel_sup_75'])
		dataset['durofel_70_75'] = normalize(dataset['durofel_70_75'])

		dataset['color_red'] = normalize(dataset['color_red'])
		dataset['color_red_brown'] = normalize(dataset['color_red_brown'])
		dataset['color_brown'] = normalize(dataset['color_brown'])
		dataset['color_black'] = normalize(dataset['color_black'])
		dataset['color_dark_brown'] = normalize(dataset['color_dark_brown'])


		dataset['brix_red'] = normalize(dataset['brix_red'])
		dataset['brix_dark_brown'] = normalize(dataset['brix_dark_brown'])
		dataset['brix_brown'] = normalize(dataset['brix_brown'])
		dataset['brix_black'] = normalize(dataset['brix_black'])

		dataset['Temperature_Outside'] = normalize(dataset['Temperature_Outside'])
		dataset['Temperature_Frut'] = normalize(dataset['Temperature_Frut'])
		print(dataset.values)
		return labels

	model = keras.models.load_model('./models/cherry_class.ker')
	labels = prepare_dataset()
	scores = model.evaluate(x=dataset.values, y=labels.values, verbose=1)
	print('Model scores : ' + str(scores))


if __name__ == "__main__":
	main()
