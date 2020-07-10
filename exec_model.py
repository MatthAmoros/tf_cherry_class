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
