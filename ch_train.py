#!/usr/bin/env python3
# CNN Classifier on CSV input dataset.
# Keras API : https://keras.io/models/model/
# pip3 install keras
# pip3 install tensorflow
import datetime
import numpy as np #Matrix / Arrays
import pandas as pd #Data Analysis
import tensorflow as tf #Tensor Flow
from tensorflow import keras

# Data sets
CHERRY_QC_TRAINING = "./data/CH_EXPORT_QC_2018-2019_CSV_FORMATED.csv"
CHERRY_TEST = "./data/CH_TEST.csv"

def main():
	dataset = pd.read_csv(CHERRY_QC_TRAINING)
	x = dataset

	model = keras.Sequential([
		keras.layers.Dense(15, input_dim=15, activation=tf.nn.relu), #15 dimensions
		keras.layers.Dense(30, activation=tf.nn.relu),
		keras.layers.Dense(60, activation=tf.nn.relu),
		keras.layers.Dense(120, activation=tf.nn.relu),
		keras.layers.Dense(240, activation=tf.nn.relu),
		keras.layers.Dense(20, activation=tf.nn.softmax) #Probabilities out of 20 labels
	])

	""" According to documentation, sparse_categorical_crossentropy loss is good for classification with labels > 2
	and representated as integers"""
	model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

	test_case = dataset.iloc[[2]]

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

		return labels

	labels = prepare_dataset()

	""" Tensorboard """
	log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

	model.fit(x=dataset.values, y=labels.values, epochs=200, callbacks=[tensorboard_callback])
	scores = model.evaluate(x=dataset.values, y=labels.values, verbose=0)

	print('Model scores : ' + str(scores))

	model.save('./models/cherry_class.ker')

if __name__ == "__main__":
	main()
