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
""" v2: We just take color and durofel as a trend and max(brix), frut temperature has been deleted due to some errors in reporting """
CHERRY_QC_TRAINING = "./data/CH_EXPORT_QC_2018-2019_CSV_FORMATED_v2.csv"
CHERRY_TEST = "./data/CH_TEST.csv"

def main():
	dataset = pd.read_csv(CHERRY_QC_TRAINING)
	x = dataset

	model = keras.Sequential([
		keras.layers.Dense(3, input_dim=3, activation=tf.nn.relu), #4 dimensions in v2
		keras.layers.Dense(9, activation=tf.nn.relu),
		keras.layers.Dense(18, activation=tf.nn.relu),
		keras.layers.Dense(18, activation=tf.nn.relu),
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

		dataset['durofel'] = normalize(dataset['durofel'])
		dataset['color'] = normalize(dataset['color'])
		dataset['max_brix'] = normalize(dataset['max_brix'])
		return labels

	labels = prepare_dataset()

	""" Tensorboard """
	log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

	model.fit(x=dataset.values, y=labels.values, epochs=5, callbacks=[tensorboard_callback])
	scores = model.evaluate(x=dataset.values, y=labels.values, verbose=0)

	print('Model scores : ' + str(scores))

	model.save('./models/cherry_class.ker')

if __name__ == "__main__":
	main()
