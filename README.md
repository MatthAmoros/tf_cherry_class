# tf_cherry_class
Classify cherries varieties using experimental quality control parameters (TensorFlow PoC)

# How ?
I collected two years of quality report from a cherry manufacturing plant.
Data has been processed to remove faulty entries/miss placed information.
The dataset represents approximatly 4000 samples of cherry samples selected at the moment of arrival at the processing plant.
Each sample is made of 100 frut, of which we express firmness (durofel), color (5 colors range classification), brix (sugar level) and core temperature (°C).
The processing facility manage 20 varieties.
To be season-independent, I compiled together two years of quality reports (2018-2019).

# Dataset processing
I reduced my dataset from :
durofel_inf_65	durofel_65_70	durofel_sup_75	durofel_70_75
color_red	color_red_brown	color_brown	color_black	color_dark_brown
brix_red	brix_dark_brown	brix_brown	brix_black
Temperature_Outside	Temperature_Frut
Variety

To:
durofel (weighted range)
color (weighted range)
brix_max
Temperature_Frut
Variety

# Result
My lack of knowledge about CNN led me to make various errors.
1) Badly selected loss function
2) Badly formated input dataset
3) Blindly designing kernel layers

But during my journey in the fabulous Keras world, I feel like I learned a lot.

I was not able to make a good CNN that could predict cherry variety according to input dataset.
BUT, my model reached an accuracy of almost 30% on my test dataset, of which I concluded that brix, firmness and color are valid parameters to infer cherry variety, I just need more samples.

# Futur
To be updated with 2020 samples !
