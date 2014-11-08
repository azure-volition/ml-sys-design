##CH02 Classification

### Problem 1: 

Given feature examples, train a model to predict Iris Types of new example

Features: Sepal Length, Sepal Width, Petal Length, Petal Width

Iris types: Iris Setosa, Iris Versicolor, Iris Virginica

### Steps:

1. data visualization --> petal length can be used to seperate Iris Setosa from the other two kinds --> train the first model, using threshold of petal width is enough

2. consideration of building a model: 
  * structure of model
  * search process
  * loss function

### Cross validation: 

1. underfitting V.S overfitting, training error V.S testing error
2. cross-validation: 
  * leave-one-out validation: cost too much
  * n-fold validation: data distribution should be balanced when generating folds
  * False Positive, False Negative and using different cost function according cost of each error type.

### Problem 2:

Given feature of examples, train a model to predict grain type of new example

Features: area, perimeter, compactness, length of grain, width of grain, asymmetry coefficient, length of grain spout

Grain types: Canadian, Kama, Rosa

### Discussion about features:

Feature engineering: design a new feature calculated from existing features, e.g.: compactness is calculated from area and perimeter

Good feature: values are varied on important aspects while kept the same on unimportant aspects

Feature selection: select good features automatically

### Steps:

1. most-nearest classification using squared euclidean distance

2. Using n-fold validation on most-nearest classification is important

3. Analysis boundary of 3 grain types on compactiness and area
   
   > plot -> compactness's affect much less compared with area ->
   	X axis(area): range 10-22; Y axis(compactness): range 0.75-1.0 -> 	normalization is needed
      
4. Z-score normalization : 
	
	> features are described by the distance from the mean value measured by 	standard deviation
	
	~~~python
	features -= features.mean(axis=0)
	features /= features.std(axis=0)
	~~~
	
	> now the boundary looks much better
	
5. most-nearest classification can be generalized to k-nearest classification

### Multi-Classification:

* Approach 1: for N types, build N 2-classification model, each model (denoted model<sub>i</sub>) output type<sub>i</sub> and other types, then make a vote

* Approach 2: build a classification tree

* Many other approaches






