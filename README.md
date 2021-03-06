# book links

[Building Machine Learning Systems with Python (Community Experience Distilled) ](http://www.amazon.com/Building-Learning-Community-Experience-Distilled/dp/1782161406/ref=sr_1_1?ie=UTF8&qid=1414288060&sr=8-1&keywords=building+machine+learning+systems+with+python)

![cover](http://ecx.images-amazon.com/images/I/51bIDil7swL._AA160_.jpg)

* [Chinese edition](http://www.ituring.com.cn/book/1192) 

* [code and dataset used in book](http://www.ituring.com.cn/book/download/ff381b73-f121-4a8b-a3cf-27ef62843333)

# content

* data cleaning
* explore and understand data
* how to find the best way to apply data to machine learning algorithms
* choose model and machine learning algorithms
* performance evaluation 

# beyond this book

* [http://blog.kaggle.com](http://blog.kaggle.com): winners of machine learning contests introduce their solutions

* [http://mlcomp.org](http://mlcomp.org): compare machine learning on multiple datasets

# notes and code comments

* [Chapter 01: Getting Started with Python Machine Learning](ch_01/)
	
	> NumPy, SciPy
	
	> predicting web traffic
	
* [Chapter 02: Learning How to Classify with Real-world Examples](ch_02/)

	> underfitting and overfitting, cross validation
	
	> adjust loss function according to the cost of False Positive and False Negative
	
	> feature engineering, good feature, feature selection automatically
	
	> normalization and z-score normalization
	
	> multi-classification
	
	> classification of Iris Data Set using thresh-hold discovery
	
	> classification of Grain Data Set using 1-nearest classification

* [Chapter 03: Clustering - Find Related Posts](ch_03/)
	
	> advantages and disadvantages of bag-of-words
	
	> generate feature vectors: min\_df, max\_df, token\_pattern, normalize, remove stop-words, use word-stem, select most important words according to TF-IDF
	
	> clustering with K-Means
	
	> other approaches which might improve clustering results
	
* [Chapter 04: Topic Modeling](ch_04/)

	> topic model
	
	> train LDA(Latent Dirichlet Allocation) model: code, parameters, representation, data preprocess, visualization of topics 
	
	> find closest topic of new documents
	
	> train LDA model on larger dataset

* [Chapter 05: Classification I - Detecting Pool Answers](ch_05/)

	> data preprocess, feature selection, label representation
	
	> classification with KNN, evaluate and find more features, problems of KNN
	
	> vias and bias, analyse and find solution when error rate can not be reduced
	
	> classification with Logistic Regression, analyse when error rate can not be reduced
	
	> trade-off between precision and callback-rate, AUC
	
	> remove useless features according the parametered trained by Logistic Regression
	
	> dump model to file for future works

* [Chapter 06: Classification II - Sentiment Analysis](ch_06/)
	
	> Naive Bayes, divison-by-zero problem, add-on-smoothing(Laplace smoothing), mathematical underflow problem
	
	> Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes
	
	> Model 1: classify positive sentiment and negative sentiment with Multinomial Naive Bayes
	
	> Model 2: evaluation and the reason of bad performance
	
	> automatically train and test on different parameter-combinations
	
	> data cleaning: emoticons, abbreviations
	
	> considering POS(Part Of Speech) and sentimental degree of each word

* [Chapter 07: Regression - Recommendations](ch_07/)
	

# packages

[Anaconda Python Distribution](http://continuum.io/downloads)(verified on my mac), including: 

* [NumPy](http://www.numpy.org/) : version 1.6.2+

* [SciPy](http://scipy.org/) : version 0.11+

* [Scikit-learn](http://scikit-learn.org/): version 0.13+

* [matplotlib](http://matplotlib.org)

~~~
export PATH="/Users/$(whoami)/anaconda/bin:$PATH"
~~~

other options:

* [enthought python](https://www.enthought.com/products/epd_free.php)

* [python(x,y)](http://code.google.com/p/pythonxy/wiki/Downloads)

* install Numpy, Scipy, Scikit-learn, matplotlib seperately

libs mentioned in some chapters

* [NLTK and PyYAML](http://nltk.org/install.html) [[Chapter 03](ch_03/)] [tutorial: Python Text Processing with NLTK 2.0 Cookbook]


# seek for help

* [http://metaoptimize.com/qa](http://metaoptimize.com/qa): Q&A site on machine learning

* [http:/stats.stackexchange.com](http:/stats.stackexchange.com): Q&A site on statistics

* [http://www.TwoToReal.com](http://www.TwoToReal.com)：Q&A instantly 

* \#machinelearning on Freenode: IRC channel

* [http://stackoverflow.com](http://stackoverflow)


# related tutorials

Scipy: 

* [http://www.scipy.org/Tentative_NumPy_Tutorial](http://www.scipy.org/Tentative_NumPy_Tutorial)

* [http://scipy-lectures.github.com](http://scipy-lectures.github.com)

* [http://docs.scipy.org/doc/scipy/reference/tutorial](http://docs.scipy.org/doc/scipy/reference/tutorial)

Numpy: 

* [NumPy Beginner's Guide - Second Edition](http://www.amazon.com/NumPy-Beginners-Guide-Second-Edition/dp/1782166084/ref=sr_1_3?ie=UTF8&qid=1414291464&sr=8-3&keywords=python+numpy) [[Chinese Edition](http://www.amazon.cn/Python%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B-NumPy%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%8D%97-%E5%8D%B0%E5%B0%BC-Ivan-Idris/dp/B00M2DL4Z8/ref=sr_1_1?ie=UTF8&qid=1414291404&sr=8-1&keywords=python%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B)]

	![](http://ecx.images-amazon.com/images/I/51CWeHVLOVL._AA160_.jpg)

Matplotlib(including pyplot): 

* [http://matplotlib.org/users/pyplot_tutorial.html](http://matplotlib.org/users/pyplot_tutorial.html)


