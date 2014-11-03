## CH06 Classification II: Sentimental Analysis

> *Opinion mining from twitters*

> *Exactly speeking, we do not intend to build a advanced sentimental classifier.  Instead, we just introduce Naive Bayes as well as POS(part of speech) tagging when solving this problem.*

### Naive Bayes

* Naive Bayes

	* beautiful and practical algorithm, can ignore unrelated features automatically, fast to learn and predict, do not cost too much memory or storage
	
	* **naive: assume features are independent on probability**, but naive bayes works well in most situation when features do not satisfy probability independence
	
	* result representation: P( C | F<sub>1</sub>,F<sub>2</sub> )
	
		> *C: classification result of an example ( e.g.: 'positive sentiment', 'negative sentiment' )*
	
		> *F<sub>1</sub>: feature 1 (e.g.: counts of 'awesome' in twitter message)*
	
		> *F<sub>2</sub>: feature 2 (e.g.: counts of 'crazy' in twitter message)*
	
	* calculate P( C | F<sub>1</sub>,F<sub>2</sub> ):
		
		> according to Bayes theorem:
		
		> P(F<sub>1</sub>,F<sub>2</sub>)\*P(C | F<sub>1</sub>,F<sub>2</sub>) = P(C)\*P(F<sub>1</sub>,F<sub>2</sub> | C)
		
		> **P(C|F<sub>1</sub>,F<sub>2</sub>) = P(C)*P(F<sub>1</sub>,F<sub>2</sub>|C) / P(F<sub>1</sub>, F<sub>2</sub>)**
		
		> **posterior\_probability = prior\_probability * likelihood / evidence**
		
		> prior\_probability( P(C) ），evidence( P(F<sub>1</sub>,F<sub>2</sub>) ) are easy to calculate from training examples
		
	* calculate likelihood( P(F<sub>1</sub>,F<sub>2</sub>) ):
	
		> **P(F<sub>1</sub>,F<sub>2</sub>|C) = P(F<sub>1</sub>|C)\*P(F<sub>2</sub>|C,F<sub>1</sub>) = P(F<sub>1</sub>|C)\*P(F<sub>2</sub>|C)**
		
		> because F<sub>1</sub>, F<sub>2</sub> are independent, P(F<sub>2</sub>|C,F<sub>1</sub>)=P(F<sub>2</sub>|C)
		
	* how to use P(C|F<sub>1</sub>,F<sub>2</sub>):
	
		> **P(C|F<sub>1</sub>,F<sub>2</sub>)=P(C)\*P(F<sub>1</sub>|C)\*(F<sub>2</sub>|C)/P(F<sub>1</sub>,F<sub>2</sub>)**
		
		> P(C="pos"|F<sub>1</sub>,F<sub>2</sub>)=P(C="pos")\*P(F<sub>1</sub>|C="pos")\*P(F<sub>2</sub>|C="pos")/P(F<sub>1</sub>,F<sub>2</sub>)
		
		> P(C="neg"|F<sub>1</sub>,F<sub>2</sub>) = P(C="neg")\*P(F<sub>2</sub>|C="neg")\*P(F<sub>2</sub>|C="neg")/P(F<sub>1</sub>,F<sub>2</sub>)
	
		> we select class with largest probility (P(F<sub>1</sub>,F<sub>2</sub>) can be ignored)
		
		> **C<sub>best</sub> = arg max<sub>c&in;C</sub>P(C=c)\*P(F<sub>1</sub>|C=c)\*P(F<sub>2</sub>|C=c)**

* Deal with division-by-zero problem when calculating probabilities
		
	* problem:
		
		> when calculating P(C|F<sub>1</sub>,F<sub>2</sub>,...,F<sub>n</sub>)=P(C)\*P(F<sub>1</sub>|C)\*P(F<sub>2</sub>|C)\*...\*P(F<sub>n</sub>|C)/P(F<sub>1</sub>,F<sub>2</sub>,...,F<sub>n</sub>) to show the probabilies, some feature (denoted by F<sub>i</sub>) did not appeared in training examples, which makes P(F<sub>1</sub>,F<sub>2</sub>...,F<sub>i</sub>,...,P<sub>n</sub>) = 0, so that we can not get P(C|F<sub>1</sub>,F<sub>2</sub>,...,F<sub>i</sub>,...,F<sub>n</sub>)
		
	* solution: use add-one smoothing, also named as Laplace smoothing (not Laplacian smoothing) to get an approximite probability
	
		> for add 1 on numerator, add 2 on denominator ( add 2 to keep P(F<sub>i</sub>=0|C="pos")+P(F<sub>i</sub>=1|C="neg")=1 and P(C="pos")+P(C="neg")=1 )
		
* Deal with mathematical underflow
	
	* problem:
	
		> when calculating **C<sub>best</sub> = arg max<sub>c&in;C</sub>P(C=c)\*P(F<sub>1</sub>|C=c)\*P(F<sub>2</sub>|C=c)**
	
		>	P(C=c<sub>i</sub>)\*P(F<sub>2</sub>|C="neg")\*P(F<sub>2</sub>|C=c<sub>i</sub>)*...*P(F<sub>n</sub>|C=c<sub>i</sub>) is too small, thus we get prediction value 0 on all class and can not make a comparation
		
	* solution:
		
		> calssification by comparing log[P(C)\*P(F<sub>1</sub>|C)\*P(F<sub>2</sub>|C)]
		
		> log[P(C)\*P(F<sub>1</sub>|C)\*P(F<sub>2</sub>|C)] = logP(C) + logP(F<sub>1</sub>)|C) + logP(F<sub>2</sub>|C)
	
		> **C<sub>best</sub> = arg max<sub>c&in;C</sub>( log P(C=c) + sum( log P(F<sub>k</sub>|C=c) )**

### Different kinds of Naive Bayes

* GaussianNB: 

	> assume features subject to Gaussian distribution (normal distribution)
	
	> e.g.: predict people's gender according to his/her height and width

* MultinomialNB:

	> assume features are appearence count
	
	> e.g.: features are word frequency or TF-IDF vector

* BernoulliNB: 
	
	> do not count word frequency, use whether each word appears as feature

### Train Model

* get Twitter data from Niek Sanders' corpus

	~~~python
	X,Y = load_sanders_data()
	for c in numpy.unique(Y):
		print("#%s: %i" % (c, sum(Y==c)))
	#irrelevant: 543
	#NEGATIVE: 535
	#NEUTRAL: 2082
	#POSITIVE: 482
	~~~ 

* data preprocess

	> assumption for simplicity : ignore neutral and irrelevant (only keep positive and negative) in dataset

	~~~python
	X, Y = load_sanders_data()
	pos_neg_idx = numpy.logical_or(Y=="positive",Y=="negative")
	X = X[pos_neg_idx]  # selecting features
	Y = Y[pos_neg_idx]	# selecting labels
	Y = Y=="positive"	# 1 for positive, 0 for negative
	~~~

* pipeline TF-IDF Vectorizer and Naive Bayes Classifier
	
	~~~python
	from sklearn.feature_extration.text import TfidfVectorizer
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.pipeline import Pipeline
	
	# return a pipeline can be used for fit() and predict()
	def create_ngram_model():
		tfidf_ngrams = TfidfVectorizer(ngram_range=(1,3), analyzer="word", binary=False)
		classifier = MultinomialNB()
		pipeline = Pipeline( [('vect', tfidf_ngrams), ('classifier', classifier)] )
		return pipeline
	~~~
	
* function for training and cross validating

	> use ShuffleSplit instead of KFold on a small example set 

	> KFold splits the data-set into K fold with continuous examples. 
	
	> In situation when data-set is small, data-set should be shuffled before split to make sure the examples are randomly appeared 
	
	~~~python
	from sklearn.metrics import precision_recall_curve, auc
	from sklearn.cross_validation import ShuffleSplit
	
	def train_model( classifier_factory, X, Y):
		# init
		cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)
		test_scores    = []
		precision_scores = []
		
		# train and test 		
		for train, test in cv:
			x_train, y_train = X[train], Y[train]
			x_test,  y_test  = X[test],  Y[test]
			
			classifier = classifier_factory()
			classifier.fit( x_train, y_train )
			
			train_score = classifier.score( x_train, y_train )
			test_score  = classifier.score( x_test,  y_test  )
			
			test_scores.append( test_score )
			proba = classifier.predict_proba( x_test )
			precision, recall, pr_thresholds = precision_recall_curve( y_test, proba[:,1] )
			precision_scores.append( auc(recall, precision) )
		
		# summary
		print "%.3f\t%.3f\t%.3f\t%.3f" % (numpy.mean(scores), numpy.std(scores), numpy.mean(precision_scores), numpy.std(precision_scores))
	~~~

* combine above all together

	~~~python
	train_moel(create_ngram_model, X, Y)
	~~~


