## CH05 Classification - Detecting Pool Answers

### Problem: Detecting Pool Answers:

* Representation of Data: Examples -> text ; Labels -> 0/1 (whether is a acceptable answer)

* Algorithms: K-Nearest? Logistic Regression? Decision Tree? SVM? Naive Bayes?

* Dataset: [http://www.clearbits.net/torrents/2076-aug-2012](http://www.clearbits.net/torrents/2076-aug-2012)

### Data Preprocess: 

* Select data: with CeationData earlier than 2011 (6,000,000 posts for training)

* Format data( *[so\_xml\_to\_tsv.py](so_xml_to_tsv.py)* ): XML -> TAB splitted,

	> so the data can be processed fast 

* **Examples**: Select useful fields ( *[choose_instance.py](choose_instance.py)* )

	* PostType(question or answer), ParentID(which question belong to)
	
	* CreationDate(time between ask question and answer)
	
	* Score
	
	* Body(content of question / answer): 
	
	* AcceptedAnswerId

	**Fields not been selected:**

	* ViewCount: not highly related with answer correctness, a more seriouse problem is that we can't get view count when the new answer is submitted

	* CommentCount: simlar with ViewCount
	
	**Other fields:**

	* OwnerUserId: is useful when we decide to consider feature about users
	
	* Title: provide more information about questions, we do not use this field yet now

* **Labels**: what is a good answer?

	* IsAccepted? : bad idea ( 1. only reflect the questioner's opinion; 2. better answer won't be accepted once another answer has already be accepted )
	
	* Best score and worst score? : bad idea (one answer is scored 2, another 4, but they are all good answer )
	
	* Good answer for score > 0, bad answer for score <= 0

### Train the first model with KNN:

* KNN usage: model training and predicting
	
	~~~python
	from sklearn import neighbors
	knn = neighbors.KNeighborsClassifier( n_neighbors=2 )
	knn.fit( [[1],[2],[3],[4],[5],[6]], [0,0,0,1,1,1])
	knn.predict( 1.5 ) 		  # array([0])
	knn.predict( 37  ) 		  # array([1])
	knn.predict( 3   ) 		  # NeighborsWarning: same distance between k+1 and k
	knn.predict_proba( 1.5 ) # array([[ 1.  , 0. ]])
	knn.predict_proba( 37  ) # array([[ 0.  , 1. ]])
	knn.predict_proba( 3.5 ) # array([[ 0.5 , 0.5]])
	~~~
	
	parameters:
	~~~python
	KNeighborsClassifier( algorithm=auto, leaf_size=30, n_neighbors=2, p=2, warn_on_equidistance=True, wights=uniform )
	~~~

* Feature Engineering: 

	* CreateDate (text) --> TimeToAnswer (int)
	
	* HTML link count
	
		> will link-count be a good feature?  (we assume that the more links in answer, the higher quality will be promised) 
		
		> extract links (packages such as BeautifulSoup are more robust than following demo code)
		
		> **is link-count a good feature?**
		
		> guess: use histgram diagram to visualiz e: most examples has no links, so this feature seems not be able to generate a good classifier
		
		> confirm: train a classifier with this single feature to validate
	
		~~~python
		# link extraction (links in normal text rather than demo code) 
		import re
		code_match = re.compile( '<pre>(.*?)</pre>', re.MULTILINE|re.DOTALL )
		link_match = re.compile( '<a href="http://.*?".*?>(.*?)</a>', re.MULTILINE|re.DOTALL )
		
		def extract_features_from_body(all_str):
			link_count_in_code = 0
			for code_str in code_match.findall(all_str):
				link_count_in_code += len(link_match.findall(code_str))
			return len(link_match.findall(all_str)) - link_count_in_code

		# train model with link-count as feature
		X = numpy.asarray( [extract_features_from_body(text) for post_id, text in fetch_posts() if post_id in all_answers] )
		knn = neighbors.KNeighborsClassifier() #k=5 by default
		knn.fit(X, Y)
		
		# evaluate the calssifier
		from sklearn.cross_validation import KFold
		scores = []
		cv = KFold( n=len(X), k=10, indices=True )
		for train, test in cv:
			x_train, y_train = X[train], Y[train] #dataset for training
			x_test,  y_test  = X[test],  Y[test]  #dataset for validation
			clf = neighbors.KNeighborsClassifier()
			clf.fit(X, Y)	 # why X,Y rather x_train, y_train, bug?
			scores.append(clf.score(x_test, y_test))
		print( "Mean(scores)=%.4f\tStddev(scores)=%.5f"%(np.mean(scores,np.std(scores)))
		# Means(scores)=0.49100 Stddev(scores)=0.02888
		~~~
		
		> with an accuracy of 49% which is worse than randomly guess, apprently only use link-count won't distinguish good answers from bad answers (at least for KNN with n=5).
		
		> we need to try more features

	* lines-of-code, text-tokens (after remove code and links) in answer: 
	
		> Mean(scores)=0.58300 Stddev(scores)=0.02216
		
		> 4 of 10 answers can not be predict correctly, but currently more features brings high accuracy. let's add more features in following steps
	
	* more features: 
	
		* AvgSentLent: we assume answers with short sentences are easy to read
		
		* AvgWordLen: we assume short words are easy to read
		
		* NumAllCaps: we assume word with all capitalized letters are not easy to read
		
		* NumExclams: number of exclamation(!)
		
	* evaluate these 4 features: **WORSE than before**
	
		> Mean(scores)=0.57650 Stddev(scores)=0.03557 
		
	* analyse the reason: **how KNN works**

		> **we set Minkowski Distance Parameter to 2 (p=2), which means KNN regard all these 7 features equally**
		
		> the fact is, some features are much more important (such as NumLinks) than others (such as NumTextTokens)
		
	* problems with KNN: 
	
		> KNN can't learn weight of each feature
		
		> the more training examples, the more performance cost on predicting new example

### What's the next?

* Possible solutions: 

	* more training data?
	
	* use more/less complicate model parameters? (e.g.: increase/decrease K parameter of KNN)
	
	* improve feature set? (e.g.: try to remove some features and add some other features)
	
	* try another algorithm?
	 
* How to analyze: Bias-Variance and trade-off

	* High Bias: model is too simple (more training examples / rememove features won't help)
	
		> symptons: testing-error keeps high when increasing data-set; training-error reduce to the same level of testing-error when increaing data-set, but also keeps high
		
		> cure: more feature, more complicated model, other algorithms
		
	* High Variance: model is too complicated
	
		> symptons: huge gap between testing-error and training-error (low training-error but hight testing-error)
	
		> cure: more training examples, less complicated model(e.g: larger K parameter of KNN), remove some features
	
* Situations of pervious 5-NN model:

	* High Variance: already over-fitting

	* Reduce feature space: use less featurs has no effect

	* Increase parameter K: can increase accuracy from 0.5765 to 0.6280 after increase K from 5 to 90, which has too much performance cost when predicting new answer
	
	* **Let's turn to other algorithms!**

### Train the second model with Logistic Regression

* A brief introduction of Logistic Regression

* Default Prarmeters

	~~~python
	>>> print( LogisticRegression() )
	LogisticRegression( C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty=12, tol=0.0001 )
	~~~
	
	C: how complicated the Logistic Regression model should be

* Train model and predict new example

	~~~python
	from sklearn.linear_model import LogisticRegression

	# train model
	classifier = LogisticRegression()
	classifier.fit(X, Y)

	# model parameters trained with X,Y
	print(numpy.exp(classifier.intercept_), numpy.exp(classifier.coef_.ravel())) # [0.09437188] [1.80094112]

	# predict: P(x=val) is prebability that 'x=val' belong class 1
	def lr_predict( classifier, X ):
		return 1 / (1+numpy.exp( -(classifier.intercept_+classifier.coef_*X) ))
	print( "P(x=-1)=%.2f"%(lr_predict(classifier,-1)) # P(x=-1)=0.05
	print( "P(x= 7)=%.2f"%(lr_predict(classifier, 7)) # P(x= 7)=0.85
	~~~

* Experiment on different parameters
	
	Base line KNN model with K=90
	Logistic Regression with C=0.1, 100, 10, 0.01, 1.0
	
	Logistic Regression with C=0.1 brings the best accuration of 0.631, with standard variance or 0.02791. But it still have a high training-error rate and high-testing error. Training-error is almostly the same with testing-error. **It is still underfitting.**
	
	Maybe there is too much noise in data, or maybe we didn't not select the proporate features.

### What's the Next?

> We don't need a model capable to find all correct answers. Acatual we expect the model to find correct answers accurately, but don't care about the call-back rate very much.

### Trade-off between precision and callback-rate, AUC

* Precision and callback-rate

	> 
|               | Output Positive     | Output Negative     |
| ------------- |:--------------------| --------------------|
| True  in Fact | True  Positive(TP)  | False Negative(FN)  |
| False in Fact | False Positive(FP)  | True  Negative(TN)  |

	> Precession = TP / (TP+FP)

	> Callback-rate = TP / (TP+FN)

* AUC

	~~~python
	from sklearn.metrics import precision_recall_curve
	
	precision, recall, thresholds = precision_recall_curve(y_test, classifier.predict(x_test)
	~~~
	
	Curve: precision on different callback-rate
	
	AUC: area under the curve

* Find the thresh-hold setting and call-back rate in AUC curve with precision larger than 80%. 

	~~~python
	threshholds = numpy.hstack(([0],thresholds[medium]))
	idx80 = precisions>=0.8
	print("P=%.2f R=%.2f thresh=%.2f" % (precision[idx80][0], recall[idx80][0], thresholds[idx80][0]))
	# P=0.81 R=0.37 thresh=0.63
	~~~
	
	> If we accept a recall-rate of 0.37, we can use the thresh-hold of 0.63 to get an precision larger than 0.8
	
* Predicate an good answer with precision larger than 0.8

	~~~python
	thresh80 = threshold[idx80][0]
	probs_for_good = classifier.predict_proba(features_to_predict)[:,1]
	answer_class = probs_for_good>thresh80
	~~~

* Validate : print out precision, recall with classification_report

	~~~python
	from sklearn.metrics import classification_report
	print( classification_report(y_test, classifier.predict_proba[:,1]>0.63, target_names=['not accepted', 'accepted']))
	~~~
	
### Remove useless features

> remove feature with lower absolute-value coefficient in clf.coef_

### Dump model for prediction in future

	~~~python
	import pickle
	pickle.dump( classifier, open("logreg.dat","w"))
	classifier = pickle.load(open("logreg.dat","r"))
	~~~

	

 	
	

	 