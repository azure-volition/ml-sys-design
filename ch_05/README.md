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

### Train first classifion model with KNN:

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

### What's the next?

* Possible solutions: 

	* more training data?
	
	* use more/less complicate model parameters? (e.g.: increase/decrease K parameter of KNN)
	
	* improve feature set? (e.g.: try to remove some features and add some other features)
	
	* try another algorithm?
	 
* Analyse the situation: Bias-Variance and trade-off

	* High Bias: model is too simple (more training examples / rememove features won't help)
	
		> symptons: high test error, 
		
		> cure: more feature, more complicated model, other algorithms
		
	* High Variance: model is too complicated
	
		> cure: more training examples, less complicated model(e.g: larger K parameter of KNN), remove some features
	
		

 	
	

	 