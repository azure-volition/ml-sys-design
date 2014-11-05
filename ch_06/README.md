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
		
		> lib such as [mpmath](http://code.google.com/p/mpmath/) is not fast enough

### Different kinds of Naive Bayes

* GaussianNB: 

	> assume features subject to Gaussian distribution (normal distribution)
	
	> e.g.: predict people's gender according to his/her height and width

* MultinomialNB:

	> assume features are appearence count
	
	> e.g.: features are word frequency or TF-IDF vector

* BernoulliNB: 
	
	> do not count word frequency, use whether each word appears as feature

### Train Model 1: only consider positive and negative sentiment

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
	# 0.805 0.024 0.878 0.016
	# 80.5 correctness, 87.8% P/R AUC
	~~~

### Train Model 2: consider all sentiment

* all sentiments:

	~~~python
	X,Y = load_sanders_data()
	for c in numpy.unique(Y):
		print("#%s: %i" % (c, sum(Y==c)))
	#irrelevant: 543
	#NEGATIVE: 535
	#NEUTRAL: 2082
	#POSITIVE: 482
	~~~ 

* lable function:

	> regard all sentiment in 'pos_sent_list' as positive examples
	
	~~~python
	def tweak_labels(Y, pos_sent_list):
		pos = (Y==pos_sent_list[0])
		for sent_label in pos_sent_list[1:]:
			pos |= (Y==sent_label)
		Y = numpy.zeros(Y.shape[0])
		Y[pos] = 1
		Y = Y.astype(int)
		return Y
	~~~

	> label all sentiment in ['positive', 'negative'] as positive examples, all other sentiment in ['irrelevant','neutral'] as negative examples
		
	~~~python
	Y = tweak_labels(Y, ["positive", "negative"])
	~~~

* train model with new labels

	~~~python
	train_model(create_ngram_model, X, Y, plot=True)
	0.767 0.014 0.670 0.022
	~~~
	
* evaluation

	> P/R AUC is 0.670 
	
	> but sent/all = (482+535)/(482+535+2082+543) = 28%, which means if we always predict sent, an AUC of 72% still can be achieved
	
	> **performance of this model is BAD**

* lable 'positive' to 1, others('negative','irrelevant','neutral') to 0

	> **performance is bad**

* lable 'negative' to 1, others('positive','irrelevant','neutral') to 0
	
	> **performance is bad**

### Improve Model 2 by trying different parameter combinations automatically

* different paramters might be helpful
	* TfidfVectorizer
		* NGrams: (1,1) / (1,2) / (1,3)
		* min_df: 1 / 2 
		* IDF: user_idf = True / False ; smooth_idf = True / False
		* stop_words
		* log calculation on word-frequency:  sublinear_tf
		* word_cnt or word_appearence: binary = True / False
	* MultinominalNB 	
		* Add-one smoothing (Laplace smoothing): alpha = 1
		* Lidstone smoothing: alpha = 0.01 / 0.05 / 0.1 / 0.5
		* no smoothing: alpha = 0

* explore parameters automatically with ssikit-learn.GridSearchCV
	* name dictionary key of parameter as required by GridSearchCV
		
		> \<estimator\>\_\_\<subestimator\>\_\_...\_\_\<param_name\>
		
		> e.g.: ngram_range of TfidfVectorizer( named as vect in Pipline later)
		
		> Param\_grid={ "vect\__ngram_range"=[(1,1),(1,2),(1,3)] }
			
	* try all combination of parameters, get optimal combination from variable 'best_estimator_'
	
	* set score_func to tell GridSearchCV how to select best parameters
		* metrics.accuracy: not suitable because of our data is asymmetric
		* metrics.f1_score: measurement according to accuracy and callback 
			
			> F = 2 * accuracy * callback / (accuracy + callback)
* code 

	~~~python
	from sklearn.grid_search import GridSearchCV
	from sklearn.metrics import f1_score
	
	def grid_search_model( clf_factory, X, Y):
	    cv = ShuffleSplit( n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)
	    param_grid = dict( vect__ngram_range=[(1,1),(1,2),(1,3)],
	                       vect__min_df=[1,2],
	                       vect__stop_words=[None, "english"],
	                       vect__smooth_idf=[False,True],
	                       vect__use_idf=[False,True],
	                       vect__sublinear_tf=[False,True],
	                       vect__binary=[False,True],
	                       classifier_alpha=[0,0.01,0.05,0.1,0.5,1],
	                       )
	    grid_search = GridSearchCV( 
	    						clf_factory(), 
	    						garam_grid=param_grid, 
	    						cv=cv,
	    						score_func=f1_score,
	    						vebose=10
	    						)
	    grid_search.fit(X,Y)
	    return grid_search.best_estimator_	
	classifier = grid_search_model(create_ngram_model, X, Y)
	# will cost several hours to try 3*2*2*2*2*2*2*6=1152 parameter combinations on 10 fold dataset
	
	print classifier
	~~~

	> P/R AUC is 70.2%, increased by 3.3%

### Imporve Model 2 by data cleaning

* use emoticons

	~~~python
	emo_repl = {
		# positive emoticons
		"<3" : "good",
		":d" : "good",
		...
		"(:" : "good",
		# negative emoticons
		":/" : "bad",
		":>" : "sad",
		...
		":-S": "bad",
	}
	~~~

* replace abbreviation with non-abbreviation

	~~~python
	re_repl = {
		r"\br\b": "are",
		r"\bu\b": "you",
		...
		r"\bcan't\b": "can not",
	}
	~~~

* add cleaning code in preprocessor

	~~~python
	def create_ngram_model(params=None):
		def preprocessor(tweet):
			global emoticons_replaced
			tweet = tweet.lower()
			for k in emo_repl_order:
				tweet = tweet.replace(k, emo_repl[k])
			for r, repl in re_repl.iteritems():
				tweet = re.sub(r, repl, tweet)
			return tweet
	
	tfidf_ngrams = TfidfVectorizer(preprocessor=preprocessor, analyzer="word")
	~~~

	> this time: P/R AUC is 70.7%, increased by 0.5%

### Improve Model 2 by considering POS(Part Of Speech) tagging and sentiment scores (from SentiWordNet)

* POS(Part Of Speech) tagging 

	* nltk.pos_tag(): trained from dataset in [Pennn Treebank Project](http://www.cis.upenn.edu/~treebank)

	* tags: see [http://americannationalcorpus.org/OANC/penn.html](http://americannationalcorpus.org/OANC/penn.html)
		
	~~~python
	import nltk
	nltk.pos_tag(nltk.word_tokenize("This is a good book"))
	# [('This','DT'), ('is','VBZ'),('a','DT'),('good','JJ'),('book','NN')]
	# DT:  determiner, e.g.: 'the'
	# VBZ:  verb, 3rd person sing. present, e.g.: 'takes'
	# JJ: adjective, e.g.: 'green'
	# NN: noun, singular or mass, e.g.: 'table'
	~~~

* SentiWordNet: [http://sentiwordnet.isti.cnr.it](http://sentiwordnet.isti.cnr.it)

	> a file to store positive_score,negative_score,synonyms of a \<word, pos_tag\> pair
	
	> netural\_score = 1 - positive\_score - negative\_score

	> a word might have different meanings, which leads different scores and need to be processed with **word sense disambiguation**. for simplicity, we just use average scores here
	
	~~~python
	import csv, collections
	def load_sent_word_net():
		sent_scores = collections.defaultdict(list)
		with open(os.path.join(DATA_DIR, SentiWordNet_3.0.0_20130122.txt"), "r") as csvfile:
			reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
			for line in reader:
				if line[0].startswith("#"):
					continue
				if len(line)==1:
					continue
			POS,ID,PosScore,NegScore,SynsetTerms,Gloss = line
			if len(POS)==0 or len(ID)==0:
				continue
			#SynsetTerms examples: fantasy#1 fantasize#1 fantasise#1
			for term in SynsetTerms.split(" "):
				term = term.split("#")[0]
				term = term.replace("-", " ").replace("_", " ")
				key  = "%s/%s" % (POS, term.split("$")[0])
				sent_scores[key].append((float(PosScore),float(NegScore)))
		for key, value in sent_scores.iteritems():
			sent_scores[key] = np.mean(value, axis=0)
		return sent_scores
	~~~
	
* implementation: LinguisticVectorizer

	~~~python
	sent_word_net = load_sent_word_net()
	
	class LinguisticVectorizer(BaseEstimator):
		def def_feature_names(self):
			return numpy.array(
				['sent_neut','sent_pos','sent_neg','nouns','adjectives'
				,'vervbs','adverbs','allcaps','exclamation','question'
				,'hashtag','mentioning'] )
		def fit(self, doc, y=None):
			# we only plan to use this function like this: fit(doc).transform(doc)
			return self
		def transform(self, documents):
			# document socres
			netural_score,positive_score,negative_score,nouns,adjectives, \
			verbs, adverbs = numpy.array( [self._get_sentiments(doc) for doc in documents]).T
			allcaps = []
			exclamation = []
			question = []
			hashtag = []
			mentioning = []
			for doc in documents:
				allcaps.append( numpy.sum( [token.issupper() for token in doc.split() if len(token)>2] ) )
				exclamation.append( doc.count("!") )
				question.append( doc.count("?") )
				hashtag.append( doc.count("#") )
				mentioning.append( doc.count("@") )
			result = numpy.array( [netural_score, positive_score, negative_score, nouns, adjectives, verbs, adverbs, allcaps, exclamation, question, hashtag,mentioning]).T
			
		
		
		def _get_sentiments(self, doc):
			pos_score_arr = []
			neg_score_arr = []
			nouns      = 0.
			adjectives = 0.
			verbs      = 0.
			adverbs    = 0.
			# analyse document
			tagged_doc = nltk.pos_tag( tuple(doc.split() )
			for word,tag in tagged_doc:
				# POS(Part Of Speech) type
				sent_pos_type = None
				if tag.startswith("NN"):
					sent_pos_type = "n"
					nouns += 1
				elif tag.startswith("JJ"):
					sent_pos_type = "a"
					adjectives += 1
				elif tag.startswith("VB"):
					sent_pos_type = "v"
					verbs += 1
				elif tag.startswith("RB"):
					sent_pos_type = "r"
					adverbs += 1
				# positive,negative score of each word
				pos_score,neg_score = 0,0
				if sent_pos_type is not None:
					sent_word = "%s/%s"%(sent_pos_type, word)
					if sent_word in sent_word_net:
						pos_score,neg_score = sent_word_net[sent_word]
					pos_score_arr.append(pos_score)
					neg_score_arr.append(neg_score)
			# documents score
			len = len(sent)
			avg_pos_score   = numpy.mean(pos_vals)
			avg_neg_score   = numpy.mean(neg_vals)
			return [ 1 - avg_pos_score - avg_neg_score
					 ,avg_pos_score, avg_neg_score
					 ,nouns/1, adjectives/1, verbs/1, adverbs/1 ]
	~~~

* combine LinguisticVectorizer into our model training:

	> [FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html): applies a list of transformer objects in parallel to the input data, then concatenates the results
	
	~~~python
	def create_union_model(params=None):
		def preprocessor(tweet):
			tweet = tweet.lower()
			for k in emo_repl_order:
				tweet = tweet.replace( k, emo_repl[k] )
			for r, repl in re_repl.iteritems():
				tweet = re.sub(r, repl, tweet)
			return tweet.replace("-"," ").replace("_", " ")
		
		tfidf_ngrams = TfidfVectorizer(preprocessor=preprocessor, analyzer="word" )
		ling_stats = LinguisticVectorizer()
		all_features = FeatureUnion( [('ling', ling_stats), ('tfidf', tfidf_ngrams)])
		classifier = MultinomialNB()
		pipeline = Pipeline( [('all',all_features), ('classifier', classifier)])
		if params:
			pipeline.set_params(**params)
		return pipeline
	~~~
	
	> we get a 0.6% increase on P/R AUC
	