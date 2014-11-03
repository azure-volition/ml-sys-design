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
		
	
	
	

* Deal with missing data

* Deal with mathematical underflow


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

* train first model
	
	
	
	
	
* 