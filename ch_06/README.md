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

* MultinomialNB:

* BernoulliNB: 


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