##CH03 Clustering

### Problem:

divide documents into several groups according to their similarity

### Preprocess: raw data --> feature vector

#### 1. Measure the similarity between documents

* [Levenshtein distance](http://en.wikipedia.org/wiki/Levenshtein_distance) ? : cost too much to compute

* [bag-of-word](http://en.wikipedia.org/wiki/Bag-of-words_model):  each document is represented with a feature vector
~~~
vector<pair<word,cnt>>
~~~

#### 2. Generate the word-bag

**(1) parameters of Scikit.CountVectorizer to filter and extract words:**
	
> min\_df (minimum documentary frequency)
	
> max\_df (max documentary frequency)
	
> token_pattern (how to cut the words) 

~~~python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer( min_df=1 )
X_train = vectorizer.fit_transform( documents )
~~~
		
**(2) sparse vector to store feature (using coo_matrix to store data)**
	
~~~python
new_document = "content of this new document"
new_document_vec = vectorizer.transform( [new_document] )
print( new_document_vec ) #print sparse vector
(0, 7)1
(0, 5)1
print( new_document_vec.toarray() )  #print raw vector
[[0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
~~~
	
> leanth of raw vector is the total number of words in all examples
	
**(3) similarity measurement between 2 feature vectors**

> use euclidean distance (naive approach)
	
~~~python
scipy.linalg.norm((vec1 - vec2).toarray())
~~~
	
> These are not enough to get good clustering result. Feature vectors need to be processed before applied to the clustering algorithm.


####3. Feature vector preprocess

**(1) normalize**

> reason: vector [[0 0 1 1 0 0 1 1 0]] [[0 0 3 3 0 0 3 3 0]] should be highly similar (think about we pasted document 1 three times in document 2)
	
~~~python
def dist_norm(v1, v2):
	v1_normalized = v1/sp.linalg.norm(v1.toarray())
	v2_normalized = v2/sp.linalg.norm(v2.toarray())
	return scipy.linalg.norm((v1_normalized - v2_normalized).toarray())
~~~

**(2) remove stop-wrods:**

> such as：'a', 'about', 'above', 'across', 'after', ...

~~~python
vectorizer = CountVectorizer(min_df=1, stop_words='english')
~~~

**(3) process stem: using NLTK package [[tutorial](http://text-processing.com/demo/stem/)]**

> e.g.: regard 'imaging' and 'images' as the same word

> for English, SnowballStemmer can be used

~~~python
import nltk.stem
s = nltk.stem.SnowballStemmer('english')
s.stem("graphics") #u'graphic
s.stem("buys")	#u'buy'
~~~

> integrate SnowballStemmer into Vectorizer

~~~python
import nltk.stem
eng_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
	def builder_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc: (eng_stemmer.stem(w) for w in analyzer(doc) )

vectorizer = StemmedCountVectorizer(min_df=1, step_words='english')
~~~

**(3) select the most important words: according to TF-IDF**

> setting max_df=0.9 is not enough: words appeared in more than 90% documents will be removed, but what if a word with df>0.9 is more important than another?

> combine both DF and IDF(inversed DF)

~~~python
import scipy as sp

def tfidf(term, doc, docset):
	tf  = float(doc.count(term))/sum(doc.count(w) for w in docset)
	idf = math.log(float(len(docset))/(len([doc for doc in docset if term in doc])))
~~~

> integrate tfidf into Vectorizer

~~~python
from sklearn.feature_extraction.text import TfidfVectorizer

class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer,self).build_analyzer()
		return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', charset_error='ignore')
# set charset_error='ignore' to skip words with UnicodeDecodeError
~~~

####4. Drawbacks of word-bag

* connections among words are not considered, such as "car hits wall" and "wall hits car"

* negative expressions are not considered, such as "I will not eat icecream" and "I will eat icecream"

* fail to process mis-spelling, such as "database" and "databas"

###Clustering

####1. introduction

* the flat clustering and hierarchical clustering

* compare clustering algorithms: [http://scikit-learn.org/dev/modules/clustering.html](http://scikit-learn.org/dev/modules/clustering.html)

* K-Means introduction

####2. use K-Means to clustering documents

* dataset: [http://mlcomp.org/datasets/379](http://mlcomp.org/datasets/379)

* get vectorized features

	> beacause real documents in this dataset have quit a lot words, min_df is set to 10 and max_df is set to 0.5

	~~~python
	vectorizer = StemmedTfidfVectorizer(min_df=10,max_df=0.5,stop_words='english',charset_error='ignore'
	vectorizered = vectorizer.fit_transform(dataset.data)
	num_samples, num_features = vectorized.shape
	~~~
	
* clustering

	> document number : 3414
	
	> feature vector length : 4331 (4331 words after preprocess)
	
	~~~python
	num_clusters = 50
	from sklearn.cluster import KMeans
	km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1)
	km.fit(vectorized)
	~~~
	
	> result 
	> km.labels: a vector of length 4311, elements id of clusters
	> km.cluster_centers_ : cluster centers
	
	~~~python
	km.labels_  			 #array([33,22,17,...,14,11,39])
	km.labels_.shape		 #(3414,)
	km.cluster_centers_	 #
	~~~

* predict new document
	
	> lable of new document
	
	~~~python
	new_doc_vec   = vectorizer.transform([new_post])
	new_doc_lable = km.predict(new_post_vec)[0]
	~~~
	
	> similar documents
	
	~~~python
	similar_indices = (km.labels_ == new_doc_lable).nonzero()[0]
	~~~
	
	~~~python
	similar = []
	for i in similar_indices:
		dist = scipy.linalg.norm((new_doc_vec - vectorized[i]).toarray())
		similar.append((dist, dataset.data[i]))
		similar = sorted(similar)
	~~~
	
	> noisy analyze with TF-IDF of words in feature vector
	
### Other things might improve the clusting performance
	
* number of clusters
	
* max_features 
	
* try different initilial cluster centers
	
* try different similarity measurments such as : Cosine Similarity, Pearson Coefficient, Jaccard Coefficient

* evaluate the result of clustering:  sklearn.metrics
	
	