##CH03 Clustering

### Problem:

Find similar 

### Preprocess: raw data --> feature vector

#### 1. Measure the similarity between posts

* [Levenshtein distance](http://en.wikipedia.org/wiki/Levenshtein_distance) ? : cost too much to compute

* [bag-of-word](http://en.wikipedia.org/wiki/Bag-of-words_model):  each post is represented with a feature vector
~~~
vector<pair<word,cnt>>
~~~

#### 2. Generate the word-bag

1. parameters of Scikit.CountVectorizer to filter and extract words:
	
	> min\_df (minimum documentary frequency)
	
	> max\_df (max documentary frequency)
	
	> token_pattern (how to cut the words)

	~~~python
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer( min_df=1 )
	X_train = vectorizer.fit_transform( posts )
	~~~
		
2. sparse vector to store feature (using coo_matrix to store data)
	
	~~~python
	new_post = "content of this new post"
	new_post_vec = vectorizer.transform( [new_post] )
	print( new_post_vec ) #print sparse vector
	(0, 7)1
	(0, 5)1
	print( new_post_vec.toarray() )  #print raw vector
	[[0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
	~~~
	
3.	
	
