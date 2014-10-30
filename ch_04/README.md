## CH04 Topic Model

### Topic Model

> discovery multiple topics among a document set

> probability of each document belong to each topic

> central topic of each document

> topic is represented by a set of words

### LDA: Latent Dirichlet Allocation

#### (1) Train LDA Model

> notice: sklearn.lda is a package of Linear Discriminant Analysis (a classification algorithm), not Latent Dirichlet Allocation

> [http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation](http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

> python package: gensim

~~~python
from gensim impor corpora, models, similarities
# examples
corpus = corpora.BleiCorpus('./data/ap/ap.dat','/data/ap/vocab.txt')
# train model
model = models.ldamodel.LdaModel( corpus, num_topics=100, id2word=corpus.id2word )
# topics
topics = [model[c] for c in corpus]
print topics[0]
# [(3,0.023607),(13, 0.116799),(19,0.075935),(92,0.107915)]
~~~
format: [(topic<sub>i</sub>, probability<sub>i</sub>),...]

> is a sparse model (only a small part of topics might appear in a document)

#### (2) Parameters

> alpha (0 < &alpha; << 1), 1.0/len(corpus) by default

> normally alpha should be set to a very small value

> the larger alpha is, the more topics each document will belong to

~~~python
model = models.ldamodel.LdaModel( corpus, num_topics=100, id2word=corpus.id2word, alpha=1.0/len(corpus) )
~~~

#### (3) Representation of a topic

> e.g.:  dress military soviet president new state capt carlucci states leader stance government 

> words with high weigh

> we could use words cloud to visualize these topics with tool such as [wordle](http://www.wordle.net)

#### (4) Preprocess

> **preprocess such remove stop-words and convert word to stem are important**

### Predict topic of a new document

#### (1) Measure distance between two topic vector

~~~python
# initial dense matrix: len(topics)*100
dense = np.zeros( (len(topics),100), float )

# convert sparse matrix to dense
for topic_idx, topic in enumerate(topics):
	for word_idx, word in topic:
		dense[topic_idx, word_idx] = word

# calculate distance between topic pairs: distance_ti_tj = sum((dense[ti]-dense[tj])**2)
for scipy.spatial import distance
distance_matrix = distance.squareform(distance.pdist(dense))

# set diagnal element larger than enough other
largest = distance_matrix.max()
for topic_idx in range(len(topics)):
	distance_matrix[topic_idx,topic_idx] = largest + 1
~~~

#### (2) Find closest topic of a new document

~~~python
def closest_to( doc_id ):
	return pairwise[doc_id].argmin()
~~~
