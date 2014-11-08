## CH08 Regression - Recommendations Improved

### New ideas

* Only considering whether a user has rated a move
	
	> use binary matrix to store the data:
	
	~~~python
	from matplotlib import pyplot as plt
	imagedata = reviews[:200, :200].todense()
	plt.imshow(imagedata, interpolation='nearest')
	~~~

* recomendation with correlated users

	> for each user: sort other users according to samilarity with this user

	> on predicting a user-movie rating: return rating-score of the most similar user who has scored this movie

	> performance: 
	* RMSE reduce by 20% compared with using avarage rating of all users
	* RESE reduce by 25% if only using data of users who frequently(top 50%) rate movies

* recomendation with movie model

	> for each movie: sort other movies according to samilarity with this movie

	> on predicting a user-movie rating: return rating-score of the most similar movie this user has scored

	* RMSE: 0.85

* ensemble learning

* basket analysis

### **Ensemble learning**

* **ensemble multiple models together and learn weight of each model:**

	> model 1 : [similar_movie](./similar_movie.py)
	
	> model 2 : [corrneighbors](./corrneighbors.py)
	
	> model 3 : [usermodel](./user_model.py)  (introduced in last Chapter)
	
	> **regard output of sub-models as features**
	
	~~~python
	import similar_movie
	import corrneighbors
	import usermodel
	from sklearn.linear_model import LinearRegression
	...
	
	# 3 sub-models
	estimate = [
		usermodel.all_estimate(),     
		corrneighbors.all_estimate(),
		similar_movie.all_estimate()
		# *.all_estimate()[u,m]: predicated rating of user u on movie m
	] 
	# parameter of each sub-model
	coefficients = []
	...
	
	# leave-one-out validation
	for uid in xrange( reviews.shape[0] ): # for all users
		# remove uid
		est_other_users = numpy.delete( estimate, uid, 1 ) # dim0: model, dim 1:user, dim2: movie)
		rev_other_users = numpy.delete(  reviews, uid, 0 ) # dim0: user; dim1: movie
		# coordinates
		pos_x, pos_y = numpy.where( rev_other_users > 0 ) # only scored [u,m]
		# training set
		X = est_other_users[:,pos_x,pos_y]  # not 'estimate' as in book?
		Y = rev_other_users[rev_other_users>0]
		# train coefficients
		regression.fit(X.T, Y)
		coefficients.append(reg.conf_)
	
	# predicate
	predication = regression.predict( estimate[:, uid, reviews[uid]>0].T )
	print coefficients.mean(0)
	# [ 0.25164062, 0.01258986, 0.60827019 ]
	~~~
	
	> RMSE is almost equal to 1, and from coefficients.mean(0) we know contribution of corrneighbors.all_estimate() is quite limited
	
* ensemble models with multiple parameters

	~~~python	
	estimate = [
		usermodel.all_estimate_all,
		similar_movie.estimate_all(k=1),
		similar_movie.estimate_all(k=2),
		similar_movie.estimate_all(k=3),
		similar_movie.estimate_all(k=4),
		similar_movie.estimate_all(k=5),
	~~~
	
### **basket analysis**

* basket analysis
	
	> consider which items tends to be bought together with higher probability **compared to the benchmark**
	
	> *without comparation with the benchmark, unrelated items might be remomended incorrectly simply because they are popular*

* load data and analyse

	~~~python
	from collections import defaultdict
	from itertools import chain
	
	# a sale record for each line, such as: item_id_1 item_id_2 ... item_id_n
	item_sale_rec = [
		[int(tok) for tok in line.strip().split()] for line in open('retail.dat')
	]
	
	sale_cnt = defaultdict(int)
	for item_id in chain(*item_sale_rec):
		sale_cnt[item_id] +=1
	~~~
	
	> long tail phenomenon: 33% items are saled less than 4 times, which accounts 1% purchase

* implementation of Apriori (Rakesh Agrawal and Ramakrishnan Srikant, 1994)
	
	> skipped

	

	





	
	
	
	