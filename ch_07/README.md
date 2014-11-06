## CH07 Regression - Recommendations

### A simple example: predict house price

* Model 1: Ordinary Least Squares (OLS) Regression

	> load data
	
	~~~python
	from sklearn.datasets import load_boston
	boston = load_boston()
	~~~

	> analyse data
	
	~~~python
	from matplotlib import pyplot as plt
	plt.scatter(boston.data[:,5], boston.target, color='r')
	~~~
	
	> train model with single dimension(input)
	
	~~~python
	import numpy
	x = boston.data[:,5]
	x = np.array([[v] for v in x])
	y = boston.target
	slope,total_error,_,_ = np.linalg.lstsq(x,y)
	rmse = numpy.sqrt(total_error[0]/len(x))
	~~~
	> visualize the data, performance is bad
	> RMSE( Root Mean Squared Error) is 7.6
	
* add a bias to our model

	~~~python
	x = boston.data[:,5]
	x = numpy.array( [[v,1] for v in x] ) # add bias=1
	y = boston.target
	(slope,bias),total_error,_,_ = numpy.linalg.lstsq(x,y)
	rmse = numpy.sqrt(total_error[0]/len(x))
	~~~
	> visualize the data, this time the performance is better
	> RMSE( Root Mean Squared Error) of 6.6, which implies the maximum error between predicted house price and real house price could be $13000, is still too large.

* **RMSE and real error**
	
	> the difference between most data and their mean are less than RMSE*2
	
	> confidence interval **[ mean - RMSE, mean + RMSE ]** 
	
	> assume the data obeys Gauss distribution (but also approximately correct on other distribution in many situations)

* regression in multi-dimension-space (multiple features)
	
	~~~python
	x = boston.data # not boston.data[:,5]
	x = numpy.array( [numpy.concatenate(v, [1]) for v in boston.data] ) # add bias=1
	y = boston.target
	s,total_error,_,_ = numpy.linalg.listsq(x,y)
	rmse = numpy.sqrt(total_error[0]/len(x))
	~~~
	> RMSE( Root Mean Squared Error ) is 4.7 now 
	> hard to visualize because the dimension number is 14

### Cross validation for regression

* use linear regression as example (OLS is implementated by LinearRegression class)

	~~~python
	from sklearn.linear_model import LinearRegression
	lr = LinearRegression(fit_intercept=True) # for add a bias
	lr.fit(x,y)
	predict_arr = map(lr.predict, x)
	error_arr   = predict_arr - y
	total_error = numpy.sum(error_arr*error_arr) # err_0^2 + err_1^2 + ... + err_n^2
	rmse_train  = numpy.sqrt(total_error/len(predict_arr))
	print( 'RMSE on trainings: {}'.format(rmse_train))
	~~~
	> the training error is 4.6, almost the same as before

* KFold cross validation for regression

	~~~python
	from sklearn.cross_validation import Kfold
	kf  = KFold(len(x), n_folds=10)
	total_err = 0
	for train, test in kf:
		lr.fit(x[train],y[train])
		predict_arr = map(lr.predict, x[test])
		error_arr   = predict_arr - y[test]
		total_err  += numpy.sum( error_arr * error_arr )
	rmse_10cv = numpy.sqrt( total_err / len(x) )
	print( 'RMSE on tranings: {}'.format(rmse_10cv))
	~~~
	> the training error is 5.6 (larger than previous 4.6), but it generalize better
	
### Penalized Regression

* L1 and L2 Penalties

	* OLS Regression without Penalty
	
		> b<sup>*</sup> = arg min<sub>b</sub> (Y - X&dot;b)<sup>2</sup> 
	
	* OLS Regression with L1 Penalty (**Lasso Regression**)
		
		> b<sup>*</sup> = arg min<sub>b</sub> (Y - X&dot;b)<sup>2</sup> + &lambda; &sum;|b<sub>i</sub>|
		
	* OLS Regression with L2 Penalty (**Ridge Regression**)
		
		> b<sup>*</sup> = arg min<sub>b</sub> (Y - X&dot;b)<sup>2</sup> + &lambda; &sum;b<sub>i</sub><sup>2</sup>

	* Elastic net 
	
		> combining Lasso and Ridge, which uses both penalties

	* These penalties will makes b<sub>i</sub> smaller.  More b<sub>i</sub> will become zero, which automatically discard some features and makes the model sparse.  The feature selections is integrated into model training.
	
	* penalty pameter: &lambda;
		
		> The smaller &lambda; is, the smaller penalty we get.  When &lambda nearly equal to 0, our model will be similar with OLS. 
	
* use Lasson or Elastic net in scikit-learn

	~~~python
	from sklearn.linear_model import ElasticNet
	en = ElasticNet(fit_intercept=True, alpha=0.5)
	~~~
	
	> training error increases to 5.0 from 4.6, but cross validation error drops to 5.4 from 5.6
	
### Regression when feature count larger than example count (P>N)

* Problem:

	> fit perfectly on training set, but generalize bad on testing set
	
	> e.g.: regard every word as a feature, thus feature count is extreme large
	
* Example: predict stocks fluctuation from 10-K reports of SEC

	> load training data: 16087 examples, 150360 features
	
	~~~python
	from sklearn.datasets import load_svmlight_file
	data,target = load_svmlight_file( 'E2006.train' )
	
	print('Min target value:{}'.format(target.min()))   # -7.89957807347
	print('Max target value:{}'.format(target.max()))   # -0.51940952695
	print('Mean target value:{}'.format(target.mean())) # -3.51405313669
	print('Std. dev. target:{}'.format(target.std()))   # 0.632278353911
	~~~
	
	> train model with OLS and evaluate performance
	
	~~~python
	from sklearn.linear_model import LinearRegression
	lr = LinearRegression(fit_intercept=True)
	lr.fit(data, target)
	predict_arr    = numpy.array( map( lr.predict, data ) ) # (1,16087) array
	predict_arr    = predict_arr.ravel() 
	error_arr      = predict_arr - target
	total_sq_error = numpy.sum( error_arr * error_arr )     # sum(dot product)
	rmse_train     = numpy.sqrt( total_sq_error/len(predict_arr) )
	print(rmse_train)
	~~~
	
	> training error is 0.0025 (extremly small)
	> but cross validating error is 0.78 (larger than standard deviation 0.632278353911, worse than always predict average value -3.51405313669)
	
* Solution: use normalization to prevent overfitting

	> e.g.: use Elastic Net with penalty parameter of 1, RMSE of 0.4 can be got from cross validation

	> panalty parameter should be selected carefully (overfit if too large, underfit if too small)
	
* **Hyperparameter: find optimal penalty parameter by training**

	* two-layer cross validation

		> e.g.: two-layer 5-fold cross validation
		
		> layer 1: split all data into 5 folds
		
		* 20% data for evaluation the generalization ability (test error)
		
		* 80% data for training model and penalty parameter
		
			> layer 2:
		
			> break these 80% data into 5 subfolds to try different penalty
			
			> on each sub-fold (16% data): a sub 5-fold-training-testing is executed
			
			* 12.8% (16%*80%) : for training
			* 3.2% (16%*20%) : for cross validation
			
			> once the optimal penatly is found, cross-validating is execute on layer2's testing fold
	
	* run two-layer cross validation with ElasticNetCV/LassoCV/RigeCV:
		
		> e.g.: ElasticNetCV
		
		~~~python
		from sklearn.linear_model import ElasticNetCV
		model_training = ElasticNetCV(fit_intercept=True)
		k_fold         = KFold(len(target), n_folds=10)
		total_err      = 0
		for train_set, test_set in k_fold:
			model_training.fit(data[train], target[train])
			predict_arr = map( model_training.predict, data[test] )
			predict_arr = numpy.array(predict_arr).ravel()
			error_arr   = predict_arr - target[test]
			total_err  += numpy.dot(error_arr, error_arr)
		rmse_k_cv = numpy.sqrt( total_err / len(target) )
		~~~
		
		> ... will run for quit a long time
		
### Rating prediction and recommendation

* problem: Netflix Challenge

	> user rate on movies (score is one of 1,2,3,4 and 5)
	
	> recomend other movies according user's rating and watching history
	
	* winners' solutions:
	
		> combining advanced machine learning algorithms
	
		> a lot of preprocess are needed, such as:
		
		* some users tend to rating high, others tend to rating a lower score
		
		* rating count received and release time

* classification or regression? 

	> classification is not suitable:
	
	* severity of errors are different between rating 5 to 4 and rating 5 to 1

	* middle value are useful, such as 4.7 is different from 4.2
	

	

	