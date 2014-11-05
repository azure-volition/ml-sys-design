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

	* rational
* 

	