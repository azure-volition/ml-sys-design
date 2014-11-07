## CH08 Regression - Recommendations Improved

### Ideas

> **basket analysis**: only considering whether a user has rated a move

> use binary matrix to store the data:

~~~python
from matplotlib import pyplot as plt
imagedata = reviews[:200, :200].todense()
plt.imshow(imagedata, interpolation='nearest')
~~~

> recomendation with user model:

* for each user: sort other users according to samilarity to this user

* on predicting a user-movie rating: scan the users array, return rating-score of 1st user who scored this movei

* performance: 
	* RMSE reduce by 20% compared with using avarage rating of all users
	* RESE reduce by 25% if only using data of users who frequently(top 50%) rate movies

> recomendation with movie model:

> combining these two approaches 