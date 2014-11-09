##CH11 Dimensionality Reduction

### Reasons to remove redundant features rather than set weight to zero

* misleading machine learning
* more parameters to tune, more risk of over-fitting
* dimesion number are larger than those really solve problems
* less dimensions, faster training
* for visualization, dimensions count should be less than 3

### Approaches

* feature selection:

	> analyse and throw away useless features

* feature extraction:

	> translate feature space from high dimension to low dimension (when feature number is too large and feature selection can not be used)
	
	> * PCA: Principal Component Analysis
	
	> * LDA: Linear Discriminant Analysis （NOT Latent Dirichlet Allocation)
	
	> * MDS: MultiDimensional Scaling

### Feature Selection

* how feature selection works:  

	* a good feature: has no dependence with other features and highly related with prediction targets

	* use scatter matrix to discover feature dependency(overlap)
	
	* when feature number increased, use more automatic and efficient approaches: filter and wraper

* **FILTER: discover feature dependancy based on statistics**

	* Workflow

		all features -> **remove redudant features** -> features for training (X:Y) -> **remove irrelevant features** -> resulting features

	* Pearson Correlation Coefficient
	
		* usage: find linear dependency between features
	
			> 
			
			~~~python
			from import scipy.stats import pearsonr
			
			pearsonr([1,2,3], [1,2,3.1]) 
			# (0.99963338516121843, 0.017498096813278487)
			# (r: correlation degree; p: un-related probability)
			# r is large, and p is small
			# 2 feature are dependent, 1 of them could be removed
			 
			pearsonr([1,2,3], [1,20,6])
			# (0.25383654128340477, 0.83661493668227405)
			# p is too large, so this Pearson r value can not be trusted
			~~~
		
		* **drawbacks: Pearson correlation coefficient can not discover non-linear dependency between features**
		
	* Mutual Information

		* Claude Shannon Entropy
		
			> **H(X) = - &sum;<sup>n</sup><sub>i=1</sub>P(X<sub>i</sub>)log<sub>2</sub>P(X<sub>i</sub>)** 			
			> uncertainty of **one** feature(events)
			
			> e.g.: toss coin
			
			> if p(x<sub>0</sub>)=p(x<sub>1</sub>)=0.5, then H(X)=-0.5\*log<sub>2</sub>(0.5)-0.5\*log<sub>2</sub>(0.5)=1.0
			
			> if p(x<sub>0</sub>)=0.6 and p(x<sub>1</sub>)=0.4, then H(X)=-0.6\*log<sub>2</sub>(0.6)-0.4\*log<sub>2</sub>(0.4)=0.97, the uncertainty decreases
			> ...
			
			> if p(x<sub>0</sub>)=1.0 and p(x<sub>0</sub>)=0.0, then H(X)=-1.0\*log<sub>2</sub>(1.0)-0.0\*log<sub>2</sub>(0.0)=0, there is no uncertainty
			
		* Mutual Information: uncertainty of **two** features(events)

			> two feature X,Y
			
			> in examples {(X<sub>1</sub>,Y<sub>1</sub>),(X<sub>2</sub>,Y<sub>2</sub>),...,(X<sub>n</sub>,Y<sub>n</sub>)}, there are M distinct values of X and N distinct values of Y

			> **I(X;Y)=&sum;<sup>m</sup><sub>i=1</sub>&sum;<sup>n</sup><sub>j=1</sub>P(X<sub>i</sub>,Y<sub>j</sub>)log<sub>2</sub>(P(X<sub>i</sub>,Y<sub>j</sub>)/P(X<sub>i</sub>)P(Y<sub>j</sub>))**
			
			> to normalize the value in [0,1]
			
			> **NI(X;Y)=I(X;Y)/(H(X)+H(Y))**
		
		* **drawback: very slow to compute because every pair of features will be calculated**
	
	* **drawback of FILTER**: may errorly remove irrelevant features (see workflow) when statistic dependancy with labels(Y). 
	> A single feature(X[i]) might be irrelevant with labels(Y), but when combined with other feature(X[j],...) it might demonstrate strong dependency with labels(Y), e.g.: Y = X[0] xor X[1]

* **WRAPER: voting by model itself to discover feature dependancy hiden from FILTER**
		 
	?? workflow
		



### 
	