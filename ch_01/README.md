## CH01 Introduction 

### preparation

* content of this book
* seek for help
* related libs and tools
* setup up the working environment 

see [../README.md](../README.md)

### brief introductions as a quick reference

* **NumPy**
	
	sample data
	
	~~~python
	>>> import numpy as np
	>>> a = np.array( [0,1,2,3,4,5] )
	>>> a
	array([0, 1, 2, 3, 4, 5])
	>>> a.shape
	(6,)
	>>> b = a.reshape( (3,2) )
	>>> b
	array([[0, 1],
	       [2, 3],
	       [4, 5]])
	>>> b.ndim
	2
	>>> b.shape
	(3, 2)
	~~~
	
	shallow copy and deep copy
	~~~python
	>>> b[1][0]=77
	>>> b
	array([[ 0,  1],
	       [77,  3],
	       [ 4,  5]])
	>>> a
	array([ 0,  1, 77,  3,  4,  5])
	>>> c = a.reshape((3,2)).copy()
	>>> c
	array([[ 0,  1],
	       [77,  3],
	       [ 4,  5]])
	>>> c[0][0]=-99
	>>> a
	array([ 0,  1, 77,  3,  4,  5])
	>>> c
	array([[-99,   1],
	       [ 77,   3],
	       [  4,   5]])
	~~~
	
	apply operation to all elements 
	
	~~~python
	>>> a*2
	array([  0,   2, 154,   6,   8,  10])
	>>> a**2
	array([   0,    1, 5929,    9,   16,   25])
	>>> [1,2,3]*2
	[1, 2, 3, 1, 2, 3]
	>>> np.array([1,2,3])*2
	array([2, 4, 6])
	>>> [1,2,3]**2
	Traceback (most recent call last):
	  File "<stdin>", line 1, in <module>
	TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'
	>>> np.array([1,2,3])**2
	array([1, 4, 9])
	~~~
	
	use numpy.array as index of another numpy.array
	~~~python
	>>> a
	array([ 0,  1, 77,  3,  4,  5])
	>>> # index array
	>>> a[np.array([2,3,4])]
	array([77,  3,  4])
	>>> # boolean array
	>>> a[np.array([True,True,False,True])]
	array([0, 1, 3])
	~~~
	
	use boolean expression to generate boolean array for indexing
	~~~python
	>>> a
	array([ 0,  1, 77,  3,  4,  5])
	>>> a>4
	array([False, False,  True, False, False,  True], dtype=bool)
	>>> a[a>4]
	array([77,  5])
	~~~
	
	fix abnormal values with boolean indexing
	~~~python
	>>> a
	array([ 0,  1, 77,  3,  4,  5])
	>>> a[a>4]=4
	>>> a
	array([0, 1, 4, 3, 4, 4])
	~~~
	
	fix abnormal values with numpy.array.clip(ge,le)
	~~~python
	>>> a
	array([0, 1, 4, 3, 4, 4])
	>>> a.clip(1,3) 
	>>> # return a copy without modify modify the original array
	array([1, 1, 3, 3, 3, 3])
	~~~
	
	NAN: non-existing values
	~~~python
	>>> c=np.array([1,2,np.NAN,3,4])
	>>> c
	array([  1.,   2.,  nan,   3.,   4.])
	>>> np.isnan(c)
	array([False, False,  True, False, False], dtype=bool)
	>>> c[~np.isnan(c)]
	array([ 1.,  2.,  3.,  4.])
	>>> np.mean(c[~np.isnan(c)])
	2.5
	~~~
	
	running time
	~~~python
	>>> import timeit
	>>> std_py_sec = timeit.timeit('sum(x*x for x in xrange(1000))', number=10000)
	>>> std_py_sec
	0.8079440593719482
	>>> np_sec = timeit.timeit('sum(na*na)', setup="import numpy as np; na=np.arange(1000)", number=10000)
	>>> np_sec
	4.368411064147949 
	>>> # only when applied to np algorithms, np.array is faster than standard array
	>>> np_dot_sec = timeit.timeit('na.dot(na)', setup="import numpy as np; na=np.arange(1000)", number=10000)
	>>> np_dot_sec
	0.0264589786529541
	~~~
	
	np.array convert all elements to the same data type
	~~~python
	>>> a = np.array([1,2,3])
	>>> a.dtype
	dtype('int64')
	>>> np.array([1,"stringy"])
	array(['1', 'stringy'], dtype='|S7')
	>>> np.array([1,"stringy",set([1,2,3])])
	array([1, 'stringy', set([1, 2, 3])], dtype=object)
	~~~
	

* **SciPy**

	packages of SciPy: 

	| package         | usage |
	| :-------------- |:----- |
	| **cluster**     | **cluster.hierarchy, cluster.vq(vector quantization), K-means, ...**|
	| constants       | convert math and physical contants |
	| fftpack         | discrete Fourier transform |
	| integrate       | integral	|
	| **interpolate** | spline, one-dimensional and multi-dimensional (univariate and multivariate) interpolation, Lagrange and Taylor polynomial interpolators, and wrappers for FITPACK and DFITPACK function |
	| io              | input & output |
	| linalg          | linear algebra lib with optimized BLAS and LAPACK lib |
	| maxentropy      | max entropy |
	| ndimage         | n-dimension visualization lib |
	| odr             | orthogonal distance regression |
	| **signal**      | **signal processing** | 
	| sparse          | sparse matrix |
	| spatial         | spatial data structures and algorithms |
	| special         | special mathatical libs, e.g.: Bessel, Jacobian |
	| **stats**       | **scientific statistics** |

### a minimum machine learning application : predict web traffic

* data cleaning : pencentage of samples with NaN fields --> decide to discard these record
* analyze data : scatter diagram --> decide to use [scipy.polyfit](http://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html) (see [curve fitting](http://en.wikipedia.org/wiki/Curve_fitting))
* define loss function, try polynominal rank from 1 in {1,2,3,10,100}: under fitting when rank in {1,2,3}; overfitting when rank in {10,100} --> what's next step? 
	* choose a polynominal ?
	* train a more complicated model ?
	* think in different way? 
* plot data, find a turinng point between week3 and week4 --> use data after week 3.5, set polynormal rank = 3, then the prefiction works well on test set 

**conclusion: understanding data is importmant**


