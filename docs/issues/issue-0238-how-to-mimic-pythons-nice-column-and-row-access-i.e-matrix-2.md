# #238: How to mimic Python's nice column and row access (i.e matrix[:, 2])?

- **URL:** https://github.com/SciSharp/NumSharp/issues/238
- **State:** OPEN
- **Author:** @henon
- **Created:** 2019-04-05T08:08:30Z
- **Updated:** 2020-02-08T11:40:24Z
- **Labels:** help wanted
- **Assignees:** @henon
- **Milestone:** v0.9 Multiple Backends

## Description

Hey all,

How would you say to do access to a column `matrix[:, i]` or access to a row `matrix[i, :]` in NumSharp?


## Comments

### Comment 1 by @henon (2019-04-05T11:21:07Z)

I need to be able to get a (n-1)-dimensional slice of a n-dimensional matrix, i.e. a 1d vector out of a 2d matrix.
First step would be to add appropriate accessors to IStorage right? 
I am not yet sure how they should look like, please comment if you have ideas.

### Comment 2 by @henon (2019-04-05T11:35:01Z)

This is not as elegantly possible as in Python, but we can do something like this, given that matrix is an NDArray:

* matrix[Range.All, 1] ... return the first column, Python: matrix[:, 1]
* matrix[new Range(0, 2), 1] ... return the first two entries of the first column, Python: matrix[:2, 1]
* matrix[new Range(1,3), new Range(1,3)] ... return a 2x2 submatrix of matrix starting at 1,1, Python: matrix[1:3, 1:3]


### Comment 3 by @Oceania2018 (2019-04-05T11:35:18Z)

Check `indexing` to help.
![image](https://user-images.githubusercontent.com/1705364/55624849-c97ce200-576c-11e9-8b03-5b4cac0766b0.png)

We can override in `Slice`
![image](https://user-images.githubusercontent.com/1705364/55624914-f03b1880-576c-11e9-9456-2b4e97c17077.png)

Or we can pass string `":1"`.

### Comment 4 by @henon (2019-04-05T11:40:45Z)

Yeah, the string idea is great for porting code from Python. I am sure it will be popular.

### Comment 5 by @Oceania2018 (2019-04-05T11:43:02Z)

Seem's like another contributor already implemented the string index. I'll found out later. @PppBr is that true?

### Comment 6 by @henon (2019-04-05T11:45:05Z)

when you slice a matrix in numpy you essentially get a view of the original data. modifying it will modify the original matrix. Do we have a view concept or implementation already?

### Comment 7 by @Oceania2018 (2019-04-05T11:50:46Z)

Not yet, we havn't have the `view` class.  I've an idea how to implement the new `view`.

`view` inherit from `ndarray`, but keep a `filter` of data index.
and `view` will implement a `IEnumerate` interface.
when interating the `view`, it will return all the element in the `filter`.
and view don't need copy the memory. change data will reflect in the original memory.

@henon What do you think of it?

or we need a new `NDStorage`?

### Comment 8 by @henon (2019-04-05T12:14:49Z)

@Oceania2018: I think the projection from the view to the original data could be handled entirely by the storage. 

This is complicated, I had to think about it for a time to wrap my head around it. If I am correct, what you  need is a function that projects a view index onto an index on the original data (which could be another view that does further projection - think about that). 

I am not yet sure how that function looks like if you i.e. for instance get a 1D slice out of a 17-dimensional matrix but once we have that generic formula we can easily project view_index to underlying data_index and view_index+1, view_index+2, ... etc. to enumerate it.







### Comment 9 by @Oceania2018 (2019-04-05T12:17:40Z)

Actually, we have the formula to convert n-d index to 1d index and wise-verse.
chech the function `Shape.GetDimIndexOutShape`
So don't worry about the dimension. the storage is persistent data in 1d array.

I still think the `view` should inherit from `ndarray` with a `filters` in `view` instance, not in `ndstorage` level. Let's keep this talk open.

### Comment 10 by @henon (2019-04-05T12:47:10Z)

Nice we have the projection and the inverse projection already, so the difficult part is actually already solved. The rest is just software design.

One thing I missed, that the view also needs to enumerate on the underlying data (or view), is a stride. If you take a row out of a matrix the stride is 1, but if you take a column, the stride is the width of the matrix.

ok, you may be right about inheriting from ndarray. I have only limited knowledge about the design at this point, so I trust your judgement. 

### Comment 11 by @Oceania2018 (2019-04-05T13:01:13Z)

I will write the first version of view, then we can discuss further.

### Comment 12 by @henon (2019-04-05T13:25:41Z)

Don't forget to make the design recursive, so that you can have a `view` of a `view` of a `view` of a 'storage'. 

### Comment 13 by @Oceania2018 (2019-04-05T14:00:16Z)

Sure

### Comment 14 by @henon (2019-04-08T09:51:45Z)

Update: Python's slice notation for accessing slices of NDArray has been implemented. However, the implementation of a view that serves that slice of the original data still needs to be done.

Here is a very simple test case that demonstrates the problem to be solved:
```[TestMethod]
    public void GetAColumnOf2dMatrix()
    {
        var x = np.arange(9).reshape(3, 3);
        var v = x[":, 1"];
        Assert.IsTrue(Enumerable.SequenceEqual( v.Data<double>(), new double[]{ 1, 4, 7 }));
        v[1] = 99;
        Assert.AreEqual(99, x[1,1]);
    }
```

### Comment 15 by @Oceania2018 (2019-04-27T16:46:05Z)

Check the `Slice3x2x2` UnitTest.

### Comment 16 by @Rikki-Tavi (2020-02-07T03:35:42Z)


I have been wrestling with the same (or similar) issue in the context of trying to generate randomized mini-batches using NumSharp. 
The python code I am trying to emulate looks like this:

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches 
	
The part I have trouble emulating with NumSharp is:
	
       shuffled_X = X[:, permutation]
       shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
	
The closest I have been able to get is:
	
       var shuffled_X = new NDArray(np.float32, X.shape);
       for (int i = 0; i < X.shape[0]; i++)
       {
           shuffled_X[i] = X[i][permutation];
       }

       var shuffled_Y = new NDArray(np.float32, Y.shape);
       for (int i = 0; i < Y.shape[0]; i++)
       {
           shuffled_Y[i] = Y[i][permutation];
       }
       shuffled_Y = shuffled_Y.reshape((Y.shape[0], m));
	   
Which works, but is much, much slower (also true when I implement the alternative in Python).
	
The problem with NumSharp is that I cannot put the indexer ":, permutation" in quotes. I've tried converting 'permutation' to its string-expanded list equivalent (so that I can have a totally string-based indexer that NumSharp likes) but I don't seem to be able to arrive at a string-serialized encoding of 'permutation' inside the resulting indexer that doesn't make NumSharp crash.
	
Am I overlooking an obvious solution here?

### Comment 17 by @Oceania2018 (2020-02-07T04:20:06Z)

@BillyDJr Try
```csharp
shuffled_X = X[Slice.All, new Slice(permutation)];
```

### Comment 18 by @Rikki-Tavi (2020-02-07T05:50:40Z)

Trying that yielded: 
`NumSharp.IncorrectShapeException: 'This method does not work with this shape or was not already implemented.'`


### Comment 19 by @henon (2020-02-07T07:50:54Z)

> @BillyDJr Try
> 
> ```cs
> shuffled_X = X[Slice.All, new Slice(permutation)];
> ```

No, he needs index access to get shuffled data. It should be

```cs
shuffled_X = X[Slice.All, new NDArray(permutation)];
```

### Comment 20 by @Rikki-Tavi (2020-02-07T21:11:11Z)

I simplified my Python target test case, showing two ways to generate the shuffled Xs and Ys.
m = 4
X = np.random.rand(3,m)
Y = np.random.rand(1,m)
permutation = list(np.random.permutation(m))
print ("perm: " + str(permutation))
print ("X: " + str(X))
print ("Y: " + str(Y))
shuffled_X1 = X[:, permutation]
shuffled_X2 = X.copy()
for i in range(0, X.shape[0]):
    shuffled_X2[i] = X[i][permutation]
    
print("shuffled_X1: " + str(shuffled_X1))
print("shuffled_X2: " + str(shuffled_X2))

shuffled_Y1 = Y[:, permutation].reshape((Y.shape[0],m))
shuffled_Y2 = Y.copy()
for i in range(0, Y.shape[0]):
    shuffled_Y2[i] = Y[i][permutation]
shuffled_Y2 = shuffled_Y2.reshape((Y.shape[0], m))

print("shuffled_Y1 " + str(shuffled_Y1))
print("shuffled_Y2 " + str(shuffled_Y2))

And the results:
perm: [2, 1, 3, 0]
X: [[0.14038694 0.19810149 0.80074457 0.96826158]
 [0.31342418 0.69232262 0.87638915 0.89460666]
 [0.08504421 0.03905478 0.16983042 0.8781425 ]]
Y: [[0.09834683 0.42110763 0.95788953 0.53316528]]
shuffled_X1: [[0.80074457 0.19810149 0.96826158 0.14038694]
 [0.87638915 0.69232262 0.89460666 0.31342418]
 [0.16983042 0.03905478 0.8781425  0.08504421]]
shuffled_X2: [[0.80074457 0.19810149 0.96826158 0.14038694]
 [0.87638915 0.69232262 0.89460666 0.31342418]
 [0.16983042 0.03905478 0.8781425  0.08504421]]
shuffled_Y1 [[0.95788953 0.42110763 0.53316528 0.09834683]]
shuffled_Y2 [[0.95788953 0.42110763 0.53316528 0.09834683]]

Here is the C# code (just showing the Xs for the sake of brevity). Note the shuffled_X2 works (as I reported before, but it runs too slowly for large datasets). As you can see, I try various stabs at arriving at shuffled_X1, all failing:

            var m = 4;
            var X = np.random.rand(3, m);
            var permutation = np.random.permutation(m); 

            print($"perm: {permutation}");
            print($"X: {X}");
            
            // In Python: shuffled_X1 = X[:, permutation]
            try
            {
                var shuffled_X1 = X[Slice.All, permutation];
                print($"shuffled_X1: {shuffled_X1}");
            }
            catch (Exception ex)
            {
                print($"Param as: raw permutation  - error: {ex.Message}");
            }

            try
            {
                var shuffled_X1 = X[Slice.All, new Slice(permutation)];
                print($"shuffled_X1: {shuffled_X1}");
            }
            catch (Exception ex)
            {
                print($"Param as: new Slice(permutation) - error: {ex.Message}");
            }

            try
            {
                var shuffled_X1 = X[Slice.All, new NDArray(permutation.ToArray<Int32>())];
                print($"shuffled_X1: {shuffled_X1}");
            }
            catch (Exception ex)
            {
                print($"Param as: new NDArray(permutation.ToArray<Int32>())  - error: {ex.Message}");
            }

            try
            {
                var shuffled_X1 = X[Slice.All, new NDArray(permutation.CloneData())];
                print($"shuffled_X1: {shuffled_X1}");
            }
            catch (Exception ex)
            {
                print($"Param as: new NDArray(permutation.CloneData())  - error: {ex.Message}");
            }

            try
            {
                var shuffled_X1 = X[Slice.All, new NDArray(Binding.list(permutation.ToArray<Int32>()))];
                print($"shuffled_X1: {shuffled_X1}");
            }
            catch (Exception ex)
            {
                print($"Param as: new NDArray(Binding.list(permutation.ToArray<Int32>()))  - error: {ex.Message}");
            }

            var shuffled_X2 = new NDArray(np.float32, X.shape);
            for (int i=0; i<X.shape[0]; i++)
                 shuffled_X2[i] = X[i][permutation];

            print($"shuffled_X2: {shuffled_X2}");


And the results:
perm: [2, 1, 0, 3]
X: [[0.5201029277034583, 0.8834703261421389, 0.034344740227956666, 0.5556772460954623],
[0.3361533942335068, 0.780603283914087, 0.6067406924472846, 0.794946916305901],
[0.8351871454320788, 0.11560027912054224, 0.601820489671929, 0.2805163801091334]]
Param as: raw permutation  - error: shape mismatch: objects cannot be broadcast to a single shape
Param as: new Slice(permutation) - error: This method does not work with this shape or was not already implemented.
Param as: new NDArray(permutation.ToArray<Int32>())  - error: shape mismatch: objects cannot be broadcast to a single shape
Param as: new NDArray(permutation.CloneData())  - error: shape mismatch: objects cannot be broadcast to a single shape
Param as: new NDArray(Binding.list(permutation.ToArray<Int32>()))  - error: shape mismatch: objects cannot be broadcast to a single shape
shuffled_X2: [[0.03434474, 0.8834703, 0.5201029, 0.55567724],
[0.6067407, 0.7806033, 0.3361534, 0.7949469],
[0.60182047, 0.11560028, 0.83518714, 0.2805164]]

Are there any other variations I should try?




### Comment 21 by @Rikki-Tavi (2020-02-08T04:45:32Z)

FWIW....It was easy to get working with Numpy.NET:

        [TestMethod]
        public void mytest() {
            var m = 4;
            var X = np.random.rand(3, m);
            var permutation = np.random.permutation(m);

            print($"perm: {permutation}");
            print($"X: {X}");

            var shuffled_X = X[":", permutation];
            print($"shuffled_X: {shuffled_X}");
        }

        private void print (string output)
        {
            System.Diagnostics.Debug.WriteLine(output);
        }

Resulted in:

```
perm: [3 0 1 2]
X: [[0.58033439 0.99341353 0.77685615 0.1893294 ]
 [0.81824309 0.7315536  0.64120989 0.97876454]
 [0.15659539 0.9045345  0.83839889 0.43254176]]
shuffled_X: [[0.1893294  0.58033439 0.99341353 0.77685615]
 [0.97876454 0.81824309 0.7315536  0.64120989]
 [0.43254176 0.15659539 0.9045345  0.83839889]]
```
Now I'm off to investigate if there is an efficient way that I can use Nump.NET NDArrays for my mini-batch randomization in TensorFlow.NET. (I'm admittedly just hacking away here....please stop me, anyone, if you think there is a better way.)


### Comment 22 by @henon (2020-02-08T09:41:39Z)

Obviously, this should also work in NumSharp, but there seems to be a bug somewhere.

### Comment 23 by @Rikki-Tavi (2020-02-08T11:40:24Z)

Ok. Thanks. I will keep my eye out for the resolution, once it appears, so that i might switch over.
