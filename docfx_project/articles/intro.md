# Introduction 

The following pages are for the users who want to use NumSharp. 

Before you read the code examples you should read this page which explain some basis concepts. 
An other reference can be numpy since we try our best to follow their APIs. 

## NDArray, NDStorgage and Shape

The 3 main classes in NumSharp are NDArray, NDStorage and Shape. 
If you want to have a better understanding for NumSharp, you can read the following lines to see how all works together. 

Let's start with the question - what is a Tensor? 

From programming point of view a tensor is a multi-dimensional array (scalar, vector, matrix, ...) mostly for numerical data like int32, int64, doubles, ...  which can be accessed via indexes like np[idx], np[idx,jdx], np[idx,jdx,kdx], ... depending on its dimension. 

Ok - in this sentence we got already some properties. 

- a tensor is an object for storing (mostly) numerical data 
- a tensor has a dimension 
- the dimension decides how many indexes are necessary to access the stored data

Each tensor type (dimension 1 - vector, dimension 2 - matrix, ...) has its own .NET type like double[,]. 

NumSharp brings its own tensor / array type called **NDArray**.

So now the question - .NET offers already multi-dimensional arrays - why a new array type?

NumSharps NDArray offers the capability of storing any tensor (independent of dimension!) into its internal storage. 
So NumSharps NDArray can store a vector, a matrix or sth with dimension 5 and higher. This is not possible with .NET arrays since each tensor type is a different class. 

Now the next question - how a NDArray can do this? 

First of all we need to be a little bit more abstract. Why we use tensors? Because we want to store data and we want to get them. How we get and set them? We get and set via indexes (which are always integers). So just this data are important and the corresponding indexes.

With this in mind we easily can understand the NDStorage of NumSharp. 

NDStorage is an object which stores the data of a tesor in a single 1D array. Since it is a 1D array independend of the tensor dimension NDStorage can be used for all kind of tensors. 

**But hold on! How the data comes into this 1D array?**

NDStorage has a property called "shape". The shape is a small but important class in NumSharp. It stores the dimensions and most important! it determines which element in the 1D array is selected by given indexes. 








