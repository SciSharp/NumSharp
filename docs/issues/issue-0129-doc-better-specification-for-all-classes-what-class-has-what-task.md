# #129: Doc : Better specification for all classes (What class has what task)

- **URL:** https://github.com/SciSharp/NumSharp/issues/129
- **State:** OPEN
- **Author:** @dotChris90
- **Created:** 2018-11-23T17:16:31Z
- **Updated:** 2018-12-14T12:25:49Z
- **Labels:** further discuss

## Description

Since now NumSharp offers a system for non generic, generic, dynamic and static we are in a good position for further dialog / discuss about classes tasks. 
Moreover this issue can be used for the developer documentation. 

Let us call it "tidy up" and think about what we really need and most important need to think about the SW design. 

At the moment I see the following classes and their tasks 

- Storage (which could be a child of interface class IStorage) 
    - store the Data somehow (abstract said - concrete store in 1D System.Array)
    - Get and Set Data in this Store (all at once or by indexing)
    - convert Data to specific type    
- Shape ( best practice would be child of IShape)
    - store the dimensions of the array 
    - compute the index of a 1D array by multiple indexes if array is a multidimensional array
    - compute the multiple indexes of multidimensional array from the index of 1D array
- Core.NDArray (non generic ) 
    - 1 NDArray has 1 Storage & 1 Shape 
    - delivers many methods for manipulate, processes, … 1 or N different Core.NDArrays.
    - since its non generic elements are ValueTypes (must be cast to double etc.)
- Generic.NDArray 
    - children of Core.NDArray which offers possibility to direct indexing since elements are of generic parameter type 
    
@Oceania2018 @fdncred you 2 ;) please correct me if I wrote something wrong.   

@Oceania2018 Sorry for my stupid questions always - is there a reason that a Storage shall have a shape? Its really just a design question. I just think for Testability Classes shall be as independent as possible from each others. If they do not need a relationship then they should not have. From my point of view since NDArray has a shape, Storage does not need a shape. Storage shall not care about its Shape and just contain the elements. Just our best friend NDArray shall use Storage and Shape to bring elements in correct form. 

;) anyway guys have a good week and nice weekend. 

 
  


## Comments

### Comment 1 by @Oceania2018 (2018-11-23T18:39:50Z)

Excellent concept describing. For shape in storage, my idea is storage should be responsible for persisting and seeking data from any dimension and position, NDArray’s main responsibility is calculating, not seek data form storage. We are open for this topic. And NumPy is responsible for exposing interfaces to external invoking.

### Comment 2 by @dotChris90 (2018-11-23T18:51:25Z)

hm good point.

from testability point of view storage should not have a shape.
from logic I am not sure.

We keep issue open and I look again the code. what fits better.

It's really just design question. user should not care this because they are not official Apis. 

### Comment 3 by @dotChris90 (2018-11-28T04:12:16Z)

I come up with decision.

I agree with you. the storage should be responsible for every data access and converting stuff.

But then we must be careful. Ndarray shall have no own shape. if necessary it shall call methods from its storage object. dtype same.
ndarray must return the dtype of storage at e. g. it's property dtype. 

### Comment 4 by @dotChris90 (2018-12-14T12:24:42Z)

@Oceania2018 @fdncred 

I added an interface for storage and one for shape. This is a future planning so the storage get easier and more clear to use. I come up with this because I noticed the storages methods sometimes not 100% well ..... most was my mistake -.- 

e.g. GetData<T>() return the internal 1D array but! if dtype is corresponding to T its the reference array and if not its a copy .... the data types are correct but the underlying pointers are different ..... and since you all so interested in performance, memory, etc. such things can not be. 


The interfaces are suggestions for next generation Storage lol https://github.com/SciSharp/NumSharp/blob/master/src/NumSharp.Core/Interfaces/IStorage.cs

As you see I made up the following convensions : 

- new property TensorOrder (maybe should call it TensorLayout) 
- Allocate will be used to choose Shape, data type and! TensorOrder
- GetData always return the reference of internal storage ALWAYS! 
- if dtype is not equal to T in GetData< T > the internal storage is converted automaticly! 
- New method CloneData which always will give you a copy of the internal storage and not storage itself
- so all in all GetData is copy by reference, CloneData is copy by value 
- GetColumWiseStorage & GetRowWiseStorage methods return the instance itself but with column wise or row wise layout 
- This layout stuff is extrem critical for LAPACK - including MKL! since they just accept rowwise and we at moment use columnwise ;)  


