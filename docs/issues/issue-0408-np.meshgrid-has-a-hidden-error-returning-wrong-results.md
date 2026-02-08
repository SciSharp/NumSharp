# #408: np.meshgrid() has a hidden error returning wrong results

- **URL:** https://github.com/SciSharp/NumSharp/issues/408
- **State:** OPEN
- **Author:** @LordTrololo
- **Created:** 2020-04-23T10:38:03Z
- **Updated:** 2021-07-15T12:26:10Z

## Description

There seems to be a bug in the `np.meshgrid()` function. It has something to do with the way the unmanaged memory is handeld but I havent tried to understand the details.

Why hidden ? Well, lets say we have the following code:
```
var meshAB = np.meshgrid(np.arange(3), np.arange(3));
```

`meshAB.Item1` seems to be ok, but `meshAB.Item2` hides a nasty problem. 
Strangely enough, when one makes a clone of Item2, the cloned values are OK! This is also my quick fix for now.

So to summary, the code below:
```
var meshAB = np.meshgrid(np.arange(3), np.arange(3));
Console.WriteLine("meshAB.Item1.flatten(): " + meshAB.Item1.flatten().ToString());
Console.WriteLine("meshAB.Item2.flatten(): " + meshAB.Item2.flatten().ToString());

NDArray a = meshAB.Item1.Clone();
NDArray b = meshAB.Item2.Clone();

Console.WriteLine("a.flatten(): " +a.flatten().ToString());
Console.WriteLine("b.flatten(): " +b.flatten().ToString());
```

Will output:
```
meshAB.Item1.flatten(): [0, 1, 2, 0, 1, 2, 0, 1, 2]    
meshAB.Item2.flatten(): [0, 1, 2, 0, 1, 2, 0, 1, 2]     <-------WRONG
a.flatten(): [0, 1, 2, 0, 1, 2, 0, 1, 2]
b.flatten(): [0, 0, 0, 1, 1, 1, 2, 2, 2]      <------ CORRECT
```
The bug is nasty because in inspector we also see ok values which is not true. Here is an image:
![image](https://user-images.githubusercontent.com/61494668/80100219-f6ff1e00-856f-11ea-8da8-91caf7e164af.png)


## Comments

### Comment 1 by @ArthHil (2021-07-15T10:48:05Z)

Same problem, but clone didnt resolve the problem 

### Comment 2 by @Oceania2018 (2021-07-15T12:24:32Z)

I tested it in [tf.net preview](https://github.com/SciSharp/TensorFlow.NET/blob/16ff7a3dafd283b07d91f61a88e0743a3c47c9fc/test/TensorFlowNET.UnitTest/Numpy/Array.Creation.Test.cs#L69). It's working good.
![image](https://user-images.githubusercontent.com/1705364/125787200-59813f3a-616e-4b44-9f1a-435fc17fe3b2.png)
New version implemented the NumPy API in [TensorFlow NumPy](https://www.tensorflow.org/guide/tf_numpy)
