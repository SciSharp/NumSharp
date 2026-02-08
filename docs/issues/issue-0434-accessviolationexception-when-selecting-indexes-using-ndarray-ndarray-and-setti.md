# #434: AccessViolationException when selecting indexes using ndarray[ndarray] and setting a scalar value

- **URL:** https://github.com/SciSharp/NumSharp/issues/434
- **State:** OPEN
- **Author:** @lijianxin520
- **Created:** 2020-12-31T02:30:16Z
- **Updated:** 2021-04-21T04:50:55Z
- **Labels:** bug

## Description

Attempted to read or write protected memory. This is often an indication that other memory is corrupt.
![2020-12-30_230218](https://user-images.githubusercontent.com/47262889/103391183-013bd800-4b53-11eb-9f76-0ad77b978ae2.png)


## Comments

### Comment 1 by @rikkitook (2021-04-01T12:43:13Z)

To my experience, problems like these can be avoided if you do not access an instance of ndarray from different threads or tasks. If it is still necessary, try copying ndarray to float[], pass it to your method and then copy to new ndarray.
Cheers!

### Comment 2 by @lijianxin520 (2021-04-13T06:43:06Z)

I'm single-threaded, so I shouldn't have any problems with multithreading

### Comment 3 by @Nucs (2021-04-13T11:19:50Z)

To my understanding - `AccessViolationException` usually occurs if you somehow lost reference to your `NDArray`.
`NDArray` only frees allocated memory when `IDisposable` is triggered by the garbage collector when there are no longer any references to it.
In addition, zero-copied `NDArray`s from any operation still hold a reference to base memory which should prevent deallocation.

Please provide here a reproducing unit test/piece of code and I'll gladly investigate. 

### Comment 4 by @lijianxin520 (2021-04-15T06:40:57Z)

这个是我测试的时候异常的内容,
![image](https://user-images.githubusercontent.com/47262889/114825071-521cc700-9df8-11eb-9125-08b91b5971fd.png)
当我对一个数组执行批处理操作时，我看到的结果是这个问题.
