# #447: np.sum() Is supported on numsharp0.20.5, but not on NumSharp0.30.0

- **URL:** https://github.com/SciSharp/NumSharp/issues/447
- **State:** OPEN
- **Author:** @lijianxin520
- **Created:** 2021-04-13T06:35:26Z
- **Updated:** 2024-02-26T22:17:28Z

## Description

np.sum() Is supported on numsharp0.20.5, but not on NumSharp0.30.0
the exception message :"Specified method is not supported."

## Comments

### Comment 1 by @Nucs (2021-04-13T11:24:36Z)

What datatype are you trying to use?

### Comment 2 by @lijianxin520 (2021-04-14T00:57:05Z)

 double s = np.sum(w * w);
w data type is NDArray;

### Comment 3 by @lijianxin520 (2021-04-14T01:16:44Z)

![image](https://user-images.githubusercontent.com/47262889/114640175-1b648500-9d02-11eb-8ef5-c75dcc150941.png)


### Comment 4 by @Nucs (2021-04-14T11:13:10Z)

I mean what will this output on NumSharp 20.5?
``` C#
Console.WriteLine(w.dtype);
Console.WriteLine(s.dtype);
```

At version 30.x+ @Oceania2018 has removed many supported DTypes which might cause `NotSupportedException` or `NotImplementedException`. Some even return null.

### Comment 5 by @lijianxin520 (2021-04-15T01:07:57Z)

Thank you very much for your attention, the following is the supplementary content.
![image](https://user-images.githubusercontent.com/47262889/114799357-07845600-9dca-11eb-9374-e26c77e0e102.png)


### Comment 6 by @bigdimboom (2021-04-18T17:14:47Z)

same problem. Is there a walkaround for now?


### Comment 7 by @Nucs (2021-04-21T04:46:24Z)

@lijianxin520 I was not able to reproduce this locally, can you or @bigdimboom provide with a simple unit test that reproduces this and I'll take a deeper look.

### Comment 8 by @ppsdatta (2021-04-23T10:14:54Z)

Hello,
I encountered this problem as well and here's some data to help investigate:
1. A sample code which reproduces the problem on my Mac - Visual Studio. https://github.com/ppsdatta/NpSumIssue
2. With version 0.30.0 the code fails with the not supported error.
![Error screen shot](https://user-images.githubusercontent.com/18713580/115856850-9e948200-a44a-11eb-8a6a-4a0665002cad.png)
3. With version 0.20.5 the code works without runtime exception. 
![No error screen shot](https://user-images.githubusercontent.com/18713580/115856971-c4ba2200-a44a-11eb-9e65-f97dbe509dbd.png)



### Comment 9 by @badjano (2021-08-19T22:31:21Z)

having exact same error, numsharp 0.30.0
any workaround besides you know... for?

### Comment 10 by @gv-collibris (2021-08-25T13:23:42Z)

same error with NumSharp 0.30.0, with `np.sum(NDArray[])`

### Comment 11 by @yuta0306 (2021-11-10T00:23:30Z)

I also encountered the same error.
When the value of elements of NDArray is `double`, this error certainly occur.
To change the type `double` to `float` works well in my case.

### Comment 12 by @guozifeng91 (2024-01-10T17:34:38Z)

same error here, sum() works for int but not for double, using version 0.30.0

### Comment 13 by @PavanSuta (2024-02-26T22:17:27Z)

I am using v4.0.30319 Numsharp. Getting the same error after calling the sum function. I am passing NDArray Double datatype.
