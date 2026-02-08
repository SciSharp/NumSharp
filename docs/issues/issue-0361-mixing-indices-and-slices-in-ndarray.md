# #361: Mixing indices and slices in NDArray[...]

- **URL:** https://github.com/SciSharp/NumSharp/issues/361
- **State:** OPEN
- **Author:** @Oceania2018
- **Created:** 2019-10-06T03:22:07Z
- **Updated:** 2019-10-12T11:35:37Z
- **Labels:** enhancement
- **Assignees:** @henon, @Nucs

## Description

Is it possible to suppor mixed index/ slice? This is what numpy doing:

![image](https://user-images.githubusercontent.com/1705364/66263777-49d4dc00-e7be-11e9-8fba-fd1014cbd922.png)

Currently, it doesn't support:

![image](https://user-images.githubusercontent.com/1705364/66263800-7688f380-e7be-11e9-933a-2dfeb71f98f3.png)


## Comments

### Comment 1 by @Nucs (2019-10-06T06:31:04Z)

```C#
label[i][Slice.Index(yind), Slice.Index(xind), Slice.Index(iou_mast), new Slice(0, 4)] = bbox_xywh;
```

### Comment 2 by @henon (2019-10-06T11:48:20Z)

or 

```C#
label[i][$"{yind}, {xind}, {iou_mast}, 0:4"] = bbox_xywh;
```

Btw: we could try implicitly convert from int to Slice.Index(int)


### Comment 3 by @Nucs (2019-10-06T12:30:22Z)

Worth mentioning that using a string inside a loop or O(n) will affect performance. 
Regex parsing is not lightning fast.

### Comment 4 by @Oceania2018 (2019-10-06T13:51:40Z)

The `iou_mask` is `NDArray`.
![image](https://user-images.githubusercontent.com/1705364/66269842-048ec980-e813-11e9-8e33-fa624271ace2.png)

It means we use different ways in different dimension to select elements.

One approach my idea is:
Define a `IIndex` interface, `NDArray` and `Slice`  implement `IIndex`.
update 
```csharp
NDArray this[params IIndex[] selectors]
{
   get; set;
}
```


### Comment 5 by @Nucs (2019-10-06T14:40:04Z)

Too messy, plus it won't solve your problem.. you can't implement implicit cast from `int` to `IIndex`.
Having index `nd[int, int, NDArray, Slice]` doesn't make sense. Our algorithm accepts either only slices or only ints.
`nd[NDArray[]]` is not for Indexing, it is for Masking.
Right now use `Slice.Index`, in the future I'll implement implicit castings to `Slice`.

### Comment 6 by @henon (2019-10-06T15:04:49Z)

We might have to extend the slicing algorithm to accept a mask (i.e. Slice.Mask(boolean_array)). He is porting Python code so it seems Python allows mixing of indices and masks. 

### Comment 7 by @Nucs (2019-10-06T15:06:57Z)

Masking has a complete separate algorithm from slicing. In fact they don't have anything in common (function-wise).

### Comment 8 by @Oceania2018 (2019-10-06T16:36:44Z)

Not only for masking, but also for any 1d array. We don’t have to have int to IIndex implicitly, we can use np.array(1) as alternative.

The goal is having people can use 1d or Slice, Scalar to select and mask elements in appropriate dimensions.

### Comment 9 by @Oceania2018 (2019-10-12T00:13:21Z)

@henon This situation doesn't work:

`iou_mask` is `NDArray`, `yind` and `xind` are `int`:
![image](https://user-images.githubusercontent.com/1705364/66691314-0addd500-ec5b-11e9-816f-be0dd63f9d0a.png)

![image](https://user-images.githubusercontent.com/1705364/66691447-0fef5400-ec5c-11e9-8368-d9f01b5ea440.png)


### Comment 10 by @henon (2019-10-12T08:32:57Z)

I know, @Nucs is removing the int[] indexer in favor of an object[] indexer so that implicit conversion operators don't get in the way and we can reliably forward the different use cases into the right implementatons ( slicing if no NDarrays are part of the params and index extraction/masking otherwise)

### Comment 11 by @Nucs (2019-10-12T11:35:37Z)

@Oceania2018 Please verify if it works on master branch.
