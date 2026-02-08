# #298: Implement numpy.random.choice

- **URL:** https://github.com/SciSharp/NumSharp/issues/298
- **State:** OPEN
- **Author:** @Plankton555
- **Created:** 2019-06-27T09:42:50Z
- **Updated:** 2019-10-05T16:12:40Z
- **Labels:** enhancement
- **Assignees:** @Nucs

## Description

Generates a random sample from a given (possible weighted) 1-D array.

NumPy docs: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html

## Comments

### Comment 1 by @Plankton555 (2019-06-27T09:48:16Z)

I needed this and implemented a subset of this functionality (random sampling based on weighted probabilities). I can clean up that code and upload here so that it maybe can work as a start for this functionality.

I've never contributed to an open-source project before though, so I might need some help when it comes to what must be implemented, and where in the architecture it should be located, and so on.

### Comment 2 by @henon (2019-06-27T10:15:05Z)

One of us (maybe @Nucs ?) can provide the method stub for you to fill in the code. You can then look at that commit to learn which files needed to be touched to add a new function.

### Comment 3 by @Plankton555 (2019-06-28T10:11:27Z)

Please do! Let me know when that is done.

### Comment 4 by @Nucs (2019-06-30T20:02:39Z)

Sorry it took so long, it should be placed in `NumSharp.Core/Random/np.random.choice.cs`
```C#
/// <summary>
/// //todo
/// </summary>
/// <param name="arr">If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)</param>
/// <param name="shape"></param>
/// <param name="probabilities">The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.</param>
/// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html</remarks>
public NDArray choice(NDArray arr, Shape shape, double[] probabilities = null) {
    throw new NotImplementedException();
}        

/// <summary>
///  //todo
/// </summary>
/// <param name="a">If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)</param>
/// <param name="shape"></param>
/// <param name="probabilities">The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.</param>
/// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html</remarks>
public NDArray choice(int a, Shape shape, double[] probabilities = null) {
    throw new NotImplementedException();
}
```
Feel free to contact us via [gitter](https://gitter.im/sci-sharp/community) if you get stuck or have a question

### Comment 5 by @Plankton555 (2019-07-02T14:16:04Z)

Working on this in https://github.com/Plankton555/NumSharp/tree/feature/np_random_choice

### Comment 6 by @Plankton555 (2019-07-04T13:52:33Z)

One of the examples in the numpy docs looks like
```
>>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
>>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],
      dtype='|S11')
```

Since the numsharp method signature either takes an integer or an NDArray I tried implementing this string[] list with an NDArray. This gives an exception (which maybe should be reported as a bug or nonimplemented feature since numpy arrays can take strings).
```
NDArray aa_milne_arr = new string[] { "pooh", "rabbit", "piglet", "Christopher" }; // throws System.NotImplementedException: implicit operator NDArray(Array array)
```

In this particular case I can solve it in some ways:
1. Let this error happen until the NDArray has support for strings.
2. Have another method signature which takes an array/enumerable of some sort. This could be reasonable since the numpy docs explicitly states that np.random.choice "Generates a random sample from a given 1-D array", which should be possible to represent as an enumerable?

Any thoughts?

### Comment 7 by @Oceania2018 (2019-07-04T13:55:59Z)

@Nucs Do you have any plan on `String` support?

### Comment 8 by @Plankton555 (2019-07-04T14:31:40Z)

Pull request at https://github.com/SciSharp/NumSharp/pull/310

### Comment 9 by @Nucs (2019-10-05T16:12:40Z)

This issue is still open because `np.random.choice` does not support multi-dimensions.
