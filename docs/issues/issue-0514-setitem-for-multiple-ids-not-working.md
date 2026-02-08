# #514: SetItem for multiple Ids not working

- **URL:** https://github.com/SciSharp/NumSharp/issues/514
- **State:** OPEN
- **Author:** @MaxOmlor
- **Created:** 2024-06-03T07:57:24Z
- **Updated:** 2024-06-03T07:57:24Z

## Description

```C#
[Test]
    public void MultiIdSetItem()
    {
        var a = np.arange(5);
        Debug.Log($"a = {a}");
        var ids = np.array(new int[] { 1, 3 });
        Debug.Log($"ids = {ids}");
        var newValues = np.array(new int[] { 10, 20 });
        Debug.Log($"newValues = {newValues}");

        a[ids] = newValues;
        Debug.Log($"after set item: a = {a}");
        
        var expected = np.array(new[] { 0, 10, 2, 20, 4 });
        Assert.AreEqual(expected, a);
    }
```
Output:
```
Expected and actual are both <NumSharp.NDArray>
  Values differ at index [1]
  Expected: 10
  But was:  1
---
a = [0, 1, 2, 3, 4]
ids = [1, 3]
newValues = [10, 20]
after set item: a = [0, 1, 2, 3, 4]
```

