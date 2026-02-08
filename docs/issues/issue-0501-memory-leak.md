# #501: Memory leak?

- **URL:** https://github.com/SciSharp/NumSharp/issues/501
- **State:** OPEN
- **Author:** @TakuNishiumi
- **Created:** 2023-12-13T07:32:29Z
- **Updated:** 2023-12-13T07:32:29Z

## Description

Hi,
I tried below code and usage rate of my memory increased up to around 10GB.

```C#
// 配列を宣言する
double[] array = new double[110];

// 乱数を生成する
Random rnd = new Random();

// 配列に乱数を代入する
// make random double array
for (int i = 0; i < array.Length; i++)
{
    array[i] = rnd.NextDouble();
}

// make new NDarray in loop
for (int i = 0; i < 1000000; i++)
{
    NDArray array2 = np.array(array);
    // NDArray array2 = np.argmin(array); also cause same trouble
}
```
This is also caused when this is used as function.
I modified this and add GC.Collect().
This make the code usage rate of my memory, but the calculation time increased.

```C#
// 配列を宣言する
double[] array = new double[110];

// 乱数を生成する
Random rnd = new Random();

// 配列に乱数を代入する
// make random double array
for (int i = 0; i < array.Length; i++)
{
    array[i] = rnd.NextDouble();
}

// make new NDarray in loop
for (int i = 0; i < 1000000; i++)
{
    NDArray array2 = np.array(array);
    if (i % 10000 == 0)
    {
        GC.Collect();
    }
}
```

What should I do next?
Do you have any ideas?
