# #509: Extremely poor performance on sum reduce

- **URL:** https://github.com/SciSharp/NumSharp/issues/509
- **State:** OPEN
- **Author:** @lucdem
- **Created:** 2024-02-19T23:32:37Z
- **Updated:** 2024-03-27T04:09:12Z

## Description

Creating a large 3 dimensional array and calling sum(axis: 0) takes an enormous amount of time on my machine (11s), far more than Numpy (around 75ms) or even just a simple C# naive implementation (around 200ms).

C# Code:

```
using NumSharp;
using System.Diagnostics;
using System.Linq;

namespace NumSharpTest;

internal class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("NumSharp");
        NumSharp();
        Console.WriteLine("#######################");
        Console.WriteLine("Naive");
        Naive();
    }

    static void NumSharp()
    {
        var randArr = np.random.uniform(0, 1, [500, 500, 500]).astype(np.float32);

        var stopwatch = new Stopwatch();
        stopwatch.Start();
        var linesSum = randArr.sum(axis: 0);
        stopwatch.Stop();
        Console.WriteLine($"Elapsed: {stopwatch.ElapsedMilliseconds} ms");
    }

    static void Naive()
    {
        var random = new Random();
        var shape = new int[] { 500, 500, 500 };
        var randArr = new float[shape[0] * shape[1] * shape[2]];
        for (int i = 0; i < randArr.Length; i++) { randArr[i] = (float)random.NextDouble(); }

        var stopwatch = new Stopwatch();
        stopwatch.Start();
        var sum = new float[shape[1] * shape[2]];
        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < shape[1]; j++)
            {
                for (int k = 0; k < shape[2]; k++)
                {
                    sum[shape[2] * j + k] += randArr[shape[1] * shape[2] * i + shape[2] * j + k];
                }
            }
        }
        stopwatch.Stop();
        Console.WriteLine($"Elapsed: {stopwatch.ElapsedMilliseconds} ms");
    }
}


```

Python Code:

```
import numpy as np
from time import time

randArr = np.random.normal(1, 0.5, (500, 500, 500))

start = time()
sum = randArr.sum(axis=0)
end = time()

print(f'Elapsed: {(end - start) * 1000} ms')
```

## Comments

### Comment 1 by @vkribo (2024-03-27T04:09:11Z)

I profiled it and the problem seems to be in the method GetOffset in the Shape class. In the method there is a list called coords being created milions of times. I tried reimplementing it using a stackalloced array instead and it went from 6287 ms to 2315 ms.
Maybe someone more familliar with the code could look into it.
