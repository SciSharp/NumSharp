# #487: linspace to Array as type float, while other functions as type double

- **URL:** https://github.com/SciSharp/NumSharp/issues/487
- **State:** OPEN
- **Author:** @changjian-github
- **Created:** 2023-02-13T12:51:52Z
- **Updated:** 2023-02-13T12:52:41Z

## Description

```
using NumSharp;

class Program
{
  static void Main(string[] args)
  {
    double[] x = np.arange(-1, 1.1, 0.1).ToArray<double>();
    float[] y = np.linspace(-1, 1, 21).ToArray<float>();
    double[] z = np.random.rand(21).ToArray<double>();
    Console.WriteLine("compile passed");
  }
}
```
Changing float to double will result in an error.
