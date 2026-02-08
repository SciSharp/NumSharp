# #426: arctan2() returning incorrect value

- **URL:** https://github.com/SciSharp/NumSharp/issues/426
- **State:** OPEN
- **Author:** @RoseberryPi
- **Created:** 2020-10-25T17:14:27Z
- **Updated:** 2020-10-29T22:22:09Z

## Description

So I'm using numsharp
`np.arctan2(np.array(-0.0012562886517319706), np.array(-0.7499033624114052))`

I get the wrong value, approximately, `-4.33e-5`

using numpy.net and C#'s Math library
`np.arctan2(np.array(-0.0012562886517319706), np.array(-0.7499033624114052))`
`Math.Atan2(-0.0012562886517319706, -0.7499033624114052)`

I get `-3.1399`, which is the correct value.. Am I just being dumb and doing something wrong or is NumSharp not actually calculating the correct value?

furthmore, `np.arctan2(1,1)` is 90deg according to numsharp. Should be 45.
`np.arctan2(1,-1)` is also 90deg....

I'm using version v0.20.5

## Comments

### Comment 1 by @ (2020-10-29T22:22:09Z)

Hello @RoseberryPi,

You are right, looking at code I don't understand the type casting done to (byte*) instead of (double*).
There must be an explanation.

```csharp
case NPTypeCode.Double:
{
   var out_addr = (double*)out_y.Address;
   var out_addr_x = (byte*)out_x.Address;
   Parallel.For(0, len, i => *(out_addr)  = Converts.ToDouble(Math.Atan2(*(out_addr) + i, *(out_addr_x) + i)));
   return out_y;
}
```
I tried casting to (double*) and it returns the expected value, 3.13991738776297 



