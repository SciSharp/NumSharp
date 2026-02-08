# #456: silent catastrophe in implicit casting singleton array to value type

- **URL:** https://github.com/SciSharp/NumSharp/issues/456
- **State:** OPEN
- **Author:** @dmacd
- **Created:** 2021-07-29T22:27:25Z
- **Updated:** 2021-07-29T22:28:14Z

## Description

Hi there! 

I love NumSharp and its been a key enabler of my current project (a neurofeedback platform built in Unity3d).
However I ran across some odd behavior I feel compelled to surface. Example:

```
var reactor_temp = <result of some NDArray operations>
{-6.62420108914375}

(float)reactor_temp
-4.03896783E-28

(int)reactor_temp
-1845493760

(double)reactor_temp
-6.6242010891437531
```

In this example, the actual dtype of the singleton array was double, yet I'm freely (and _implicitly_) able to cast the object to other numeric types with no warning or error. This is....counterintuitive...to say the least, has cost me several hours of debugging time, and I'm frankly lucky to have even caught it at all. Fortunately, I'm only using numsharp to manipulate people's brainwaves and not fissile material just yet :) 


(As an aside: The issue surfaced for me when I changed an operation np.sum to np.mean, which changed the output type on me, contrary to expectation)

If the implicit conversions cant be type-guarded or converted in a sane manner, would it make more sense to just remove them?

Once again, grateful for all hard work that went in to this library!
Daniel



