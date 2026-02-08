# #401: How to convert NDArray to list

- **URL:** https://github.com/SciSharp/NumSharp/issues/401
- **State:** OPEN
- **Author:** @Sullivanecidi
- **Created:** 2020-03-18T00:26:17Z
- **Updated:** 2020-03-30T02:57:02Z

## Description

It is really a big surprise for me to find the Numsharp.
Here is little question with using the NumSharp: How to convert the NDArray data to list ? since in VB.net, list is the frequently used.
For example:
dim x_data, y_data as NDArray
dim y() as double
x_data = np.linspace(0,100,500)
y_data = x_data * 2 + np.random.randn(500)

how to convert y_data to y() ?

Thanks!

## Comments

### Comment 1 by @Oceania2018 (2020-03-18T02:35:02Z)

`x_data.ToArray<double>()`

### Comment 2 by @Sullivanecidi (2020-03-18T03:05:49Z)

> `x_data.ToArray<double>()`

Do you mean y = y_data.ToArray(of double)()
it doesn't work at all......

### Comment 3 by @Jiuyong (2020-03-18T05:45:07Z)

VB.Net don't support Of **unmanaged**
目前看到的情况是，VB.Net 不支持 C# 7.3 新的 **unmanaged** 泛型约束。
如果想要使用，可能需要变通一下了。

### Comment 4 by @pepure (2020-03-18T06:57:27Z)

> > `x_data.ToArray<double>()`
> 
> Do you mean y = y_data.ToArray(of double)()
> it doesn't work at all......

I offer a lower level solution：）
--------------------------------------------------------
Dim x_data, y_data As NDArray
x_data = np.linspace(0, 100, 500)
y_data = x_data * 2 + np.random.randn(500)

Dim y(y_data.size - 1) As Double
For i = 0 To y_data.size - 1
      y(i) = y_data(i)
Next i
--------------------------------------------------------
Debugging results：
![image](https://user-images.githubusercontent.com/53322892/76933706-b1c04e80-6928-11ea-9825-aa453c80421b.png)
Please confirm if it can solve your problem。


### Comment 5 by @Sullivanecidi (2020-03-18T07:23:03Z)

> VB.Net don't support Of **unmanaged**
> 目前看到的情况是，VB.Net 不支持 C# 7.3 新的 **unmanaged** 泛型约束。
> 如果想要使用，可能需要变通一下了。

那就可能是按照楼上这位了，一个一个取出来。谢谢你了~

### Comment 6 by @Sullivanecidi (2020-03-18T07:25:10Z)

Thanks very much! this is the alternative way, since VB.net don't support the unmanaged constraint.
 

### Comment 7 by @Sullivanecidi (2020-03-18T08:38:54Z)

> VB.Net don't support Of **unmanaged**
> 目前看到的情况是，VB.Net 不支持 C# 7.3 新的 **unmanaged** 泛型约束。
> 如果想要使用，可能需要变通一下了。

我刚试了下c#，是可以用toArray<double>()实现的。

### Comment 8 by @Jiuyong (2020-03-30T02:57:02Z)

是的，C#肯定没问题啊。
还有个稍微好一点的解决方案。
就是用C#弄一个缝合项目，然后VB项目引用。
