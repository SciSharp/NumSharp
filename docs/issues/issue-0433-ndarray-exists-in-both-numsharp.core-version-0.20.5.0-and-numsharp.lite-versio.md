# #433: NDArray exists in both NumSharp.Core, Version=0.20.5.0 and NumSharp.Lite, Version=0.1.9.0

- **URL:** https://github.com/SciSharp/NumSharp/issues/433
- **State:** OPEN
- **Author:** @gscheck
- **Created:** 2020-12-09T16:50:56Z
- **Updated:** 2020-12-09T20:14:45Z

## Description

I get the following error when trying to compile an example:

Severity	Code	Description	Project	File	Line	Suppression State
Error	CS0433	The type 'NDArray' exists in both 'NumSharp.Core, Version=0.20.5.0, Culture=neutral, PublicKeyToken=cc7b13ffcd2ddd51' and 'NumSharp.Lite, Version=0.1.9.0, Culture=neutral, PublicKeyToken=cc7b13ffcd2ddd51'	Tensorflow1	I:\Operations\Test Engineering\Test Eng\Software Development\AOI\Tensorflow1\Tensorflow1\Form1.cs	47	Active


I am only using TensorFlow.NET and NumSharp libraries.

See screen shots below:

![image](https://user-images.githubusercontent.com/32424716/101659915-6c363b00-39fb-11eb-8c40-e86669bd195c.png)

![image](https://user-images.githubusercontent.com/32424716/101659977-7eb07480-39fb-11eb-8586-8ffa642f1059.png)



## Comments

### Comment 1 by @Oceania2018 (2020-12-09T17:59:09Z)

Remove NumSharp reference, just reference TensorFlow.NET project. It will include NumSharp automatically.

### Comment 2 by @gscheck (2020-12-09T18:29:12Z)

If I remove the NumSharp reference, I get the following error.

Severity	Code	Description	Project	File	Line	Suppression State
Error	CS0246	The type or namespace name 'NDArray' could not be found (are you missing a using directive or an assembly reference?)	Tensorflow1	I:\Operations\Test Engineering\Test Eng\Software Development\AOI\Tensorflow1\Tensorflow1\Form1.cs	46	Active


![image](https://user-images.githubusercontent.com/32424716/101671436-5891d100-3a09-11eb-8217-918b6444a1e0.png)

![image](https://user-images.githubusercontent.com/32424716/101671376-431ca700-3a09-11eb-930c-5fb6c5e4dfa7.png)




### Comment 3 by @Oceania2018 (2020-12-09T20:14:35Z)

Remove project reference means remove it from package.
![image](https://user-images.githubusercontent.com/1705364/101682026-ae29a600-3a28-11eb-8fad-6b9496827314.png)

You still need:
```csharp
using Numsharp;
```

The easiest step is just follow this [sample project](https://github.com/SciSharp/SciSharp-Stack-Examples).
