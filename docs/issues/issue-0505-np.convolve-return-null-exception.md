# #505: `np.convolve` return null exception 

- **URL:** https://github.com/SciSharp/NumSharp/issues/505
- **State:** OPEN
- **Author:** @behroozbc
- **Created:** 2023-12-25T14:42:33Z
- **Updated:** 2023-12-25T14:42:33Z

## Description

I am new to this repository, and I want to test a convolve funcation but I got `System.NullReferenceException: Object reference not set to an instance of an object.`. I am using the NumSharp version: 0.30.0 and .net sdk version: 8.0.100 my code is simple, which I wrote in a console app.
```
using NumSharp;
var f = np.array(new int[] { 3, 3, 2, 1, 2 });
var g = np.array(new int[] { -1, 2, 1 });
var outp=np.convolve(f,g, "valid");
Console.WriteLine(outp.ToString());
```
the full error message
```
Unhandled exception. System.NullReferenceException: Object reference not set to an instance of an object.
   at Program.<Main>$(String[] args) in E:\repos\ConsoleApp7\ConsoleApp7\Program.cs:line 7
```
