# #498: Is there an example on how to use it with IronPython?

- **URL:** https://github.com/SciSharp/NumSharp/issues/498
- **State:** OPEN
- **Author:** @william19941994
- **Created:** 2023-08-22T01:52:28Z
- **Updated:** 2023-08-22T01:52:28Z

## Description

I have a big project using c# + wpf +.net framework 4. and I can upgrade to 4.8 or 4.7.2
and a supplier send some python examples to me. 
I have to add some python script to the old project. I don't want to rewrite.

I added IronPython yesterday, but it uses hardware communication lib, I replaced with a c# object.
and then, I found that it uses numpy.
I added another small c# class.
and then, I found that it uses lots of numpy functions.
I download this project and added a wrapper class to redirect the result.
and then, this project need System.Memory.dll 4.0.1.1, while my project has 4.5.5 .... 
and I changed the prj.exe.config to lower dll version. 
my exe can't run now.

I'm re-compiling this csprj now.

Does anyone have done this before? and where can I find a whole example or blog?

