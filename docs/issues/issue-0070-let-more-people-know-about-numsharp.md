# #70: Let more people know about NumSharp

- **URL:** https://github.com/SciSharp/NumSharp/issues/70
- **State:** OPEN
- **Author:** @Oceania2018
- **Created:** 2018-11-08T03:07:08Z
- **Updated:** 2019-08-06T13:03:10Z
- **Labels:** help wanted

## Description

I've written an article [here](https://medium.com/@haiping008/numsharp-numerical-net-7c6ab6edfe27) which introduce NumSharp. 

@dotChris90 Can you start with the [docs](https://numsharp.readthedocs.io).

## Comments

### Comment 1 by @dotChris90 (2018-11-08T06:24:14Z)

Would start maybe at weekends. Next 2 days have little bit more work to do. :) 

### Comment 2 by @dotChris90 (2018-11-08T10:19:18Z)

@Oceania2018 just one little thing. Sorry I mention it.

Shouldn't it be:

var a = np.arange(9).Reshape(3,3)?

So with capital R instead of small r?
The output of arange is a NDArray object. So we call its Reshape method.

What we are doing is method chaining. So we do not use the np.reshape method (if exist didn't look ^^') but the NDArray Reshape method. 

### Comment 3 by @dotChris90 (2018-11-08T10:46:38Z)

@Oceania2018 and question : shall we start doc with python sphinx? Or maybe take docfx? https://dotnet.github.io/docfx/

Just idk if sphinx can parse c# for api documentations by doc comments. 

### Comment 4 by @Oceania2018 (2018-11-08T11:34:21Z)

We should use both of them, docfx is for generating md files, Sphinx is for organizing them, and make docs readable on readthedocs.io. 


I prefer arange().reshape(), keep same as numpy as much as possible.

### Comment 5 by @dotChris90 (2018-11-17T18:19:34Z)

@Oceania2018 at the moment I experiment with docfx - it does not look as bad as i though. Especially the API documentation and the style looks promising. Moreover docfx is the one which is used from Microsoft I though. The hosting we could do on Github Pages. 

Just feel annoyed at sphinx python doc, that it highlight my C# code but it thinks it is python lol so the code looks not good. And was investigating for C# docs at moment. 

How you think of docfx? i tried at https://dotchris90.github.io/NumSharp/ I will not merge it to Scisharp/NumSharp - so dont worry but I will see how well this can look like. 

### Comment 6 by @Oceania2018 (2018-11-17T18:30:20Z)

Does docfx generate markdown file? I think docfx is used to generate developer's API reference, because it is generated from souce code directly. Sphinx is often used to make tutorial document, like quick start. Sphinx support markdown file, it's easy to use.

### Comment 7 by @dotChris90 (2018-11-17T18:58:24Z)

Yes they are now able to do it. I was also surprised. Seems they developed quite well.

You can write so called articles which is equivalent sphinx tutorials. Using markdown. Very easy seems. So tutorials and api and swagger rest api and more seems (ok rest API is not important for us....).

I can maybe experiment with it and upload to my repo. If we think it looks good we can go with it. If not we take sphinx. 

### Comment 8 by @dotChris90 (2018-11-18T17:01:07Z)

@Oceania2018 an example for our api 
https://dotchris90.github.io/NumSharp/api/NumSharp.NDArray-1.html

And example for article : 
https://dotchris90.github.io/NumSharp/articles/NDArray.Creation.html

@Oceania2018 if u not too angry with me. I would like to go with docfx. They plan to support multiple languages besides c# in future and using 100% md for articles. Plus the generated docs folder can be hosted 100% on github for free. 

The only thing that I am not satisfied is... The style. I look now for better template... 




### Comment 9 by @Oceania2018 (2018-11-19T02:29:48Z)

@dotChris90 Docfx looks awesome. keep going.

### Comment 10 by @dotChris90 (2018-11-19T02:48:29Z)

@Oceania2018 thanks. Before thinking about pull requests. Is NumSharps github page active? If not, could you activate it? Not sure if I can since u the founder / constructor of SciSharp organizations. :) 

### Comment 11 by @Oceania2018 (2018-11-19T03:22:12Z)

@dotChris90 It's active. https://scisharp.github.io/NumSharp/

### Comment 12 by @dotChris90 (2018-11-19T03:29:29Z)

@Oceania2018 thanks a lot. Then today maybe write some more user tutorials. ;)
And push. 

### Comment 13 by @Oceania2018 (2018-11-19T03:32:43Z)

Sounds great. NumSharp got 80+ stars now. Seems like people  really need it.

### Comment 14 by @dotChris90 (2018-11-19T03:42:26Z)

Totally. Before c# was not the language of numeric. Since this year Microsoft push this AI and ML topics so hard... But Microsoft didn't give us a numerical stack.

But people deserve and need a numerical stack like NumSharp. :) 

### Comment 15 by @dotChris90 (2018-11-19T11:52:18Z)

Ok it worked. Just merged the docs which I already made. Nothing new. But will write some tutorials for user and also a small Readme how to use docfx best.

It's time we dotnet developers go deeper to science, machine learning and numerics. 

### Comment 16 by @fdncred (2018-11-19T13:54:26Z)

docfx is looking good.

### Comment 17 by @Nucs (2019-08-06T13:03:10Z)

@henon @Oceania2018, Should we implement a documentation page like discussed here?
https://dotnet.github.io/docfx/
