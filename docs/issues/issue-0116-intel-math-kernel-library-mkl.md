# #116: Intel Math Kernel Library (MKL)

- **URL:** https://github.com/SciSharp/NumSharp/issues/116
- **State:** OPEN
- **Author:** @Oceania2018
- **Created:** 2018-11-17T15:12:25Z
- **Updated:** 2019-05-02T16:53:31Z
- **Labels:** enhancement, help wanted
- **Assignees:** @dotChris90
- **Milestone:** v0.7

## Description

Performance-sensitive algorithms can be swapped with alternative implementations by the concept of providers like Intel MKL.
https://software.intel.com/en-us/performance-libraries

## Comments

### Comment 1 by @dotChris90 (2018-12-12T20:01:13Z)

Intels MKL library implement the LAPACK interfaces - so it would be logical to think about a layer with different LAPACK providers - in other words - a strategy pattern. Let users decide which LAPACK provider they want to use. 

It will be not much work since LAPACK usual means --> all interfaces are the same. So the PInvoke class would be the same - just the native lib is different.

As fdncred already said. :) 

### Comment 2 by @Oceania2018 (2018-12-12T20:01:52Z)

Sounds great.

### Comment 3 by @fdncred (2018-12-12T20:05:26Z)

This is exactly what I was saying in #145 when I mentioned and linked to Armadillo.

### Comment 4 by @dotChris90 (2018-12-13T07:27:39Z)

@fdncred sorry yes i know but I want to mention it here at this place ;) 

### Comment 5 by @Oceania2018 (2018-12-13T18:04:00Z)

@dotChris90 Please add MKL as a new provider when you have a chance.

### Comment 6 by @dotChris90 (2018-12-13T18:29:30Z)

yes. but after seeing Intel website I think we go strategy that mkl provider expects the lib installed. on their site they said u have to registry etc..... So.... I don't wanna quarrel with them. in case we simple copy their DLLs and publish them with NumSharp.... think Intel will be not happy. 

so installation must be done manually by  user.

At moment not sure what is most comfortable way to change providers.

My suggestion is that static numpy class has a static property lapack provider which is an enum. 

The static lapack class will check this enum and will call the specific static lapack provider classes.

The user can change provider global. I think most comfortable. 

### Comment 7 by @fdncred (2018-12-13T19:04:47Z)

@dotChris90, I'll research this a bit because other people, namely microsoft, point to dlls such as this https://docs.microsoft.com/en-us/cognitive-toolkit/setup-mkl-on-windows.

Now that I look closer, this one ls mklml and is opensource by intel. Perhaps that is different.

### Comment 8 by @fdncred (2018-12-13T19:12:12Z)

Check this out. Intel allows one to redistribute MKL. It's in the license.
https://software.intel.com/en-us/mkl/license-faq

And here is the full license.
https://software.intel.com/en-us/license/intel-simplified-software-license


### Comment 9 by @fdncred (2018-12-13T19:20:21Z)

Here's a few Github projects that wrap MKL in C#. May be easier just to use something that already exists.
https://github.com/DNRY/CSIntelPerfLibs
https://github.com/Rafka86/SharpMKL
https://github.com/Proxem/BlasNet


### Comment 10 by @dotChris90 (2018-12-13T19:40:01Z)

@fdncred much thanks for investigation. This makes the situation much better. 👍 

### Comment 11 by @dotChris90 (2018-12-14T07:26:10Z)

Hm - ok it will not be so easy as I though. 

The problem is that MKL implement just a subset of LAPACK and it implement LAPACKE - not LAPACK. They are little bit different from interfaces. 

The LAPACK we have at moment (Standard) has about 1500 Functions - MKL has 450. But is much bigger .... NetLib is about 8MB and their MKL 150MB -.- 

Sorry I say - this issue will be open little bit longer because Intel just implement what they want .... 

### Comment 12 by @dotChris90 (2018-12-14T07:36:25Z)

But do not worry - the Provider strategy I still want to follow. ;) 

### Comment 13 by @dotChris90 (2019-01-05T08:36:59Z)

ok some news from my side. 

MKL is installable on Linux via https://gist.github.com/pachamaltese/afc4faef2f191b533556f261a46b3aa8
and on Windows via PIP with command"pip install mkl". 

With this Libs can check what functions **are really exported**.

@Oceania2018 @fdncred  just one question to you 2. What you think about "put providers in separate packages?" Like "NumSharp.MKLProvider" and so on. The benefit would be --> do unit Tests and maybe automatic benchmark Tests just when the provider project made some changes and not NumSharp. 
Appveyor would be capable of this. 

@Oceania2018 this would be maybe also a solution for Tensorflow.NET --> package the Libs as Provider in new Nuget Package. 

### Comment 14 by @Oceania2018 (2019-01-05T11:21:41Z)

Seems like if we separate NumSharp.MKLProvider, we need to separate an Interface project too, then we will have 3 separate NuGet packages, right?

### Comment 15 by @fdncred (2019-01-05T13:36:58Z)

I like the idea of separate packages because it gives users a choice.

### Comment 16 by @dotChris90 (2019-01-05T14:40:46Z)

@Oceania2018 I think so. at least easiest solution. also agree with @fdncred. people can free choose. maybe we even can pack mkl in nuget package like python do. since pip offers nuget also could do. and we could offer nuget package with tensorflow C Api. 

 I think it could be best solution because people can use mkl if they like or not. or take standard lapack.

by the way on Friday I checked the CI of the standard open source lapack. with appveyor was able to build the DLLs and shared objects myself.

If appveyor offers mac OS images we also could support Mac OS.


### Comment 17 by @Oceania2018 (2019-01-05T14:53:49Z)

@dotChris90 you can try to add a new project.

### Comment 18 by @dotChris90 (2019-01-05T20:34:53Z)

Ok before do the stuff just want to clear with you 2 (@Oceania2018 @fdncred ) the plan. 

Our NumSharp repo would consist of 

- NumSharp.Core => our main project with no provider just abstraction (like an interface ILAPACKProvider)
- NumSharp.NetLibLAPACK => it provides the default LAPACK implementation from https://github.com/Reference-LAPACK/lapack (we could fork it and build C# wrapper classes - already was able to set it up on appveyor and generate some *.dlls 
- NumSharp.IntelMKL => it provides the MKL from Intel with LAPACK (just can deliver the binaries - no source code)
- Ironpython and Powershell projects but they are not relevant for our providers

Beside This I suggest one more thing 
- folder "NumSharp.Meta" with a nuget spec file 
- the NumSharp.Meta folder with its nuget spec file shall be used to generate an metapackage called simple "NumSharp" (this NumSharp package should add multiple references to users project like NumSharp.Core + NumSharp.NetLibLAPACK - in future we can add more but for now this 2 references are enough)
- why we should do it? --> users just need to do "dotnet add package NumSharp" (so 1 instead of 2 add package commands) and can start! but! important! they can remove the reference to NetLibLAPACK by themselves and add the IntelMKL - and everybody is happy. ;)

@Oceania2018 for Tensorflow.NET I also suggest this solution with meta-package - so we deliver Tensorflow C API as package for users. 

### Comment 19 by @dotChris90 (2019-01-05T20:45:28Z)

But also the annoying news .... Since we take a provider strategy we must take an OOP strategy pattern and ... they do not work with static methods. This just means our providers must be implemented as non-static classes with constructor - or other said - they must implement an interface. 

This is no bad thing just brings more code. we need our static methods to adapt the native libraries and we need the non static methods to implement interface.  

### Comment 20 by @Oceania2018 (2019-01-05T22:08:17Z)

@dotChris90 Where is the interface project? We should have another Interface project dll.

### Comment 21 by @dotChris90 (2019-01-07T19:46:35Z)

hm I am not sure at moment. 

We could make a separate project and package or keep it in NumSharp.Core. 
At moment think about pros and cons. 

### Comment 22 by @Oceania2018 (2019-01-07T19:52:20Z)

OK, do more research.

### Comment 23 by @buybackoff (2019-01-07T19:59:49Z)

Have you looked at Math.Net MKL provider? Or at Math.Net integration at all? As far as I remember Math.Net has its own native wrapper over  which they interop, but it should be trivial to amend if all parties agree. I'm following this project/stack for a while and happy if Python stack is repeated 1-to-1 in  .NET, but having several options that basically do the same stuff seems sad. Especially given the size of MKL dependency. 

Cc @cdrnet

### Comment 24 by @Oceania2018 (2019-01-07T20:57:15Z)

@dotChris90 Do you know how much size of the MKL dll? As my experience of TensorFlow.NET, tensorflow.dll is about 65M, luckly the NuGet compress it to about 16M. So I pack the dll directly into NuGet, It's acceptable.

### Comment 25 by @pkingwsd (2019-01-08T02:46:16Z)

as below from MathNet


Licensing Restrictions
Be aware that unlike the core of Math.NET Numerics including the native wrapper, which are both open source under the terms of the MIT/X11 license, the Intel MKL binaries themselves are closed source and non-free.

The Math.NET Numerics project does own an Intel MKL license (for Windows, no longer for Linux) and thus does have the right to distribute it along Math.NET Numerics. You can therefore use the Math.NET Numerics MKL native provider for free for your own use. However, it does not give you any right to redistribute it again yourself to customers of your own product. If you need to redistribute, buy a license from Intel. If unsure, contact the Intel sales team to clarify.

@Oceania2018 
MathNet.Numerics.MKL.Win-x64 nuget package, the size about 136 MB,  mathnet.numerics.mkl.linux-x64 about 134m


### Comment 26 by @Oceania2018 (2019-01-08T02:50:27Z)

@pkingwsd It means we can't pack MKL into NumSharp's package? @dotChris90 so we just have to place the download link for user? 

### Comment 27 by @pkingwsd (2019-01-08T02:57:28Z)

@Oceania2018  i thank so that ,just make NumSharp's MKL for windows package.   

### Comment 28 by @dotChris90 (2019-01-08T07:00:00Z)

@buybackoff thanks for mention it. Actually I was thinking about this **BUT** (there is always a but ... ) I am really unsure about Intel Licensing etc. Not sure if we can use their Math.NET Libs .... thats the only reason. 
To be honest - we plan that at the end is just one meta-data package called **NumSharp** and this contains references to **NumSharp.Core** and **NetLibLAPACK** package. 

For NetLibLAPACK Standard - I already tried out their build system with my personal account https://github.com/dotChris90/lapack. I am sure if we mention them in our Readme its ok for them. 

MKL is tricky. For standard user who just want to learn and get things done I prefer the NetLibLAPACK Lib. which will be reference when using "dotnet add package NumSharp" in future. But .... when we cleared this discussion about MKL .... we want to give user the chance to remove this NetLibLAPACK reference from their project and use MKL intel. This will reduce the space in memory and hard disk space.  

### Comment 29 by @dotChris90 (2019-01-08T07:04:37Z)

@Oceania2018 @pkingwsd damn ... this license stuff get more complex .... but yes he is right. **If you do not mind - I will write an email to intel sales team to clear things.**

I am little confused at moment since the Linux MKL is a simple debian based package and can be downloaded any time. So I though there is no restriction for windows also. But I am not sure so far. Because I can not find official windows repo for MKL .....

### Comment 30 by @dotChris90 (2019-01-08T15:00:49Z)

@Oceania2018 @pkingwsd what about this section https://software.intel.com/en-us/mkl/license-faq ? 
In the FAQ stand Yes, redistribution is allowed per the terms of the ISSL. and also no payment etc. 

### Comment 31 by @Oceania2018 (2019-01-08T15:18:30Z)

Great, so we don't need to worry about the license problem per the official statement.

### Comment 32 by @dotChris90 (2019-01-08T15:41:53Z)

still wrote to them. I want a statement of them.... BTW fdncred also mentioned it before lol the faq but as I said I wait for their response. 

### Comment 33 by @fdncred (2019-01-08T16:04:00Z)

yeah! someone reads my comments. LOL! ;)

### Comment 34 by @dotChris90 (2019-01-08T16:09:44Z)

@fdncred definitely. I am just highly confused about the mkl libraries out there.

is anaconda mkl the same like normal mkl and why there is no package for windows... the apt get installer also just download the file and put at specific locations...

drives me crazy.... 

### Comment 35 by @fdncred (2019-01-08T16:47:58Z)

@dotChris90 it appears to me that anaconda's MKL is just the regular MKL. It also looks to me like there is a windows package. See the links below.
https://anaconda.org/anaconda/mkl
https://docs.anaconda.com/accelerate/mkl-overview/
https://repo.anaconda.com/pkgs/main/win-64/
For my two cents, I think Proxem's BlasNet is the best wrapper for MKL. It looks pretty mature.
https://github.com/Proxem/BlasNet

### Comment 36 by @dotChris90 (2019-01-08T21:13:48Z)

@fdncred  Thanks for the update. ;)

From the URL I agree with you. 👍 
I already downloaded the package via pip and checked it but the package brings multiple MKL dlls. In anaconda location you will find binaries with mkl_corem mkl_xyz, …. about 6 or 7 files. But I though the mkl thing is one single DLL …. but anaconda shows multiple .... this is extrem strange behaviour... 

I agree Proxems BlasNet looks fine but it does not solve our main problem. Proxem just brings Wrapper code in C# - please do not misunderstand me but I am not very impressed by this …. because there are a lot of C# Code-Generators who do the same job - e.g. ClangSharp, T4, SWIG and if I check the Mono team I am 100% sure I can find more generators. xD Such automatic wrapper always simple take a C header file and generate Code …. I really do not see any benefit to use it BlasNet. I saw the Code and as far as I can see the main methods we can use could be also generated automaticly by ClangSharp (or T4) so not sure why add package for something we can auto generate xD.  

Our (at least what makes me feel uncomfortable) problem is : Proxem does not bring a native MKL Lib package. I have to install it manually. In Linux its easy - "sudo apt-get install" but on windows it is impossible to do this. This makes CI Testing impossible. How do we want to do unit testing or automatic benchmark testing on Appveyor? Unfortunately -.- we can not do this - except we package this MKL DLL into a nuget package. damn windows has no apt get.... 


### Comment 37 by @fdncred (2019-01-08T22:08:23Z)

Probably not impossible for CI. Have you looked at using chocolatey to install MKL? It looks like one of microsoft's packages includes it https://chocolatey.org/packages/microsoft-r-open. It may not be perfect but it may work.

I've had a lot of experience with ClangSharp and NClang as well as other generators to take a C/C++ library to C# and I'm quite sure that they are not easy. I've found with one for months trying to get the library ported and it's very time consuming. You can get close pretty easily but you can't get complete easily. It'll take work for sure.

### Comment 38 by @dotChris90 (2019-01-09T05:16:00Z)

damn choco package management didn't think on it yes. can try. thanks.

If u say it is hard maybe I was too easy thinking. the only complex thing we could have with code generation is.... lapack 100 % uses pointer. so generator can not know what's array and what pass by reference.

hm...yes could be trickier. but as long as we just using some few functions from lapack.... writing own is easier. but yes maybe later need some more professional things. 

### Comment 39 by @fdncred (2019-01-09T13:56:05Z)

Yes, IntPtr from C/C++ can be tricky and requires hand massaging for every call, from my experience. Marshalling arrays isn't too complex if you know the count of the array and the size of the array. Ref, In, Out params can be tricky, especially if you have an [In, Out] parameter in C/C++, where you pass in out value in a parameter and you return a different value out of the same parameter. If you get stuck on something reach out to me and I may be able to help.

### Comment 40 by @Oceania2018 (2019-01-09T14:11:46Z)

@fdncred can you help with 
https://github.com/SciSharp/TensorFlow.NET/issues/86?

### Comment 41 by @fdncred (2019-01-09T14:17:14Z)

@Oceania2018, yes, i'll look at it. I'm not promising i can fix it but I'll try. ;)

### Comment 42 by @Oceania2018 (2019-01-09T14:22:30Z)

Anyway I’d appreciate. Try to make it work. Enjoy tensorflow.net.

### Comment 43 by @dotChris90 (2019-01-10T06:06:45Z)

for traceability. Intel just offer you help if u have customer service.. yeah every company need money I know....

But post in forum and got answer direct from Intel.

https://software.intel.com/en-us/comment/1932183#comment-1932183

as we can read there. don't worry take the mkl lib.

answers was (2019 Jan 10th, 7:09 Berlin time)

please refer here : license FAQ

you won't have a problem redistributing it. 

### Comment 44 by @mzhukova (2019-05-02T16:53:31Z)

Hi folks,
I saw on the Intel forum that you were asking about distributing MKL via nuget.
It is available now in nuget channel as well. I was just wondering, if this may be useful to you: https://www.nuget.org/packages?q=intelmkl

Best regards,
Maria
