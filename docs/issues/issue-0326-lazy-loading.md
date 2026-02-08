# #326: Lazy loading

- **URL:** https://github.com/SciSharp/NumSharp/issues/326
- **State:** OPEN
- **Author:** @aidevnn
- **Created:** 2019-08-01T13:30:31Z
- **Updated:** 2019-09-25T17:58:40Z
- **Labels:** further discuss
- **Assignees:** @Nucs

## Description

Hi
Your lib NumSharp was inspired me and i tried to write a other approach of NumPy more easy to use.
The big idea is the Lazy Loading for the front end and this is my repo.
https://github.com/aidevnn/DesertLand

My lib isnt optimized for speed at this time, but i can improve it with ease by rewritting NDarray backend class and methods WITHOUT changing nothing from the NDview struct frontend which use the lazy loading.

Actually your Backend is very very powerfull and maybe the frontend can be also improved with Lazy Loading.

Best regards.

## Comments

### Comment 1 by @Oceania2018 (2019-08-01T13:46:01Z)

@aidevnn Join us, build Deep Learning library based on current work.

### Comment 2 by @aidevnn (2019-08-01T13:52:23Z)

I started to fork your repo, it takes me some month to understand in deep your great work. 
Best regards.

### Comment 3 by @Nucs (2019-08-01T15:01:17Z)

Hey @aidevnn, Thank you.
A. Are you aware we are rewriting our backend in a [separate branch](https://github.com/SciSharp/NumSharp/tree/unmanaged-bytes-storage)?
B. Can you elaborate more about what do you mean by a front-end lazy loading?
If I understand correctly, lazy-loading is something that numpy does not implement therefore if you wish to contribute a lazy-loading front-end when NumSharp is the backend - it'll have to be in a seperate project.

### Comment 4 by @aidevnn (2019-08-01T19:23:20Z)

Hello @Nucs 

A. Ok a devel branch is more collaborative than a fork

B. Lazy Loading can separate the SharpNum library onto frontend with simple to write specification which will be used by the rest of SciSharp ML/AI framework and the pure SharpNum backend for fast and speed computation. The SciSharp ML/AI team can write their own unit test to integrate SharpNum progression.
Lazy Loading is a design pattern wich allow to build complex expressions and he acts sometime like -proxy-. It is also a possible abstraction for integrating multiple different backend but it isnt a good idea at this moment for the framework SciSharp.

For example, the expression 
NDproxy d = a+2*b+np.log(c);
NDarray e = d;
will only be evaluated during the implicit type conversion.
I am a self taugh and i discovered the lazy loading just now to reduce computation. I have some intuition about some other usage of the Lazy Loading, but i cannot explain it.

### Comment 5 by @Oceania2018 (2019-08-01T19:35:35Z)

@aidevnn Is `NDProxy` like a kind of Expression/ Func/ Delegate ? It can be executed after all operations have been decided ?

### Comment 6 by @aidevnn (2019-08-01T19:56:43Z)

@Oceania2018  Yes it is a struct that can be defined like this during internal computation of the backend 
```
NDproxy a = new NDproxy(()=>ndArray0)
```

Its contains only one field / property : 
```
Func<NDarray> fnc
```
which can be called explicitly 
```
NDarray b = a.fnc() 
```
during internal computation of the backend 
or implicitly during type conversion by the frontend.

In my repo i used delegate + struct.
https://github.com/aidevnn/DesertLand/blob/master/NDarray/NDview.cs#L4

### Comment 7 by @Nucs (2019-08-01T20:04:54Z)

@aidevnn 
A. You can still use that branch from a fork
B. Thanks for laying it out for me, I am too a self taught.
Honestly - there is no fast way to stack lazy loading ops. Any basic implementation in C++ will surpass C#'s performance by far.
Your best luck is to put efforts in [Tensorflow.NET](https://github.com/SciSharp/TensorFlow.NET) since tensorflow holds a somewhat similar API to numpy and is capable of computation on a GPU and it provides a pretty impressive performance (both CPU and GPU).

if you do insist on writing it on your own then you better get started with IL generation that will link the math ops together.
Mainly because every math-op method you delegate to the next math-op method is adding more overhead than you would think.

`NumSharp` itself is not the fastest it can be. Any library that utilizes the performance of a low-level languages will be faster than our unmanaged algorithms.

### Comment 8 by @aidevnn (2019-08-01T20:11:27Z)

@Nucs  you are right! Lazy Loading adds some stacks operations because its a kind of managing expression pattern.
Managing code have advantage for simplifying coding and maintenance, but with loss of speed and performance. Its a tradeoff.

### Comment 9 by @aidevnn (2019-08-02T19:38:38Z)

@Nucs The idea behind the lazy loading is to introduce step by step some symbolic neural network algorithms with the numerical algorithms.
For beginning, a precompilation (or may be caching) of all requirement like shape / strides / slices can be done without recreating them each time and its my first objectif. I will let you being informed on my progress

### Comment 10 by @aidevnn (2019-08-03T02:50:11Z)

I forget to comment an important thing about some counters.
```
var a = ND.Uniform<double>(1, 10, 4, 4); // Counters. DataAccess:16 / MethCall:16
var b = 3 * ND.Sq(a - 1) - 4; // Counters. DataAccess:16 / MethCall:80
var c = 3 * ND.Sq(a - 1) - 4 * ND.Sqrt(a + 5); // Counters. DataAccess:32 / MethCall:144
```
In the above expression, my actual code and symbolic approch reduce rawdata access only to one time for expression b. Also in a more complexe expression, the lazy loading reduce the data access to the minimum effort for expression c. But i am trying to solve some performances problems.

We can say that a lazy loading pattern can improve the performance in theory but it introduces some other execution stack problems. I will continue to search how to take advantage of it.

### Comment 11 by @aidevnn (2019-09-25T17:58:40Z)

I discovered in this article 
https://techdecoded.intel.io/resources/parallelism-in-python-directing-vectorization-with-numexpr/#gs.63gvwl

Numexpr : it is very usefull and it already use lazy-loading and a lot of symbolics optimisations to enhance performance.
