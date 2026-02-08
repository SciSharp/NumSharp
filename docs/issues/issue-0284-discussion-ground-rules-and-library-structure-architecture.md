# #284: [Discussion] Ground Rules and Library Structure/Architecture

- **URL:** https://github.com/SciSharp/NumSharp/issues/284
- **State:** OPEN
- **Author:** @Nucs
- **Created:** 2019-06-14T15:33:31Z
- **Updated:** 2021-06-29T07:16:05Z
- **Labels:** further discuss

## Description

I couldn't but notice the inconsistencies around the library, There are functions that returns copies while some don't while some are not even complete.
The strong-typing of C# does surely makes it harder to get stuff done so `.tt` generators does a nice job helping with that.

I would like to discuss the following:
1. What are the supported C# primitive types compared to numpy? 
Looking at the implicit conversions of `NDArray` and other math operations such as `np.sum`,
I found that the type support is inconsistent.
note: Numpy supports both signed and unsigned types. [see this](https://www.numpy.org/devdocs/user/basics.types.html).

    ![image](https://user-images.githubusercontent.com/649919/59519128-ddad3000-8ecf-11e9-9091-552d86184504.png)

2. Do we support `Complex`? 
because aside of the basic math operations, it is not supported anywhere.

3. To what rank/ndim the algorithms in C# support?
I saw some functions that support 2-3 and some up to 6.
[Numpy limitation is 32 dimensions.](https://github.com/numpy/numpy/issues/5744)

4. NDArray mutable vs immutable
 I think it should be up to the user to decide via a bool parameter.
   * Arithmetic operators should always return a copy
   * Following numpy's lead is the priority ofcouse.

5. Seperation between `np`, `NDArray` and `backend engine`
There is a lot of mixup between whats computed where. 
The reality is that they all redirect calls to each other.
In a perfect scenario: 
we would refer all operations low-level operations to backend engine to perform the wanted task. 
Having a fallback to each method that will compute in C#.
`NDArray` is only responsible for casting, initialization, storing data and serialization
And `np` is the high-level zone that uses backend.

6. The incomplete methods should be logged somewhere, preferably in Issues or project here.
Heres just a few examples from my findings:
    * `np.transpose` supports only to rank 2 and theres a `//todo` in the unit test that fails
    * `np.flatten` wasn't implemented completely but was shipped on release
    * `np.sum` doesn't support all data types.
We should start by creating issues for those problems in code and we'll get on from there

7. Means of communication between the developing team via instant messaging?
Edit: Did not see the gitter tag on readme.md

Those issues are important to address, simply because this library is and will be the backend of many high-level applications.
That's about it, Best regards.

## Comments

### Comment 1 by @henon (2019-06-14T17:00:28Z)

Hi Nucs,
you are right, there are many inconsistencies due to the fact that people just added what they needed and left the rest undone. We should define stricter rules and postulate them in the readme.

I think the most important guidance principle must be compatibility with the original NumPy. Only that way we can guarantee that porting Python scripts using numpy will work in NumSharp also. In this regard there is a lot of work to be done because until now it wasn't enforced very strongly.

I agree that some kind of management of what we have and what we are missing would be nice. Creating issues is of course a good idea. I have another one: create as many test cases as possible which will reflect what already works and what not. I have recently ported a lot of tests from the Numpy documentation for NumSharp's sister project [Numpy.NET](https://github.com/SciSharp/Numpy.NET) which is a auto-generated wrapper around numpy. While doing that I also generated over 600 test cases which could be repurposed for testing NumSharp. In addition to that, we can generate further test cases that test both Numpy.NET and NumSharp against all the things you mentioned:
* number of supported ranks
* supported data types
* support for all backend engines
etc. 

A complete test suite is the ultimate todo list and helps keeping NumSharp in sync with numpy.

PS: I could help generating more test cases from the Numpy.NET project. 

### Comment 2 by @henon (2019-06-14T17:25:44Z)

ad 7: you can reach us on Gitter for quick questions, small talk, etc, but of course important stuff that should be accessible to all (like this) is of course best discussed in issues. I am sure @Oceania2018 will have answers for the other questions.

### Comment 3 by @Oceania2018 (2019-06-16T00:02:33Z)

ad 1: Ideally, support all primitive data type.
ad 2: Not support, but can add easily.
ad 3: In theory, no limitation.
ad 4: Following numpy's lead.
ad 5: Should do the calculation in `engine`, move current code into `engine` gradually.
ad 6: Agree.
ad 7: https://gitter.im/sci-sharp/community

### Comment 4 by @Nucs (2019-08-06T12:47:33Z)

All of these suggestions were implemented in the [new release](https://github.com/SciSharp/NumSharp/pull/336) of NumSharp 0.11.0-alpha2 and available on [nuget](https://www.nuget.org/packages/NumSharp/0.11.0-alpha2).

### Comment 5 by @dcuccia (2021-06-29T00:51:14Z)

ad 2: Would love to have Complex support. Just got to the end of a port and realized this was a non-starter.
