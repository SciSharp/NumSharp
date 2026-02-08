# #368: Masking a slice ("...") returns null

- **URL:** https://github.com/SciSharp/NumSharp/issues/368
- **State:** OPEN
- **Author:** @ohjerm
- **Created:** 2019-11-12T09:48:54Z
- **Updated:** 2019-11-12T14:56:29Z
- **Assignees:** @henon, @Nucs

## Description

```
NDArray positive_boxes = batch_item["...,0"] != 0;
Debug.Log(positive_boxes);
```
The output is Null. In numpy I would expect an boolean mask NDArray of shape (...,1) here, right? Is masking implemented another way? 

I'm on the latest nuget release (0.20.4) if that helps.

## Comments

### Comment 1 by @henon (2019-11-12T14:16:57Z)

@ChaiKnight: I can not work with the example you have given here to reproduce the problem. Can you please write me a piece of code that is complete and reproduces the bug?

Like this:

```C#
var x=np.arange(10).reshape(2,5);
var y=x["...,0"]; 
// what does it give and what do you expect instead?
```
I don't know your setup, the above is just an example. Thanks.

### Comment 2 by @ohjerm (2019-11-12T14:42:55Z)

I don't know how much more I can give you without just dumping my entire codebase on you. My batch_item is an NDArray of size (7160,6). When I test I currently just feed it zeroes, so it basically looks like np.zero((7160,6)) to me when I print it.

Your example seems to work just fine, I am able to print out [0,5], but I am also able to print out a long array of zeroes when I do `Debug.Log(batch_item["...,0"]);`, so somewhere in the masking process is where it goes wrong. I figured the masking should return an empty array like numpy does, of size (0,6). Is that at all possible?



### Comment 3 by @henon (2019-11-12T14:55:51Z)

Alright, I understand. This is a known problem then. Masking is currently under construction. @Nucs is working on it. 
