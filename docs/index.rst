.. NumSharp documentation master file, created by
   sphinx-quickstart on Thu Oct 11 20:29:54 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome Seeker to NumSharp's documentation!
===========================================

Welcome Seeker to NumSharp. 

Since you reached our documentation, you must be interested into .NET or into machine learning. 
No matter what you was searching, you will be satisfied. 

NumSharp is quite new open source project with one big goal: adapt the famous and well-known python library numpy and bring it into .NET world. 
On top of this, we try to bring the whole Scipy Stack to .NET world. 
Sounds crazy? 
Yes maybe a little bit. As crazy as bring .NET on mobile phones (Xamarin) or into browser (Blazor). ;)   

Why we should do that?

- Because we can do (hey we are .NET developer we can do cloud computing, do fancy web side stuff, are No. 1 for GUI programming in windows area, experiment with WASM – a.k.a. the Blazor project, do mobile things with Xamarin – yes, so how hard it can be to improve our numeric stack?)  
- Because Microsoft also try to let .NET framework become an important framework for machine learning area. Therefore, it is time for us to support the F# developers who always were the leading experts in numeric area and machine learning. Let us help to shape .NET for numeric area. 
- Because we can reach more languages than any other framework can do. It is not just a project for C#. We are not just C# developers – we are .NET developers and we are like a big family. Code one time and give it to all other languages. Because of this we are not just interested into C# - we also want to deliver packages special for F#, Powershell, VB.NET, Ironpython, PHP, … we want to write the core in C# but we want to give each language the sugar it deserves. 

What does Numsharp make different than the other numeric frameworks?

- We deliver a new class of array, which has a good performance in all different situations. The NDArray follows the idea of numpy and Quantstack, which store all elements of a multidimensional array (independent of the number of dimensions) into one large array. The indexing depends totally on the Shape of the NDArray, which determines if it is a matrix, a tensor, a vector or something else.
- We try to implement the numpy APIs as well as possible so that people who come from numpy feels like be at home. More over people can find easier tutorials if they are new to machine learning or numerical stuff in general.    

So I hope seeker you got a quite well impression of what this is all about.

Curious? 
- dotnet add package NumSharp

Want new features?
- 

Want to support? 
- fork us on Github ;) 

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================
The main documentation for the site is organized into a couple sections:

:ref:`User Documentation <user-docs>`
:ref:`Developer's Documentation <developer-docs>`

.. _user-docs:

.. toctree::
   :maxdepth: 3
   :caption: User Documentation:
   
   user-docs/overview
   user-docs/NDArray.Creation

.. _developer-docs:
   :maxdepth: 3
   :caption: Developer's Documentation:
   
   developer-docs/overview

* :ref:`search`
