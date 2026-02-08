# #422: Index of element with a condiction

- **URL:** https://github.com/SciSharp/NumSharp/issues/422
- **State:** OPEN
- **Author:** @EnricoBeltramo
- **Created:** 2020-09-11T05:45:48Z
- **Updated:** 2020-09-11T05:45:48Z

## Description

I'm implementing a filter on array. In python it works well, but if I try to replicate in numsharp, doesn't:

var threshold = 0.01
var filter1 = XYZ1[2, Slice.All] < -threshold;
var XYZ_1_filter = XYZ1[Slice.All, filter1];
var XYZ_2_filter = XYZ2[Slice.All, filter1];

In particular way, filter1  return null. 
How can I do? In general, how i can find indexes that respect a particular condiction (i.e. greater than, minor than) in numsharp?
