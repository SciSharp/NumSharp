# #510: How to save a nested dictionary with `save_npz`

- **URL:** https://github.com/SciSharp/NumSharp/issues/510
- **State:** OPEN
- **Author:** @rqx110
- **Created:** 2024-03-07T01:20:45Z
- **Updated:** 2024-03-07T01:20:45Z

## Description

If i have a data struct like:
```json
{
   "port1": {
                 "time": ["xxxx", "yyyy"],
                 "data": [0.0, 0.0]
                },
   "port2": {
                 "time": ["xxxx", "yyyy"],
                 "data": [0.0, 0.0]
                },
}
```
how to save it with `save_npz`?
can you give same sample codes? Thanks!
