# #461: np.save incorrectly saves System.Byte arrays as signed

- **URL:** https://github.com/SciSharp/NumSharp/issues/461
- **State:** OPEN
- **Author:** @rikkitook
- **Created:** 2021-08-04T12:06:31Z
- **Updated:** 2023-01-23T09:04:19Z

## Description

in function GetDtypeFromType
...
if (type == typeof(Byte))
                return "|i1";
...
i1 gets translated to signed integer 
https://numpy.org/doc/stable/user/basics.types.html

## Comments

### Comment 1 by @Cle-O (2023-01-23T09:03:24Z)

Bump. This issue is still existing.
I am saving a numpy array (an image) with positive values only using _np.save()_.
Positive values only are verified calling _amin()_ on the array.
After reading the file again, the array contains negative values. 
Is there a workaround for this?
