# #520: Can't convert to Vector3

- **URL:** https://github.com/SciSharp/NumSharp/issues/520
- **State:** OPEN
- **Author:** @xiaoshux
- **Created:** 2025-03-12T06:09:51Z
- **Updated:** 2025-12-16T13:45:12Z

## Description

In Unity, this bug trapped me for 5 days—I couldn't pass values into Vector3. I don't recommend using NumSharp!

![Image](https://github.com/user-attachments/assets/1e1ee948-510f-4373-b02d-a8b7bc0bef4f)

![Image](https://github.com/user-attachments/assets/35dfd893-3a91-4496-abda-9ce46aa10268)

## Comments

### Comment 1 by @zhuoshui-AI (2025-12-16T13:45:12Z)

Bcausse in this a[1,1,1] return tpye is NDarray of one element.Not a scalar. try to use .getInt32() np.asscalar()to get a pure number
