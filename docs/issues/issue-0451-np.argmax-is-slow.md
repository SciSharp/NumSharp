# #451: np.argmax is slow

- **URL:** https://github.com/SciSharp/NumSharp/issues/451
- **State:** OPEN
- **Author:** @feiyuhuahuo
- **Created:** 2021-06-18T01:46:57Z
- **Updated:** 2021-06-18T01:46:57Z

## Description

![image](https://user-images.githubusercontent.com/32631344/122493133-bb4bd100-d019-11eb-9a89-728c4465d4e0.png)
`nd` is an array with shape of (1, 13, 512, 512), the time consumption of doing argmax on it is about 500ms. That's really slow. Any way to improve it?
