# #483: How to convert List<NDArray> to NDArray

- **URL:** https://github.com/SciSharp/NumSharp/issues/483
- **State:** OPEN
- **Author:** @williamlzw
- **Created:** 2022-12-13T14:12:14Z
- **Updated:** 2022-12-13T14:13:02Z

## Description

str = '00000.jpg 130,83,205,108,0 130,137,154,161,1 255,137,279,160,2 125,186,177,208,3 210,186,236,208,4 285,186,311,208,5 130,237,400,292,6 230,328,555,354,7'
line = str.split()
box = np.array([np.array(list(map(int, box.split(','))))for box in line[1:]])
print(box)

string str = "00000.jpg 130,83,205,108,0 130,137,154,161,1 255,137,279,160,2 125,186,177,208,3 210,186,236,208,4 285,186,311,208,5 130,237,400,292,6 230,328,555,354,7";
var line = str.Split();
List<List<int>> allList = new List<List<int>>();
var boxarr = line.Skip(1).Select(box => np.array(box.Split(',').Select(int.Parse))).ToList();
var aa = boxarr.ToArray();//How to convert List\<NDArray\> to NDArray


