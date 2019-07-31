import math
import numpy as np

# The asCode2D function generates NDArray declarations in C# for use in unit testing.
# This avoids some of the tedium and errors of hand-generation.
# For example, calling the function like this generates C# static variables named
# 'a53' and 'b53' from numpy's mgrid:
#   aa, bb = np.mgrid[0:5, 0:3]
#   cSharp.asCode2D("a53", aa)
#   cSharp.asCode2D("b53", bb)


class cSharp:
    def asCode2D(varName, v):
        if v.dtype.name == "int32":
            vType = "Int32"
        elif v.dtype.name == "float64":
            vType = "double"
        else:
            vType = "unknown"
        print("        static NDArray {0} = new NDArray(new {1}[] {{".format(varName, vType))
        valstr = ""
        commasToPrint = v.shape[0] * v.shape[1] - 1
        for i, row in enumerate(v):
            rowStr = "            "
            for j, item in enumerate(row):
                rowStr = rowStr + "{}".format(item)
                if commasToPrint > 0:
                    rowStr = rowStr + ", "
                commasToPrint -= 1
                #if (i < v)
            print(rowStr)
        print("            }}, new Shape(new int[] {{ {}, {} }}));".format(v.shape[0], v.shape[1]))
        print("")

    def asCode1D(varName, v):
        if v.dtype.name == "int32":
            vType = "Int32"
        elif v.dtype.name == "float64":
            vType = "double"
        else:
            vType = "unknown"
        print("        static NDArray {0} = new NDArray(new {1}[] {{".format(varName, vType))
        rowStr = "            "
        commasToPrint = v.shape[0] - 1
        for j, item in enumerate(v):
            rowStr = rowStr + "{}".format(item)
            if commasToPrint > 0:
                rowStr = rowStr + ", "
            commasToPrint -= 1
        print(rowStr)
        print("            }}, new Shape(new int[] {{ {} }}));".format(v.shape[0]))
        print("")
