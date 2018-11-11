import clr 
from System.IO import Path
from System import Array
from System import Double as double 
import sys

numSharpFolder = Path.GetDirectoryName(__file__)
sys.path.append(numSharpFolder)
clr.AddReferenceToFile('NumSharp.dll')
import NumSharp as ns

def HelloWorld():

    print ("Hello World from"  )

def array(listArg):

    dotNetArray = Array.CreateInstance(double,len(listArg))

    for idx in range(0,len(listArg)):
        dotNetArray[idx] = listArg[idx]

    numSharpArray = ns.NDArray[double]()

    numSharpArray.Data = dotNetArray
    numSharpArray.Shape = ns.Shape(dotNetArray.Length)

    return numSharpArray