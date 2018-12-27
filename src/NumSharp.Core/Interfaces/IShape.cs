using NumSharp;
using System;
using System.Collections.Generic;

namespace NumSharp.Core.Interfaces
{
    public interface IShape
    {
        int Size {get;}
        int NDim {get;}
        int TensorLayout {get;}
        int[] DimOffset {get;}
        int[] Dimensions {get;}
        int GetIndexInShape(params int[] select); 
        int[] GetDimIndexOutShape(int select);
        int UniShape {get;}
        (int, int) BiShape {get;}
        (int, int, int) TriShape {get;}
        void ChangeTensorLayout(int order);
        void ReShape(params int[] dims);
    }
}