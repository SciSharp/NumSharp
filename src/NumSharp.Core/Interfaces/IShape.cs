using NumSharp;
using System;
using System.Collections.Generic;

namespace NumSharp.Interfaces
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
        void ChangeTensorLayout(int order);
        void ReShape(params int[] dims);
    }
}