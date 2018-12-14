using NumSharp;
using System;
using System.Collections.Generic;

namespace NumSharp.Core.Interfaces
{
    public interface IShape
    {
        int Size {get;}
        int NDim {get;}
        int TensorOrder {get;}
        IReadOnlyList<int> DimOffset {get;}
        IReadOnlyList<int> Shapes {get;}
        int ReShape(params int[] dimensions);
        int GetIndexInShape(params int[] select); 
        int[] GetDimIndexOutShape(int select);
        int UniShape {get;}
        (int, int) BiShape {get;}
        (int, int, int) TriShape {get;}
        bool ChangeTensorOrder(int order);
    }
}