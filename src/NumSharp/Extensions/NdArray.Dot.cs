using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp.Extensions
{
    internal enum MultiplicationCase 
    {
        ArrayArray,
        MatrixArray,
        MatrixMatrix
    }
    public static partial class NDArrayExtensions
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="np1"></param>
        /// <param name="np2"></param>
        /// <typeparam name="TData"></typeparam>
        /// <returns></returns>
        public static NDArray<TData> Dot<TData>(this NDArray<TData> np1, NDArray<TData> np2)
        {
            NDArray<TData> prod = new NDArray<TData>();

            // Shape 
            var prodShape = new Shape(0);
            var multipliationCase = ShapeDetermination(np1.Shape,np2.Shape,ref prodShape);
            
            prod.Shape = prodShape;
            int dataLength = (prodShape.Length == 1) ? ( prodShape.Shapes[0] ) : (prodShape.Shapes[0] * prodShape.Shapes[1]);
            prod.Data = new TData[dataLength];

            switch (np1.Data)
            {
                case double[] np1Array : 
                {
                    double[] result = prod.Data as double[];
                    double[] np2Array = np2.Data as double[];

                    if (multipliationCase == MultiplicationCase.ArrayArray)
                    {    
                        for (int idx = 0; idx < np1Array.Length;idx++)
                            result[0] += np1Array[idx] * np2Array[idx]; 
                    }
                    else 
                    {
                        for (int idx = 0; idx < prod.Data.Length;idx++)
                            for (int kdx = 0; kdx < np1.Shape.Shapes[1];kdx++)
                                result[idx] += np1Array[(idx / np1.Shape.Shapes[0]) * np1.Shape.Shapes[1] + kdx] * np2Array[np2.Shape.Shapes[1] * kdx + (idx % np2.Shape.Shapes[1])];    
                    }
                    break;
                }
                case float[] np1Array : 
                {
                    float[] result = prod.Data as float[];
                    float[] np2Array = np2.Data as float[];

                    if (multipliationCase == MultiplicationCase.ArrayArray)
                    {    
                        for (int idx = 0; idx < np1Array.Length;idx++)
                            result[0] += np1Array[idx] * np2Array[idx]; 
                    }
                    else 
                    {
                        for (int idx = 0; idx < prod.Data.Length;idx++)
                            for (int kdx = 0; kdx < np1.Shape.Shapes[1];kdx++)
                                result[idx] += np1Array[(idx / np1.Shape.Shapes[0]) * np1.Shape.Shapes[1] + kdx] * np2Array[np2.Shape.Shapes[1] * kdx + (idx % np2.Shape.Shapes[1])];    
                    }
                    break;
                }
                case Complex[] np1Array : 
                {
                    Complex[] result = prod.Data as Complex[];
                    Complex[] np2Array = np2.Data as Complex[];

                    if (multipliationCase == MultiplicationCase.ArrayArray)
                    {    
                        for (int idx = 0; idx < np1Array.Length;idx++)
                            result[0] += np1Array[idx] * np2Array[idx]; 
                    }
                    else 
                    {
                        for (int idx = 0; idx < prod.Data.Length;idx++)
                            for (int kdx = 0; kdx < np1.Shape.Shapes[1];kdx++)
                                result[idx] += np1Array[(idx / np1.Shape.Shapes[0]) * np1.Shape.Shapes[1] + kdx] * np2Array[np2.Shape.Shapes[1] * kdx + (idx % np2.Shape.Shapes[1])];    
                    }
                    break;
                }
                case Quaternion[] np1Array : 
                {
                    Quaternion[] result = prod.Data as Quaternion[];
                    Quaternion[] np2Array = np2.Data as Quaternion[];

                    if (multipliationCase == MultiplicationCase.ArrayArray)
                    {    
                        for (int idx = 0; idx < np1Array.Length;idx++)
                            result[0] += np1Array[idx] * np2Array[idx]; 
                    }
                    else 
                    {
                        for (int idx = 0; idx < prod.Data.Length;idx++)
                            for (int kdx = 0; kdx < np1.Shape.Shapes[1];kdx++)
                                result[idx] += np1Array[(idx / np1.Shape.Shapes[0]) * np1.Shape.Shapes[1] + kdx] * np2Array[np2.Shape.Shapes[1] * kdx + (idx % np2.Shape.Shapes[1])];    
                    }
                    break;
                }
                default : 
                {
                    throw new Exception("The Dot method is not implemented for the "  + typeof(TData).Name);
                }
            }

            return ((NDArray<TData>) prod);
        }
        
        internal static NumSharp.Extensions.MultiplicationCase ShapeDetermination(Shape shape1, Shape shape2, ref Shape shapeResult)
        {
            NumSharp.Extensions.MultiplicationCase returnValue = 0;
            
            if ((shape1.Length == 1 ) & (shape2.Length == 1))
                if (shape1.Shapes[0] != shape2.Shapes[0])
                    throw new Exception("The Dot method does not work with this shape or was not already implemented."); 
                else {}
            else
                if (shape1.Shapes[1] != shape2.Shapes[0])
                    throw new Exception("The Dot method does not work with this shape or was not already implemented.");
                

            // matrix multiplication
            if( (shape1.Length == 2 ) && (shape2.Length == 2 )  )
            {
                shapeResult = new Shape(shape1.Shapes[0],shape2.Shapes[1]);
                returnValue = MultiplicationCase.MatrixMatrix;
            }
            // matrix array multiplication
            else if ((shape1.Length == 2) && (shape2.Length == 1))
            {
                shapeResult = new Shape(shape1.Shapes[0]);
                returnValue = MultiplicationCase.MatrixArray;
            }
            // scalar product 
            else if ((shape1.Length == 1) && (shape2.Length == 1))
            {
                shapeResult = new Shape(1);
                returnValue = MultiplicationCase.ArrayArray;
            }
            else 
            {
                throw new Exception("The Dot method does not work with this shape or was not already implemented.");
            }
            return returnValue;
        }
    }
}
