using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Shared;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<TData> Dot<TData>(this NDArray<TData> np1, NDArray<TData> np2)
        {
            dynamic prod = new NDArray<TData>();
            
            dynamic np1Dyn = np1.Data.ToArray();
            dynamic np2Dyn = np2.Data.ToArray();

            var dataType = typeof(TData);

            int dimensionSum = np1.Shape.Length + np2.Shape.Length;

            switch (dimensionSum)
            {
                case 2 : 
                {
                    prod.Shape = new Shape(new int[] {1});

                    switch (dataType.Name)
                    {
                        case ("Double") : prod.Data = ScalarProduct1D.MuliplyScalarProd1DDouble((double[])np1Dyn,(double[])np2Dyn); break;
                        case ("Float") : prod.Data = ScalarProduct1D.MuliplyScalarProd1Dfloat((float[])np1Dyn,(float[])np2Dyn); break;
                        //case ("Complex") : prod.Data = ScalarProduct1D.Mult((float[])np1Dyn,(float[])np2Dyn); break;
                        //case ("Quaternion") : prod.Data = ScalarProduct1D.MuliplyScalarProd1Dfloat((float[])np1Dyn,(float[])np2Dyn); break;
                    }
                    break;
                }
                case 3 : 
                {
                    int[] dim0 = np1.Shape.Shapes;
                    int[] dim1 = np2.Shape.Shapes;
                    int iterator = np1.Shape.Shapes[1];

                    prod.Shape = new Shape(new int[] {dim0[0],dim1[1]});
                    
                    switch (dataType.Name)
                    {
                        case ("Double"): prod.Data = MatrixMultiplication.MatrixMultiplyDoubleMatrix((double[])np1Dyn,(double[])np2Dyn,dim0,dim1); break;
                        case ("Float"): prod.Data = MatrixMultiplication.MatrixMultiplyfloatMatrix((float[])np1Dyn,(float[])np2Dyn,dim0,dim1); break;
                        case ("Complex"): prod.Data = MatrixMultiplication.MatrixMultiplyComplexMatrix((Complex[])np1Dyn,(Complex[])np2Dyn,dim0,dim1); break;
                        case ("Quaternion"): prod.Data = MatrixMultiplication.MatrixMultiplyQuaternionMatrix((Quaternion[])np1Dyn,(Quaternion[]) np2Dyn,dim0,dim1) ; break;
                    }
                    break;
                }
                case 4 : 
                {
                    int[] dim0 = np1.Shape.Shapes;
                    int[] dim1 = np2.Shape.Shapes;
                    int iterator = np1.Shape.Shapes[1];

                    prod.Shape = new Shape(new int[] {dim0[0],dim1[1]});
                    
                    switch (dataType.Name)
                    {
                        case ("Double"): prod.Data = MatrixMultiplication.MatrixMultiplyDoubleMatrix((double[])np1Dyn,(double[])np2Dyn,dim0,dim1); break;
                        case ("Float"): prod.Data = MatrixMultiplication.MatrixMultiplyfloatMatrix((float[])np1Dyn,(float[])np2Dyn,dim0,dim1); break;
                        case ("Complex"): prod.Data = MatrixMultiplication.MatrixMultiplyComplexMatrix((Complex[])np1Dyn,(Complex[])np2Dyn,dim0,dim1); break;
                        case ("Quaternion"): prod.Data = MatrixMultiplication.MatrixMultiplyQuaternionMatrix((Quaternion[])np1Dyn,(Quaternion[]) np2Dyn,dim0,dim1) ; break;
                    }
                    break;
                }
            }

            return ((NDArray<TData>) prod);
        }
        
    }
}
