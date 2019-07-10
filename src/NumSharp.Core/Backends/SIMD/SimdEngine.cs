using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace NumSharp.Backends
{
    public class SimdEngine : DefaultEngine
    {
        public override NDArray Add(NDArray x, NDArray y)
        {
            return null;
            //return base.Add(x, y);
            //if (x.ndim == y.ndim && x.ndim == 1)
            //{
            //    switch (Type.GetTypeCode(x.dtype))
            //    {
            //        case TypeCode.Int32:
            //            {
            //                var lhs = x.Data<int>();
            //                var rhs = y.Data<int>();

            //                var simdLength = Vector<int>.Count;
            //                var result = new int[lhs.Length];
            //                var i = 0;
            //                for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            //                {
            //                    var va = new Vector<int>(lhs, i);
            //                    var vb = new Vector<int>(rhs, i);

            //                    (va + vb).CopyTo(result, i);
            //                }

            //                for (; i < lhs.Length; ++i)
            //                    result[i] = lhs[i] + rhs[i];

            //                return result;
            //            }

            //        case TypeCode.Single:
            //            {
            //                var lhs = x.Data<float>();
            //                var rhs = y.Data<float>();

            //                var simdLength = Vector<float>.Count;
            //                var result = new float[lhs.Length];
            //                var i = 0;
            //                for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            //                {
            //                    var va = new Vector<float>(lhs, i);
            //                    var vb = new Vector<float>(rhs, i);
            //                    (va + vb).CopyTo(result, i);
            //                }

            //                for (; i < lhs.Length; ++i)
            //                    result[i] = lhs[i] + rhs[i];

            //                return result;
            //            }

            //        case TypeCode.Double:
            //            {
            //                var lhs = x.Data<double>();
            //                var rhs = y.Data<double>();

            //                var simdLength = Vector<double>.Count;
            //                var result = new double[lhs.Length];
            //                var i = 0;
            //                for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            //                {
            //                    var va = new Vector<double>(lhs, i);
            //                    var vb = new Vector<double>(rhs, i);
            //                    (va + vb).CopyTo(result, i);
            //                }

            //                for (; i < lhs.Length; ++i)
            //                    result[i] = lhs[i] + rhs[i];

            //                return result;
            //            }

            //        default:
            //            throw new NotImplementedException($"SIMD Add {x.dtype.Name} {y.dtype.Name}");
            //    }
            //}
            //else
            //{
            //    return base.Add(x, y);
            //}
        }

        public override NDArray Dot(NDArray x, NDArray y)
        {
            return null;
            //return base.Dot(x, y);
            //var dtype = x.dtype;

            //switch (dtype.Name)
            //{
            //    case "Int32":
            //        if(x.ndim == 2 && y.ndim == 2)
            //        {
            //            var vx = new Vector<int>(x.Data<int>());
            //            var vy = new Vector<int>(y.Data<int>());
            //            var vec = Vector.Dot(vx, vy);
            //        }

            //        break;
            //}

            //throw new NotImplementedException("SimdEngine.dot");
        }
    }
}
