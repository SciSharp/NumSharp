using System;
using NumSharp.Generic;
using NumSharp.Utilities;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Linq;
using System.Threading.Tasks;
using NumSharp.Backends;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test whether all array elements evaluate to True.
        /// </summary>
        /// <param name="nd"></param>
        /// <returns></returns>
        public override NDArray<int>[] NonZero(in NDArray nd)
        {
#if _REGEN
            #region Compute
		    switch (nd.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return nonzeros<#2>(nd.MakeGeneric<#2>());
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
		    switch (nd.typecode)
		    {
			    case NPTypeCode.Boolean: return nonzeros<bool>(nd.MakeGeneric<bool>());
			    case NPTypeCode.Byte: return nonzeros<byte>(nd.MakeGeneric<byte>());
			    case NPTypeCode.Int16: return nonzeros<short>(nd.MakeGeneric<short>());
			    case NPTypeCode.UInt16: return nonzeros<ushort>(nd.MakeGeneric<ushort>());
			    case NPTypeCode.Int32: return nonzeros<int>(nd.MakeGeneric<int>());
			    case NPTypeCode.UInt32: return nonzeros<uint>(nd.MakeGeneric<uint>());
			    case NPTypeCode.Int64: return nonzeros<long>(nd.MakeGeneric<long>());
			    case NPTypeCode.UInt64: return nonzeros<ulong>(nd.MakeGeneric<ulong>());
			    case NPTypeCode.Char: return nonzeros<char>(nd.MakeGeneric<char>());
			    case NPTypeCode.Double: return nonzeros<double>(nd.MakeGeneric<double>());
			    case NPTypeCode.Single: return nonzeros<float>(nd.MakeGeneric<float>());
			    case NPTypeCode.Decimal: return nonzeros<decimal>(nd.MakeGeneric<decimal>());
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }

        private static unsafe NDArray<int>[] nonzeros<T>(NDArray<T> x) where T : unmanaged
        {
            x = np.atleast_1d(x).MakeGeneric<T>();
            var nonzeroCoords = new List<int[]>(x.size / 3);
            var size = x.size;
            Debug.Assert(size > 0);
#if _REGEN
            #region Compute
            Func<int[], int> getOffset = x.Shape.GetOffset;
            switch (x.typecode) {
                %foreach supported_dtypes, supported_dtypes_lowercase%
                case NPTypeCode.#1: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (#2*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(#2)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                %
                default: throw new NotSupportedException();
            }
            #endregion
#else
            #region Compute
            Func<int[], int> getOffset = x.Shape.GetOffset;
            switch (x.typecode) {
                case NPTypeCode.Boolean: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (bool*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(bool)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.Byte: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (byte*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(byte)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.Int16: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (short*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(short)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.UInt16: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (ushort*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(ushort)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.Int32: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (int*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(int)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.UInt32: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (uint*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(uint)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.Int64: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (long*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(long)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.UInt64: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (ulong*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(ulong)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.Char: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (char*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(char)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.Double: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (double*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(double)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.Single: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (float*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(float)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                case NPTypeCode.Decimal: {
                    var incr = new ValueCoordinatesIncrementor(x.shape);
                    var coords = incr.Index;
                    var src = (decimal*)x.Address;
                    int offset;
                    do
                    {
                        offset = getOffset(coords);
                        if (!(src[offset] == default(decimal)))
                            nonzeroCoords.Add(coords.CloneArray());
                    } while (incr.Next() != null);

                    break;
                }
                default: throw new NotSupportedException();
            }
            #endregion
#endif

            var len = nonzeroCoords.Count;
            var ndim = x.ndim;
            //create ndarray for each dimension
            var ret = new NDArray<int>[ndim];
            for (int i = 0; i < x.ndim; i++)
                ret[i] = new NDArray<int>(len);

            //create address for each dimension
            var addresses = new int*[ndim];
            for (int i = 0; i < ndim; i++)
                addresses[i] = (int*)ret[i].Address;

            //extract coordinates
            for (int i = 0; i < len; i++)
            {
                var coords = nonzeroCoords[i];
                for (int axis = 0; axis < ndim; axis++)
                {
                    addresses[axis][i] = coords[axis];
                }
            };

            return ret;
        }
        
    }
}
