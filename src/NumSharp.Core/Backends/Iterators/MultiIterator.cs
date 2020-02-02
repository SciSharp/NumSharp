using System;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public static class MultiIterator
    {
        /// <summary>
        ///     Assigns rhs values to lhs.
        /// </summary>
        /// <remarks>Stops at first iterator stop.</remarks>
        public static void Assign(NDArray lhs, NDArray rhs)
        {
            Assign(lhs.Storage, rhs.Storage);
        }
        
        /// <summary>
        ///     Assigns rhs values to lhs.
        /// </summary>
        /// <remarks>Stops at first iterator stop.</remarks>
        public static void Assign(UnmanagedStorage lhs, UnmanagedStorage rhs)
        {
#if _REGEN1
            #region Compute
		    switch (lhs.TypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1:
			    {
                    var (l, r)= GetIterators<#2>(lhs, rhs, true);
                    AssignBroadcast<#2>(l, r);
                    break;
			    }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
		    switch (lhs.TypeCode)
		    {
			    case NPTypeCode.Boolean:
			    {
                    var (l, r)= GetIterators<bool>(lhs, rhs, true);
                    AssignBroadcast<bool>(l, r);
                    break;
			    }
			    case NPTypeCode.Byte:
			    {
                    var (l, r)= GetIterators<byte>(lhs, rhs, true);
                    AssignBroadcast<byte>(l, r);
                    break;
			    }
			    case NPTypeCode.Int32:
			    {
                    var (l, r)= GetIterators<int>(lhs, rhs, true);
                    AssignBroadcast<int>(l, r);
                    break;
			    }
			    case NPTypeCode.Int64:
			    {
                    var (l, r)= GetIterators<long>(lhs, rhs, true);
                    AssignBroadcast<long>(l, r);
                    break;
			    }
			    case NPTypeCode.Single:
			    {
                    var (l, r)= GetIterators<float>(lhs, rhs, true);
                    AssignBroadcast<float>(l, r);
                    break;
			    }
			    case NPTypeCode.Double:
			    {
                    var (l, r)= GetIterators<double>(lhs, rhs, true);
                    AssignBroadcast<double>(l, r);
                    break;
			    }
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }

        /// <summary>
        ///     Assigns rhs values to lhs.
        /// </summary>
        /// <remarks>Stops at first iterator stop.</remarks>
        public static void AssignBroadcast<T>(NDIterator lhs, NDIterator rhs) where T : unmanaged
        {
            if (!lhs.BroadcastedShape.HasValue || !rhs.BroadcastedShape.HasValue)
                throw new InvalidOperationException("MultiIterator can only accept broadcasted shapes.");

            var len = lhs.BroadcastedShape.Value.size;

            var Rhs_MoveNext = rhs.MoveNext<T>();
            var Lhs_MoveNextReference = lhs.MoveNextReference<T>();

            for (int i = 0; i < len; i++) 
                Lhs_MoveNextReference() = Rhs_MoveNext();
        }

        /// <summary>
        ///     Gets the iterators of <paramref name="lhs"/> and <paramref name="rhs"/>.
        /// </summary>
        /// <param name="broadcast"></param>
        public static (NDIterator, NDIterator) GetIterators(UnmanagedStorage lhs, UnmanagedStorage rhs, bool broadcast) 
        {
            if (broadcast)
            {
                var (leftShape, rightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);

#if _REGEN1
                #region Compute
		        switch (lhs.TypeCode)
		        {
			        %foreach supported_dtypes,supported_dtypes_lowercase%
			        case NPTypeCode.#1: return (new NDIterator<#2>(lhs.InternalArray, lhs.Shape, leftShape, false), new NDIterator<#2>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        %
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#else

                #region Compute
		        switch (lhs.TypeCode)
		        {
			        case NPTypeCode.Boolean: return (new NDIterator<bool>(lhs.InternalArray, lhs.Shape, leftShape, false), new NDIterator<bool>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        case NPTypeCode.Byte: return (new NDIterator<byte>(lhs.InternalArray, lhs.Shape, leftShape, false), new NDIterator<byte>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        case NPTypeCode.Int32: return (new NDIterator<int>(lhs.InternalArray, lhs.Shape, leftShape, false), new NDIterator<int>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        case NPTypeCode.Int64: return (new NDIterator<long>(lhs.InternalArray, lhs.Shape, leftShape, false), new NDIterator<long>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        case NPTypeCode.Single: return (new NDIterator<float>(lhs.InternalArray, lhs.Shape, leftShape, false), new NDIterator<float>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        case NPTypeCode.Double: return (new NDIterator<double>(lhs.InternalArray, lhs.Shape, leftShape, false), new NDIterator<double>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#endif
            }
            else
            {
#if _REGEN1
                #region Compute
		        switch (lhs.TypeCode)
		        {
			        %foreach supported_dtypes,supported_dtypes_lowercase%
			        case NPTypeCode.#1: return (new NDIterator<#2>(lhs, false), new NDIterator<#2>(false));
			        %
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#else

                #region Compute
		        switch (lhs.TypeCode)
		        {
			        case NPTypeCode.Boolean: return (new NDIterator<bool>(lhs, false), new NDIterator<bool>(false));
			        case NPTypeCode.Byte: return (new NDIterator<byte>(lhs, false), new NDIterator<byte>(false));
			        case NPTypeCode.Int32: return (new NDIterator<int>(lhs, false), new NDIterator<int>(false));
			        case NPTypeCode.Int64: return (new NDIterator<long>(lhs, false), new NDIterator<long>(false));
			        case NPTypeCode.Single: return (new NDIterator<float>(lhs, false), new NDIterator<float>(false));
			        case NPTypeCode.Double: return (new NDIterator<double>(lhs, false), new NDIterator<double>(false));
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#endif
            }
        }


        /// <summary>
        ///     Assigns rhs values to lhs.
        /// </summary>
        public static (NDIterator<TOut>, NDIterator<TOut>) GetIterators<TOut>(UnmanagedStorage lhs, UnmanagedStorage rhs, bool broadcast) where TOut : unmanaged
        {
            if (broadcast)
            {
                var (leftShape, rightShape) = lhs.Shape == rhs.Shape ? (lhs.Shape, rhs.Shape) : DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);

#if _REGEN1
                #region Compute
		        switch (InfoOf<TOut>.NPTypeCode)
		        {
			        %foreach supported_dtypes,supported_dtypes_lowercase%
			        case NPTypeCode.#1: return ((NDIterator<TOut>)(object)new NDIterator<#2>(lhs.InternalArray, lhs.Shape, leftShape, false), (NDIterator<TOut>)(object)new NDIterator<#2>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        %
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#else

                #region Compute
		        switch (InfoOf<TOut>.NPTypeCode)
		        {
			        case NPTypeCode.Boolean: return ((NDIterator<TOut>)(object)new NDIterator<bool>(lhs.InternalArray, lhs.Shape, leftShape, false), (NDIterator<TOut>)(object)new NDIterator<bool>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        case NPTypeCode.Byte: return ((NDIterator<TOut>)(object)new NDIterator<byte>(lhs.InternalArray, lhs.Shape, leftShape, false), (NDIterator<TOut>)(object)new NDIterator<byte>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        case NPTypeCode.Int32: return ((NDIterator<TOut>)(object)new NDIterator<int>(lhs.InternalArray, lhs.Shape, leftShape, false), (NDIterator<TOut>)(object)new NDIterator<int>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        case NPTypeCode.Int64: return ((NDIterator<TOut>)(object)new NDIterator<long>(lhs.InternalArray, lhs.Shape, leftShape, false), (NDIterator<TOut>)(object)new NDIterator<long>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        case NPTypeCode.Single: return ((NDIterator<TOut>)(object)new NDIterator<float>(lhs.InternalArray, lhs.Shape, leftShape, false), (NDIterator<TOut>)(object)new NDIterator<float>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        case NPTypeCode.Double: return ((NDIterator<TOut>)(object)new NDIterator<double>(lhs.InternalArray, lhs.Shape, leftShape, false), (NDIterator<TOut>)(object)new NDIterator<double>(rhs.InternalArray, rhs.Shape, rightShape, false));
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#endif
            }
            else
            {
#if _REGEN1
                #region Compute
		        switch (lhs.TypeCode)
		        {
			        %foreach supported_dtypes,supported_dtypes_lowercase%
			        case NPTypeCode.#1: return ((NDIterator<TOut>)(object)new NDIterator<#2>(lhs, false), (NDIterator<TOut>)(object)new NDIterator<#2>(false));
			        %
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#else

                #region Compute
		        switch (lhs.TypeCode)
		        {
			        case NPTypeCode.Boolean: return ((NDIterator<TOut>)(object)new NDIterator<bool>(lhs, false), (NDIterator<TOut>)(object)new NDIterator<bool>(false));
			        case NPTypeCode.Byte: return ((NDIterator<TOut>)(object)new NDIterator<byte>(lhs, false), (NDIterator<TOut>)(object)new NDIterator<byte>(false));
			        case NPTypeCode.Int32: return ((NDIterator<TOut>)(object)new NDIterator<int>(lhs, false), (NDIterator<TOut>)(object)new NDIterator<int>(false));
			        case NPTypeCode.Int64: return ((NDIterator<TOut>)(object)new NDIterator<long>(lhs, false), (NDIterator<TOut>)(object)new NDIterator<long>(false));
			        case NPTypeCode.Single: return ((NDIterator<TOut>)(object)new NDIterator<float>(lhs, false), (NDIterator<TOut>)(object)new NDIterator<float>(false));
			        case NPTypeCode.Double: return ((NDIterator<TOut>)(object)new NDIterator<double>(lhs, false), (NDIterator<TOut>)(object)new NDIterator<double>(false));
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#endif
            }
        }
    }
}
