namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Returns the discrete, linear convolution of two one-dimensional sequences.
        ///
        /// The convolution operator is often seen in signal processing, where it models the effect of a linear time-invariant system on a signal[1]. In probability theory, the sum of two independent random variables is distributed according to the convolution of their individual distributions.
        /// 
        /// If v is longer than a, the arrays are swapped before computation.
        /// </summary>
        /// <param name="rhs"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public NDArray convolve(NDArray rhs, string mode = "full")
        {
            var lhs = this;
            int nf = lhs.shape[0];
            int ng = rhs.shape[0];

            if (ndim > 1 || rhs.ndim > 1)
                throw new IncorrectShapeException();
            var retType = np._FindCommonType(lhs, rhs);
            return null;
#if _REGEN
            #region Output
            %mod = "%"
            switch (lhs.GetTypeCode)
            {
	            %foreach supported_numericals,supported_numericals_lowercase%
	            case NPTypeCode.#1:
	            {
                    ArraySlice<#2> lhsarr = lhs.Storage.GetData<#2>();
		            switch (rhs.GetTypeCode)
                    {
	                    %foreach supported_numericals,supported_numericals_lowercase%
	                    case NPTypeCode.#101: 
                        {
                            ArraySlice<#102> rhsarr = rhs.Storage.GetData<#102>();
	                        %foreach supported_numericals,supported_numericals_lowercase%
		                    switch (retType)
                            {
                                case NPTypeCode.#201:
                                {
            #region Compute
                                    switch (mode.ToLowerInvariant())
                                    {
                    
                                        case "full":
                                        {
                                            int n = nf + ng - 1;

                                            var ret = new NDArray<#201>(Shape.Vector(n), true);
                                            var outArray = (ArraySlice<#202>)ret.Array;

                                            for (int idx = 0; idx < n; ++idx)
                                            {
                                                int jmn = (idx >= ng - 1) ? (idx - (ng - 1)) : 0;
                                                int jmx = (idx < nf - 1) ? idx : nf - 1;

                                                for (int jdx = jmn; jdx <= jmx; ++jdx)
                                                {
                                                    outArray[idx] += Converts.To#201(lhsarr[jdx] * rhsarr[idx - jdx]);
                                                }
                                            }

                                            return ret;
                                        }

                                        case "valid":
                                        {
                                            var min_v = (nf < ng) ? lhsarr : rhsarr;
                                            var max_v = (nf < ng) ? rhsarr : lhsarr;

                                            int n = Math.Max(nf, ng) - Math.Min(nf, ng) + 1;

                                            var ret = new NDArray(retType, Shape.Vector(n), true);
                                            var outArray = (ArraySlice<#202>)ret.Array;

                                            for (int idx = 0; idx < n; ++idx)
                                            {
                                                int kdx = idx;

                                                for (int jdx = (min_v.Count - 1); jdx >= 0; --jdx)
                                                {
                                                    outArray[idx] += Converts.To#202(min_v[jdx] * max_v[kdx]);
                                                    ++kdx;
                                                }
                                            }

                                            return ret;
                                        }

                                        case "same":
                                        {
                                            // followed the discussion on 
                                            // https://stackoverflow.com/questions/38194270/matlab-convolution-same-to-numpy-convolve
                                            // implemented numpy convolve because we follow numpy
                                            var npad = rhs.shape[0] - 1;

                                            if (npad #(mod) 2 == 1)
                                            {
                                                unsafe
                                                {
                                                    npad = (int)Math.Floor(((double)npad) / 2.0);

                                                    var arr = ArraySlice<#202>.Allocate(npad + lhsarr.Count);
                                                    lhsarr.CopyTo(arr.AsSpan, npad);
                                                    var retnd = new NDArray(new UnmanagedStorage(arr, Shape.Vector(lhsarr.Count)));

                                                    return retnd.convolve(rhs, "valid");
                                                }
                                            }
                                            else
                                            {
                                                {
                                                    unsafe
                                                    {
                                                        npad = npad / 2;

                                                        var puffer = new NDArray(retType, Shape.Vector(npad + lhsarr.Count), true);
                                                        lhsarr.CopyTo(puffer.Storage.AsSpan<#202>(), npad);
                                                        var np1New = puffer;

                                                        puffer = new NDArray(retType, Shape.Vector(npad + np1New.size), true);
                                                        var cpylen = np1New.size * sizeof(#202);
                                                        Buffer.MemoryCopy(np1New.Address, ((#202*)puffer.Address) + npad, cpylen, cpylen);

                                                        return puffer.convolve(rhs, "valid");
                                                    }
                                                }
                                            }
                                        }
                                        default:
                                            throw new ArgumentOutOfRangeException(nameof(mode));
                                    }
            #endregion
                                }
                            }

                            %
                            break;
                        }
                        %
                    }
                    break;
	            }

	            %
	            default:
		            throw new NotSupportedException();
            }

            #endregion
#else

#endif

        }
    }
}
