// NumPy's numpy/_core/function_base.py functions that compute, as DefaultEngine's members: np.linspace, np.logspace and
// np.geomspace (real and complex ends), each function's statements as its np.* facade held them, with their helpers.

using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        public override NDArray Linspace(double start, double stop, long num, bool endpoint = true, DType dtype = null, string device = null)
        {
            // NumPy's first statements are `num = operator.index(num); if num < 0: raise ValueError(...)`; the device is
            // read later, by the np.arange that builds the ramp. So a negative count wins over a bad device, and neither
            // reaches the allocation (whose own text would be `negative dimensions are not allowed`).
            ValidateSampleCount(num);
            DeviceArguments.ValidateDevice(device);
            // A text dtype (S, U or StringDType: the String storage lane) has no typed fill. NumPy computes the float64
            // samples (issubdtype(dtype, integer) is false: no floor) and its final astype formats them, so
            // np.linspace(0, 1, 5, dtype='U5') is ['0.0', '0.25', '0.5', '0.75', '1.0'].
            if (dtype != null && dtype.GetTypeCode() == NPTypeCode.String)
            {
                // The cast to text always copies (float64 is never a text dtype), so the float64 samples - a view of
                // the fill's buffer for two or more samples - are released here, whatever astype does.
                using NDArray samples = LinspaceCore(start, stop, num, endpoint, NPTypeCode.Double);
                return samples.astype(dtype, copy: false);
            }
            return LinspaceCore(start, stop, num, endpoint, dtype?.GetTypeCode() ?? NPTypeCode.Double);
        }

        /// <inheritdoc />
        public override NDArray Linspace(NDArray start, NDArray stop, long num, bool endpoint, DType dtype, int axis, string device)
        {
            // The facades refuse a null end before they pick an engine; this guards a direct engine call.
            if (start is null)
                throw new ArgumentNullException(nameof(start));
            if (stop is null)
                throw new ArgumentNullException(nameof(stop));

            // num = operator.index(num); if num < 0: raise ValueError(...) - NumPy's first statements, before the ends are read.
            ValidateSampleCount(num);
            long div = endpoint ? num - 1 : num;

            // conv = _array_converter(start, stop); dt = conv.result_type(ensure_inexact=True).
            DType dt = LinspaceComputeType(start, stop);
            // integer_dtype = issubdtype(dtype, integer), read before dtype defaults to dt (which is never an integer).
            bool integerDtype = dtype != null && IsSpacingIntegerDtype(dtype.typecode);

            // Two 0-d ends that compute in float64, with the sample axis first, are the scalar member's case: its ramp is
            // NumPy's float64 statements value for value (the same roundings, the exact last sample, the gh-5437 branch, the
            // same ownership), without the arange / delta / step arrays. The C# number ends of every existing call land here.
            if (axis == 0 && start.ndim == 0 && stop.ndim == 0 && dt.typecode == NPTypeCode.Double)
                return Linspace(Converts.ToDouble(start.GetAtIndex(0)), Converts.ToDouble(stop.GetAtIndex(0)), num, endpoint, dtype, device);

            return LinspaceArrays(start, stop, num, endpoint, div, dt, integerDtype, dtype ?? dt, axis, device);
        }

        /// <summary>
        ///     NumPy's <c>np.linspace</c> statements after the dtype is known, over arrays: <c>delta = np.subtract(stop, start,
        ///     dtype=dt)</c>, the ramp <c>arange(0, num, dtype=dt)</c> shaped <c>(num, 1, ...)</c> against delta's rank, the
        ///     step and its gh-5437 branch, <c>y += start</c>, the exact last sample, the axis move, the integer floor and the
        ///     final <c>astype(dtype, copy=False)</c>.
        /// </summary>
        /// <param name="start">The first end (any shape that broadcasts with <paramref name="stop"/>).</param>
        /// <param name="stop">The second end.</param>
        /// <param name="num">The number of samples (validated non-negative).</param>
        /// <param name="endpoint">Whether the last sample is <paramref name="stop"/> itself.</param>
        /// <param name="div"><c>num - 1</c> with the endpoint, <c>num</c> without; no step when it is not positive.</param>
        /// <param name="dt">The computation dtype (<see cref="LinspaceComputeType"/>).</param>
        /// <param name="integerDtype">Whether the requested dtype is an integer one (the samples are floored first).</param>
        /// <param name="dtype">The result dtype (<paramref name="dt"/> when none was requested).</param>
        /// <param name="axis">The destination of the sample axis.</param>
        /// <param name="device">The Array-API device, checked where NumPy's ramp checks it.</param>
        /// <returns>The samples; a view of the ramp buffer where NumPy's result is one (scalar ends, a moved axis).</returns>
        /// <exception cref="IncorrectShapeException">Ends that do not broadcast (the subtract's text).</exception>
        /// <exception cref="ValueError">A <paramref name="device"/> other than <c>"cpu"</c>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range for the result's rank.</exception>
        /// <exception cref="TypeError">An integer <paramref name="dtype"/> over complex samples (floor has no complex loop).</exception>
        /// <remarks>Every intermediate is released before returning: on success only the result's buffer stays referenced,
        ///     and on a failure the partial ramp goes with the rest.</remarks>
        private NDArray LinspaceArrays(NDArray start, NDArray stop, long num, bool endpoint, long div, DType dt,
            bool integerDtype, DType dtype, int axis, string device)
        {
            // delta = np.subtract(stop, start, dtype=type(dt)): the ends broadcast here (NumPy's broadcast error is raised by
            // this subtract), and a weak Python-number end takes dt.
            NDArray delta = Subtract(stop, start, dt, null, null);
            NDArray step = null, y = null;
            try
            {
                // y = _nx.arange(0, num, dtype=dt, device=device).reshape((-1,) + (1,) * ndim(delta)): the ramp in dt (the
                // dtype's own fill, as NumPy's arange), as a VIEW shaped to broadcast against delta.
                DeviceArguments.ValidateDevice(device);
                y = LinspaceRamp(num, dt, delta.ndim);

                // _mult_inplace = _nx.isscalar(delta): two scalar ends multiply in place (y stays the ramp's view), array ends
                // into a new array; the values are the same, NumPy's result ownership is not.
                bool scalarDelta = delta.ndim == 0;
                if (div > 0)
                {
                    // step = delta / div: the Python int div is weak, so the division runs in dt.
                    using var divisor = NDArray.PythonLiteral(div);
                    step = Divide(delta, divisor, null, null, null);
                    bool anyStepZero;
                    using (var zero = step == 0)
                        anyStepZero = Any(zero);
                    if (anyStepZero)
                    {
                        // Special handling for denormal numbers, gh-5437: y /= div; y *= delta (y = y * delta for arrays),
                        // so a step that underflowed to zero does not collapse every sample onto the start.
                        Divide(y, divisor, null, y, null);
                        if (scalarDelta)
                            Multiply(y, delta, null, y, null);
                        else
                            y = ReplaceLinspaceStage(y, Multiply(y, delta, null, null, null));
                    }
                    else if (scalarDelta)
                        Multiply(y, step, null, y, null);
                    else
                        y = ReplaceLinspaceStage(y, Multiply(y, step, null, null, null));
                }
                else
                {
                    // No samples, or one with the endpoint: no step (NumPy's nan); y = y * delta keeps NumPy's 0 * delta, a NaN
                    // for an infinite end.
                    y = ReplaceLinspaceStage(y, Multiply(y, delta, null, null, null));
                }

                // y += start (in place: the result keeps dt; an integer or weak start is cast into it).
                Add(y, start, null, y, null);

                // if endpoint and num > 1: y[-1, ...] = stop - the last sample is the end itself, never a rounded ramp value.
                if (endpoint && num > 1)
                    y["-1, ..."] = stop;

                // if axis != 0: y = _nx.moveaxis(y, 0, axis) - raises NumPy's "destination: axis ..." AxisError.
                if (axis != 0)
                    y = ReplaceLinspaceStage(y, MoveAxis(y, new[] { 0 }, new[] { axis }));

                // if integer_dtype: _nx.floor(y, out=y) - NumPy 1.20+ rounds toward -inf before the integer cast.
                if (integerDtype)
                    Floor(y, null, y, null);

                // y = conv.wrap(y.astype(dtype, copy=False)): the wrap is this member's [ArrayWrap] declaration, which the
                // woven np.linspace applies to a subclass operand's result.
                NDArray result = y.astype(dtype, copy: false);
                if (!ReferenceEquals(result, y))
                    y.Dispose();
                return result;
            }
            catch
            {
                y?.Dispose();
                throw;
            }
            finally
            {
                delta.Dispose();
                step?.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's linspace ramp, <c>_nx.arange(0, num, dtype=dt).reshape((-1,) + (1,) * ndim)</c>: the counts
        ///     <c>0 .. num-1</c> written by <paramref name="dt"/>'s own fill, as a view with <paramref name="ndim"/> trailing
        ///     axes of length 1.
        /// </summary>
        /// <param name="num">The number of samples.</param>
        /// <param name="dt">The computation dtype.</param>
        /// <param name="ndim">The rank of delta: the ramp broadcasts against it from the front.</param>
        /// <returns>The ramp view (it holds the arange buffer alive; the arange's own reference is released here).</returns>
        private static NDArray LinspaceRamp(long num, DType dt, int ndim)
        {
            var dims = new long[1 + ndim];
            dims[0] = num;
            for (int i = 1; i < dims.Length; i++)
                dims[i] = 1;
            using NDArray counts = UnmanagedStorage.ArangeObj(0L, num, 1L, dt);
            return counts.reshape(new Shape(dims, copy: false));
        }

        /// <summary>
        ///     Swaps one stage of the linspace ramp for the next (NumPy rebinding <c>y</c>): the stage that is replaced is
        ///     released, and the new one holds whatever buffer it still needs (a view keeps its source alive).
        /// </summary>
        /// <param name="previous">The stage being replaced; disposed unless the next stage is the same object.</param>
        /// <param name="next">The new stage.</param>
        /// <returns><paramref name="next"/>.</returns>
        private static NDArray ReplaceLinspaceStage(NDArray previous, NDArray next)
        {
            if (!ReferenceEquals(previous, next))
                previous.Dispose();
            return next;
        }

        /// <summary>
        ///     The dtype NumPy's <c>np.linspace</c> computes in: <c>_array_converter(start, stop).result_type(ensure_inexact=True)</c>,
        ///     the promotion of the two ends' classes with the abstract Python float. A NumPy array or scalar end is strong (its
        ///     dtype counts: two float32 ends give float32, a float32 with a float64 end float64); a C# number that reached the
        ///     <see cref="NDArray"/> parameter through its implicit conversion is a weak Python number (its kind counts, not its
        ///     width: a Python int beside a float32 end gives float32). Integer and bool ends give float64.
        /// </summary>
        /// <param name="start">The first end.</param>
        /// <param name="stop">The second end.</param>
        /// <returns>The computation dtype (always inexact).</returns>
        /// <exception cref="DTypePromotionError">No common DType (a text end: NumPy's text lists the classes and
        ///     <c>_PyFloatDType</c>).</exception>
        /// <remarks>NumSharp's <c>char</c> lane, which NumPy lacks, promotes as its uint16 proxy, as np.pad's linear ramp
        ///     reads it.</remarks>
        internal static DType LinspaceComputeType(NDArray start, NDArray stop)
            => DTypePromotion.ResultType(LinspacePromotionOperand(start), LinspacePromotionOperand(stop), DTypeRegistry.PyFloat);

        /// <summary>The operand <see cref="LinspaceComputeType"/> promotes for one end: the weak Python number a C# literal
        ///     stands for, the uint16 proxy of a char end, or the end itself (strong).</summary>
        /// <param name="end">The end.</param>
        /// <returns>The <see cref="DTypePromotion.ResultType"/> operand.</returns>
        private static object LinspacePromotionOperand(NDArray end)
            => end.typecode == NPTypeCode.Char ? np.uint16 : end.Nep50Operand;

        /// <summary>
        ///     NumPy's <c>issubdtype(dtype, np.integer)</c> over NumSharp's lanes, for the floor linspace applies before an
        ///     integer cast: the signed and unsigned integers, and NumSharp's <c>char</c>, which casts like uint16. Bool is not
        ///     an integer there (it casts the samples directly: nonzero is True).
        /// </summary>
        /// <param name="typeCode">The requested dtype's lane.</param>
        /// <returns>True when the samples are floored before the cast.</returns>
        private static bool IsSpacingIntegerDtype(NPTypeCode typeCode)
            => typeCode is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32
                or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Char;

        /// <summary>
        ///     The storage-lane core behind every public <c>linspace(..., dtype)</c> overload (which all take the single
        ///     <see cref="DType"/> descriptor parameter, like NumPy's <c>dtype=</c>).
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The end value of the sequence, unless endpoint is set to False. In that case, the sequence consists of all but the last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size changes when endpoint is False.</param>
        /// <param name="num">Number of samples to generate. Default is 50. Must be non-negative.</param>
        /// <param name="endpoint">If True, stop is the last sample. Otherwise, it is not included. Default is True.</param>
        /// <param name="typeCode">The type of the output array.</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples; a float64 result whose NumPy counterpart is a view
        ///     (<see cref="TensorEngine.Linspace"/>) is an alias that does not own its buffer.</returns>
        /// <exception cref="ArgumentException"><paramref name="typeCode"/> is <see cref="NPTypeCode.Empty"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="typeCode"/> has no fixed-size storage lane (the text
        ///     lane), or no fill below.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        ///     <para>Each sample is NumPy's float64 ramp value (<see cref="LinspaceValue"/>), floored for an integer type,
        ///     then converted in one typed loop per storage lane.</para></remarks>
        private NDArray LinspaceCore(double start, double stop, long num, bool endpoint, NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentException("Invalid typeCode", nameof(typeCode));

            // NumPy 1.20+ floors the inexact linspace values before casting to an explicitly
            // requested integer dtype (rather than truncating toward zero). Char follows UInt16.
            bool floorInteger = typeCode == NPTypeCode.SByte || typeCode == NPTypeCode.Byte
                                || typeCode == NPTypeCode.Int16 || typeCode == NPTypeCode.UInt16
                                || typeCode == NPTypeCode.Int32 || typeCode == NPTypeCode.UInt32
                                || typeCode == NPTypeCode.Int64 || typeCode == NPTypeCode.UInt64
                                || typeCode == NPTypeCode.Char;

            NDArray ret = NewArray(typeCode, new Shape(num), false);

            if (num == 0)
                return ret;

            // NumPy's quantities: delta = stop - start, and div = num - 1 with the endpoint, num without. One sample with
            // the endpoint has div 0 and no step; LinspaceValue takes NumPy's branch for it (0 * delta + start), so the
            // division here may give an infinity or a NaN that is never read.
            double delta = stop - start;
            double div = endpoint ? num - 1.0 : num;
            double step = delta / div;

            double Sample(long i)
            {
                // NumPy's float64 ramp value, then the floor NumPy applies before an integer cast.
                double value = LinspaceValue(i, num, endpoint, start, stop, delta, div, step);
                return floorInteger ? Math.Floor(value) : value;
            }

            switch (ret.GetTypeCode)
            {
	            // case NPTypeCode.Boolean:
	            // {
                    // unsafe
                    // {
                        // var addr = (bool*)ret.Address;
                        // for (long i = 0; i < num; i++) addr[i] = (start + i * step) != 0;
                    // }

                    // return ret;
	            // }
	            // %foreach except(supported_dtypes, "Boolean"),except(supported_dtypes_lowercase, "bool")%
	            // case NPTypeCode.#1:
	            // {
                    // unsafe
                    // {
                        // var addr = (#2*)ret.Address;
                         // for (long i = 0; i < num; i++) addr[i] = Converts.To#1(start + i * step);
                    // }

                    // return ret;
	            // }
	            // %
	            // default:
		            // throw new NotSupportedException();
	            case NPTypeCode.Boolean:
	            {
                    unsafe
                    {
                        var addr = (bool*)ret.Address;
                        for (long i = 0; i < num; i++) addr[i] = Sample(i) != 0;
                    }

                    return ret;
	            }
	            case NPTypeCode.Byte:
	            {
                    unsafe
                    {
                        var addr = (byte*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToByte(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.SByte:
	            {
                    unsafe
                    {
                        var addr = (sbyte*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToSByte(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Int16:
	            {
                    unsafe
                    {
                        var addr = (short*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToInt16(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.UInt16:
	            {
                    unsafe
                    {
                        var addr = (ushort*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToUInt16(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Int32:
	            {
                    unsafe
                    {
                        var addr = (int*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToInt32(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.UInt32:
	            {
                    unsafe
                    {
                        var addr = (uint*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToUInt32(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Int64:
	            {
                    unsafe
                    {
                        var addr = (long*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToInt64(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.UInt64:
	            {
                    unsafe
                    {
                        var addr = (ulong*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToUInt64(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Char:
	            {
                    unsafe
                    {
                        var addr = (char*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToChar(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Half:
	            {
                    unsafe
                    {
                        var addr = (Half*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = (Half)Sample(i);
                    }

                    return ret;
	            }
	            case NPTypeCode.Double:
	            {
                    unsafe
                    {
                        var addr = (double*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToDouble(Sample(i));
                    }

                    // One sample with the endpoint: NumPy's div is 0, so it takes the rebinding `y = y * delta` branch
                    // and its result owns its buffer (probed 2.4.2).
                    if (div == 0)
                        return ret;

                    // NumPy's float64 linspace is a reshape-VIEW of its internal arange buffer
                    // mutated in place (y *= step; y += start), and astype(float64, copy=False)
                    // returns that view verbatim — so the result reports OWNDATA=False (probed
                    // 2.4.2; div > 0 here — two or more samples, or one without the endpoint — so the
                    // in-place branch is the one NumPy took). Every other dtype rebinds through a real
                    // astype copy and owns.
                    var view = new NDArray(ret.Storage.Alias()) { TensorEngine = ret.TensorEngine };
                    ret.Dispose();
                    return view;
	            }
	            case NPTypeCode.Single:
	            {
                    unsafe
                    {
                        var addr = (float*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToSingle(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Decimal:
	            {
                    unsafe
                    {
                        var addr = (decimal*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToDecimal(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Complex:
	            {
                    unsafe
                    {
                        var addr = (System.Numerics.Complex*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = new System.Numerics.Complex(Sample(i), 0);
                    }

                    return ret;
	            }
	            default:
		            throw new NotSupportedException();
            }
        }

        /// <inheritdoc />
        public override NDArray Logspace(double start, double stop, long num = 50, bool endpoint = true, double @base = 10.0, DType dtype = null, int axis = 0)
        {
            // NumPy order: linspace validates `num` (ValueError) BEFORE the trailing moveaxis validates
            // `axis`, so the sample-count error wins when both are bad.
            ValidateSampleCount(num);
            ValidateSpacingAxis(axis);

            // NumPy computes `power(base, y)` in the inferred (float64 for scalar-double inputs) dtype and
            // ONLY THEN casts to `dtype` — computing directly in a narrow dtype would round differently.
            NDArray result = LogspaceCore(start, stop, num, endpoint, @base);
            return CastSpacingResult(result, dtype);
        }

        /// <summary>
        ///     The float64 core behind every <c>logspace</c> overload: fills a fresh 1-D buffer with
        ///     <c>base ** (start + i*step)</c> in a single fused pass (no intermediate <c>linspace</c> array),
        ///     bit-identical to NumPy's <c>power(base, linspace(start, stop, num, endpoint))</c>.
        /// </summary>
        /// <param name="start">Log-space start exponent.</param>
        /// <param name="stop">Log-space stop exponent (the last sample when <paramref name="endpoint"/> is True).</param>
        /// <param name="num">Number of samples (already validated non-negative by the caller).</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the final exponent.</param>
        /// <param name="base">The base raised to each exponent.</param>
        /// <returns>A fresh, C-contiguous, owning float64 array of length <paramref name="num"/>.</returns>
        /// <remarks>
        ///     The exponents are NumPy's linspace values (<see cref="LinspaceValue"/>): the endpoint is <c>stop</c>
        ///     itself (NumPy overwrites <c>y[-1] = stop</c> before the power), a single exponent is <c>0 * delta + start</c>
        ///     (NaN for an infinite or NaN stop, so <c>base ** y</c> is NaN too), and a step that underflows to zero takes
        ///     NumPy's <c>(i / div) * delta + start</c>. Addition/multiplication commute bit-for-bit with NumPy's
        ///     <c>arange*step + start</c>, and <see cref="Math.Pow(double,double)"/> is the same <c>ucrtbase</c> routine
        ///     NumPy's <c>npy_pow</c> calls on win-amd64, so the float64 result is byte-exact.
        /// </remarks>
        private NDArray LogspaceCore(double start, double stop, long num, bool endpoint, double @base)
        {
            // fillZeros:false — every element is written below, so zeroing the buffer first would be wasted work.
            NDArray ret = NewArray(NPTypeCode.Double, new Shape(num), false);

            if (num == 0)
                return ret;

            // NumPy's logspace is power(base, linspace(start, stop, num, endpoint)): the linspace quantities, with div 0
            // (one exponent and the endpoint) handled by LinspaceValue before the step is read.
            double delta = stop - start;
            double div = endpoint ? num - 1.0 : num;
            double step = delta / div;

            unsafe
            {
                double* addr = (double*)ret.Address;
                for (long i = 0; i < num; i++)
                    addr[i] = Math.Pow(@base, LinspaceValue(i, num, endpoint, start, stop, delta, div, step));
            }

            return ret;
        }

        /// <summary>
        ///     Sample <paramref name="i"/> of NumPy's float64 linspace ramp, the value NumPy's statements leave in
        ///     <c>y[i]</c> before an integer dtype's floor and the final cast. Shared by <see cref="LinspaceCore"/> and
        ///     <see cref="LogspaceCore"/> (whose exponents are NumPy's linspace).
        /// </summary>
        /// <param name="i">The sample's index, in <c>[0, num)</c>.</param>
        /// <param name="num">The number of samples, at least 1.</param>
        /// <param name="endpoint">Whether the last of two or more samples is <paramref name="stop"/>.</param>
        /// <param name="start">The first end.</param>
        /// <param name="stop">The second end.</param>
        /// <param name="delta"><c>stop - start</c>.</param>
        /// <param name="div"><c>num - 1</c> with the endpoint, <c>num</c> without.</param>
        /// <param name="step"><c>delta / div</c>; not read when <paramref name="div"/> is 0.</param>
        /// <returns>The sample as NumPy computes it.</returns>
        /// <remarks>
        ///     NumPy's branches, each observable in the result:
        ///     <list type="bullet">
        ///       <item><description>two or more samples with the endpoint: the last is <paramref name="stop"/> itself
        ///       (<c>y[-1] = stop</c>), so no rounding of the ramp reaches it;</description></item>
        ///       <item><description>one sample with the endpoint (<paramref name="div"/> 0, no step): <c>y = y * delta;
        ///       y += start</c>, so <c>0 * delta + start</c> - a NaN when an end is infinite or NaN, and +0.0 for a start of
        ///       -0.0 (one sample without the endpoint gives the same value through the step, which is delta);</description></item>
        ///       <item><description>a step that underflows to zero (gh-5437: a subnormal delta over many samples):
        ///       <c>y /= div; y *= delta</c>, so <c>(i / div) * delta + start</c>, which keeps the spread
        ///       <c>i * step</c> loses;</description></item>
        ///       <item><description>otherwise <c>i * step + start</c>.</description></item>
        ///     </list>
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // called once per sample from the fill loops
        private static double LinspaceValue(long i, long num, bool endpoint, double start, double stop, double delta, double div, double step)
        {
            if (endpoint && num > 1 && i == num - 1)
                return stop;
            if (div == 0)
                return i * delta + start;
            if (step == 0)
                return i / div * delta + start;
            return start + i * step;
        }

        /// <summary>
        ///     Validates a sample count for the spacing generators (<c>logspace</c>/<c>geomspace</c>), matching NumPy's
        ///     <c>linspace</c> guard verbatim so a negative count fails before any allocation.
        /// </summary>
        /// <param name="num">The requested number of samples.</param>
        /// <exception cref="ValueError">If <paramref name="num"/> is negative (NumPy's <c>ValueError</c>, verbatim message).</exception>
        internal static void ValidateSampleCount(long num)
        {
            // No paramName: NumPy's ValueError message has no ".NET (Parameter 'num')" suffix, so keep it verbatim.
            if (num < 0)
                throw new ValueError(string.Create(CultureInfo.InvariantCulture, $"Number of samples, {num}, must be non-negative."));
        }

        /// <summary>
        ///     Validates the <c>axis</c> argument for the scalar spacing generators. Their output is always 1-D, so —
        ///     like NumPy's trailing <c>moveaxis(y, 0, axis)</c> — only destinations 0 and -1 are in range; anything
        ///     else raises with NumPy's verbatim "destination:" message reporting the ORIGINAL (un-normalized) axis.
        /// </summary>
        /// <param name="axis">The requested destination axis.</param>
        /// <exception cref="AxisError">If <paramref name="axis"/> is neither 0 nor -1.</exception>
        internal static void ValidateSpacingAxis(int axis)
        {
            if (axis != 0 && axis != -1)
                throw new AxisError(string.Create(CultureInfo.InvariantCulture, $"destination: axis {axis} is out of bounds for array of dimension 1"));
        }

        /// <summary>
        ///     Applies NumPy's final <c>result.astype(dtype, copy=False)</c> to a freshly-computed spacing result:
        ///     returns it unchanged when no dtype is requested or the result is already that dtype, otherwise casts
        ///     (float→integer TRUNCATES toward zero; complex→float drops the imaginary part, matching NumPy).
        /// </summary>
        /// <param name="result">
        ///     The freshly-computed array in the computation domain (float64 or complex128). OWNERSHIP TRANSFERS in:
        ///     it is returned as-is when no cast is needed, and DISPOSED here when a cast replaces it, so the caller
        ///     must not touch it afterwards.
        /// </param>
        /// <param name="dtype">The requested output dtype, or null to keep the computation domain.</param>
        /// <returns>The result in the requested dtype (the same instance when the dtype already matches or is null).</returns>
        /// <exception cref="NotSupportedException">The descriptor's class has no storage lane (propagated from <see cref="NDArray.astype(DType, bool)"/>).</exception>
        /// <remarks>
        ///     Domain-agnostic on purpose: <c>logspace</c> feeds a float64 result while <c>geomspace</c>'s complex path
        ///     feeds a complex128 one, and both want "cast only if the dtype actually differs" — so it compares the
        ///     result's own type code, not a hard-coded float64.
        /// </remarks>
        private static NDArray CastSpacingResult(NDArray result, DType dtype)
        {
            // Keep the computation-domain array when no cast is needed (NumPy's astype(..., copy=False) no-op).
            if (dtype == null || result.GetTypeCode == dtype.GetTypeCode())
                return result;
            // A genuine dtype change always allocates fresh storage; copy:false mirrors NumPy's copy=False.
            NDArray cast = result.astype(dtype, copy: false);
            // The computation-domain array is now a dead intermediate: every caller passes a fresh array
            // it owns and returns only what this method hands back. Release it now, or its pooled buffer
            // waits for the finalizer (the FuzzMatrix scope audit caught exactly one escaped buffer per
            // non-float64 logspace/geomspace call). The guard keeps a same-instance astype from being
            // disposed out from under the caller.
            if (!ReferenceEquals(cast, result))
                result.Dispose();
            return cast;
        }

        // =====================================================================
        // np.geomspace — numbers spaced evenly on a log scale (a GEOMETRIC
        // progression), port of NumPy 2.4.2 numpy/_core/function_base.py::geomspace.
        //
        // NumPy's algorithm, followed here operation-for-operation:
        //     if any(start==0) or any(stop==0): raise "Geometric sequence cannot include zero"
        //     out_sign = sign(start)                 # rotate start onto the positive real axis
        //     start /= out_sign;  stop /= out_sign   # so log10 is real for real inputs
        //     result = logspace(log10(start), log10(stop), num, endpoint, base=10)
        //     if num>0: result[0]=start; if num>1 and endpoint: result[-1]=stop   # exact endpoints
        //     result *= out_sign                     # undo the rotation
        //
        // REAL path (the double overload) is BIT-EXACT vs NumPy: out_sign is
        // ±1.0, so the rotate (start/out_sign) and the final multiply are exact, and
        // result[0]=start/result[-1]=stop land the ORIGINAL endpoints byte-for-byte
        // (start/out_sign*out_sign == start). The interior rides logspace's bit-exact
        // Math.Pow/Math.Log10 == npy_pow/npy_log10 (win-amd64). Mixed-sign real inputs
        // (e.g. geomspace(-1,1)) reproduce NumPy's [-1, nan, nan, 1] automatically:
        // rotating makes stop negative, log10(negative)=NaN poisons the interior, and
        // the endpoints are overwritten.
        //
        // COMPLEX path (dtype=complex OR the Complex-input overload) computes in the
        // complex128 domain — the interior differs from the real path (NumPy's
        // geomspace(1,8,4,dtype=complex)[1] == 1.9999999999999998, not the real path's
        // exact 2.0), so the domain genuinely matters. It composes NumSharp's
        // NumPy-tuned complex power (npy_cpow port) and complex log10, so it is
        // ACCURATE to NumPy within the documented ≤3-ULP complex-unary envelope but
        // NOT byte-reproducible (like np.sinc's complex path) — allclose, not bit-exact.
        //
        // ARRAY ends (API-parity batch 10): any other pair - array ends, or 0-d ends whose dt is
        // float16 / float32 / decimal - runs NumPy's statements over arrays (GeomspaceArrays):
        // the ends cast to dt, sign, true_divide, log10, the NDArray linspace, power, the
        // end writes, multiply, moveaxis. Two 0-d ends that compute in float64 / complex128
        // (every C# number pair) keep the fused scalar cores below, value for value.
        // =====================================================================

        /// <inheritdoc />
        public override NDArray Geomspace(NDArray start, NDArray stop, long num = 50, bool endpoint = true, DType dtype = null, int axis = 0)
        {
            // The facade refuses a null end before it picks an engine; this guards a direct engine call.
            if (start is null)
                throw new ArgumentNullException(nameof(start));
            if (stop is null)
                throw new ArgumentNullException(nameof(stop));

            // start = asanyarray(start); stop = asanyarray(stop): a C# number end is a Python number, whose array NumPy
            // builds in the number's default dtype; an array end keeps its own.
            DType startDType = GeomspaceEndDType(start), stopDType = GeomspaceEndDType(stop);
            bool scalarEnds = start.ndim == 0 && stop.ndim == 0;

            // if _nx.any(start == 0) or _nx.any(stop == 0): raise - NumPy's FIRST check, so a zero end wins over a negative
            // num, a text dtype and a bad axis. Two 0-d ends are read directly (no comparison arrays).
            if (scalarEnds ? IsZeroScalar(start) || IsZeroScalar(stop) : AnyZero(start) || AnyZero(stop))
                throw new ValueError("Geometric sequence cannot include zero");
            // A text dtype fails NumPy's next statements, before the count and the axis are read.
            if (dtype != null && dtype.GetTypeCode() == NPTypeCode.String)
                RefuseGeomspaceTextDtype(start, startDType, stopDType, num, dtype);

            // dt = result_type(start, stop, float(num), _nx.zeros((), dtype)): the ends and the 0-d zeros strong, float(num)
            // weak - so dt is inexact, and float64 for C# number ends unless a complex dtype makes it complex128.
            DType dt = DTypePromotion.ResultType(startDType, stopDType, (double)num, dtype ?? DType.Double);

            if (scalarEnds && dt.GetTypeCode() == NPTypeCode.Double)
            {
                // The scalar real path: NumPy's float64 statements fused into one pass (GeomspaceRealCore), value for value.
                ValidateSampleCount(num);
                ValidateSpacingAxis(axis);
                // NumPy's tail: `if axis != 0: result = moveaxis(result, 0, axis)`, then `result.astype(dtype, copy=False)`.
                NDArray ret = SpacingAxisView(GeomspaceRealCore(ScalarAsDouble(start), ScalarAsDouble(stop), num, endpoint), axis);
                return CastSpacingResult(ret, dtype);
            }
            if (scalarEnds && dt.GetTypeCode() == NPTypeCode.Complex)
            {
                // The scalar spiral (a complex end or a complex dtype): NumPy's complex128 statements, GeomspaceComplexCore.
                ValidateSampleCount(num);
                ValidateSpacingAxis(axis);
                return GeomspaceComplexCore(ScalarAsComplex(start), ScalarAsComplex(stop), num, endpoint, dtype, axis);
            }

            return GeomspaceArrays(start, stop, num, endpoint, dt, dtype ?? dt, axis);
        }

        /// <summary>
        ///     The dtype of an end as NumPy's <c>asanyarray(end)</c> builds it: an array keeps its dtype; a C# number (the 0-d
        ///     array NDArray's implicit conversion marks as a Python number) takes the Python number's default - bool,
        ///     int64 (uint64 for an integer above the int64 range), float64, complex128 - as NumPy's Python number would,
        ///     whatever its C# width.
        /// </summary>
        /// <param name="end">The end (not null).</param>
        /// <returns>The dtype NumPy's array of the end has.</returns>
        /// <remarks>Only the result dtype and the text of a promotion error read it: the values are cast from the end
        ///     itself, and a C# number's value is the Python number's.</remarks>
        private static DType GeomspaceEndDType(NDArray end)
        {
            if (end.WasPythonBool)
                return DType.Boolean;
            if (!end.WasPythonLiteral)
                return end.dtype;
            switch (end.typecode)
            {
                case NPTypeCode.Complex:
                    return DType.Complex;
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal:
                    return DType.Double;
                case NPTypeCode.UInt64:
                    // np.asanyarray(2**64 - 1) is uint64: a Python int beyond int64 takes the unsigned type.
                    return Converts.ToUInt64(end.GetAtIndex(0)) > long.MaxValue ? DType.UInt64 : DType.Int64;
                default:
                    return DType.Int64;
            }
        }

        /// <summary>NumPy's <c>end == 0</c> for a 0-d end, read directly: a complex end is zero only at <c>0+0j</c>, a NaN
        ///     never is, <c>False</c> is (bool compares as 0), and a text end never is.</summary>
        /// <param name="end">The 0-d end.</param>
        /// <returns>True when the end equals zero.</returns>
        private static bool IsZeroScalar(NDArray end)
        {
            switch (end.typecode)
            {
                case NPTypeCode.Complex:
                    return (Complex)end.GetAtIndex(0) == Complex.Zero;
                case NPTypeCode.String:
                    return false;
                default:
                    return Converts.ToDouble(end.GetAtIndex(0)) == 0.0;
            }
        }

        /// <summary>NumPy's <c>_nx.any(end == 0)</c> over an array end.</summary>
        /// <param name="end">The end.</param>
        /// <returns>True when any element equals zero.</returns>
        private bool AnyZero(NDArray end)
        {
            // The comparison array is a temporary of this check alone.
            using var zero = end == 0;
            return Any(zero);
        }

        /// <summary>The float64 value of a real 0-d end (a C# number, or a 0-d array of any real dtype), as NumPy's
        ///     <c>start.astype(float64)</c> reads it.</summary>
        /// <param name="end">The 0-d end.</param>
        /// <returns>The value.</returns>
        private static double ScalarAsDouble(NDArray end) => Converts.ToDouble(end.GetAtIndex(0));

        /// <summary>The complex128 value of a 0-d end, as NumPy's <c>start.astype(complex128)</c> reads it (a real end's
        ///     imaginary part is zero).</summary>
        /// <param name="end">The 0-d end.</param>
        /// <returns>The value.</returns>
        private static Complex ScalarAsComplex(NDArray end)
            => end.typecode == NPTypeCode.Complex ? (Complex)end.GetAtIndex(0) : new Complex(ScalarAsDouble(end), 0.0);

        /// <summary>
        ///     NumPy's <c>np.geomspace</c> statements after <c>dt</c> is known, over array ends: <c>start.astype(dt,
        ///     copy=True)</c> and <c>stop</c>'s, <c>out_sign = sign(start)</c>, <c>start /= out_sign</c>, <c>stop = stop /
        ///     out_sign</c>, <c>logspace(log10(start), log10(stop), num, endpoint, base=10.0, dtype=dt)</c> (NumPy's
        ///     <c>power(10.0, linspace(log_start, log_stop, num, endpoint))</c> then <c>astype(dt, copy=False)</c>), the ends
        ///     written into the first and last samples, <c>result *= out_sign</c>, <c>moveaxis(result, 0, axis)</c> and
        ///     <c>astype(dtype, copy=False)</c>.
        /// </summary>
        /// <param name="start">The first end (any shape that broadcasts with <paramref name="stop"/>).</param>
        /// <param name="stop">The second end.</param>
        /// <param name="num">The number of samples (validated by the linspace, as NumPy's logspace does).</param>
        /// <param name="endpoint">Whether the last sample is <paramref name="stop"/>.</param>
        /// <param name="dt">The computation dtype.</param>
        /// <param name="dtype">The result dtype (<paramref name="dt"/> when none was requested).</param>
        /// <param name="axis">The destination of the sample axis.</param>
        /// <returns>The samples; a moved axis is a view of the computed buffer, as NumPy's moveaxis is.</returns>
        /// <exception cref="ValueError">A negative <paramref name="num"/> (the linspace's check).</exception>
        /// <exception cref="IncorrectShapeException">Ends that do not broadcast.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range for the result's rank.</exception>
        /// <remarks>Every intermediate is released before returning; on a failure the partial result goes with them.</remarks>
        private NDArray GeomspaceArrays(NDArray start, NDArray stop, long num, bool endpoint, DType dt, DType dtype, int axis)
        {
            NDArray s = null, e = null, outSign = null, logStart = null, logStop = null, y = null, result = null;
            try
            {
                // start = start.astype(dt, copy=True); stop = stop.astype(dt, copy=True): private copies, changed in place below.
                s = start.astype(dt, copy: true);
                e = stop.astype(dt, copy: true);

                // out_sign = _nx.sign(start): +-1 for a real end (NaN for NaN), z / |z| for a complex one.
                outSign = Sign(s);
                // start /= out_sign (in place); stop = stop / out_sign (a new array: the ends broadcast here).
                Divide(s, outSign, null, s, null);
                NDArray rotatedStop = Divide(e, outSign, null, null, null);
                e.Dispose();
                e = rotatedStop;

                // log_start = _nx.log10(start); log_stop = _nx.log10(stop).
                logStart = Log10(s);
                logStop = Log10(e);

                // result = logspace(log_start, log_stop, num=num, endpoint=endpoint, base=10.0, dtype=dt):
                //     y = linspace(start, stop, num=num, endpoint=endpoint, axis=0); power(base, y).astype(dt, copy=False).
                y = Linspace(logStart, logStop, num, endpoint, null, 0, null);
                // base=10.0 is a weak Python float: the power runs in y's dtype.
                using (var ten = NDArray.PythonLiteral(10.0))
                    result = Power(ten, y, null, null, null);
                result = CastSpacingResult(result, dt);

                // if num > 0: result[0] = start; if num > 1 and endpoint: result[-1] = stop - the ends themselves, never
                // 10 ** log10(x) (which need not round back to x).
                if (num > 0)
                {
                    result["0, ..."] = s;
                    if (num > 1 && endpoint)
                        result["-1, ..."] = e;
                }

                // result *= out_sign: undo the rotation (the sign broadcasts against every sample).
                Multiply(result, outSign, null, result, null);

                // if axis != 0: result = _nx.moveaxis(result, 0, axis) - NumPy's "destination: axis ..." AxisError, and a view.
                if (axis != 0)
                {
                    NDArray moved = MoveAxis(result, new[] { 0 }, new[] { axis });
                    // The view holds its own reference to the buffer: only the computed wrapper goes.
                    if (!ReferenceEquals(moved, result))
                        result.Dispose();
                    result = moved;
                }

                // return result.astype(dtype, copy=False)
                NDArray ret = CastSpacingResult(result, dtype);
                result = null; // ownership left with ret (CastSpacingResult released a replaced array)
                return ret;
            }
            finally
            {
                // The working copies and stages; `result` is non-null here only when a statement failed.
                s?.Dispose();
                e?.Dispose();
                outSign?.Dispose();
                logStart?.Dispose();
                logStop?.Dispose();
                y?.Dispose();
                result?.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's geomspace tail <c>if axis != 0: result = _nx.moveaxis(result, 0, axis)</c> on a 1-D result: a view of
        ///     the same buffer, which the final <c>result.astype(dtype, copy=False)</c> returns as it is when the dtype
        ///     already matches - so <c>np.geomspace(1, 1000, 4, axis=-1)</c> does not own its data, where axis 0 does.
        /// </summary>
        /// <param name="result">The fresh 1-D result. Ownership transfers in: it is returned itself for axis 0, and disposed
        ///     when the view replaces it (the view keeps the buffer alive).</param>
        /// <param name="axis">The destination, already validated (0 or -1).</param>
        /// <returns><paramref name="result"/> for axis 0; otherwise a new alias of its buffer that does not own it.</returns>
        private static NDArray SpacingAxisView(NDArray result, int axis)
        {
            if (axis == 0)
                return result;
            // moveaxis of a 1-D array is the same layout over the same buffer; only the ownership flag tells them apart.
            var view = new NDArray(result.Storage.Alias()) { TensorEngine = result.TensorEngine };
            result.Dispose();
            return view;
        }

        /// <summary>
        ///     NumPy's geomspace statements after the zero check, for a text dtype - they always raise, before the sample
        ///     count and the axis are read: <c>dt = result_type(start, stop, float(num), zeros((), dtype))</c> finds no
        ///     common DType for <c>StringDType</c>; <c>S</c> and <c>U</c> promote to text, NumPy casts the ends to it
        ///     (<c>start.astype(dt, copy=True)</c>), and <c>out_sign = sign(start)</c> has no string loop.
        /// </summary>
        /// <param name="start">The first end (the caller's; only read).</param>
        /// <param name="startDType">The dtype NumPy's <c>asanyarray(start)</c> has (<see cref="GeomspaceEndDType"/>).</param>
        /// <param name="stopDType">The dtype NumPy's <c>asanyarray(stop)</c> has.</param>
        /// <param name="num">The sample count, which NumPy promotes as <c>float(num)</c>, a weak Python float.</param>
        /// <param name="dtype">The text dtype (the String storage lane).</param>
        /// <exception cref="DTypePromotionError">A <c>StringDType</c>: NumPy's text, which lists the four DType classes
        ///     (<c>The DType &lt;class 'numpy.dtypes.StringDType'&gt; could not be promoted by &lt;class
        ///     'numpy.dtypes.Float64DType'&gt;. ...</c>).</exception>
        /// <exception cref="TypeError">An <c>S</c> or <c>U</c> dtype: sign's refusal (<c>ufunc 'sign' did not contain a loop
        ///     with signature matching types &lt;class 'numpy.dtypes.StrDType'&gt; -&gt; None</c>, a UFuncTypeError in
        ///     NumPy).</exception>
        /// <exception cref="InvalidOperationException">Never: np.sign computing text would be a library bug.</exception>
        private void RefuseGeomspaceTextDtype(NDArray start, DType startDType, DType stopDType, long num, DType dtype)
        {
            // The ends' arrays are strong, float(num) is weak, the dtype's 0-d zeros strong: DTypePromotion.ResultType takes
            // the same operands as NumPy's call, in its order (the DTypePromotionError text lists them).
            DType dt = DTypePromotion.ResultType(startDType, stopDType, (double)num, dtype);
            // start.astype(dt, copy=True), then out_sign = sign(start): the text copy is this check's own temporary.
            using (NDArray text = start.astype(dt))
            using (NDArray sign = Sign(text))
            {
            }
            throw new InvalidOperationException("np.sign computed the sign of text; NumPy has no string loop for it.");
        }

        /// <summary>
        ///     The bit-exact float64 core: fills a fresh 1-D buffer following NumPy's geomspace fused into ONE pass —
        ///     the sign rotation, the log-space interior (<c>out_sign * 10 ** (log_start + i*step)</c>), and the exact
        ///     endpoint overwrites — with NO intermediate arrays.
        /// </summary>
        /// <param name="start">The (non-zero) starting value.</param>
        /// <param name="stop">The (non-zero) final value.</param>
        /// <param name="num">Number of samples (already validated non-negative).</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the final sample.</param>
        /// <returns>A fresh, C-contiguous, owning float64 array of length <paramref name="num"/>.</returns>
        /// <remarks>
        ///     out_sign is ±1.0 (sign of a non-zero real), so writing the ORIGINAL <paramref name="start"/> /
        ///     <paramref name="stop"/> at the endpoints equals NumPy's <c>(x/out_sign)*out_sign</c> byte-for-byte, and
        ///     <c>out_sign * Math.Pow(10, log_start + i*step)</c> equals NumPy's <c>power(10, linspace_val) * out_sign</c>
        ///     (multiplication commutes; ±1 is exact). Mixed-sign inputs give NaN interiors exactly as NumPy does.
        /// </remarks>
        private NDArray GeomspaceRealCore(double start, double stop, long num, bool endpoint)
        {
            // out_sign = np.sign(start): ±1 for a finite/inf non-zero, but NaN for a NaN start (start != 0 is
            // guaranteed and -0.0 was rejected by the ==0.0 check). Propagating NaN here is load-bearing — NumPy's
            // trailing `result *= out_sign` turns EVERY element (endpoints included) into NaN for a NaN start.
            double outSign = double.IsNaN(start) ? start : start < 0.0 ? -1.0 : 1.0;
            double startR = start / outSign;                 // |start| on the positive real axis (NaN if start is NaN)
            double stopR = stop / outSign;                   // may be negative if start,stop have opposite signs
            double logStart = Math.Log10(startR);
            double logStop = Math.Log10(stopR);              // NaN when stopR < 0 (mixed-sign inputs)

            NDArray ret = NewArray(NPTypeCode.Double, new Shape(num), false);
            if (num == 0)
                return ret;

            // step is undefined for num==1 (div==0); the loop never reads it there (only the i==0 endpoint runs).
            double step = num > 1 ? (logStop - logStart) / (endpoint ? num - 1.0 : num) : 0.0;

            unsafe
            {
                double* addr = (double*)ret.Address;
                for (long i = 0; i < num; i++)
                {
                    // NumPy computes the ROTATED sequence (startR at 0, stopR at the endpoint, 10**exponent in
                    // between) and only THEN does `result *= out_sign` — so the multiply is applied UNIFORMLY,
                    // including the endpoints. For out_sign = ±1 this is bit-identical to writing the original
                    // start/stop (startR*out_sign == start exactly — double negation is exact), and for out_sign =
                    // NaN it correctly yields NaN everywhere, matching NumPy's NaN-start behaviour.
                    double baseV;
                    if (i == 0)
                        baseV = startR;
                    else if (endpoint && i == num - 1)
                        baseV = stopR;
                    else
                        baseV = Math.Pow(10.0, logStart + i * step);
                    addr[i] = baseV * outSign;
                }
            }

            return ret;
        }

        /// <summary>
        ///     The complex128 core (spiral): composes NumSharp's NumPy-tuned complex power over a complex linspace of
        ///     log-space exponents, then applies the exact endpoints and the sign rotation in one in-place pass.
        /// </summary>
        /// <param name="start">The (non-zero) complex starting value.</param>
        /// <param name="stop">The (non-zero) complex final value.</param>
        /// <param name="num">Number of samples (already validated non-negative).</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the final sample.</param>
        /// <param name="dtype">The requested output dtype (null → complex128); applied as the final cast.</param>
        /// <param name="axis">The validated moveaxis destination (0 or -1): -1 makes the complex128 result NumPy's view
        ///     (<see cref="SpacingAxisView"/>) before the cast.</param>
        /// <returns>A fresh array of length <paramref name="num"/> in the requested dtype; it does not own its buffer when
        ///     <paramref name="axis"/> is -1 and no cast copies it, as NumPy's moveaxis view does not.</returns>
        /// <remarks>
        ///     Accurate but not byte-reproducible (see the class comment). Follows NumPy's exact operation order:
        ///     result[0]=start_r / result[-1]=stop_r are set, THEN the whole array is multiplied by out_sign — so the
        ///     endpoints become <c>start_r*out_sign</c> / <c>stop_r*out_sign</c> (not necessarily the original inputs,
        ///     since complex divide-then-multiply is not exact), matching NumPy; then the moveaxis and the cast.
        /// </remarks>
        private NDArray GeomspaceComplexCore(Complex start, Complex stop, long num, bool endpoint, DType dtype, int axis)
        {
            // sign(z) = z / |z| (NumPy 2.x complex sign); z != 0 guaranteed by the caller.
            Complex outSign = start / Complex.Abs(start);
            Complex startR = start / outSign;
            Complex stopR = stop / outSign;
            Complex logStart = NDComplexMath.Log10(startR);
            Complex logStop = NDComplexMath.Log10(stopR);

            // Complex linspace of the log-space exponents (a straight line between the two log10 values).
            NDArray cLin = ComplexLinspaceCore(logStart, logStop, num, endpoint);
            // 10 ** exponent through NumSharp's npy_cpow port (best parity with NumPy's own complex power).
            NDArray base10 = NewScalar(new Complex(10.0, 0.0));
            // np.power(base10, cLin): the operand's engine member (a static helper).
            NDArray result = base10.TensorEngine.Power(base10, cLin);
            // cLin/base10 were pure inputs to power (which materialises a fresh result); reclaim them now.
            base10.Dispose();
            cLin.Dispose();

            if (num > 0)
            {
                unsafe
                {
                    Complex* addr = (Complex*)result.Address;
                    for (long i = 0; i < num; i++)
                    {
                        // Endpoints take the ROTATED start/stop; the sign multiply below then un-rotates them, exactly
                        // as NumPy's `result[0]=start; result[-1]=stop; result *= out_sign` does.
                        Complex v = i == 0 ? startR
                                  : endpoint && i == num - 1 ? stopR
                                  : addr[i];
                        addr[i] = v * outSign;
                    }
                }
            }

            // NumPy's tail: the moveaxis view for axis != 0, then astype(dtype, copy=False).
            result = SpacingAxisView(result, axis);
            return CastSpacingResult(result, dtype);
        }

        /// <summary>
        ///     A complex128 <c>linspace(start, stop, num, endpoint)</c> fill loop (used only by the geomspace spiral):
        ///     <c>start + i*step</c> with the last sample pinned to <paramref name="stop"/> under <paramref name="endpoint"/>.
        /// </summary>
        /// <param name="start">The first sample.</param>
        /// <param name="stop">The last sample when <paramref name="endpoint"/> is True.</param>
        /// <param name="num">Number of samples (already validated non-negative).</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the final sample.</param>
        /// <returns>A fresh, C-contiguous, owning complex128 array of length <paramref name="num"/>.</returns>
        private NDArray ComplexLinspaceCore(Complex start, Complex stop, long num, bool endpoint)
        {
            NDArray ret = NewArray(NPTypeCode.Complex, new Shape(num), false);
            if (num == 0)
                return ret;
            if (num == 1)
            {
                // linspace(a, b, 1) == [a] for either endpoint (mirrors the real linspace core).
                ret.SetAtIndex(start, 0);
                return ret;
            }

            Complex step = (stop - start) / (endpoint ? num - 1.0 : num);
            unsafe
            {
                Complex* addr = (Complex*)ret.Address;
                for (long i = 0; i < num; i++)
                    addr[i] = endpoint && i == num - 1 ? stop : start + (double)i * step;
            }

            return ret;
        }
    }
}
