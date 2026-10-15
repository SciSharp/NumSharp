using System;
using System.Globalization;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    public static partial class NDArrayPythonInterop
    {
        // =====================================================================================================
        //  Text that cannot be a view: NEP 55 StringDType (T) arrays, object arrays of str, and numpy's text
        //  scalars. The fixed-width S / U arrays are NOT here — they are ordinary buffers and travel through the
        //  numeric view/copy paths with a sized descriptor.
        //
        //  A StringDType array crosses as an exact COPY (owner's decision D7, 2026-10-13): each element is
        //  duplicated with NumPy's NpyString_dup rules (NumpyStringApi) — an NA stays NA, an empty string stays
        //  empty, a short (<= 15-byte) string moves as its 16-byte cell, a longer one is loaded from the source's
        //  allocator and packed into the destination's. Both sides port static_string.c (NumSharp's
        //  VStringAllocator statement for statement), so the cell layouts agree and the destination's cells are
        //  the ones the side's own copy would build (numpy's arr.copy() for an import, np.array(list, dtype) for
        //  an export).
        // =====================================================================================================

        /// <summary>
        ///     Whether <paramref name="obj"/> is a numpy array of NEP 55 <c>StringDType</c> (its descriptor's type character is
        ///     <c>'T'</c>). Such an array exports no buffer, so the buffer paths fail on it before this check runs; asked only
        ///     then, it costs the numeric paths nothing. Call under the GIL.
        /// </summary>
        /// <param name="obj">The Python object that refused a <c>memoryview</c>.</param>
        /// <returns>True for a numpy <c>StringDType</c> array (or subclass).</returns>
        /// <remarks>
        ///     The order keeps the buffer route numpy-agnostic and the check cheap (about 0.7 µs): an object without a
        ///     <c>dtype</c> attribute is rejected before numpy is touched, the descriptor's type character is read next, and
        ///     only a <c>'T'</c> descriptor — which exists only where numpy 2 is loaded — leads to the <c>isinstance</c> test
        ///     against <c>numpy.ndarray</c>. That test is what makes the copy safe: it reads the array's shape, strides, data
        ///     address and descriptor allocator, which another object carrying a <c>StringDType</c> attribute does not have.
        ///     (<c>hasattr(obj, '__array_interface__')</c>, the earlier test, builds the whole interface dict.)
        /// </remarks>
        private static bool IsStringDTypeArray(PyObject obj)
        {
            if (!obj.HasAttr(PythonRuntimeInterop.NameDtype))
                return false;
            using PyObject dtype = obj.GetAttr(PythonRuntimeInterop.NameDtype);
            if (!dtype.HasAttr(PythonRuntimeInterop.NameChar))
                return false;
            using PyObject typeChar = dtype.GetAttr(PythonRuntimeInterop.NameChar);
            return PyString.IsStringType(typeChar) && typeChar.As<string>() == "T" && obj.IsInstance(PythonRuntimeInterop.NpNdarray);
        }

        /// <summary>
        ///     Copies a numpy <c>StringDType</c> array of any layout (C, F, strided, reversed, broadcast, 0-d, empty) into a
        ///     fresh C-contiguous NumSharp <c>StringDType</c> array with the same shape and the same descriptor parameters
        ///     (<c>na_object</c>, <c>coerce</c>). Call under the GIL.
        /// </summary>
        /// <param name="array">The numpy array (kept alive by the caller for the call).</param>
        /// <param name="descrOverride">The NumSharp descriptor to give the result instead of the source's (the object-array
        ///     route, whose temporary carries a validation-only <c>coerce=False</c>); null takes the source's.</param>
        /// <returns>The owning NumSharp copy; every element is the source element (NA stays NA).</returns>
        /// <exception cref="NotSupportedException">The running numpy has no <c>StringDType</c>, or the descriptor's
        ///     <c>na_object</c> has no C# form (see <see cref="MissingValueFromPython"/>).</exception>
        /// <exception cref="InvalidOperationException">numpy could not read one of its own cells (a corrupt array).</exception>
        /// <exception cref="OutOfMemoryException">NumSharp's allocator could not hold the text (NumPy's <c>MemoryError</c>
        ///     texts); the half-built result is disposed.</exception>
        private static unsafe NDArray ImportStringDType(PyObject array, DType descrOverride = null)
        {
            NumpyStringApi.EnsureAvailable();
            using PyObject numpyDtype = array.GetAttr(PythonRuntimeInterop.NameDtype);
            DType descr = descrOverride ?? StringDTypeFromNumpy(numpyDtype);
            long[] dims = GetLongTuple(array, PythonRuntimeInterop.NameShape);
            long[] byteStrides = GetLongTuple(array, PythonRuntimeInterop.NameStrides);
            Shape shape = dims.Length == 0 ? new Shape() : new Shape(dims);

            // A StringDType allocation is always zero-filled (an all-zero cell is the empty string), which the copy below
            // relies on: every destination cell starts fresh, so no old text needs freeing.
            var result = new NDArray(descr, shape);
            try
            {
                if (shape.size == 0)
                    return result;
                byte* source = NumpyStringApi.DataAddress(array, dims.Length);
                VStringArraySlice destination = result.Storage.VStringSlice;
                VStringAllocator outAllocator = destination.Allocator;
                IntPtr npAllocator = NumpyStringApi.Acquire(numpyDtype);
                try
                {
                    outAllocator.Acquire();
                    try
                    {
                        // Pass 1 settles every cell that allocates nothing (NA, empty, short) and sums the arena bytes the
                        // others need; pass 2 runs only when there are such cells, after one exact reservation. Only the
                        // long cells allocate, and pass 2 visits them in the same C order a single pass would, so every
                        // arena offset is the one numpy's own copy assigns (the reservation changes capacity, never an
                        // offset: see VStringAllocator.TryReserve).
                        var settle = new ImportSettleVisitor(outAllocator);
                        WalkCells(source, dims, byteStrides, destination.Address, ref settle);
                        if (settle.LongCells > 0)
                        {
                            outAllocator.TryReserve(settle.ArenaBytes);
                            var duplicate = new ImportLongVisitor(npAllocator, outAllocator);
                            WalkCells(source, dims, byteStrides, destination.Address, ref duplicate);
                        }
                    }
                    finally
                    {
                        outAllocator.Release();
                    }
                }
                finally
                {
                    NumpyStringApi.Release(npAllocator);
                }

                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Copies a NumSharp <c>StringDType</c> array of any layout into a fresh C-contiguous numpy <c>StringDType</c> array
        ///     with the same shape and the same descriptor parameters. Call under the GIL.
        /// </summary>
        /// <param name="source">The NumSharp array (its buffer is guarded by the caller for the call).</param>
        /// <returns>The owning numpy copy (owned by the caller).</returns>
        /// <exception cref="NotSupportedException">The running numpy has no <c>StringDType</c>, or the descriptor's
        ///     <c>na_object</c> has no Python form (see <see cref="MissingValueToPython"/>).</exception>
        /// <exception cref="InvalidOperationException">NumSharp could not read one of its own cells (a corrupt array).</exception>
        /// <exception cref="OutOfMemoryException">numpy's allocator could not hold the text; the half-built numpy array is
        ///     released.</exception>
        private static unsafe PyObject ExportStringDType(NDArray source)
        {
            NumpyStringApi.EnsureAvailable();
            UnmanagedStorage storage = source.Storage;
            VStringArraySlice cells = storage.VStringSlice;
            Shape shape = storage.Shape;

            PyObject result;
            using (PyObject numpyDtype = StringDTypeToNumpy(source.dtype))
            using (PyTuple dims = MakeTuple(shape.dimensions))
                result = np.empty(dims, numpyDtype);
            try
            {
                if (shape.size == 0)
                    return result;

                // NumSharp strides count cells; the walk takes byte strides.
                var sourceStrides = new long[shape.strides.Length];
                for (int i = 0; i < sourceStrides.Length; i++)
                    sourceStrides[i] = shape.strides[i] * VStringAllocator.CellSize;
                byte* first = storage.Address + shape.offset * VStringAllocator.CellSize;
                byte* destination = NumpyStringApi.DataAddress(result, shape.NDim);

                // numpy gives every new array a descriptor INSTANCE of its own (its allocator lives there), so the allocator
                // to lock is the result's, not the one passed to np.empty.
                using PyObject resultDtype = result.GetAttr(PythonRuntimeInterop.NameDtype);
                IntPtr npAllocator = NumpyStringApi.Acquire(resultDtype);
                try
                {
                    cells.Allocator.Acquire();
                    try
                    {
                        var copy = new ExportVisitor(cells.Allocator, npAllocator);
                        WalkCells(first, shape.dimensions, sourceStrides, destination, ref copy);
                    }
                    finally
                    {
                        cells.Allocator.Release();
                    }
                }
                finally
                {
                    NumpyStringApi.Release(npAllocator);
                }

                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Imports a numpy OBJECT array whose every element is a Python <c>str</c> as a NumSharp <c>StringDType()</c> copy
        ///     (owner's decision, 2026-10-13: object arrays are refused, except an array of <c>str</c>, which is text that
        ///     cannot be viewed and so is copied). Call under the GIL.
        /// </summary>
        /// <param name="array">The numpy object array.</param>
        /// <returns>The owning NumSharp copy, descriptor <c>StringDType()</c> (no missing-data object, coercion on).</returns>
        /// <remarks>
        ///     <para>
        ///     numpy's own conversion validates and converts in one C pass: <c>astype(StringDType(coerce=False))</c> accepts an
        ///     exact <c>str</c> or <c>np.str_</c> element and raises for any other object — <c>None</c>, NaN, numbers,
        ///     <c>pd.NA</c> — so nothing is coerced to text and no missing-value object is guessed.
        ///     </para>
        ///     <para>
        ///     A <c>str</c> SUBCLASS is a <c>str</c> too, but numpy's <c>coerce=False</c> refuses it (<c>as_pystring</c> takes
        ///     only the exact type). When that cast fails and every element is still a <c>str</c> instance, the array goes
        ///     through numpy's coercing cast instead, which stores <c>str(x)</c> of a subclass element — what
        ///     <c>np.array(arr, dtype=StringDType())</c> stores, so a subclass that overrides <c>__str__</c> crosses as that
        ///     text. Nothing that is not a <c>str</c> reaches the coercing cast.
        ///     </para>
        /// </remarks>
        /// <exception cref="NotSupportedException">An element is not a <c>str</c> (the message names the first one by flat
        ///     C-order index, type and repr), or a <c>str</c> holds a lone surrogate (UTF-8 cannot encode it, so neither
        ///     numpy nor NumSharp can store it), or a <c>str</c> subclass's <c>__str__</c> raises, or the running numpy has no
        ///     <c>StringDType</c>.</exception>
        private static NDArray ImportObjectArrayOfStr(PyObject array)
        {
            NumpyStringApi.EnsureAvailable();
            PyObject converted;
            try
            {
                converted = ObjectArrayAsStringDType(array, coerce: false);
            }
            catch (PythonException strictError)
            {
                string notStr = DescribeFirstElementThatIsNotStr(array, strictError.Message);
                if (notStr != null)
                    throw new NotSupportedException(notStr + ObjectArrayGuidance, strictError);
                try
                {
                    // Every element is a str instance: the strict cast refused a str subclass (or text UTF-8 cannot hold).
                    converted = ObjectArrayAsStringDType(array, coerce: true);
                }
                catch (PythonException e)
                {
                    string how = e.Type?.Name == "UnicodeEncodeError" ? "as UTF-8" : "as text";
                    throw new NotSupportedException(
                        $"the object array holds only str elements, but numpy cannot store one of them {how} ({e.Message})." +
                        ObjectArrayGuidance, e);
                }
            }

            using (converted)
                return ImportStringDType(converted, global::NumSharp.np.dtypes.StringDType());
        }

        /// <summary>The guidance every object-array refusal ends with: what crosses, and the conversion that makes the rest
        ///     cross.</summary>
        private const string ObjectArrayGuidance =
            " Only an object array whose every element is a str crosses (as a StringDType copy); " +
            "convert it in Python first, e.g. np.array(arr, dtype=np.dtypes.StringDType(na_object=None)).";

        /// <summary>
        ///     <c>array.astype(StringDType(coerce=coerce))</c> — numpy's own conversion of an object array to text. Call under
        ///     the GIL.
        /// </summary>
        /// <param name="array">The numpy object array.</param>
        /// <param name="coerce">numpy's <c>coerce</c>: false takes only an exact <c>str</c> or <c>np.str_</c> element; true
        ///     stores <c>str(x)</c> of anything else.</param>
        /// <returns>The new numpy <c>StringDType()</c> array (owned by the caller).</returns>
        /// <exception cref="PythonException">numpy refuses an element (its error, unchanged).</exception>
        private static PyObject ObjectArrayAsStringDType(PyObject array, bool coerce)
        {
            using PyObject astype = array.GetAttr(PythonRuntimeInterop.NameAstype);
            using PyObject descr = NewNumpyStringDType(hasNA: false, na: null, coerce: coerce);
            return astype.Invoke(descr);
        }

        /// <summary>
        ///     The refusal text of an object array that is not an array of <c>str</c>: the first element that is not a
        ///     <c>str</c> instance (flat C-order index, type and a short repr). Runs only on the failure path.
        /// </summary>
        /// <param name="array">The numpy object array.</param>
        /// <param name="numpyError">numpy's message from the failed conversion (for an element whose <c>repr()</c> raises).</param>
        /// <returns>The text without the guidance, or <see langword="null"/> when every element is a <c>str</c> instance (a
        ///     <c>str</c> subclass or <c>np.str_</c> included).</returns>
        private static string DescribeFirstElementThatIsNotStr(PyObject array, string numpyError)
        {
            try
            {
                using PyModule scope = Py.CreateScope();
                scope.Set("a", array);
                scope.Exec(
                    "_bad = next(((i, type(x).__name__, repr(x)) for i, x in enumerate(a.flat) if not isinstance(x, str)), None)\n" +
                    "_bad = None if _bad is None else (_bad[0], _bad[1], _bad[2] if len(_bad[2]) <= 60 else _bad[2][:57] + '...')\n");
                using PyObject bad = scope.Get("_bad");
                if (bad.IsNone())
                    return null;
                using var tuple = PyTuple.AsTuple(bad);
                using PyObject index = tuple[0];
                using PyObject typeName = tuple[1];
                using PyObject repr = tuple[2];
                return $"the object array holds a {typeName.As<string>()} at flat index {index.As<long>()} ({repr.As<string>()}), which is not a str.";
            }
            catch (PythonException)
            {
                // The description is a courtesy; the refusal itself must survive an object whose repr() raises (only a
                // non-str element's repr is taken, so one exists).
                return $"the object array holds an element that is not a str ({numpyError}).";
            }
        }

        // -----------------------------------------------------------------------------------------------------
        //  numpy's text scalars: np.str_ exports its own 0-d UCS-4 buffer (format '<n>w', shape ()), so it maps to
        //  a 0-d U<n> through the ordinary paths. np.bytes_ is a bytes SUBCLASS and exports the bytes' buffer
        //  (format 'B', shape (n,)), but numpy reads it as the 0-d |S<n> it is (np.asarray(np.bytes_(b'ab')) is
        //  dtype 'S2', 0-d): the bridge does the same, where a plain bytes object stays raw uint8 bytes.
        // -----------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Whether <paramref name="source"/> is a numpy <c>bytes_</c> scalar whose buffer (<paramref name="format"/>,
        ///     <paramref name="dims"/>) is its byte string — then numpy's reading is a 0-d <c>S&lt;n&gt;</c>.
        /// </summary>
        /// <param name="source">The exporter.</param>
        /// <param name="format">Its <c>memoryview.format</c>.</param>
        /// <param name="dims">Its <c>memoryview.shape</c>.</param>
        /// <returns>True for a <c>numpy.bytes_</c> object exporting its 1-D byte buffer.</returns>
        private static bool IsNumpyBytesScalar(PyObject source, string format, long[] dims)
        {
            // The format/shape test first: only a 1-D unsigned-byte buffer can be one, so other exporters pay nothing more.
            if (format != "B" || dims is null || dims.Length != 1)
                return false;
            using PyType type = source.GetPythonType();
            return type.Name == "numpy.bytes_";
        }

        /// <summary>
        ///     The descriptor a zero-width text element is COPIED as: numpy grows an empty <c>np.str_</c> /
        ///     <c>np.bytes_</c> to one character (<c>np.asarray(np.str_(''))</c> is <c>&lt;U1</c>), and a NumSharp string
        ///     element needs a positive width too. A view is impossible (there is no byte to share).
        /// </summary>
        /// <param name="descr">A text descriptor, possibly of width 0.</param>
        /// <returns>The descriptor itself, or its one-character form when it is unsized.</returns>
        private static DType GrowZeroWidthText(DType descr)
            => descr.Meta is StringDTypeMeta text && descr.itemsize == 0 ? TextDescr(text.IsUnicode ? 'U' : 'S', 1) : descr;

        // -----------------------------------------------------------------------------------------------------
        //  Descriptor parameters: numpy's StringDType(na_object=?, coerce=?) <-> NumSharp's np.dtypes.StringDType.
        //  The missing-data object crosses only in the C# forms NumSharp's boundary map gives Python objects
        //  (decision D5): None <-> null, bool, int <-> long / BigInteger, float <-> double, complex <-> Complex,
        //  str <-> string, bytes <-> byte[], np.float16 <-> Half. Anything else (pd.NA, an arbitrary object, a
        //  decimal or container on the NumSharp side) is refused (owner's decision, 2026-10-13).
        // -----------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The NumSharp descriptor of a numpy <c>StringDType</c> instance: the same <c>coerce</c>, and the same missing-data
        ///     object (or none — "not given" stays different from <c>None</c>). Call under the GIL.
        /// </summary>
        /// <param name="numpyDtype">The numpy descriptor.</param>
        /// <returns>The equivalent NumSharp descriptor.</returns>
        /// <exception cref="NotSupportedException">The missing-data object has no C# form.</exception>
        private static DType StringDTypeFromNumpy(PyObject numpyDtype)
        {
            bool coerce;
            using (PyObject flag = numpyDtype.GetAttr(PythonRuntimeInterop.NameCoerce))
                coerce = flag.IsTrue();
            // numpy omits the attribute entirely when no missing-data object was given; None is a real one.
            if (!numpyDtype.HasAttr(PythonRuntimeInterop.NameNaObject))
                return global::NumSharp.np.dtypes.StringDType(coerce);
            using PyObject na = numpyDtype.GetAttr(PythonRuntimeInterop.NameNaObject);
            return global::NumSharp.np.dtypes.StringDType(MissingValueFromPython(na), coerce);
        }

        /// <summary>
        ///     The numpy <c>StringDType</c> instance with a NumSharp descriptor's parameters. Call under the GIL.
        /// </summary>
        /// <param name="descr">A NumSharp <c>StringDType</c> descriptor.</param>
        /// <returns>A new numpy descriptor (owned by the caller).</returns>
        /// <exception cref="NotSupportedException">The missing-data object has no Python form.</exception>
        private static PyObject StringDTypeToNumpy(DType descr)
        {
            var parameters = descr.VString;
            if (!parameters.HasNAObject)
                return NewNumpyStringDType(hasNA: false, na: null, parameters.coerce);
            PyObject na = MissingValueToPython(parameters.na_object);
            try
            {
                return NewNumpyStringDType(hasNA: true, na, parameters.coerce);
            }
            finally
            {
                // None is the session's cached literal; every other object was made here.
                if (!ReferenceEquals(na, PythonRuntimeInterop.NoneLiteral))
                    na.Dispose();
            }
        }

        /// <summary>
        ///     <c>numpy.dtypes.StringDType(na_object=na, coerce=coerce)</c>, or <c>StringDType(coerce=coerce)</c> without a
        ///     missing-data object. Call under the GIL.
        /// </summary>
        /// <param name="hasNA">Whether to pass <c>na_object</c>.</param>
        /// <param name="na">The missing-data object (ignored when <paramref name="hasNA"/> is false; not disposed here).</param>
        /// <param name="coerce">numpy's <c>coerce</c>.</param>
        /// <returns>The new numpy descriptor (owned by the caller).</returns>
        private static PyObject NewNumpyStringDType(bool hasNA, PyObject na, bool coerce)
        {
            // The default descriptor needs no keyword dict (building one costs about as much as the call itself).
            if (!hasNA && coerce)
                return PythonRuntimeInterop.NpStringDType.Invoke();
            using var kwargs = new PyDict();
            if (hasNA)
                kwargs[PythonRuntimeInterop.NameNaObject] = na;
            kwargs[PythonRuntimeInterop.NameCoerce] = coerce ? PythonRuntimeInterop.TrueLiteral : PythonRuntimeInterop.FalseLiteral;
            return PythonRuntimeInterop.NpStringDType.Invoke(Array.Empty<PyObject>(), kwargs);
        }

        /// <summary>
        ///     The C# form of a Python missing-data object (the boundary map of decision D5). Call under the GIL.
        /// </summary>
        /// <param name="na">The Python object.</param>
        /// <returns><see langword="null"/> for <c>None</c>; a <see cref="bool"/>, <see cref="long"/> (or
        ///     <see cref="BigInteger"/> past it), <see cref="double"/>, <see cref="Complex"/>, <see cref="string"/>,
        ///     <c>byte[]</c> or <see cref="Half"/> otherwise.</returns>
        /// <remarks>
        ///     A numpy float64 / complex128 / str_ / bytes_ object maps like the Python type it subclasses (its <c>str()</c> and
        ///     its equality — the two things a missing-data object is used for — are the same; only the descriptor's repr
        ///     loses the <c>np.</c> prefix), and numpy itself calls such descriptors equal.
        /// </remarks>
        /// <exception cref="NotSupportedException">Any other object (<c>pd.NA</c>, a user class, a tuple, …): it has no C#
        ///     form, so the descriptor cannot cross.</exception>
        internal static object MissingValueFromPython(PyObject na)
        {
            if (na.IsNone())
                return null;
            string typeName;
            using (PyType type = na.GetPythonType())
                typeName = type.Name;
            switch (typeName)
            {
                case "bool":
                    return na.IsTrue();
                case "int":
                {
                    var value = BigInteger.Parse(na.ToString(), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
                    return value >= long.MinValue && value <= long.MaxValue ? (object)(long)value : value;
                }
                case "float":
                case "numpy.float64":
                    return na.As<double>();
                case "numpy.float16":
                {
                    // pythonnet converts only a Python float (or subclass) to double, and np.float16 is neither: go through
                    // Python's float(). Every float16 is exactly a double, so the round trip is exact (NaN stays NaN).
                    using PyObject asFloat = PythonRuntimeInterop.Builtins.GetAttr("float");
                    using PyObject value = asFloat.Invoke(na);
                    return (Half)value.As<double>();
                }
                case "complex":
                case "numpy.complex128":
                {
                    using PyObject real = na.GetAttr(PythonRuntimeInterop.NameReal);
                    using PyObject imag = na.GetAttr(PythonRuntimeInterop.NameImag);
                    return new Complex(real.As<double>(), imag.As<double>());
                }
                case "str":
                    return na.As<string>();
                case "numpy.str_":
                {
                    // pythonnet's string conversion compares the object's type with str exactly, so it refuses a subclass
                    // ("'numpy.str_' value cannot be converted to System.String"): str() returns the text as a plain str.
                    using PyObject asStr = PythonRuntimeInterop.Builtins.GetAttr("str");
                    using PyObject value = asStr.Invoke(na);
                    return value.As<string>();
                }
                case "bytes":
                case "numpy.bytes_":
                {
                    using PyObject hex = na.InvokeMethod("hex");
                    return Convert.FromHexString(hex.As<string>());
                }
                default:
                    throw new NotSupportedException(
                        $"the StringDType's na_object {na.Repr()} ({typeName}) has no C# form in NumSharp (None, bool, int, float, complex, str, " +
                        "bytes and np.float16 do), so the array cannot cross with it. Convert it in Python first, e.g. " +
                        "arr.astype(np.dtypes.StringDType(na_object=None)) or arr.astype(np.dtypes.StringDType(na_object=np.nan)).");
            }
        }

        /// <summary>
        ///     The Python form of a NumSharp missing-data object (the reverse of <see cref="MissingValueFromPython"/>). Call
        ///     under the GIL.
        /// </summary>
        /// <param name="na">The C# value (a <see cref="char"/> object was already normalised to a one-character string by the
        ///     descriptor).</param>
        /// <returns>A new Python object owned by the caller, or the session's <c>None</c> literal (never dispose it).</returns>
        /// <exception cref="NotSupportedException">A value with no Python form (a <see cref="decimal"/>, an
        ///     <see cref="NDArray"/>, a container, an arbitrary object).</exception>
        internal static PyObject MissingValueToPython(object na)
        {
            switch (na)
            {
                case null:
                    return PythonRuntimeInterop.NoneLiteral;
                case bool b:
                    return b.ToPython();
                case sbyte or byte or short or ushort or int or uint or long:
                    return new PyInt(Convert.ToInt64(na, CultureInfo.InvariantCulture));
                case ulong u:
                    return new PyInt(new BigInteger(u));
                case BigInteger big:
                    return new PyInt(big);
                case float f:
                    return new PyFloat(f);
                case double d:
                    return new PyFloat(d);
                case Half h:
                {
                    using PyObject float16 = PythonRuntimeInterop.Numpy.GetAttr("float16");
                    using var value = new PyFloat((double)h);
                    return float16.Invoke(value);
                }
                case Complex z:
                {
                    using PyObject complex = PythonRuntimeInterop.Builtins.GetAttr("complex");
                    using var real = new PyFloat(z.Real);
                    using var imag = new PyFloat(z.Imaginary);
                    return complex.Invoke(real, imag);
                }
                case string s:
                    return new PyString(s);
                case byte[] bytes:
                {
                    using PyObject type = PythonRuntimeInterop.Builtins.GetAttr("bytes");
                    using PyObject fromHex = type.GetAttr("fromhex");
                    using var hex = new PyString(Convert.ToHexString(bytes));
                    return fromHex.Invoke(hex);
                }
                default:
                    throw new NotSupportedException(
                        $"the StringDType's na_object {na} ({na.GetType().Name}) has no Python form (None, bool, integers, float, double, Complex, " +
                        "string, byte[] and Half do), so the array cannot cross with it. Convert it first, e.g. " +
                        "nd.astype(np.dtypes.StringDType((object)null)).");
            }
        }

        // -----------------------------------------------------------------------------------------------------
        //  The cell walk: the SOURCE is visited in C order through arbitrary byte strides (numpy's strides for an
        //  import, NumSharp's cell strides x 16 for an export), the DESTINATION is a fresh C-contiguous run of
        //  16-byte cells. A struct visitor keeps the inner loop free of delegate calls (the JIT specializes the
        //  generic walk per visitor).
        // -----------------------------------------------------------------------------------------------------

        /// <summary>One step of a cell walk: a source cell and the destination cell it maps to.</summary>
        private unsafe interface ICellVisitor
        {
            /// <summary>Handles one source cell.</summary>
            /// <param name="source">The source cell.</param>
            /// <param name="destination">The destination cell (fresh, all zero before the walk).</param>
            void Visit(byte* source, byte* destination);
        }

        /// <summary>
        ///     Visits every element of a strided source in C order, pairing it with consecutive 16-byte destination cells.
        /// </summary>
        /// <typeparam name="TVisitor">The per-cell action (a struct, so the call is specialized and inlined).</typeparam>
        /// <param name="source">The source's element 0.</param>
        /// <param name="dims">The source's dimensions (empty for 0-d; no dimension is 0 — the callers return earlier).</param>
        /// <param name="byteStrides">The source's strides in BYTES (0 broadcasts, negative walks backward).</param>
        /// <param name="destination">The first destination cell.</param>
        /// <param name="visitor">The action, by reference (it accumulates state).</param>
        private static unsafe void WalkCells<TVisitor>(byte* source, long[] dims, long[] byteStrides, byte* destination, ref TVisitor visitor)
            where TVisitor : struct, ICellVisitor
        {
            int nd = dims.Length;
            long inner = nd == 0 ? 1 : dims[nd - 1];
            long innerStride = nd == 0 ? 0 : byteStrides[nd - 1];
            long rows = 1;
            for (int axis = 0; axis < nd - 1; axis++)
                rows *= dims[axis];

            // The odometer over the leading axes moves `row` to each innermost run; the run itself is a tight loop.
            var index = nd > 1 ? new long[nd - 1] : null;
            byte* row = source;
            byte* d = destination;
            for (long r = 0; r < rows; r++)
            {
                byte* s = row;
                for (long j = 0; j < inner; j++, s += innerStride, d += VStringAllocator.CellSize)
                    visitor.Visit(s, d);
                for (int axis = nd - 2; axis >= 0; axis--)
                {
                    row += byteStrides[axis];
                    if (++index[axis] < dims[axis])
                        break;
                    row -= byteStrides[axis] * dims[axis];
                    index[axis] = 0;
                }
            }
        }

        /// <summary>Whether a cell is a short string with text (<c>INITIALIZED | OUTSIDE_ARENA</c>, length 1–15): its 16
        ///     bytes ARE the value in either allocator's world (no arena offset, no heap pointer).</summary>
        /// <param name="flags">Byte 15 of the cell.</param>
        /// <returns>True when the cell moves as a plain 16-byte block.</returns>
        private static bool IsShortWithText(byte flags)
            => (flags & VStringAllocator.FlagMask) == (VStringAllocator.Initialized | VStringAllocator.OutsideArena)
               && (flags & VStringAllocator.ShortSizeMask) != 0;

        /// <summary>Copies one 16-byte cell.</summary>
        /// <param name="source">The source cell.</param>
        /// <param name="destination">The destination cell.</param>
        private static unsafe void CopyCell(byte* source, byte* destination)
        {
            *(ulong*)destination = *(ulong*)source;
            *(ulong*)(destination + 8) = *(ulong*)(source + 8);
        }

        /// <summary>
        ///     Import pass 1 (<c>NpyString_dup</c> for every cell that allocates nothing): an NA cell packs NA, a short string
        ///     moves as its cell, an empty string leaves the fresh (empty) cell; a medium / long / heap string is counted for
        ///     pass 2 with the arena bytes its duplicate appends.
        /// </summary>
        private unsafe struct ImportSettleVisitor : ICellVisitor
        {
            /// <summary>The destination's allocator (locked).</summary>
            private readonly VStringAllocator _out;

            /// <summary>The arena bytes pass 2 will append.</summary>
            internal long ArenaBytes;

            /// <summary>The number of cells pass 2 must duplicate.</summary>
            internal long LongCells;

            /// <summary>Creates the visitor.</summary>
            /// <param name="outAllocator">The destination's allocator (locked by the caller).</param>
            internal ImportSettleVisitor(VStringAllocator outAllocator)
            {
                _out = outAllocator;
                ArenaBytes = 0;
                LongCells = 0;
            }

            /// <inheritdoc/>
            /// <exception cref="OutOfMemoryException">NumSharp could not pack an NA cell.</exception>
            public void Visit(byte* source, byte* destination)
            {
                byte flags = source[15];
                if (IsShortWithText(flags))
                {
                    CopyCell(source, destination);
                    return;
                }
                if (VStringAllocator.IsNull(source))
                {
                    if (_out.PackNull(destination) < 0)
                        throw new OutOfMemoryException("Failed to pack null string while copying a numpy StringDType array");
                    return;
                }
                // Empty (an uninitialized all-zero cell, a short empty string, or a vstring of size 0): NumPy's dup packs the
                // empty string, which leaves a fresh cell as it is.
                if (VStringAllocator.Size(source) == 0)
                    return;
                ArenaBytes += VStringAllocator.ArenaBytesForDuplicate(source);
                LongCells++;
            }
        }

        /// <summary>
        ///     Import pass 2: every medium / long / heap string is loaded through NUMPY's allocator (<c>NpyString_load</c>) and
        ///     packed into the fresh destination cell (<c>NpyString_newsize</c> — an arena allocation, as numpy's copy does).
        /// </summary>
        private unsafe struct ImportLongVisitor : ICellVisitor
        {
            /// <summary>The source array's numpy allocator (locked).</summary>
            private readonly IntPtr _numpy;

            /// <summary>The destination's allocator (locked).</summary>
            private readonly VStringAllocator _out;

            /// <summary>Creates the visitor.</summary>
            /// <param name="numpyAllocator">The source's numpy allocator (acquired by the caller).</param>
            /// <param name="outAllocator">The destination's allocator (locked by the caller).</param>
            internal ImportLongVisitor(IntPtr numpyAllocator, VStringAllocator outAllocator)
            {
                _numpy = numpyAllocator;
                _out = outAllocator;
            }

            /// <inheritdoc/>
            /// <exception cref="InvalidOperationException">numpy could not read the cell.</exception>
            /// <exception cref="OutOfMemoryException">NumSharp could not allocate the text.</exception>
            public void Visit(byte* source, byte* destination)
            {
                // Pass 1 settled every other cell; this test is the same partition, so each cell is handled once.
                if (IsShortWithText(source[15]) || VStringAllocator.IsNull(source) || VStringAllocator.Size(source) == 0)
                    return;
                NumpyStringApi.NpyStaticString text;
                if (NumpyStringApi.Load(_numpy, source, &text) != 0)
                    throw new InvalidOperationException("numpy could not load a StringDType element (NpyString_load failed: a corrupt cell)");
                if (_out.NewSize(text.buf, (long)text.size, destination) < 0)
                    throw new OutOfMemoryException("Failed to allocate string while copying a numpy StringDType array");
            }
        }

        /// <summary>
        ///     The export's single pass (<c>NpyString_dup</c> into numpy's fresh cells): an NA cell is packed NA by numpy, a
        ///     short string moves as its cell, an empty string leaves the fresh cell, and a longer string is loaded through
        ///     NumSharp's allocator and packed by numpy (<c>NpyString_pack</c>) — numpy's arena then grows as it does for
        ///     <c>np.array(list, dtype)</c>.
        /// </summary>
        private unsafe struct ExportVisitor : ICellVisitor
        {
            /// <summary>The source's NumSharp allocator (locked).</summary>
            private readonly VStringAllocator _in;

            /// <summary>The destination array's numpy allocator (locked).</summary>
            private readonly IntPtr _numpy;

            /// <summary>Creates the visitor.</summary>
            /// <param name="inAllocator">The source's allocator (locked by the caller).</param>
            /// <param name="numpyAllocator">The destination's numpy allocator (acquired by the caller).</param>
            internal ExportVisitor(VStringAllocator inAllocator, IntPtr numpyAllocator)
            {
                _in = inAllocator;
                _numpy = numpyAllocator;
            }

            /// <inheritdoc/>
            /// <exception cref="InvalidOperationException">NumSharp could not read the cell.</exception>
            /// <exception cref="OutOfMemoryException">numpy could not store the text or the NA value.</exception>
            public void Visit(byte* source, byte* destination)
            {
                byte flags = source[15];
                if (IsShortWithText(flags))
                {
                    CopyCell(source, destination);
                    return;
                }
                if (VStringAllocator.IsNull(source))
                {
                    if (NumpyStringApi.PackNull(_numpy, destination) < 0)
                        throw new OutOfMemoryException("Failed to pack null string while copying a StringDType array to numpy");
                    return;
                }
                if (VStringAllocator.Size(source) == 0)
                    return;
                if (_in.Load(source, out byte* text, out long size) < 0)
                    throw new InvalidOperationException("NumSharp could not load a StringDType element (a corrupt cell)");
                if (NumpyStringApi.Pack(_numpy, destination, text, size) < 0)
                    throw new OutOfMemoryException("Failed to pack string while copying a StringDType array to numpy");
            }
        }
    }
}
