using System;
using System.Globalization;
using System.Numerics;
using System.Text;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Printing;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>MemoryError</c>: an allocation that cannot be satisfied. NumSharp raises it where NumPy does, with
    ///     NumPy's text: a <c>StringDType</c> cell that cannot be allocated, loaded or packed (<c>Failed to allocate string in
    ///     add</c>, <c>Failed to load string in string comparison</c>), an iterator whose buffers cannot be allocated
    ///     (<c>Could not allocate the iterator's buffers</c>), CPython's bare <c>PyErr_NoMemory</c> (an empty text), and an
    ///     array allocation, as the subclass <see cref="ArrayMemoryError"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It derives from .NET's <see cref="OutOfMemoryException"/>, the exception with the same meaning, so a
    ///     <c>catch (OutOfMemoryException)</c> catches it, and .NET's own allocation failures (which stay
    ///     <see cref="OutOfMemoryException"/>) and NumSharp's Python-named ones are handled by one catch.
    ///     </para>
    ///     <para>
    ///     Footgun: a <c>catch (MemoryError)</c> does NOT catch a plain .NET <see cref="OutOfMemoryException"/> (the
    ///     runtime's own, a <see cref="System.Runtime.InteropServices.NativeMemory"/> failure NumSharp did not translate);
    ///     catch <see cref="OutOfMemoryException"/> to handle both.
    ///     </para>
    /// </remarks>
    public class MemoryError : OutOfMemoryException, INumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise MemoryError</c> (CPython's <c>PyErr_NoMemory</c>): an
        ///     empty <see cref="Exception.Message"/>, as <c>str(MemoryError())</c> is <c>''</c> (not .NET's
        ///     <see cref="OutOfMemoryException"/> default, "Insufficient memory to continue the execution of the program.").
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>MemoryError</c>.</remarks>
        public MemoryError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error (<c>Failed to allocate string in
        ///     add</c>).</param>
        public MemoryError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise MemoryError(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one (the .NET <see cref="OutOfMemoryException"/>
        ///     of the failed allocation, for example), or null for none.</param>
        public MemoryError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     NumPy's <c>_ArrayMemoryError</c>: an array of a given shape and dtype could not be allocated. Its text is NumPy's
    ///     <c>Unable to allocate 7.45 GiB for an array with shape (1000000000,) and data type float64</c>, the size in binary
    ///     units from the shape and the dtype's item size.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A subclass of <see cref="MemoryError"/> (and so of .NET's <see cref="OutOfMemoryException"/>), as NumPy's is a
    ///     subclass of Python's <c>MemoryError</c>; catch <see cref="MemoryError"/>, as NumPy advises (<c>_ArrayMemoryError</c>
    ///     is private to NumPy and its traceback shows it as <c>MemoryError</c>). NumSharp drops NumPy's leading underscore:
    ///     a C# type name cannot hide behind <c>_display_as_base</c>.
    ///     </para>
    ///     <para>
    ///     It reports a request NumPy's size checks accept (a byte count that fits <c>intp</c>) but the memory cannot hold; a
    ///     byte count past <c>intp</c> is NumPy's <see cref="ValueError"/> <c>array is too big; …</c> instead.
    ///     </para>
    /// </remarks>
    public class ArrayMemoryError : MemoryError
    {
        /// <summary>The shape as given; <see cref="shape"/> hands out copies so the error's text stays true.</summary>
        private readonly long[] _shape;

        /// <summary>
        ///     NumPy's <c>_ArrayMemoryError(shape, dtype)</c>: the message is NumPy's <c>__str__</c> of the two values.
        /// </summary>
        /// <param name="shape">The dimensions of the array that could not be allocated (a 0-d array: empty).</param>
        /// <param name="dtype">Its dtype; the item size times the dimensions is the byte count reported.</param>
        /// <exception cref="ArgumentNullException"><paramref name="shape"/> or <paramref name="dtype"/> is null (NumPy fails
        ///     only when the text is formatted; NumSharp formats it here).</exception>
        public ArrayMemoryError(long[] shape, DType dtype) : this(shape, dtype, null) { }

        /// <summary>
        ///     NumPy's <c>_ArrayMemoryError(shape, dtype)</c> with the exception that caused it: NumSharp's translation of a
        ///     .NET allocation failure keeps the .NET exception in <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="shape">The dimensions of the array that could not be allocated (a 0-d array: empty).</param>
        /// <param name="dtype">Its dtype.</param>
        /// <param name="innerException">The exception that led to this one (the allocator's
        ///     <see cref="OutOfMemoryException"/>), or null for none.</param>
        /// <exception cref="ArgumentNullException"><paramref name="shape"/> or <paramref name="dtype"/> is null.</exception>
        public ArrayMemoryError(long[] shape, DType dtype, Exception innerException)
            : base(Format(shape ?? throw new ArgumentNullException(nameof(shape)),
                          dtype ?? throw new ArgumentNullException(nameof(dtype))), innerException)
        {
            // A copy: the caller's array may be a live dimensions array that a later reshape reuses.
            _shape = (long[])shape.Clone();
            this.dtype = dtype;
        }

        /// <summary>
        ///     The dimensions of the array that could not be allocated (NumPy's <c>shape</c> attribute), as a fresh copy on
        ///     every read: NumPy's is an immutable tuple, so a caller that changes the copy cannot change the error.
        /// </summary>
        public long[] shape => (long[])_shape.Clone();

        /// <summary>The dtype of the array that could not be allocated (NumPy's <c>dtype</c> attribute).</summary>
        public DType dtype { get; }

        /// <summary>
        ///     NumPy's <c>_ArrayMemoryError.__str__</c>: <c>Unable to allocate {size} for an array with shape {shape} and
        ///     data type {dtype}</c>, the shape as Python prints a tuple (<c>(3, 4)</c>, <c>(1000000000,)</c>, <c>()</c>) and
        ///     the dtype as <c>str(dtype)</c> (<c>float64</c>, <c>&lt;U3</c>, <c>|S5</c>, <c>StringDType()</c>).
        /// </summary>
        /// <param name="shape">The dimensions.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The text.</returns>
        private static string Format(long[] shape, DType dtype)
        {
            // _total_size: itemsize times every dimension, in Python ints (no wrap however large the product).
            BigInteger total = dtype.itemsize;
            var tuple = new StringBuilder("(");
            for (int i = 0; i < shape.Length; i++)
            {
                total *= shape[i];
                if (i > 0)
                    tuple.Append(", ");
                tuple.Append(shape[i].ToString(CultureInfo.InvariantCulture));
            }
            // Python's 1-tuple keeps its trailing comma.
            if (shape.Length == 1)
                tuple.Append(',');
            tuple.Append(')');
            return $"Unable to allocate {SizeToString(total)} for an array with shape {tuple} and data type {dtype}";
        }

        /// <summary>
        ///     NumPy's <c>_ArrayMemoryError._size_to_string</c>: a byte count in binary units (<c>bytes</c>, <c>KiB</c>, …
        ///     <c>EiB</c>), with three significant figures in Python's alternate form (<c>'#.3g'</c>: <c>8.00</c>,
        ///     <c>16.0</c>, <c>149.</c>) below 1000 units, every integer digit (<c>'#.0f'</c>: <c>1000.</c>) from there on,
        ///     and no decimal point for plain bytes (<c>1023 bytes</c>).
        /// </summary>
        /// <param name="numBytes">The byte count (a Python int: any size).</param>
        /// <returns>The size text, e.g. <c>64.0 GiB</c>.</returns>
        /// <remarks>Sizes past the largest unit stay in EiB (<c>1024. EiB</c>, <c>1048576. EiB</c>), as NumPy prints them;
        ///     <c>1023.6 KiB</c> rounds to <c>1.00 MiB</c> through NumPy's "unit after rounding" step.</remarks>
        internal static string SizeToString(BigInteger numBytes)
        {
            const int log2Step = 10, step = 1024;
            string[] units = { "bytes", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB" };

            // Python's int.bit_length() is the bit length of the absolute value (.NET's GetBitLength of a negative value
            // is not), so a negative count from a negative dimension reads as NumPy prints it.
            int unitI = (int)(Math.Max((long)BigInteger.Abs(numBytes).GetBitLength() - 1, 1) / log2Step);
            // n_units = num_bytes / unit_val: Python's int / int, correctly rounded.
            double nUnits = PyScalar.IntTrueDivide(numBytes, BigInteger.One << (unitI * log2Step));

            // Ensure the unit is right after rounding (Python's round() is half-to-even).
            if (Math.Round(nUnits, MidpointRounding.ToEven) == step)
            {
                unitI++;
                nUnits /= step;
            }
            // Sizes too large for the units: scale back to the last one.
            if (unitI >= units.Length)
            {
                int newUnitI = units.Length - 1;
                nUnits *= Math.Pow(2, (unitI - newUnitI) * log2Step);
                unitI = newUnitI;
            }

            string unitName = units[unitI];
            string number = unitI == 0
                ? PrintfFormatter.Format("%.0f", nUnits, PercentFormat.PyObject)
                : Math.Round(nUnits, MidpointRounding.ToEven) < 1000
                    ? PrintfFormatter.Format("%#.3g", nUnits, PercentFormat.PyObject)
                    : PrintfFormatter.Format("%#.0f", nUnits, PercentFormat.PyObject);
            return number + " " + unitName;
        }
    }
}
