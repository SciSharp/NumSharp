using System;
using NumSharp.Backends;
using NumSharp.Generic;

// =============================================================================
// NDListArray.TextOperands.cs — the operators of NumPy's object array of lists with a Python str / bytes operand
// =============================================================================
//
// A C# string (or byte[]) converts implicitly to BOTH NDArray (a 0-d U / S array) and NDCharArray (a chararray stand-in),
// and C# overload resolution prefers the more derived NDCharArray — which would run the chararray's own refusals
// ("comparison of non-string arrays", "Can only create a chararray from string data.") where NumPy compares or combines
// each list with the str. These exact-type overloads bind first and treat the text as the Python value it stands for:
// np.asarray(text), a 0-d U / S operand of the object loop.
// =============================================================================

namespace NumSharp
{
    public sealed partial class NDListArray
    {
        /// <summary>NumPy's <c>lists == str</c> with a C# string (a Python <c>str</c>): every element False (a list never equals a str or bytes).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        public static NDArray<bool> operator ==(NDListArray left, string right) => CompareObject(PyCompareOp.Eq, left, right, reflected: false);

        /// <summary>NumPy's <c>str == lists</c> with a C# string (a Python <c>str</c>): every element False (a list never equals a str or bytes).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        public static NDArray<bool> operator ==(string left, NDListArray right) => CompareObject(PyCompareOp.Eq, right, left, reflected: true);

        /// <summary>NumPy's <c>lists == bytes</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): every element False (a list never equals a str or bytes).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        public static NDArray<bool> operator ==(NDListArray left, byte[] right) => CompareObject(PyCompareOp.Eq, left, right, reflected: false);

        /// <summary>NumPy's <c>bytes == lists</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): every element False (a list never equals a str or bytes).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        public static NDArray<bool> operator ==(byte[] left, NDListArray right) => CompareObject(PyCompareOp.Eq, right, left, reflected: true);

        /// <summary>NumPy's <c>lists != str</c> with a C# string (a Python <c>str</c>): every element True.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        public static NDArray<bool> operator !=(NDListArray left, string right) => CompareObject(PyCompareOp.Ne, left, right, reflected: false);

        /// <summary>NumPy's <c>str != lists</c> with a C# string (a Python <c>str</c>): every element True.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        public static NDArray<bool> operator !=(string left, NDListArray right) => CompareObject(PyCompareOp.Ne, right, left, reflected: true);

        /// <summary>NumPy's <c>lists != bytes</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): every element True.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        public static NDArray<bool> operator !=(NDListArray left, byte[] right) => CompareObject(PyCompareOp.Ne, left, right, reflected: false);

        /// <summary>NumPy's <c>bytes != lists</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): every element True.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        public static NDArray<bool> operator !=(byte[] left, NDListArray right) => CompareObject(PyCompareOp.Ne, right, left, reflected: true);

        /// <summary>NumPy's <c>lists &lt; str</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair (<c>'&lt;' not supported between instances of 'list' and 'str'</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator <(NDListArray left, string right) => CompareObject(PyCompareOp.Lt, left, right, reflected: false);

        /// <summary>NumPy's <c>str &lt; lists</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair (<c>'&lt;' not supported between instances of 'list' and 'str'</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator <(string left, NDListArray right) => CompareObject(PyCompareOp.Lt, right, left, reflected: true);

        /// <summary>NumPy's <c>lists &lt; bytes</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair (<c>'&lt;' not supported between instances of 'list' and 'str'</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator <(NDListArray left, byte[] right) => CompareObject(PyCompareOp.Lt, left, right, reflected: false);

        /// <summary>NumPy's <c>bytes &lt; lists</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair (<c>'&lt;' not supported between instances of 'list' and 'str'</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator <(byte[] left, NDListArray right) => CompareObject(PyCompareOp.Lt, right, left, reflected: true);

        /// <summary>NumPy's <c>lists &lt;= str</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator <=(NDListArray left, string right) => CompareObject(PyCompareOp.Le, left, right, reflected: false);

        /// <summary>NumPy's <c>str &lt;= lists</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator <=(string left, NDListArray right) => CompareObject(PyCompareOp.Le, right, left, reflected: true);

        /// <summary>NumPy's <c>lists &lt;= bytes</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator <=(NDListArray left, byte[] right) => CompareObject(PyCompareOp.Le, left, right, reflected: false);

        /// <summary>NumPy's <c>bytes &lt;= lists</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator <=(byte[] left, NDListArray right) => CompareObject(PyCompareOp.Le, right, left, reflected: true);

        /// <summary>NumPy's <c>lists &gt; str</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator >(NDListArray left, string right) => CompareObject(PyCompareOp.Gt, left, right, reflected: false);

        /// <summary>NumPy's <c>str &gt; lists</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator >(string left, NDListArray right) => CompareObject(PyCompareOp.Gt, right, left, reflected: true);

        /// <summary>NumPy's <c>lists &gt; bytes</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator >(NDListArray left, byte[] right) => CompareObject(PyCompareOp.Gt, left, right, reflected: false);

        /// <summary>NumPy's <c>bytes &gt; lists</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator >(byte[] left, NDListArray right) => CompareObject(PyCompareOp.Gt, right, left, reflected: true);

        /// <summary>NumPy's <c>lists &gt;= str</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator >=(NDListArray left, string right) => CompareObject(PyCompareOp.Ge, left, right, reflected: false);

        /// <summary>NumPy's <c>str &gt;= lists</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator >=(string left, NDListArray right) => CompareObject(PyCompareOp.Ge, right, left, reflected: true);

        /// <summary>NumPy's <c>lists &gt;= bytes</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator >=(NDListArray left, byte[] right) => CompareObject(PyCompareOp.Ge, left, right, reflected: false);

        /// <summary>NumPy's <c>bytes &gt;= lists</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape.</returns>
        /// <exception cref="TypeError">The first pair's ordering (an empty array compares nothing).</exception>
        public static NDArray<bool> operator >=(byte[] left, NDListArray right) => CompareObject(PyCompareOp.Ge, right, left, reflected: true);

        /// <summary>NumPy's <c>lists + str</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair (<c>can only concatenate list (not "str") to list</c>; with the text on the left, <c>can only concatenate str (not "list") to str</c> / <c>can't concat list to bytes</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape (empty: nothing was computed).</returns>
        /// <exception cref="TypeError">The first pair's operation (an empty array computes nothing).</exception>
        public static NDListArray operator +(NDListArray left, string right) => AddObject(left, right, reflected: false);

        /// <summary>NumPy's <c>str + lists</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair (<c>can only concatenate list (not "str") to list</c>; with the text on the left, <c>can only concatenate str (not "list") to str</c> / <c>can't concat list to bytes</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape (empty: nothing was computed).</returns>
        /// <exception cref="TypeError">The first pair's operation (an empty array computes nothing).</exception>
        public static NDListArray operator +(string left, NDListArray right) => AddObject(right, left, reflected: true);

        /// <summary>NumPy's <c>lists + bytes</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair (<c>can only concatenate list (not "str") to list</c>; with the text on the left, <c>can only concatenate str (not "list") to str</c> / <c>can't concat list to bytes</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape (empty: nothing was computed).</returns>
        /// <exception cref="TypeError">The first pair's operation (an empty array computes nothing).</exception>
        public static NDListArray operator +(NDListArray left, byte[] right) => AddObject(left, right, reflected: false);

        /// <summary>NumPy's <c>bytes + lists</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair (<c>can only concatenate list (not "str") to list</c>; with the text on the left, <c>can only concatenate str (not "list") to str</c> / <c>can't concat list to bytes</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape (empty: nothing was computed).</returns>
        /// <exception cref="TypeError">The first pair's operation (an empty array computes nothing).</exception>
        public static NDListArray operator +(byte[] left, NDListArray right) => AddObject(right, left, reflected: true);

        /// <summary>NumPy's <c>lists * str</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair (<c>can't multiply sequence by non-int of type 'str'</c>; with the text on the left, <c>... of type 'list'</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape (empty: nothing was computed).</returns>
        /// <exception cref="TypeError">The first pair's operation (an empty array computes nothing).</exception>
        public static NDListArray operator *(NDListArray left, string right) => MultiplyObject(left, right, reflected: false);

        /// <summary>NumPy's <c>str * lists</c> with a C# string (a Python <c>str</c>): CPython's TypeError for the first pair (<c>can't multiply sequence by non-int of type 'str'</c>; with the text on the left, <c>... of type 'list'</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape (empty: nothing was computed).</returns>
        /// <exception cref="TypeError">The first pair's operation (an empty array computes nothing).</exception>
        public static NDListArray operator *(string left, NDListArray right) => MultiplyObject(right, left, reflected: true);

        /// <summary>NumPy's <c>lists * bytes</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair (<c>can't multiply sequence by non-int of type 'str'</c>; with the text on the left, <c>... of type 'list'</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape (empty: nothing was computed).</returns>
        /// <exception cref="TypeError">The first pair's operation (an empty array computes nothing).</exception>
        public static NDListArray operator *(NDListArray left, byte[] right) => MultiplyObject(left, right, reflected: false);

        /// <summary>NumPy's <c>bytes * lists</c> with a C# <c>byte[]</c> (a Python <c>bytes</c>): CPython's TypeError for the first pair (<c>can't multiply sequence by non-int of type 'str'</c>; with the text on the left, <c>... of type 'list'</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result over the list array's shape (empty: nothing was computed).</returns>
        /// <exception cref="TypeError">The first pair's operation (an empty array computes nothing).</exception>
        public static NDListArray operator *(byte[] left, NDListArray right) => MultiplyObject(right, left, reflected: true);
    }
}
