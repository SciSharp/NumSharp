using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;
using NumSharp.Generic;

// =============================================================================
// NDListArray.Compare.cs — the comparison operators of NumPy's object array of lists, its truth members (all / any /
// nonzero) and astype
// =============================================================================
//
// Comparisons. `a < b` of two object arrays is np.less's OBJECT loop: the operands broadcast (NumPy's ufunc error text),
// the result is a bool array in the ufunc's output layout, and every element pair is CPython's list_richcompare (the
// engine partial). Against a value that is not a list array, NumPy converts it (np.asarray) and the loop compares a list
// with that array's elements as Python objects: == / != are False / True for every element (a list never equals a
// number or a str), an ordering is CPython's TypeError "'<' not supported between instances of 'list' and 'int'" as soon
// as one pair is compared (an empty broadcast compares nothing and gives an empty result). With the value on the LEFT,
// Python's reflected operator runs (1 < lists is lists > 1), whose TypeError names '>'. A chararray beside a list array
// is the chararray's own operator (a subclass's reflected method goes first): NumPy's "comparison of non-string
// arrays". The C# null keeps NDArray's reference test for == / != (a 0-d bool), and an ordering against it is the
// TypeError against None.
//
// Truth. A Python list is true when it is not empty, so all / any / nonzero / astype(bool) read the lists' lengths:
// NumPy's object reductions convert each element with PyObject_IsTrue, and their results are the bool reductions of
// those truths (the reductions themselves — axis, keepdims, out=, where=, their errors — are NDArray's).
//
// astype. object → anything is an 'unsafe' cast only (casting='safe' and stricter raise "Cannot cast array data from
// dtype('O') to dtype('bool') according to the rule 'safe'"). The conversion is the element's Python conversion: bool
// its truth, StringDType its str() (the list's repr), an integer dtype int(list) (TypeError), a float dtype float(list)
// (ValueError "setting an array element with a sequence."), complex complex(list) (TypeError), S / U a sequence stored
// into a string (ValueError without the period); an EMPTY array converts nothing and gives an empty array of the dtype.
// =============================================================================

namespace NumSharp
{
    public sealed partial class NDListArray
    {
        // =============================================================================================================
        // Comparison operators
        // =============================================================================================================

        /// <summary>NumPy's <c>a == b</c> of two object arrays of lists: each pair compared as Python lists (equal lengths
        ///     and equal items), broadcast together.</summary>
        /// <param name="left">The left list array (null: NDArray's reference test).</param>
        /// <param name="right">The right list array (null: NDArray's reference test).</param>
        /// <returns>The bool result (a 0-d one for a null operand).</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast (NumPy's ValueError text).</exception>
        public static NDArray<bool> operator ==(NDListArray left, NDListArray right) => CompareListArrays(PyCompareOp.Eq, left, right);

        /// <summary>NumPy's <c>a != b</c> of two object arrays of lists (see <c>==</c>).</summary>
        /// <param name="left">The left list array.</param>
        /// <param name="right">The right list array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        public static NDArray<bool> operator !=(NDListArray left, NDListArray right) => CompareListArrays(PyCompareOp.Ne, left, right);

        /// <summary>NumPy's <c>a &lt; b</c> of two object arrays of lists: Python's list ordering (the first unequal items
        ///     decide, then the lengths).</summary>
        /// <param name="left">The left list array.</param>
        /// <param name="right">The right list array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">A str item meets a bytes item (<c>'&lt;' not supported between instances of 'str' and
        ///     'bytes'</c>), or an operand is null (the ordering against None).</exception>
        public static NDArray<bool> operator <(NDListArray left, NDListArray right) => CompareListArrays(PyCompareOp.Lt, left, right);

        /// <summary>NumPy's <c>a &lt;= b</c> of two object arrays of lists (see <c>&lt;</c>).</summary>
        /// <param name="left">The left list array.</param>
        /// <param name="right">The right list array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">A str item meets a bytes item, or a null operand.</exception>
        public static NDArray<bool> operator <=(NDListArray left, NDListArray right) => CompareListArrays(PyCompareOp.Le, left, right);

        /// <summary>NumPy's <c>a &gt; b</c> of two object arrays of lists (see <c>&lt;</c>).</summary>
        /// <param name="left">The left list array.</param>
        /// <param name="right">The right list array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">A str item meets a bytes item, or a null operand.</exception>
        public static NDArray<bool> operator >(NDListArray left, NDListArray right) => CompareListArrays(PyCompareOp.Gt, left, right);

        /// <summary>NumPy's <c>a &gt;= b</c> of two object arrays of lists (see <c>&lt;</c>).</summary>
        /// <param name="left">The left list array.</param>
        /// <param name="right">The right list array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">A str item meets a bytes item, or a null operand.</exception>
        public static NDArray<bool> operator >=(NDListArray left, NDListArray right) => CompareListArrays(PyCompareOp.Ge, left, right);

        /// <summary>NumPy's <c>lists == x</c> with an ndarray (or a C# value its implicit conversion made one): every
        ///     element False (a list never equals a number or a str), broadcast together.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        public static NDArray<bool> operator ==(NDListArray left, NDArray right) => CompareValue(PyCompareOp.Eq, left, right, reflected: false);

        /// <summary>NumPy's <c>lists != x</c> with an ndarray: every element True.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        public static NDArray<bool> operator !=(NDListArray left, NDArray right) => CompareValue(PyCompareOp.Ne, left, right, reflected: false);

        /// <summary>NumPy's <c>lists &lt; x</c> with an ndarray: CPython's TypeError for the first pair (an empty broadcast
        ///     gives an empty result).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The array.</param>
        /// <returns>An empty bool result (nothing was compared).</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError"><c>'&lt;' not supported between instances of 'list' and 'int'</c> (the element's
        ///     Python type).</exception>
        public static NDArray<bool> operator <(NDListArray left, NDArray right) => CompareValue(PyCompareOp.Lt, left, right, reflected: false);

        /// <summary>NumPy's <c>lists &lt;= x</c> with an ndarray (see <c>&lt;</c>).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        public static NDArray<bool> operator <=(NDListArray left, NDArray right) => CompareValue(PyCompareOp.Le, left, right, reflected: false);

        /// <summary>NumPy's <c>lists &gt; x</c> with an ndarray (see <c>&lt;</c>).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        public static NDArray<bool> operator >(NDListArray left, NDArray right) => CompareValue(PyCompareOp.Gt, left, right, reflected: false);

        /// <summary>NumPy's <c>lists &gt;= x</c> with an ndarray (see <c>&lt;</c>).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        public static NDArray<bool> operator >=(NDListArray left, NDArray right) => CompareValue(PyCompareOp.Ge, left, right, reflected: false);

        /// <summary>NumPy's <c>x == lists</c> with an ndarray on the left (the reflected comparison).</summary>
        /// <param name="left">The array.</param>
        /// <param name="right">The list array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        public static NDArray<bool> operator ==(NDArray left, NDListArray right) => CompareValue(PyCompareOp.Eq, right, left, reflected: true);

        /// <summary>NumPy's <c>x != lists</c> with an ndarray on the left (the reflected comparison).</summary>
        /// <param name="left">The array.</param>
        /// <param name="right">The list array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        public static NDArray<bool> operator !=(NDArray left, NDListArray right) => CompareValue(PyCompareOp.Ne, right, left, reflected: true);

        /// <summary>NumPy's <c>x &lt; lists</c>: Python runs <c>lists &gt; x</c>, whose TypeError names <c>'&gt;'</c>.</summary>
        /// <param name="left">The array.</param>
        /// <param name="right">The list array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError"><c>'&gt;' not supported between instances of 'list' and 'int'</c>.</exception>
        public static NDArray<bool> operator <(NDArray left, NDListArray right) => CompareValue(PyCompareOp.Lt, right, left, reflected: true);

        /// <summary>NumPy's <c>x &lt;= lists</c> (the reflected <c>&gt;=</c>).</summary>
        /// <param name="left">The array.</param>
        /// <param name="right">The list array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        public static NDArray<bool> operator <=(NDArray left, NDListArray right) => CompareValue(PyCompareOp.Le, right, left, reflected: true);

        /// <summary>NumPy's <c>x &gt; lists</c> (the reflected <c>&lt;</c>).</summary>
        /// <param name="left">The array.</param>
        /// <param name="right">The list array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        public static NDArray<bool> operator >(NDArray left, NDListArray right) => CompareValue(PyCompareOp.Gt, right, left, reflected: true);

        /// <summary>NumPy's <c>x &gt;= lists</c> (the reflected <c>&lt;=</c>).</summary>
        /// <param name="left">The array.</param>
        /// <param name="right">The list array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        public static NDArray<bool> operator >=(NDArray left, NDListArray right) => CompareValue(PyCompareOp.Ge, right, left, reflected: true);

        /// <summary>NumPy's <c>lists == x</c> with a C# value (a <c>byte[]</c> is Python bytes, a string a Python str,
        ///     <c>null</c> NDArray's reference test).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The value.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator ==(NDListArray left, object right) => CompareObject(PyCompareOp.Eq, left, right, reflected: false);

        /// <summary>NumPy's <c>lists != x</c> with a C# value.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The value.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator !=(NDListArray left, object right) => CompareObject(PyCompareOp.Ne, left, right, reflected: false);

        /// <summary>NumPy's <c>lists &lt; x</c> with a C# value (null is Python's None).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The value.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator <(NDListArray left, object right) => CompareObject(PyCompareOp.Lt, left, right, reflected: false);

        /// <summary>NumPy's <c>lists &lt;= x</c> with a C# value.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The value.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator <=(NDListArray left, object right) => CompareObject(PyCompareOp.Le, left, right, reflected: false);

        /// <summary>NumPy's <c>lists &gt; x</c> with a C# value.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The value.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator >(NDListArray left, object right) => CompareObject(PyCompareOp.Gt, left, right, reflected: false);

        /// <summary>NumPy's <c>lists &gt;= x</c> with a C# value.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The value.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator >=(NDListArray left, object right) => CompareObject(PyCompareOp.Ge, left, right, reflected: false);

        /// <summary>NumPy's <c>x == lists</c> with a C# value on the left.</summary>
        /// <param name="left">The value.</param>
        /// <param name="right">The list array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator ==(object left, NDListArray right) => CompareObject(PyCompareOp.Eq, right, left, reflected: true);

        /// <summary>NumPy's <c>x != lists</c> with a C# value on the left.</summary>
        /// <param name="left">The value.</param>
        /// <param name="right">The list array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator !=(object left, NDListArray right) => CompareObject(PyCompareOp.Ne, right, left, reflected: true);

        /// <summary>NumPy's <c>x &lt; lists</c> with a C# value on the left (the reflected <c>&gt;</c>).</summary>
        /// <param name="left">The value.</param>
        /// <param name="right">The list array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator <(object left, NDListArray right) => CompareObject(PyCompareOp.Lt, right, left, reflected: true);

        /// <summary>NumPy's <c>x &lt;= lists</c> with a C# value on the left (the reflected <c>&gt;=</c>).</summary>
        /// <param name="left">The value.</param>
        /// <param name="right">The list array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator <=(object left, NDListArray right) => CompareObject(PyCompareOp.Le, right, left, reflected: true);

        /// <summary>NumPy's <c>x &gt; lists</c> with a C# value on the left (the reflected <c>&lt;</c>).</summary>
        /// <param name="left">The value.</param>
        /// <param name="right">The list array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator >(object left, NDListArray right) => CompareObject(PyCompareOp.Gt, right, left, reflected: true);

        /// <summary>NumPy's <c>x &gt;= lists</c> with a C# value on the left (the reflected <c>&lt;=</c>).</summary>
        /// <param name="left">The value.</param>
        /// <param name="right">The list array.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDArray<bool> operator >=(object left, NDListArray right) => CompareObject(PyCompareOp.Ge, right, left, reflected: true);

        /// <summary>NumPy's <c>lists == c</c> beside a chararray: the chararray's own comparison runs first (a subclass's
        ///     reflected method), which refuses an object array.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The chararray.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator ==(NDListArray left, NDCharArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>lists != c</c> beside a chararray (see <c>==</c>).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The chararray.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator !=(NDListArray left, NDCharArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>lists &lt; c</c> beside a chararray (see <c>==</c>).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The chararray.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator <(NDListArray left, NDCharArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>lists &lt;= c</c> beside a chararray (see <c>==</c>).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The chararray.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator <=(NDListArray left, NDCharArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>lists &gt; c</c> beside a chararray (see <c>==</c>).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The chararray.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator >(NDListArray left, NDCharArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>lists &gt;= c</c> beside a chararray (see <c>==</c>).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The chararray.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator >=(NDListArray left, NDCharArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>c == lists</c>: the chararray's comparison refuses the object array.</summary>
        /// <param name="left">The chararray.</param>
        /// <param name="right">The list array.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator ==(NDCharArray left, NDListArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>c != lists</c> (see <c>==</c>).</summary>
        /// <param name="left">The chararray.</param>
        /// <param name="right">The list array.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator !=(NDCharArray left, NDListArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>c &lt; lists</c> (see <c>==</c>).</summary>
        /// <param name="left">The chararray.</param>
        /// <param name="right">The list array.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator <(NDCharArray left, NDListArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>c &lt;= lists</c> (see <c>==</c>).</summary>
        /// <param name="left">The chararray.</param>
        /// <param name="right">The list array.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator <=(NDCharArray left, NDListArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>c &gt; lists</c> (see <c>==</c>).</summary>
        /// <param name="left">The chararray.</param>
        /// <param name="right">The list array.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator >(NDCharArray left, NDListArray right) => throw NonStringComparison();

        /// <summary>NumPy's <c>c &gt;= lists</c> (see <c>==</c>).</summary>
        /// <param name="left">The chararray.</param>
        /// <param name="right">The list array.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError"><c>comparison of non-string arrays</c>.</exception>
        public static NDArray<bool> operator >=(NDCharArray left, NDListArray right) => throw NonStringComparison();

        /// <summary>Reference equality, as for <see cref="NDArray"/>: the operators compare elements, these compare objects.</summary>
        /// <param name="obj">The other object.</param>
        /// <returns>True for the same list array object.</returns>
        public override bool Equals(object obj) => ReferenceEquals(this, obj);

        /// <summary>The object's identity hash (see <see cref="Equals(object)"/>).</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(this);

        // =============================================================================================================
        // Truth
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>all(axis=None, out=None, keepdims=False, *, where=True)</c>: whether every list is non-empty
        ///     (Python's truth of a list), reduced as NDArray's <see cref="NDArray.all"/> reduces a bool array.
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="out">NumPy's <c>out=</c>: a bool (or castable) array of the result's shape.</param>
        /// <param name="keepdims">Keep the reduced axis as length 1.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <returns>A 0-d bool for <c>axis=None</c>, else the reduced bool array (or <paramref name="out"/>).</returns>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <exception cref="Exception">NDArray.all's own errors for <paramref name="out"/> and <paramref name="where"/>.</exception>
        public NDArray all(int? axis = null, NDArray @out = null, bool keepdims = false, NDArray where = null)
        {
            using var truth = Truth('K');
            return truth.all(axis, @out, keepdims, where);
        }

        /// <summary>
        ///     NumPy's <c>any(axis=None, out=None, keepdims=False, *, where=True)</c>: whether some list is non-empty (see
        ///     <see cref="all"/>).
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="out">NumPy's <c>out=</c>.</param>
        /// <param name="keepdims">Keep the reduced axis as length 1.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <returns>A 0-d bool for <c>axis=None</c>, else the reduced bool array (or <paramref name="out"/>).</returns>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <exception cref="Exception">NDArray.any's own errors for <paramref name="out"/> and <paramref name="where"/>.</exception>
        public NDArray any(int? axis = null, NDArray @out = null, bool keepdims = false, NDArray where = null)
        {
            using var truth = Truth('K');
            return truth.any(axis, @out, keepdims, where);
        }

        /// <summary>NumPy's <c>nonzero()</c>: the indices of the non-empty lists, one int64 array per axis, in C order.</summary>
        /// <returns>The index arrays (one for a 0-d array, as NumPy's 0-d nonzero gives).</returns>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray[] nonzero()
        {
            ThrowIfDisposed();
            // NumPy 2's array_nonzero refuses a 0-d array outright.
            if (ndim == 0)
                throw new ValueError("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.");
            using var truth = Truth('K');
            return np.nonzero(truth);
        }

        // =============================================================================================================
        // astype
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>astype(dtype, order='K', casting='unsafe', subok=True, copy=True)</c> of an object array of lists
        ///     (see the file header): bool is each list's truth, <c>StringDType</c> its <c>str()</c>; every other dtype is
        ///     NumPy's conversion error for the first element, and an empty array the empty array of the dtype.
        /// </summary>
        /// <param name="dtype">The target dtype.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c> (any case): the result's layout.</param>
        /// <param name="casting">The casting rule: only <c>'unsafe'</c> allows a cast from object.</param>
        /// <param name="subok">NumPy's <c>subok</c> (the result is never a subclass here).</param>
        /// <param name="copy">NumPy's <c>copy</c> (a conversion always makes a new array).</param>
        /// <returns>The converted array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="ValueError">An unknown order or casting name; a float, <c>S</c> or <c>U</c> dtype for a non-empty
        ///     array (<c>setting an array element with a sequence.</c> / without the period for text).</exception>
        /// <exception cref="TypeError">A rule other than <c>'unsafe'</c> (<c>Cannot cast array data from dtype('O') to
        ///     dtype('bool') according to the rule 'safe'</c>); an integer or complex dtype for a non-empty array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray astype(DType dtype, char order = 'K', string casting = "unsafe", bool subok = true, bool copy = true)
        {
            ThrowIfDisposed();
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            char o = char.ToUpperInvariant(order);
            if (o is not ('C' or 'F' or 'A' or 'K'))
                throw new ValueError($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
            // array_astype's casting converter, then PyArray_CanCastArrayTo: object converts to anything, but only unsafely.
            switch (casting)
            {
                case "unsafe":
                    break;
                case "no":
                case "equiv":
                case "safe":
                case "same_kind":
                case "same_value":
                    // PyArray_CanCastArrayTo's text calls a 0-d array a scalar.
                    throw new TypeError($"Cannot cast {(ndim == 0 ? "scalar" : "array data")} from dtype('O') to {dtype.ToString(true)} " +
                                        $"according to the rule '{casting}'");
                default:
                    throw new ValueError(
                        $"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe', 'same_value' (got '{casting}')");
            }
            var layout = LayoutFor(o);
            if (dtype.typecode == NPTypeCode.Boolean)
                return Truth(o);
            if (dtype.kind == 'T')
                return ReprTexts(dtype, layout);
            if (size == 0)
                return new NDArray(dtype, layout, true);
            // The first element's Python conversion (NumPy stops at it): int() / float() / complex() of a list, or a list
            // stored into a fixed-width string.
            throw dtype.kind switch
            {
                'f' => new ValueError("setting an array element with a sequence."),
                'c' => new TypeError("must be real number, not list"),
                'S' or 'U' => new ValueError("setting an array element with a sequence"),
                _ when dtype.typecode == NPTypeCode.Decimal => new ValueError("setting an array element with a sequence."),
                _ => new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'list'"),
            };
        }

        // =============================================================================================================
        // Internals
        // =============================================================================================================

        /// <summary>
        ///     The layout NumPy gives a conversion of this array (<c>PyArray_NewLikeArray</c>): C, F, F for an F-contiguous
        ///     array under <c>'A'</c>, the id array's memory order under <c>'K'</c>.
        /// </summary>
        /// <param name="order">The normalized order letter.</param>
        /// <returns>A fresh dense shape of the outer dimensions.</returns>
        private Shape LayoutFor(char order)
        {
            var dims = (long[])Ids.Shape.dimensions.Clone();
            return order switch
            {
                'C' => Shape.Adopt(dims),
                'F' => Shape.Adopt(dims, 'F'),
                'A' => Ids.Shape.IsFContiguous && !Ids.Shape.IsContiguous ? Shape.Adopt(dims, 'F') : Shape.Adopt(dims),
                _ => Ids.Shape.KeepOrder(),
            };
        }

        /// <summary>Each list's Python truth (non-empty), as a fresh bool array in <paramref name="order"/>'s layout.</summary>
        /// <param name="order">The normalized order letter (see <see cref="LayoutFor"/>).</param>
        /// <returns>The truth array.</returns>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        private unsafe NDArray Truth(char order)
        {
            ThrowIfDisposed();
            var result = new NDArray(NPTypeCode.Boolean, LayoutFor(order), false);
            var heap = Heap;
            Walk(new[] { Ids, result }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY },
                (dataptrs, strides, n, aux) =>
                {
                    byte* pi = (byte*)dataptrs[0];
                    byte* po = (byte*)dataptrs[1];
                    long si = strides[0], so = strides[1];
                    for (long k = 0; k < n; k++, pi += si, po += so)
                        *(bool*)po = heap.Length(*(long*)pi) != 0;
                });
            return result;
        }

        /// <summary>Each list's <c>str()</c> (its Python repr) as a <c>StringDType</c> array in <paramref name="layout"/>.</summary>
        /// <param name="dtype">The <c>StringDType</c> descriptor (its NA object, if any, is kept).</param>
        /// <param name="layout">The result's layout.</param>
        /// <returns>The text array.</returns>
        private NDArray ReprTexts(DType dtype, Shape layout)
        {
            using var flat = Ids.flatten();
            var texts = new string[flat.size];
            for (long i = 0; i < texts.Length; i++)
                texts[i] = ListRepr(flat.GetInt64(i));
            using var c = np.array(texts, dtype);
            using var shaped = c.reshape(Shape.Adopt((long[])Ids.Shape.dimensions.Clone()));
            var result = new NDArray(dtype, layout, true);
            try
            {
                np.copyto(result, shaped);
            }
            catch
            {
                result.Dispose();
                throw;
            }
            return result;
        }

        /// <summary>
        ///     The elementwise comparison of two list arrays (np.less's object loop over the broadcast operands), with
        ///     NDArray's null rule.
        /// </summary>
        /// <param name="op">The operator.</param>
        /// <param name="left">The left operand (null: see the operators).</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">An unorderable item pair, or an ordering against null.</exception>
        private static unsafe NDArray<bool> CompareListArrays(PyCompareOp op, NDListArray left, NDListArray right)
        {
            if (left is null || right is null)
            {
                if (op == PyCompareOp.Eq || op == PyCompareOp.Ne)
                    return BoolScalar(ReferenceEquals(left, right) == (op == PyCompareOp.Eq));
                // An ordering against None: the list array's elements against Python's None.
                return left is null
                    ? CompareObject(op, right, null, reflected: true)
                    : CompareObject(op, left, null, reflected: false);
            }
            left.ThrowIfDisposed();
            right.ThrowIfDisposed();
            // A Python list (a stand-in) facing an ARRAY is np.asarray'd into its items by the ufunc; two lists compare as
            // Python lists, which is the 0-d elementwise comparison below.
            if (left.IsStandIn != right.IsStandIn)
            {
                var (list, array) = left.IsStandIn ? (left, right) : (right, left);
                using var items = list.ItemsArray(list.FlatId(0));
                return CompareValue(op, array, items, reflected: left.IsStandIn);
            }
            var dims = BroadcastDims(left.Ids.Shape, right.Ids.Shape);
            var result = new NDArray<bool>(Shape.NpyIterOutputShape(dims, new[] { left.Ids.Shape, right.Ids.Shape }), false);
            try
            {
                NDListHeap ha = left.Heap, hb = right.Heap;
                Walk(new NDArray[] { left.Ids, right.Ids, result },
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY },
                    (dataptrs, strides, n, aux) =>
                    {
                        byte* pa = (byte*)dataptrs[0];
                        byte* pb = (byte*)dataptrs[1];
                        byte* po = (byte*)dataptrs[2];
                        long sa = strides[0], sb = strides[1], so = strides[2];
                        for (long k = 0; k < n; k++, pa += sa, pb += sb, po += so)
                            *(bool*)po = RichCompare(op, ha, *(long*)pa, hb, *(long*)pb);
                    });
            }
            catch
            {
                result.Dispose();
                throw;
            }
            return result;
        }

        /// <summary>
        ///     A list array compared with a C# value (null is Python's None; == / != with null is NDArray's reference test).
        /// </summary>
        /// <param name="op">The operator as written.</param>
        /// <param name="lists">The list array.</param>
        /// <param name="value">The other operand.</param>
        /// <param name="reflected">True when the value was the LEFT operand.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        private static NDArray<bool> CompareObject(PyCompareOp op, NDListArray lists, object value, bool reflected)
        {
            switch (value)
            {
                case NDListArray other:
                    return reflected ? CompareListArrays(op, other, lists) : CompareListArrays(op, lists, other);
                case NDCharArray:
                    throw NonStringComparison();
                case NDArray array:
                    return CompareValue(op, lists, array, reflected);
                case null when op == PyCompareOp.Eq || op == PyCompareOp.Ne:
                    // NDArray's rule for the C# null: a reference test, a 0-d bool.
                    return BoolScalar(ReferenceEquals(lists, null) == (op == PyCompareOp.Eq));
                case null:
                    return CompareNone(op, lists, reflected);
            }
            using var converted = PythonOperand(value);
            return CompareValue(op, lists, converted, reflected);
        }

        /// <summary>
        ///     A list array against an ndarray's elements as Python objects: == / != constant over the broadcast shape, an
        ///     ordering CPython's TypeError for the first pair (nothing compared: an empty result).
        /// </summary>
        /// <param name="op">The operator as written.</param>
        /// <param name="lists">The list array.</param>
        /// <param name="array">The other operand.</param>
        /// <param name="reflected">True when the array was the LEFT operand (Python's reflected comparison runs).</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's ordering.</exception>
        /// <exception cref="ArgumentNullException">An ordering with a null list array.</exception>
        private static NDArray<bool> CompareValue(PyCompareOp op, NDListArray lists, NDArray array, bool reflected)
        {
            if (lists is null)
            {
                // A null list array beside an array: NDArray's rule for the C# null, a reference test (never equal to an
                // array), and no ordering.
                if (op == PyCompareOp.Eq || op == PyCompareOp.Ne)
                    return BoolScalar((array is null) == (op == PyCompareOp.Eq));
                throw new ArgumentNullException(nameof(lists));
            }
            if (array is NDCharArray)
                throw NonStringComparison();
            lists.ThrowIfDisposed();
            if (lists.IsStandIn)
            {
                // A Python list against an ndarray: list's comparison is NotImplemented, the ndarray's reflected one runs
                // np.asarray(list) — its items — against the array's elements (a text or numeric comparison).
                using var items = lists.ItemsArray(lists.FlatId(0));
                return ItemsComparison(op, items, array, reflected);
            }
            var dims = BroadcastDims(lists.Ids.Shape, array.Shape);
            var layout = Shape.NpyIterOutputShape(dims, new[] { lists.Ids.Shape, array.Shape });
            long count = 1;
            foreach (var d in dims)
                count *= d;
            if (op == PyCompareOp.Eq || op == PyCompareOp.Ne)
            {
                // A list is never equal to a number, a str, bytes or None.
                var result = new NDArray<bool>(layout, false);
                if (count > 0)
                    result.fill(op == PyCompareOp.Ne);
                return result;
            }
            if (count == 0)
                return new NDArray<bool>(layout, false);
            // Python tries the list's comparison with the element: list.__lt__(int) is NotImplemented, so is the reflected
            // int.__gt__(list), and CPython raises naming the operator the loop called (the reflected one for x < lists).
            var called = reflected ? Reflected(op) : op;
            throw new TypeError($"'{Symbol(called)}' not supported between instances of 'list' and '{PyElementTypeName(array.dtype)}'");
        }

        /// <summary>
        ///     The comparison of a Python list's items (<c>np.asarray(list)</c>) with an ndarray, NumPy's own (NDArray's
        ///     comparison of the two arrays), as a bool array.
        /// </summary>
        /// <param name="op">The operator as written.</param>
        /// <param name="items">The list's items.</param>
        /// <param name="array">The other operand.</param>
        /// <param name="reflected">True when the ARRAY was the left operand.</param>
        /// <returns>The bool result.</returns>
        private static NDArray<bool> ItemsComparison(PyCompareOp op, NDArray items, NDArray array, bool reflected)
        {
            // list's comparison is NotImplemented for an ndarray, so the ARRAY's operator runs with the items as its
            // operand: `lst < a` is a.__gt__(lst), np.greater(a, items) — its error names greater and (a, items).
            var called = reflected ? op : Reflected(op);
            NDArray result = called switch
            {
                PyCompareOp.Eq => np.equal(array, items),
                PyCompareOp.Ne => np.not_equal(array, items),
                PyCompareOp.Lt => np.less(array, items),
                PyCompareOp.Le => np.less_equal(array, items),
                PyCompareOp.Gt => np.greater(array, items),
                _ => np.greater_equal(array, items),
            };
            return result.AsGeneric<bool>();
        }

        /// <summary>A list array ordered against Python's None (the C# null): CPython's TypeError for the first element, an
        ///     empty result for an empty array.</summary>
        /// <param name="op">The operator as written.</param>
        /// <param name="lists">The list array.</param>
        /// <param name="reflected">True when None was the LEFT operand.</param>
        /// <returns>An empty bool result.</returns>
        /// <exception cref="TypeError">The first element's ordering.</exception>
        private static NDArray<bool> CompareNone(PyCompareOp op, NDListArray lists, bool reflected)
        {
            lists.ThrowIfDisposed();
            if (lists.size == 0)
                return new NDArray<bool>(lists.Ids.Shape.KeepOrder(), false);
            var called = reflected ? Reflected(op) : op;
            throw new TypeError($"'{Symbol(called)}' not supported between instances of 'list' and 'NoneType'");
        }

        /// <summary>NumPy's refusal of a chararray beside an object array (the chararray's comparison runs first).</summary>
        /// <returns>The exception.</returns>
        private static TypeError NonStringComparison() => new TypeError("comparison of non-string arrays");

        /// <summary>A fresh 0-d bool array holding <paramref name="value"/> (the result of NDArray's null rule), created typed
        ///     so no untyped wrapper is left holding a reference.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The 0-d array.</returns>
        private static NDArray<bool> BoolScalar(bool value)
        {
            var result = new NDArray<bool>(Shape.NewScalar(), false);
            result.fill(value);
            return result;
        }
    }
}
