using System;
using System.Globalization;
using System.Text;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>axis</c> argument as one C# type: a single axis (<c>axis=1</c>) or a tuple of axes
    ///     (<c>axis=(0, 2)</c>, <c>axis=(1,)</c>, <c>axis=()</c>). A function whose NumPy default is <c>axis=None</c> takes
    ///     <c>NDAxis? axis = null</c>, so <c>null</c> is NumPy's <c>None</c>, and the four spellings below convert implicitly:
    ///     <code>
    ///     np.all(a, 1);                 // axis=1
    ///     np.all(a, (0, 2));            // axis=(0, 2)        - a C# tuple of two or three ints
    ///     np.all(a, new[] { 0, 2 });    // axis=(0, 2)        - an int[]: NumPy's tuple of any length, () included
    ///     np.all(a, null);              // axis=None          - the default
    ///     np.all(a, new NDAxis(0, 2));  // axis=(0, 2)        - the constructors build the same values explicitly
    ///     </code>
    ///     The value only HOLDS the axes, exactly as given: negative axes stay negative and nothing is range-checked until the
    ///     function that receives it normalizes them against its operand (<see cref="Normalize"/>), so the error texts are
    ///     the ones NumPy's function raises.
    /// </summary>
    /// <remarks>
    ///     <para>A single axis and a one-element tuple are different values, as in Python (<c>1 != (1,)</c>): most functions
    ///     answer both alike (NumPy's <c>normalize_axis_tuple</c> turns an int into a 1-tuple), but a few read the
    ///     difference - an integer axis of a 0-d array may be 0 or -1 in a reduction (<c>np.all(np.array(5), 0)</c>) where the
    ///     tuple <c>(0,)</c> is an AxisError, and <c>np.squeeze</c> checks a tuple's repeats as it reads it. Use
    ///     <see cref="IsTuple"/> to tell them apart.</para>
    ///     <para>Footguns. <c>default(NDAxis)</c> and <c>new NDAxis()</c> are the EMPTY tuple <c>()</c> (no axis at all, like
    ///     <c>new NDAxis(new int[0])</c>), not axis 0 - a reduction over <c>()</c> reduces nothing. A <c>null</c>
    ///     <c>int[]</c> converts to a null <c>NDAxis?</c> (None), but to a non-nullable <c>NDAxis</c> it throws
    ///     <see cref="ArgumentNullException"/>: the null literal converts to <c>NDAxis</c> through the <c>int[]</c>
    ///     conversion, so <c>flag ? null : someAxis</c> is typed <c>NDAxis</c> and throws when <c>flag</c> is true - write
    ///     <c>flag ? (NDAxis?)null : someAxis</c>. A <c>List&lt;int&gt;</c> has no conversion: pass <c>list.ToArray()</c>.</para>
    ///     <para>The array a tuple was built from is copied, so changing that array afterwards never changes the value.</para>
    /// </remarks>
    public readonly struct NDAxis : IEquatable<NDAxis>
    {
        /// <summary>The axes of a tuple (a private copy, never handed out), or null for a single axis and for the empty
        ///     tuple <c>default(NDAxis)</c>.</summary>
        private readonly int[] _axes;

        /// <summary>The axis of a single-axis value (0 for a tuple).</summary>
        private readonly int _axis;

        /// <summary>True for a single axis; false for every tuple, <c>default(NDAxis)</c>'s empty one included.</summary>
        private readonly bool _single;

        /// <summary>
        ///     A single axis: NumPy's <c>axis=axis</c>. The same value as the implicit conversion from <see cref="int"/>.
        /// </summary>
        /// <param name="axis">The axis as given; a negative axis counts from the end and is resolved by the function that
        ///     receives the value.</param>
        public NDAxis(int axis)
        {
            _axes = null;
            _axis = axis;
            _single = true;
        }

        /// <summary>
        ///     A tuple of axes: NumPy's <c>axis=(…)</c>, the order kept (<c>new NDAxis(2, 0)</c> is <c>(2, 0)</c>). With no
        ///     argument it is the empty tuple <c>()</c>; with exactly one <see cref="int"/> argument C# calls
        ///     <see cref="NDAxis(int)"/> instead (a single axis) - pass <c>new[] { 1 }</c> for the one-element tuple
        ///     <c>(1,)</c>.
        /// </summary>
        /// <param name="axes">The axes as given (negative ones count from the end; repeats are reported by the function that
        ///     receives the value). The array is copied.</param>
        /// <exception cref="ArgumentNullException"><paramref name="axes"/> is null (a null array is None only through the
        ///     conversion to <c>NDAxis?</c>).</exception>
        public NDAxis(params int[] axes)
        {
            if (axes is null)
                throw new ArgumentNullException(nameof(axes));
            // A private copy: the caller may reuse or change its array after the call, and the value must not change with it.
            _axes = axes.Length == 0 ? Array.Empty<int>() : (int[])axes.Clone();
            _axis = 0;
            _single = false;
        }

        /// <summary>A tuple of two axes from a C# tuple: NumPy's <c>axis=(a, b)</c>.</summary>
        /// <param name="axes">The two axes as given.</param>
        public NDAxis((int, int) axes)
        {
            _axes = new[] { axes.Item1, axes.Item2 };
            _axis = 0;
            _single = false;
        }

        /// <summary>A tuple of three axes from a C# tuple: NumPy's <c>axis=(a, b, c)</c>.</summary>
        /// <param name="axes">The three axes as given.</param>
        public NDAxis((int, int, int) axes)
        {
            _axes = new[] { axes.Item1, axes.Item2, axes.Item3 };
            _axis = 0;
            _single = false;
        }

        /// <summary>
        ///     True for a tuple of axes (<c>(0, 2)</c>, <c>(1,)</c>, <c>()</c>, and <c>default(NDAxis)</c>), false for a single
        ///     axis. The functions whose NumPy answer differs between <c>1</c> and <c>(1,)</c> read it (see the type's
        ///     remarks).
        /// </summary>
        public bool IsTuple => !_single;

        /// <summary>The number of axes: 1 for a single axis, the tuple's length otherwise (0 for <c>()</c>).</summary>
        public int Count => _single ? 1 : _axes?.Length ?? 0;

        /// <summary>The axis at <paramref name="index"/>, as given (not normalized).</summary>
        /// <param name="index">0 for a single axis; a position in the tuple otherwise.</param>
        /// <returns>The axis.</returns>
        /// <exception cref="IndexOutOfRangeException"><paramref name="index"/> is not in <c>[0, Count)</c>.</exception>
        public int this[int index]
        {
            get
            {
                if (_single)
                {
                    if (index != 0)
                        throw new IndexOutOfRangeException();
                    return _axis;
                }
                // The empty tuple has no array (default(NDAxis)): every index is out of range there.
                if (_axes is null)
                    throw new IndexOutOfRangeException();
                return _axes[index];
            }
        }

        /// <summary>
        ///     The axes as a new array, as given (not normalized): one element for a single axis, the tuple's elements
        ///     otherwise. The array is the caller's to change.
        /// </summary>
        /// <returns>A new array of <see cref="Count"/> axes.</returns>
        public int[] ToArray()
        {
            if (_single)
                return new[] { _axis };
            // A copy, so the caller can never change this value's own array.
            return _axes is null || _axes.Length == 0 ? Array.Empty<int>() : (int[])_axes.Clone();
        }

        /// <summary>
        ///     NumPy's <c>numpy.lib.array_utils.normalize_axis_tuple(axis, ndim, argname, allow_duplicate)</c>: every axis
        ///     wrapped into <c>[0, ndim)</c>, in order, then the repeat check. A single axis is the one-element tuple, as NumPy
        ///     turns an int into one.
        /// </summary>
        /// <param name="ndim">The rank the axes address.</param>
        /// <param name="argname">The argument name NumPy prints before the AxisError text and inside the repeat text
        ///     (<c>"repeated axis in `source` argument"</c>), or null for the plain texts.</param>
        /// <param name="allowDuplicate">When false (NumPy's default) a repeated axis raises.</param>
        /// <returns>A new array of the normalized axes, in the given order.</returns>
        /// <exception cref="AxisError">An axis is outside <c>[-ndim, ndim)</c> (the first one in order, reported as given),
        ///     checked for every axis before any repeat.</exception>
        /// <exception cref="ValueError">Two axes are the same after normalization and <paramref name="allowDuplicate"/> is
        ///     false (<c>repeated axis</c>).</exception>
        /// <remarks>Not every NumPy function normalizes this way: a ufunc reduction (<c>np.all</c>, <c>np.ptp</c>, the sum
        ///     behind <c>np.count_nonzero</c>) reports a repeat as <c>duplicate value in 'axis'</c> and lets a 0-d operand
        ///     take the INTEGER axis 0 / -1, and <c>np.squeeze</c> checks each axis' range and repeat together, in order.
        ///     Those functions read the axes themselves.</remarks>
        public int[] Normalize(int ndim, string argname = null, bool allowDuplicate = false)
            => Shape.normalize_axis_tuple(_single ? new[] { _axis } : _axes ?? Array.Empty<int>(), ndim, argname, allowDuplicate);

        /// <summary>A single axis: NumPy's <c>axis=axis</c> (the same value as <see cref="NDAxis(int)"/>).</summary>
        /// <param name="axis">The axis as given.</param>
        /// <returns>The single-axis value.</returns>
        /// <remarks>C# lifts this conversion, so an <c>int?</c> converts to <c>NDAxis?</c> as well: a null <c>int?</c> is
        ///     None.</remarks>
        public static implicit operator NDAxis(int axis) => new NDAxis(axis);

        /// <summary>A tuple of axes from an <see cref="int"/> array, for a non-nullable <c>NDAxis</c> parameter (the array is
        ///     copied).</summary>
        /// <param name="axes">The axes as given; an empty array is the empty tuple <c>()</c>.</param>
        /// <returns>The tuple.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="axes"/> is null: a non-nullable <c>NDAxis</c> has no
        ///     None. Convert to <c>NDAxis?</c> instead, where a null array is None.</exception>
        public static implicit operator NDAxis(int[] axes) => new NDAxis(axes);

        /// <summary>
        ///     A tuple of axes from an <see cref="int"/> array, for an <c>NDAxis?</c> parameter: a null array is None (the
        ///     meaning a null <c>int[] axis</c> had in the overloads this type replaced), any other array a tuple (copied).
        /// </summary>
        /// <param name="axes">The axes, or null for None.</param>
        /// <returns>The tuple, or null.</returns>
        public static implicit operator NDAxis?(int[] axes)
        {
            // Not `axes is null ? null : new NDAxis(axes)`: the null literal converts to NDAxis through the int[] operator
            // above, so that conditional is typed NDAxis and its null branch would call that operator, which throws.
            if (axes is null)
                return default(NDAxis?);
            return new NDAxis(axes);
        }

        /// <summary>
        ///     One integer axis or None, as an <c>NDAxis?</c>: what the kept <c>int?</c> forwards of the one-integer-axis
        ///     functions (argmax / argmin / nanargmax / nanargmin) hand their NDAxis member. It is ONE call where the
        ///     cast <c>(NDAxis?)axis</c> of an <c>int?</c> is the compiler's lifted conversion, a branch inside the argument
        ///     list; NumSharp.Build's subclass weaver recognises a forward to the same function only when its operands
        ///     pass straight on, and that branch hid it, so the forward got a dispatch of its own (np.argmax(matrix, 1)
        ///     was then the 1 x 4 row, not NumPy's 4 x 1 column).
        /// </summary>
        /// <param name="axis">The axis, or null for None.</param>
        /// <returns>The single axis, or null.</returns>
        /// <remarks>Not an implicit conversion: an <c>int? -> NDAxis?</c> operator would make <c>NDAxis? x = 1</c> ambiguous
        ///     (C#'s most-specific-operator rule finds neither it nor the lifted <c>int -> NDAxis</c> operator converting
        ///     from <c>int</c> to <c>NDAxis?</c> exactly).</remarks>
        internal static NDAxis? FromNullable(int? axis) => axis is { } single ? new NDAxis(single) : default(NDAxis?);

        /// <summary>A tuple of two axes from a C# tuple: <c>(0, 2)</c> is NumPy's <c>axis=(0, 2)</c>.</summary>
        /// <param name="axes">The two axes as given.</param>
        /// <returns>The tuple.</returns>
        public static implicit operator NDAxis((int, int) axes) => new NDAxis(axes);

        /// <summary>A tuple of three axes from a C# tuple: <c>(0, 1, 2)</c> is NumPy's <c>axis=(0, 1, 2)</c>.</summary>
        /// <param name="axes">The three axes as given.</param>
        /// <returns>The tuple.</returns>
        public static implicit operator NDAxis((int, int, int) axes) => new NDAxis(axes);

        /// <summary>Whether two values hold the same axes in the same form: a single axis never equals a tuple, as
        ///     <c>1 != (1,)</c> in Python.</summary>
        /// <param name="other">The other value.</param>
        /// <returns>True when both are single axes with the same value, or tuples with the same axes in the same
        ///     order.</returns>
        public bool Equals(NDAxis other)
        {
            if (_single != other._single)
                return false;
            if (_single)
                return _axis == other._axis;
            // default(NDAxis) (a null array) and new NDAxis(new int[0]) are both the empty tuple.
            int count = Count;
            if (count != other.Count)
                return false;
            for (int i = 0; i < count; i++)
                if (_axes[i] != other._axes[i])
                    return false;
            return true;
        }

        /// <inheritdoc cref="Equals(NDAxis)"/>
        /// <param name="obj">The other object.</param>
        /// <returns>True for an equal <see cref="NDAxis"/>.</returns>
        public override bool Equals(object obj) => obj is NDAxis other && Equals(other);

        /// <summary>A hash consistent with <see cref="Equals(NDAxis)"/>.</summary>
        /// <returns>The hash of the form and the axes.</returns>
        public override int GetHashCode()
        {
            if (_single)
                return HashCode.Combine(1, _axis);
            var hash = new HashCode();
            hash.Add(2);
            for (int i = 0; i < Count; i++)
                hash.Add(_axes[i]);
            return hash.ToHashCode();
        }

        /// <summary>Whether two values are equal (<see cref="Equals(NDAxis)"/>).</summary>
        /// <param name="left">The first value.</param>
        /// <param name="right">The second value.</param>
        /// <returns>True when they are equal.</returns>
        public static bool operator ==(NDAxis left, NDAxis right) => left.Equals(right);

        /// <summary>Whether two values differ (<see cref="Equals(NDAxis)"/>).</summary>
        /// <param name="left">The first value.</param>
        /// <param name="right">The second value.</param>
        /// <returns>True when they differ.</returns>
        public static bool operator !=(NDAxis left, NDAxis right) => !left.Equals(right);

        /// <summary>The value as Python prints the argument: <c>1</c>, <c>(0, 2)</c>, <c>(1,)</c> or <c>()</c>.</summary>
        /// <returns>The Python repr of the axis or the tuple.</returns>
        public override string ToString()
        {
            if (_single)
                return _axis.ToString(CultureInfo.InvariantCulture);
            int count = Count;
            var text = new StringBuilder("(");
            for (int i = 0; i < count; i++)
            {
                if (i > 0)
                    text.Append(", ");
                text.Append(_axes[i].ToString(CultureInfo.InvariantCulture));
            }
            // Python's one-element tuple keeps its trailing comma: (1,).
            if (count == 1)
                text.Append(',');
            return text.Append(')').ToString();
        }
    }
}
