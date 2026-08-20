using System;
using System.Linq;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Printing;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns the NumPy <c>str()</c> representation of this array
        ///     (equivalent to <c>np.array_str</c>), e.g. <c>[1 2 3]</c>.
        /// </summary>
        /// <remarks>
        ///     Matches NumPy 2.4.2 exactly: space separators, decimal-point alignment for floats,
        ///     summarization at <see cref="PrintOptions.threshold"/>, and line wrapping at
        ///     <see cref="PrintOptions.linewidth"/>. Use <see cref="ToString(bool)"/> with
        ///     <c>flat: true</c> for the <c>repr()</c> form (<c>array([1, 2, 3], dtype=…)</c>).
        /// </remarks>
        public override string ToString()
        {
            return ToString(flat: false);
        }

        /// <summary>
        ///     Returns the array as a string. When <paramref name="flat"/> is <c>false</c> this is the
        ///     NumPy <c>str()</c> form (<c>np.array_str</c>); when <c>true</c> it is the NumPy
        ///     <c>repr()</c> form (<c>np.array_repr</c>, i.e. <c>array([…], dtype=…)</c>).
        /// </summary>
        public string ToString(bool flat)
        {
            // NumSharp's Char dtype is used for string storage and has no NumPy equivalent; keep the
            // legacy string-oriented rendering for it so the string APIs are unaffected.
            if (typecode == NPTypeCode.Char)
                return LegacyCharToString(flat);

            var opts = PrintOptions.Current;
            return flat ? ArrayFormatter.ArrayRepr(this, opts) : ArrayFormatter.ArrayStr(this, opts);
        }

        #region legacy char/string rendering

        private string LegacyCharToString(bool flat)
        {
            var s = new StringBuilder();
            if (shape.Length == 0)
            {
                s.Append($"{Storage.GetAtIndex(0)}");
            }
            else if (shape.Length == 1)
            {
                s.Append(string.Join("", GetData<char>()));
            }
            else
            {
                if (flat)
                    s.Append("array(");
                PrettyPrintChar(s, flat);
                if (flat)
                    s.Append(")");
            }

            return s.ToString();
        }

        private void PrettyPrintChar(StringBuilder s, bool flat = false)
        {
            if (Shape.IsScalar)
            {
                s.Append($"{Storage.GetAtIndex(0)}");
                return;
            }

            if (this.size == 0)
            {
                s.Append("[]");
                return;
            }

            if (shape.Length == 1)
            {
                s.Append("[");
                s.Append("\"");
                for (long i = 0; i < this.size; i++)
                    s.Append((char)GetAtIndex(i));
                s.Append("\"");
                s.Append("]");
                return;
            }

            var size = shape[0];
            s.Append("[");

            for (long i = 0; i < size; i++)
            {
                var n_minus_one_dim_slice = this[i];
                n_minus_one_dim_slice.PrettyPrintChar(s, flat);
                if (i < size - 1)
                {
                    if (flat)
                        s.Append(", ");
                    else
                    {
                        s.Append(",");
                        s.AppendLine();
                    }
                }
            }

            s.Append("]");
        }

        #endregion
    }
}
