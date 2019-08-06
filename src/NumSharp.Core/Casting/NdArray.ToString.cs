using System;
using System.Linq;
using System.Text;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        public override string ToString()
        {
            return ToString(flat: false);
        }

        public string ToString(bool flat)
        {
            var s = new StringBuilder();
            if (shape.Length == 0)
            {
                s.Append($"{Storage.GetAtIndex(0)}");
            }
            else if (shape.Length == 1 && Type.GetTypeCode(dtype) == TypeCode.Char)
            {
                s.Append(string.Join("", GetData<char>()));
            }
            else
            {
                if (flat)
                    s.Append("array(");
                PrettyPrint(s, flat);
                if (flat)
                    s.Append(")");
            }

            return s.ToString();
        }

        private void PrettyPrint(StringBuilder s, bool flat = false)
        {
            if (Shape.IsScalar)
            {
                s.Append($"{Storage.GetAtIndex(0)}");
                return;
            }

            if (shape.Length == 1)
            {
                s.Append("[");
                if (typecode == NPTypeCode.Char)
                {
                    s.Append("\"");
                    s.Append(string.Join("", this.AsIterator<char>()));
                    s.Append("\"");
                }
                else
                    s.Append(string.Join(", ", this.AsIterator().Cast<object>().Select(v => v.ToString())));

                s.Append("]");
                return;
            }

            var size = shape[0];
            s.Append("[");

            for (int i = 0; i < size; i++)
            {
                var n_minus_one_dim_slice = this[i];
                n_minus_one_dim_slice.PrettyPrint(s, flat);
                if (i < size - 1)
                {
                    s.Append(", ");
                    if (!flat)
                        s.AppendLine();
                }
            }

            s.Append("]");
        }
    }
}
