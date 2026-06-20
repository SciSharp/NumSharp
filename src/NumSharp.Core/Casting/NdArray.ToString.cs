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

            // Handle empty arrays (size == 0)
            if (this.size == 0)
            {
                s.Append("[]");
                return;
            }

            if (shape.Length == 1)
            {
                s.Append("[");
                if (typecode == NPTypeCode.Char)
                {
                    s.Append("\"");
                    for (long i = 0; i < this.size; i++)
                        s.Append((char)GetAtIndex(i));
                    s.Append("\"");
                }
                else
                {
                    for (long i = 0; i < this.size; i++)
                    {
                        if (i > 0) s.Append(", ");
                        s.Append(GetAtIndex(i)?.ToString());
                    }
                }

                s.Append("]");
                return;
            }

            var size = shape[0];
            s.Append("[");

            for (long i = 0; i < size; i++)
            {
                var n_minus_one_dim_slice = this[i];
                n_minus_one_dim_slice.PrettyPrint(s, flat);
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
    }
}
