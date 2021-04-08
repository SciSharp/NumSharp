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
            else if (shape.Length == 1 && typecode == NPTypeCode.Char)
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
                {
                    if (this.size <= 10)
                    {
                        switch (typecode)
                        {
                            case NPTypeCode.Boolean:
                                s.Append(string.Join(", ", Read<bool>().ToArray().Select(v => v.ToString())));
                                break;
                            case NPTypeCode.Int32:
                                s.Append(string.Join(", ", Read<int>().ToArray().Select(v => v.ToString())));
                                break;
                            case NPTypeCode.Single:
                                s.Append(string.Join(", ", Read<float>().ToArray().Select(v => v.ToString())));
                                break;
                            case NPTypeCode.Double:
                                s.Append(string.Join(", ", Read<double>().ToArray().Select(v => v.ToString())));
                                break;
                            default:
                                break;
                        }
                    }
                    else
                    {
                        switch (typecode)
                        {
                            case NPTypeCode.Boolean:
                                var items_bool = Read<bool>().ToArray();
                                s.Append(string.Join(", ", items_bool.Take(5).Select(v => v.ToString())));
                                s.Append(", ..., ");
                                s.Append(string.Join(", ", items_bool.Skip(this.size - 10).Select(v => v.ToString())));
                                break;
                            case NPTypeCode.Int32:
                                var items_int32 = Read<int>().ToArray();
                                s.Append(string.Join(", ", items_int32.Take(5).Select(v => v.ToString())));
                                s.Append(", ..., ");
                                s.Append(string.Join(", ", items_int32.Skip(this.size - 10).Select(v => v.ToString())));
                                break;
                            case NPTypeCode.Double:
                                var items_double = Read<double>().ToArray();
                                s.Append(string.Join(", ", items_double.Take(5).Select(v => v.ToString())));
                                s.Append(", ..., ");
                                s.Append(string.Join(", ", items_double.Skip(this.size - 10).Select(v => v.ToString())));
                                break;
                            default:
                                break;
                        }
                    }
                }

                s.Append("]");
                return;
            }

            var size = shape[0];
            s.Append("[");

            if (size <= 10)
            {
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
            }
            else
            {
                for (int i = 0; i < 5; i++)
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

                s.Append(" ... ");
                s.AppendLine();

                for (int i = size - 5; i < size; i++)
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
            }

            s.Append("]");
        }
    }
}
