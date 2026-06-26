using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     Formats a single array element to its aligned string, mirroring the per-dtype formatter
    ///     classes in <c>numpy/_core/arrayprint.py</c> (BoolFormat, IntegerFormat, FloatingFormat,
    ///     ComplexFloatingFormat).
    /// </summary>
    internal interface IElementFormatter
    {
        string Format(object value);
    }

    /// <summary>Port of NumPy's <c>BoolFormat</c>.</summary>
    internal sealed class BoolFormat : IElementFormatter
    {
        private readonly string _true;

        // add an extra space so " True" and "False" align, except in 0d arrays.
        public BoolFormat(int rank) => _true = rank != 0 ? " True" : "True";

        public string Format(object value) => Convert.ToBoolean(value) ? _true : "False";
    }

    /// <summary>Port of NumPy's <c>IntegerFormat</c>.</summary>
    internal sealed class IntegerFormat : IElementFormatter
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;
        private readonly char _sign;
        private readonly int _width;

        public IntegerFormat(IReadOnlyList<object> values, char sign)
        {
            // If sign is ' ' but there are negatives, NumPy drops the space (treats it as '-').
            bool anyNegative = false;
            foreach (var v in values)
            {
                if (DecimalString(v)[0] == '-')
                {
                    anyNegative = true;
                    break;
                }
            }

            if (sign == ' ' && anyNegative)
                sign = '-';
            _sign = sign;

            int width = 0;
            foreach (var v in values)
                width = Math.Max(width, SignedString(v).Length);
            _width = width;
        }

        public string Format(object value) => SignedString(value).PadLeft(_width);

        private string SignedString(object value)
        {
            string s = DecimalString(value);
            if (s[0] == '-')
                return s; // negative already carries its sign
            return _sign switch
            {
                '+' => "+" + s,
                ' ' => " " + s,
                _ => s
            };
        }

        // Plain base-10 string of an integer-like boxed value (no overflow on long.MinValue / UInt64).
        internal static string DecimalString(object value) => value switch
        {
            sbyte x => x.ToString(CI),
            byte x => x.ToString(CI),
            short x => x.ToString(CI),
            ushort x => x.ToString(CI),
            int x => x.ToString(CI),
            uint x => x.ToString(CI),
            long x => x.ToString(CI),
            ulong x => x.ToString(CI),
            char x => ((int)x).ToString(CI),
            bool x => x ? "1" : "0",
            _ => Convert.ToInt64(value, CI).ToString(CI)
        };
    }

    /// <summary>Port of NumPy's <c>FloatingFormat</c> (the two-pass <c>fillFormat</c> column sizer).</summary>
    internal sealed class FloatingFormat : IElementFormatter
    {
        private readonly FloatKind _kind;
        private readonly bool _signPlus;
        private readonly char _sign;
        private readonly PrintOptions _opts;

        private bool _expFormat;
        private int _padLeft;
        private int _padRight;
        private int _expSize;
        private int _precision;
        private int _minDigits;
        private bool _unique;
        private TrimMode _trim;

        public FloatingFormat(IReadOnlyList<object> values, FloatKind kind, int finfoPrecision,
            PrintOptions opts, char sign)
        {
            _kind = kind;
            _opts = opts;
            _sign = sign;
            _signPlus = sign == '+';

            if (opts.floatmode == "unique")
                _precision = -1;
            else
                _precision = opts.precision;

            FillFormat(values, finfoPrecision);
        }

        private double NativeRatio(double maxAbs, double minAbs) => _kind switch
        {
            // numpy float32/float16 divide by upcasting to float, dividing, then (for half) narrowing.
            FloatKind.Single => (float)maxAbs / (float)minAbs,
            FloatKind.Half => (double)(Half)((float)maxAbs / (float)minAbs),
            _ => maxAbs / minAbs
        };

        private double AsDouble(object v) => v switch
        {
            double d => d,
            float f => f,
            Half h => (double)h,
            decimal m => (double)m,
            _ => Convert.ToDouble(v, CultureInfo.InvariantCulture)
        };

        private object Coerce(object v)
        {
            if (_kind == FloatKind.Double && v is decimal dm)
                return (double)dm;
            return v;
        }

        private void FillFormat(IReadOnlyList<object> data, int finfoPrecision)
        {
            var finite = new List<object>();
            double maxAbs = double.NegativeInfinity, minAbs = double.PositiveInfinity;
            bool anyNonZero = false;
            bool anyNegInf = false;
            bool anyNegSign = false;

            foreach (var raw in data)
            {
                var v = Coerce(raw);
                double d = AsDouble(v);
                if (double.IsFinite(d))
                {
                    finite.Add(v);
                    if (double.IsNegative(d))
                        anyNegSign = true;
                    if (d != 0.0)
                    {
                        double a = Math.Abs(d);
                        anyNonZero = true;
                        if (a > maxAbs) maxAbs = a;
                        if (a < minAbs) minAbs = a;
                    }
                }
                else if (double.IsNegativeInfinity(d))
                {
                    anyNegInf = true;
                }
            }

            // pass 1: exponential vs positional
            if (anyNonZero)
            {
                double expCutoff = Math.Pow(10.0, Math.Min(8, finfoPrecision));
                // NumPy computes max/min in the array's native float precision (np.max/np.min return
                // the native dtype and the division happens at that precision), so the ratio must be
                // evaluated there too — float16/float32 ratios differ from the double ratio at the
                // 1000 boundary.
                if (maxAbs >= expCutoff ||
                    (!_opts.suppress && (minAbs < 0.0001 || NativeRatio(maxAbs, minAbs) > 1000.0)))
                {
                    _expFormat = true;
                }
            }

            if (finite.Count == 0)
            {
                _padLeft = 0;
                _padRight = 0;
                _trim = TrimMode.Zeros;
                _expSize = -1;
                _unique = true;
                _minDigits = -1;
            }
            else if (_expFormat)
            {
                FillExponential(finite);
            }
            else
            {
                FillPositional(finite);
            }

            // legacy > 113: account for sign == ' ' by adding one to pad_left when no negatives.
            if (_sign == ' ' && !anyNegSign)
                _padLeft += 1;

            // non-finite values may need wider pad_left so nan/inf fit the field.
            if (data.Count != finite.Count)
            {
                bool neginf = _sign != '-' || anyNegInf;
                int offset = _padRight + 1; // +1 for decimal point
                _padLeft = Math.Max(_padLeft,
                    Math.Max(_opts.nanstr.Length - offset,
                        _opts.infstr.Length + (neginf ? 1 : 0) - offset));
            }
        }

        private void FillPositional(List<object> finite)
        {
            var trim = TrimMode.Zeros;
            bool unique = true;
            if (_opts.floatmode == "fixed")
            {
                trim = TrimMode.Keep;
                unique = false;
            }

            int padLeft = 0, padRight = 0;
            foreach (var v in finite)
            {
                string s = Dragon4.FormatPositional(v, _kind, _precision, 0, unique, trim, _signPlus, -1, -1);
                int dot = s.IndexOf('.');
                string intPart = dot >= 0 ? s.Substring(0, dot) : s;
                string fracPart = dot >= 0 ? s.Substring(dot + 1) : "";
                padLeft = Math.Max(padLeft, intPart.Length);
                padRight = Math.Max(padRight, fracPart.Length);
            }

            _padLeft = padLeft;
            _padRight = padRight;
            _expSize = -1;
            _unique = unique;

            if (_opts.floatmode == "fixed" || _opts.floatmode == "maxprec_equal")
            {
                _precision = _minDigits = _padRight;
                _trim = TrimMode.Keep;
            }
            else
            {
                _trim = TrimMode.Zeros;
                _minDigits = 0;
            }
        }

        private void FillExponential(List<object> finite)
        {
            var trim = TrimMode.Zeros;
            bool unique = true;
            if (_opts.floatmode == "fixed")
            {
                trim = TrimMode.Keep;
                unique = false;
            }

            int padLeft = 0, fracMax = 0, expSize = 0;
            foreach (var v in finite)
            {
                string s = Dragon4.FormatScientific(v, _kind, _precision, 0, unique, trim, _signPlus, -1, -1);
                int e = s.IndexOf('e');
                string mant = s.Substring(0, e);
                string expStr = s.Substring(e + 1); // includes sign char
                int dot = mant.IndexOf('.');
                string intPart = dot >= 0 ? mant.Substring(0, dot) : mant;
                string fracPart = dot >= 0 ? mant.Substring(dot + 1) : "";
                padLeft = Math.Max(padLeft, intPart.Length);
                fracMax = Math.Max(fracMax, fracPart.Length);
                expSize = Math.Max(expSize, expStr.Length - 1); // minus the sign char
            }

            _expSize = expSize;
            _trim = TrimMode.Keep;
            _precision = fracMax;
            _minDigits = fracMax;
            _unique = unique;
            _padLeft = padLeft;
            _padRight = _expSize + 2 + _precision;
        }

        public string Format(object value)
        {
            var v = Coerce(value);
            double d = AsDouble(v);

            if (!double.IsFinite(d))
            {
                string ret;
                if (double.IsNaN(d))
                {
                    string s = _sign == '+' ? "+" : "";
                    ret = s + _opts.nanstr;
                }
                else
                {
                    string s = d < 0 ? "-" : (_sign == '+' ? "+" : "");
                    ret = s + _opts.infstr;
                }

                int field = _padLeft + _padRight + 1;
                int deficit = field - ret.Length;
                return deficit > 0 ? new string(' ', deficit) + ret : ret;
            }

            if (_expFormat)
                return Dragon4.FormatScientific(v, _kind, _precision, _minDigits, _unique, _trim, _signPlus, _padLeft, _expSize);

            return Dragon4.FormatPositional(v, _kind, _precision, _minDigits, _unique, _trim, _signPlus, _padLeft, _padRight);
        }
    }

    /// <summary>Port of NumPy's <c>ComplexFloatingFormat</c>.</summary>
    internal sealed class ComplexFloatingFormat : IElementFormatter
    {
        private readonly FloatingFormat _real;
        private readonly FloatingFormat _imag;

        public ComplexFloatingFormat(IReadOnlyList<object> values, PrintOptions opts, char sign)
        {
            var reals = new List<object>(values.Count);
            var imags = new List<object>(values.Count);
            foreach (var v in values)
            {
                var c = (Complex)v;
                reals.Add(c.Real);
                imags.Add(c.Imaginary);
            }

            // complex128 -> double parts (finfo precision 15). imag always uses sign '+'.
            _real = new FloatingFormat(reals, FloatKind.Double, 15, opts, sign);
            _imag = new FloatingFormat(imags, FloatKind.Double, 15, opts, '+');
        }

        public string Format(object value)
        {
            var c = (Complex)value;
            string r = _real.Format(c.Real);
            string i = _imag.Format(c.Imaginary);

            // insert 'j' just before the trailing whitespace of the imaginary part.
            int sp = i.Length;
            while (sp > 0 && i[sp - 1] == ' ')
                sp--;
            i = i.Substring(0, sp) + "j" + i.Substring(sp);

            return r + i;
        }
    }
}
