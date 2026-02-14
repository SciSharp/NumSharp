using System;
using System.Numerics;
using System.Text.RegularExpressions;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Implements Python utility functions that are often used in connection with numpy
    /// </summary>
    public static class py
    {
        public static int[] range(int n)
        {
            var a = new int[n];
            for (int i = 0; i < n; i++)
                a[i] = i;
            return a;
        }
        /// <summary>
        /// 解析单个Python风格的复数字符串为Complex对象
        /// </summary>
        private static readonly Regex _pythonComplexRegex = new Regex(
@"^(?<real>-?\d+(\.\d+)?)?((?<imagSign>\+|-)?(?<imag>\d+(\.\d+)?)?)?j$|^(?<onlyReal>-?\d+(\.\d+)?)$",
RegexOptions.IgnoreCase | RegexOptions.Compiled | RegexOptions.ExplicitCapture);
        public static Complex complex(string input)
        {
            var match = _pythonComplexRegex.Match(input);
            if (!match.Success)
                throw new FormatException($"Invalid Python complex format: '{input}'. Expected format like '10+5j', '3-2j', '4j' or '5'.");

            // 解析仅实部的场景
            if (match.Groups["onlyReal"].Success)
            {
                double real = double.Parse(match.Groups["onlyReal"].Value);
                return new Complex(real, 0);
            }

            // 解析实部（默认0）
            double realPart = 0;
            if (double.TryParse(match.Groups["real"].Value, out double r))
                realPart = r;

            // 解析虚部（处理特殊情况：j / -j / +j）
            double imagPart = 0;
            string imagStr = match.Groups["imag"].Value;
            string imagSign = match.Groups["imagSign"].Value;

            if (string.IsNullOrEmpty(imagStr) && !string.IsNullOrEmpty(input.TrimEnd('j', 'J')))
            {
                // 处理仅虚部的情况：j → 1j, -j → -1j, +j → 1j
                imagStr = "1";
            }

            if (double.TryParse(imagStr, out double im))
            {
                imagPart = im * (imagSign == "-" ? -1 : 1);
            }

            return new Complex(realPart, imagPart);
        }
    }
}
