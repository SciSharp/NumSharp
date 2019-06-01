using System;
using System.Collections.Generic;
using System.Text;
using Python.Runtime;

namespace Numpy
{
    public static partial class np
    {
        /// <summary>
        /// IEEE 754 floating point representation of (positive) infinity.
        /// </summary>
        public static float inf => NumPy.Instance.self.GetAttr("inf").As<float>();

        /// <summary>
        /// IEEE 754 floating point representation of (positive) infinity.
        /// 
        /// Use np.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float Inf => NumPy.Instance.self.GetAttr("inf").As<float>();

        /// <summary>
        /// IEEE 754 floating point representation of (positive) infinity.
        /// 
        /// Use np.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float Infinity => NumPy.Instance.self.GetAttr("inf").As<float>();

        /// <summary>
        /// IEEE 754 floating point representation of (positive) infinity.
        /// 
        /// Use np.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float PINF => NumPy.Instance.self.GetAttr("inf").As<float>();

        /// <summary>
        /// IEEE 754 floating point representation of (positive) infinity.
        /// 
        /// Use np.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float infty => NumPy.Instance.self.GetAttr("inf").As<float>();

        /// <summary>
        /// IEEE 754 floating point representation of (positive) infinity.
        /// </summary>
        public static float NINF => NumPy.Instance.self.GetAttr("NINF").As<float>();

        /// <summary>
        /// IEEE 754 floating point representation of Not a Number(NaN).
        /// </summary>
        public static float nan => NumPy.Instance.self.GetAttr("nan").As<float>();

        /// <summary>
        /// IEEE 754 floating point representation of Not a Number(NaN).
        /// 
        /// NaN and NAN are equivalent definitions of nan.Please use nan instead of NAN.
        /// </summary>
        public static float NaN => NumPy.Instance.self.GetAttr("nan").As<float>();

        /// <summary>
        /// IEEE 754 floating point representation of Not a Number(NaN).
        /// 
        /// NaN and NAN are equivalent definitions of nan.Please use nan instead of NAN.
        /// </summary>
        public static float NAN => NumPy.Instance.self.GetAttr("nan").As<float>();

        /// <summary>
        /// IEEE 754 floating point representation of negative zero.
        /// </summary>
        public static float NZERO => NumPy.Instance.self.GetAttr("NZERO").As<float>();

        /// <summary>
        /// IEEE 754 floating point representation of positive zero.
        /// </summary>
        public static float PZERO => NumPy.Instance.self.GetAttr("PZERO").As<float>();

        /// <summary>
        /// Euler’s constant, base of natural logarithms, Napier’s constant.
        /// </summary>
        public static float e => NumPy.Instance.self.GetAttr("e").As<float>();

        /// <summary>
        /// γ = 0.5772156649015328606065120900824024310421...
        /// https://en.wikipedia.org/wiki/Euler-Mascheroni_constant
        /// </summary>
        public static float euler_gamma => NumPy.Instance.self.GetAttr("e").As<float>();

        /// <summary>
        /// A convenient alias for None, useful for indexing arrays.
        /// </summary>
        public static object newaxis => NumPy.Instance.self.GetAttr("newaxis");

        /// <summary>
        /// pi = 3.1415926535897932384626433...
        /// </summary>
        public static float pi => NumPy.Instance.self.GetAttr("pi").As<float>();

    }
}
