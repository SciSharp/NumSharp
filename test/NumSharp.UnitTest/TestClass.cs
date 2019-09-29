using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.UnitTest
{
    public class TestClass
    {
        public void AssertAreEqual(object expected, object given, string msg = null)
        {
            if (expected is string)
                Assert.AreEqual(expected, given, msg ?? $"Expected '{expected}', given '{given}'");
            else if (expected is Array && given is Array)
                AssertSequenceEqual(expected as Array, given as Array);
            else if (expected is ICollection && given is ICollection)
                AssertSequenceEqual(expected as ICollection, given as ICollection);
            else if (expected is IArraySlice && given is IArraySlice)
                AssertSequenceEqual((expected as IArraySlice).ToArray(), (given as IArraySlice).ToArray());
            else if (expected is IArraySlice && given is Array)
                AssertSequenceEqual((expected as IArraySlice).ToArray(), given as Array);
            else if (expected is Array && given is IArraySlice)
                AssertSequenceEqual((expected as Array), (given as IArraySlice).ToArray());
            else
                Assert.AreEqual(expected, given, msg ?? $"Expected '{expected}', given '{given}'");
        }

        private void AssertSequenceEqual(ICollection a, ICollection b)
        {
            AssertSequenceEqual(a.OfType<object>().ToArray(), b.OfType<object>().ToArray());
        }

        private void AssertSequenceEqual(Array a, Array b)
        {
            Assert.AreEqual(a.Length, b.Length, $"Arrays are not of same length. Expected '{a.Length}', given '{b.Length}'");
            for (int i = 0; i < a.Length; i++)
                AssertAreEqual(a.GetValue(i), b.GetValue(i), $"Elements at index {i} differ. Expected '{a.GetValue(i)}', given '{b.GetValue(i)}'");
        }

        public NDArray arange(params int[] dims)
        {
            var rshape = new Shape(dims);
            return np.arange(rshape.size).reshape(rshape);
        }
        public NDArray arange(ITuple tup)
        {
            var rshape = new Shape(yield(tup).ToArray());
            return np.arange(rshape.size).reshape(rshape);
        }

        public NDArray array(ITuple tuple, params int[] vals)
        {
            return np.array(vals).reshape(yield(tuple).ToArray());
        }

        private IEnumerable<int> yield(ITuple tup)
        {
            for (int i = 0; i < tup.Length; i++)
            {
                yield return (int) tup[i];
            }
        }

        private static string _tostring(object obj)
        {
            switch (obj)
            {
                case NDArray nd:
                    return nd.ToString(false);
                case Array arr:
                    if (arr.Rank!=1 || arr.GetType().GetElementType()?.IsArray == true)
                        arr = Arrays.Flatten(arr);
                    var objs = toObjectArray(arr);
                    return $"[{string.Join(", ", objs.Select(_tostring))}]";
                default:
                    return obj?.ToString() ?? "null";
            }

            object[] toObjectArray(Array arr)
            {
                var len = arr.LongLength;
                var ret = new object[len];
                for (long i = 0; i < len; i++)
                {
                    ret[i] = arr.GetValue(i);
                }

                return ret;
            }
        }

        public static void print(object obj)
        {
            Console.WriteLine(_tostring(obj));
        }
    }
}
