using System;
using System.Linq;
using System.Linq.Expressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Maths
{
    [TestClass]
    public class npmaximumTest
    {
        [TestMethod]
        public void kek()
        {
            Expression<Func<NDArray, NDArray, bool>> expr = (lhs, rhs) => lhs > rhs;
            var parameters = expr.Parameters.ToArray();
            if (parameters.Length != 2)
                throw new ArgumentException("...");
            if (!(expr.Body is UnaryExpression uexpr))
            {
                throw new ArgumentException("...");
            }

            var oper = uexpr.Operand.NodeType;
            ;
        }
    }
}
