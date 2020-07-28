using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;

namespace NumSharp.Utilities.Maths
{
    using OperatorIndex = ConcurrentDictionary<(NPTypeCode argl, NPTypeCode argr), BinaryOperator>;

    /// <summary>
    /// Encapsulates the binary operator metadata index for the specified static container type and operaton method name.
    /// </summary>
    /// <remarks>This class uses reflection, but it takes time only at instance initialization that should be a static by design, otherwise, it's used improperly.
    /// Therefore it's going to have the performance hit only on first invocation of an operator index.</remarks>
    public class BinaryOperatorIndex : OperationMetadataIndex
    {
        private readonly string _operatorName;
        private readonly OperatorIndex _operators = new OperatorIndex();

        /// <summary>
        /// Creates the new instance of a binary operator metadata index and scans the static container type for operators.
        /// </summary>
        /// <param name="type">The static container <see cref="Type"/> thet contains the operator delegates.</param>
        /// <param name="methodName">The operator delegates method info name to create the index for.</param>
        public BinaryOperatorIndex(Type type, string methodName)
        {
            if (type is null)
                throw new ArgumentNullException(nameof(type));

            _operatorName = methodName;
            foreach (var method in type.GetMethods().Where(x => x.Name == methodName))
            {
                if (method.GetParameters() is ParameterInfo[] args && args.Length == 2)
                {
                    if (args[0].ParameterType is var argl && !IsValid(argl))
                        throw new InvalidOperationException($"The left operator argument type is not supported: {argl.GetTypeCode()}.");
                    if (args[1].ParameterType is var argr && !IsValid(argr))
                        throw new InvalidOperationException($"The left operator argument type is not supported: {argr.GetTypeCode()}.");
                    if (method.ReturnType is var ret && !IsValid(ret))
                        throw new InvalidOperationException($"The left operator argument type is not supported: {argr.GetTypeCode()}.");
                    var key = (argl.GetTypeCode(), argr.GetTypeCode());
                    _operators.TryAdd((argl.GetTypeCode(), argr.GetTypeCode()), Create(key, method));
                }
                else
                    throw new InvalidOperationException($"The method {method} does not match the required parameter signature.");
            }
        }

        /// <summary>
        /// Creates the <see cref="BinaryOperator"/> operator helper based on the argument type codes.
        /// </summary>
        /// <param name="key">The tuple encapsulating the type codes for arguments.</param>
        /// <param name="method">The <see cref="MethodInfo"/> reflection method that should be used as a delegate.</param>
        /// <returns>The <see cref="BinaryOperator"/> operation created.</returns>
        private BinaryOperator Create((NPTypeCode argl, NPTypeCode argr) key, MethodInfo method)
        {
            if (method is null)
                throw new ArgumentNullException(nameof(method));

            //var (argl, argr, ret) = (Translate(key.argl), Translate(key.argr), Translate(key.ret));
            //var method = _method.MakeGenericMethod(argl, argr);
            //var caller = typeof(Operation<,,>).MakeGenericType(argl, argr, ret);
            //var ls = Expression.Parameter(typeof(NDArray), "ls");
            //var rs = Expression.Parameter(typeof(NDArray), "rs");
            //var @operator = Expression.Parameter(typeof(Operator), "operator");
            //var lambda = Expression.Lambda(Expression.Call(method, ls, rs, @operator), true, ls, rs, @operator);
            //return Activator.CreateInstance(caller, lambda.Compile()) as Operation;

            var (argl, argr, ret) = (Translate(key.argl), Translate(key.argr), Translate(method.ReturnType.GetTypeCode()));
            var @operator = typeof(BinaryOperator<,,>).MakeGenericType(argl, argr, ret);
            var signature = typeof(Func<,,>).MakeGenericType(argl, argr, ret);
            return Activator.CreateInstance(@operator, Delegate.CreateDelegate(signature, method)) as BinaryOperator;
        }

        /// <summary>
        /// Gets the <see cref="BinaryOperator"/> operator for the requested argument type codes.
        /// </summary>
        /// <param name="argl">The left operation argument type code.</param>
        /// <param name="argr">The right operation argument type code.</param>
        /// <returns>The <see cref="BinaryOperator"/> operation returned.</returns>
        public BinaryOperator Get(NPTypeCode argl, NPTypeCode argr)
        {
            if (_operators.TryGetValue((argl, argr), out var @operator))
                return @operator;
            else
                throw new ArgumentException($"The operator {_operatorName} is not supported for arguments: {argl}, {argr}.");
        }
    }
}
