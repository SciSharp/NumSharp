using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;

namespace NumSharp.Utilities.Maths
{
    using OperationIndex = ConcurrentDictionary<(NPTypeCode argl, NPTypeCode argr, NPTypeCode ret), BinaryOperation>;

    /// <summary>
    /// Encapsulates the binary operation metadata index for the specified static container type and operaton method name.
    /// </summary>
    /// <remarks>This class uses reflection, but only at first operation call. This index that should be a static by design, otherwise,
    /// it's used improperly. Therefore it's going to have the performance hit only on first invocation of an operation index.</remarks>
    public class BinaryOperationIndex : OperationMetadataIndex
    {
        private readonly MethodInfo _method;
        private readonly OperationIndex _operations = new OperationIndex();
        private readonly Type _parameter1;
        private readonly Type _parameter2;
        private readonly Type _operator;
        private readonly Type _return;

        /// <summary>
        /// Initializes the new instance of a binary operation metadata index and sets the static operation method.
        /// </summary>
        /// <param name="type">The containing <see cref="Type"/> that holds the operation method.</param>
        /// <param name="methodName">The operation method info name to create the lambda functions for.</param>
        public BinaryOperationIndex(Type type, string methodName)
        {
            if (type is null)
                throw new ArgumentNullException(nameof(type));

            if (type.GetMethod(methodName, Filter) is MethodInfo method)
            {
                _method = method;
                var parameters = _method.GetParameters().Select(x => x.ParameterType).ToList();
                if (parameters.Count == 3)
                    if (parameters[2].GetGenericTypeDefinition() is var @operator && @operator == typeof(BinaryOperator<,,>))
                        if (_method.ReturnType != typeof(void))
                        {
                            _parameter1 = parameters[0];
                            _parameter2 = parameters[1];
                            _operator = @operator;
                            _return = _method.ReturnType;
                        }
                        else
                            throw new ArgumentException($"The operation {method} is supposed to have return value.", nameof(methodName));
                    else
                        throw new ArgumentException($"The operation {method} last argument should be a gneric operator.", nameof(methodName));
                else
                    throw new ArgumentException($"The operation {method} is supposed to have {3} arguments.", nameof(methodName));
            }
            else
                throw new ArgumentException($"The operation method does not exist on type {type}: {methodName}.");
        }

        /// <summary>
        /// Creates the <see cref="BinaryOperation"/> operation helper based on the argument type codes.
        /// </summary>
        /// <param name="key">The tuple encapsulating the type codes for arguments.</param>
        /// <returns>The <see cref="BinaryOperation"/> operation created.</returns>
        private BinaryOperation Create((NPTypeCode argl, NPTypeCode argr, NPTypeCode ret) key)
        {
            var (argl, argr, ret) = (Translate(key.argl), Translate(key.argr), Translate(key.ret));
            var @operator = typeof(BinaryOperation<,,>);
            switch ((l: _parameter1, r: _parameter2))
            {
                case var p when p.l.IsGenericParameter && p.r.IsGenericParameter:
                    throw new InvalidOperationException($"The operation {_method} does not support scalar arguments.");
                case var p when p.l.IsGenericParameter:
                    @operator = typeof(BinaryROperation<,,>);
                    break;
                case var p when p.r.IsGenericParameter:
                    @operator = typeof(BinaryLOperation<,,>);
                    break;
            };
            var parameters = new[] {
                _parameter1.IsGenericParameter ? argl : _parameter1,
                _parameter2.IsGenericParameter ? argr : _parameter1,
                _operator.MakeGenericType(argl, argr, ret),
                _return
            };
            var signature = typeof(Func<,,,>).MakeGenericType(parameters.ToArray());
            var operation = Delegate.CreateDelegate(signature, _method.MakeGenericMethod(argl, argr, ret));
            return Activator.CreateInstance(@operator.MakeGenericType(argl, argr, ret), operation) as BinaryOperation;
        }

        /// <summary>
        /// Gets the <see cref="BinaryOperation"/> operation for the requested argument type codes.
        /// </summary>
        /// <param name="argl">The left operation argument type code.</param>
        /// <param name="argr">The right operation argument type code.</param>
        /// <param name="ret">The return type code.</param>
        /// <returns>The <see cref="BinaryOperation"/> operation returned.</returns>
        public BinaryOperation Get(NPTypeCode argl, NPTypeCode argr, NPTypeCode ret)
        {
            return _operations.GetOrAdd((argl, argr, ret), Create);
        }

        /// <summary>
        /// Invokes the operation/operator combination on <see cref="NDArray"/> by matching an <see cref="BinaryOperator"/>.
        /// </summary>
        /// <param name="lhs">The left scalar operand to call the operator for.</param>
        /// <param name="rhs">The right <see cref="NDArray"/> operand to call the operator for.</param>
        /// <param name="operator">The <see cref="BinaryOperator"/> operator that maps the operator signature.</param>
        /// <returns>The resulting <see cref="NDArray"/> returned.</returns>
        public NDArray Invoke(ValueType lhs, NDArray rhs, BinaryOperatorIndex operators)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));

            var rtc = lhs.GetType().GetTypeCode();
            var ltc = rhs.GetTypeCode;

            if (operators.Get(ltc, rtc) is BinaryOperator @operator)
                if (Get(ltc, rtc, @operator.ReturnCode) is BinaryOperation operation)
                    return operation.Invoke(lhs, rhs, @operator);
                else
                    throw new InvalidOperationException($"The operation is not found for types: {lhs}, {rhs}, {@operator.ReturnCode}.");
            else
                throw new InvalidOperationException($"The operation is not found for types: {lhs}, {rhs}.");
        }

        /// <summary>
        /// Invokes the operation/operator combination on <see cref="NDArray"/> by matching an <see cref="BinaryOperator"/>.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> operand to call the operator for.</param>
        /// <param name="rhs">The right scalar operand to call the operator for.</param>
        /// <param name="operator">The <see cref="BinaryOperator"/> operator that maps the operator signature.</param>
        /// <returns>The resulting <see cref="NDArray"/> returned.</returns>
        public NDArray Invoke(NDArray lhs, ValueType rhs, BinaryOperatorIndex operators)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetType().GetTypeCode();

            if (operators.Get(ltc, rtc) is BinaryOperator @operator)
                if (Get(ltc, rtc, @operator.ReturnCode) is BinaryOperation operation)
                    return operation.Invoke(lhs, rhs, @operator);
                else
                    throw new InvalidOperationException($"The operation is not found for types: {lhs}, {rhs}, {@operator.ReturnCode}.");
            else
                throw new InvalidOperationException($"The operation is not found for types: {lhs}, {rhs}.");
        }

        /// <summary>
        /// Invokes the operation/operator combination for on <see cref="NDArray"/> by matching an <see cref="BinaryOperator"/>.
        /// </summary>
        /// <param name="lhs">The left <see cref="NDArray"/> operand to call the operator for.</param>
        /// <param name="rhs">The right <see cref="NDArray"/> operand to call the operator for.</param>
        /// <param name="operator">The <see cref="BinaryOperator"/> operator that maps the operator signature.</param>
        /// <returns>The resulting <see cref="NDArray"/> returned.</returns>
        public NDArray Invoke(NDArray lhs, NDArray rhs, BinaryOperatorIndex operators)
        {
            if (lhs is null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs is null)
                throw new ArgumentNullException(nameof(rhs));

            var ltc = lhs.GetTypeCode;
            var rtc = rhs.GetTypeCode;

            if (operators.Get(ltc, rtc) is BinaryOperator @operator)
                if (Get(ltc, rtc, @operator.ReturnCode) is BinaryOperation operation)
                    return operation.Invoke(lhs, rhs, @operator);
                else
                    throw new InvalidOperationException($"The operation is not found for types: {lhs}, {rhs}, {@operator.ReturnCode}.");
            else
                throw new InvalidOperationException($"The operation is not found for types: {lhs}, {rhs}.");
        }
    }
}
