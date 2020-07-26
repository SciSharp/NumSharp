using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;

namespace NumSharp.Utilities.Maths
{
    using OperationCallerIndex = ConcurrentDictionary<(NPTypeCode argl, NPTypeCode argr, NPTypeCode ret), BinaryOperation>;

    /// <summary>
    /// Encapsulates the binary operation metadata index for the specified static container type and operaton method name.
    /// </summary>
    /// <remarks>This class uses reflection, but only at first operation call. This index that should be a static by design, otherwise,
    /// it's used improperly. Therefore it's going to have the performance hit only on first invocation of an operation index.</remarks>
    public class BinaryOperationIndex : OperationMetadataIndex
    {
        private readonly MethodInfo _method;
        private readonly OperationCallerIndex _operations = new OperationCallerIndex();
        private readonly Type _parameter1;
        private readonly Type _parameter2;
        private readonly Type _caller;
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
                    if (parameters[2].GetGenericTypeDefinition() is var caller && caller == typeof(BinaryOperator<,,>))
                        if (_method.ReturnType != typeof(void))
                        {
                            _parameter1 = parameters[0];
                            _parameter2 = parameters[1];
                            _caller = caller;
                            _return = _method.ReturnType;
                        }
                        else
                            throw new ArgumentException($"The operation {method} is supposed to have return value.", nameof(methodName));
                    else
                        throw new ArgumentException($"The operation {method} last argument should be a gneric operator caller.", nameof(methodName));
                else
                    throw new ArgumentException($"The operation {method} is supposed to have {3} arguments.", nameof(methodName));
            }
            else
                throw new ArgumentException($"The operation method does not exist on type {type}: {methodName}.");
        }

        /// <summary>
        /// Creates the <see cref="BinaryOperation"/> operation caller helper based on the argument type codes.
        /// </summary>
        /// <param name="key">The tuple encapsulating the type codes for arguments.</param>
        /// <returns>The <see cref="BinaryOperation"/> operation caller created.</returns>
        private BinaryOperation CreateCaller((NPTypeCode argl, NPTypeCode argr, NPTypeCode ret) key)
        {
            var (argl, argr, ret) = (Translate(key.argl), Translate(key.argr), Translate(key.ret));
            var caller = typeof(BinaryOperation<,,>);
            switch ((l: _parameter1, r: _parameter2))
            {
                case var p when p.l.IsGenericParameter && p.r.IsGenericParameter:
                    throw new InvalidOperationException($"The operation {_method} does not support scalar arguments.");
                case var p when p.l.IsGenericParameter:
                    caller = typeof(BinaryROperation<,,>);
                    break;
                case var p when p.r.IsGenericParameter:
                    caller = typeof(BinaryLOperation<,,>);
                    break;
            };
            var parameters = new[] {
                _parameter1.IsGenericParameter ? argl : _parameter1,
                _parameter2.IsGenericParameter ? argr : _parameter1,
                _caller.MakeGenericType(argl, argr, ret),
                _return
            };
            var signature = typeof(Func<,,,>).MakeGenericType(parameters.ToArray());
            var operation = Delegate.CreateDelegate(signature, _method.MakeGenericMethod(argl, argr, ret));
            return Activator.CreateInstance(caller.MakeGenericType(argl, argr, ret), operation) as BinaryOperation;
        }

        /// <summary>
        /// Gets the <see cref="BinaryOperation"/> operation caller for the requested argument type codes.
        /// </summary>
        /// <param name="argl">The left operation argument type code.</param>
        /// <param name="argr">The right operation argument type code.</param>
        /// <param name="ret">The return type code.</param>
        /// <returns>The <see cref="BinaryOperation"/> operation caller returned.</returns>
        public BinaryOperation GetCaller(NPTypeCode argl, NPTypeCode argr, NPTypeCode ret)
        {
            return _operations.GetOrAdd((argl, argr, ret), CreateCaller);
        }
    }
}
