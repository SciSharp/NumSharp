using System;
using System.Runtime.CompilerServices;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     pythonnet codec for tuples: a C# <see cref="ValueTuple"/> or <see cref="Tuple"/> crosses into
    ///     Python as a <c>tuple</c>, and a Python <c>tuple</c> decodes into a C# tuple of the same arity.
    ///     Registered by <see cref="NDArrayPythonInterop.RegisterCodec()"/> (opt out with
    ///     <see cref="NumpyCodecOptions.ConvertTuples"/>).
    ///
    ///     <para>pythonnet has no tuple conversion of its own: a shape written as <c>numpy.zeros((2, 3))</c>
    ///     reaches numpy as an opaque wrapped <c>System.ValueTuple</c> ("expected a sequence of integers or a
    ///     single integer"), and <c>(long, long) shape = a.shape</c> fails with "cannot convert PyObject to
    ///     ValueTuple". With this codec every place a tuple is idiomatic Python — shapes, axes, strides, the
    ///     multi-index of <c>a.item((1, 2))</c>, a <c>(value, index)</c> pair returned from Python — reads the
    ///     same in C#.</para>
    ///
    ///     <para><b>Encode.</b> Any arity (8+ elements ride the <c>Rest</c> slot transparently), nested tuples,
    ///     and any element types: every element crosses through the registered codecs, so an
    ///     <see cref="NDArray"/> element becomes a zero-copy numpy view exactly as it would on its own, a
    ///     <c>null</c> element becomes <c>None</c>, and a CLR object with no Python form is wrapped as
    ///     pythonnet always wraps it.</para>
    ///
    ///     <para><b>Decode.</b> Only a Python <c>tuple</c> (or a subclass — a namedtuple decodes too) into a
    ///     C# tuple whose arity matches EXACTLY; the element types drive the per-element conversion
    ///     (<c>(NDArray, long)</c> decodes the array element through the numpy codec). A length mismatch or
    ///     a non-tuple source (a <c>list</c>) is DECLINED, so the conversion fails the ordinary pythonnet
    ///     way — a tuple decode never silently truncates or pads.</para>
    /// </summary>
    public sealed class TupleCodec : IPyObjectEncoder, IPyObjectDecoder
    {
        /// <summary>The shared instance <see cref="NDArrayPythonInterop.RegisterCodec()"/> registers.</summary>
        public static TupleCodec Instance { get; } = new TupleCodec();

        // ---- CLR -> Python -----------------------------------------------------------------------

        /// <inheritdoc/>
        public bool CanEncode(Type type) => typeof(ITuple).IsAssignableFrom(type);

        /// <inheritdoc/>
        public PyObject TryEncode(object value)
        {
            if (value is not ITuple tuple)
                return null;

            // ITuple flattens the Rest chain of an 8+ element ValueTuple/Tuple, so Length and the indexer
            // already walk the whole logical tuple. Each element takes the ordinary conversion route
            // (registered codecs included), and the element wrappers are released once the PyTuple has
            // taken its own references.
            var items = new PyObject[tuple.Length];
            try
            {
                for (int i = 0; i < items.Length; i++)
                {
                    object element = tuple[i];
                    items[i] = element is null ? PyObject.None : element.ToPython();
                }

                return EncoderHandoff.Hand(new PyTuple(items));   // pythonnet takes its own reference; the handoff bounds the wrapper's
            }
            finally
            {
                for (int i = 0; i < items.Length; i++)
                    items[i]?.Dispose();
            }
        }

        // ---- Python -> CLR -----------------------------------------------------------------------

        /// <inheritdoc/>
        public bool CanDecode(PyType objectType, Type targetType)
        {
            if (!IsTupleType(targetType))
                return false;

            try
            {
                using PyObject builtins = Py.Import("builtins");
                using PyObject tupleType = builtins.GetAttr("tuple");
                return objectType.IsSubclass(tupleType);   // tuple and its subclasses (namedtuples); never a list
            }
            catch
            {
                return false;   // CanDecode must never throw into pythonnet's conversion pipeline
            }
        }

        /// <inheritdoc/>
        public bool TryDecode<T>(PyObject pyObj, out T value)
        {
            value = default;
            if (!IsTupleType(typeof(T)))
                return false;

            using var tuple = PyTuple.AsTuple(pyObj);
            if (!TryDecodeInto(tuple, 0, typeof(T), out object result))
                return false;

            value = (T)result;
            return true;
        }

        /// <summary>
        ///     Builds <paramref name="target"/> from <paramref name="tuple"/>'s elements starting at
        ///     <paramref name="offset"/>. An 8+ element target is <c>ValueTuple&lt;T1..T7, TRest&gt;</c> (or the
        ///     <see cref="Tuple"/> twin) whose <c>TRest</c> is itself a tuple type, so the tail decodes
        ///     recursively into the Rest slot. Declines on any arity mismatch.
        /// </summary>
        private static bool TryDecodeInto(PyTuple tuple, int offset, Type target, out object result)
        {
            result = null;
            long remaining = tuple.Length() - offset;

            if (!target.IsGenericType)
            {
                if (target != typeof(ValueTuple) || remaining != 0)
                    return false;
                result = default(ValueTuple);   // the empty tuple ()
                return true;
            }

            Type[] elementTypes = target.GetGenericArguments();
            bool hasRest = elementTypes.Length == 8 && IsTupleType(elementTypes[7]);
            int direct = hasRest ? 7 : elementTypes.Length;
            if (hasRest ? remaining <= direct : remaining != direct)
                return false;

            var args = new object[elementTypes.Length];
            for (int i = 0; i < direct; i++)
            {
                using PyObject item = tuple[offset + i];
                args[i] = item.AsManagedObject(elementTypes[i]);
            }

            if (hasRest)
            {
                if (!TryDecodeInto(tuple, offset + direct, elementTypes[7], out object rest))
                    return false;
                args[7] = rest;
            }

            result = Activator.CreateInstance(target, args);
            return true;
        }

        private static bool IsTupleType(Type type)
            => typeof(ITuple).IsAssignableFrom(type) &&
               (type == typeof(ValueTuple) || type.IsGenericType) &&
               type.Namespace == "System";   // ValueTuple<...> / Tuple<...> only, never a user type that happens to implement ITuple
    }
}
