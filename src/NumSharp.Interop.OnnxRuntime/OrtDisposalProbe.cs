using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq.Expressions;
using System.Reflection;
using Microsoft.ML.OnnxRuntime;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     Reads ONNX Runtime's private disposal flags, so a call on a disposed <see cref="InferenceSession"/> or a
    ///     disposed <see cref="OrtValue"/> throws <see cref="ObjectDisposedException"/> instead of ending the process.
    /// </summary>
    /// <remarks>
    ///     <para>ONNX Runtime guards neither type: <c>Dispose</c> releases the native object and zeroes the handle, and no
    ///     member checks for it afterwards. <c>session.Run</c> on a disposed session and
    ///     <c>GetTensorTypeAndShape()</c> on a disposed value are access violations (<c>0xC0000005</c>, exit 139):
    ///     measured with ONNX Runtime 1.31.1, raw ORT and through this package alike. Both types record the disposal in a
    ///     private <c>bool _disposed</c> at the floor and at the current release (1.16.0 and 1.31.1), which this class
    ///     reads through a compiled field getter built once per type (the ML.NET bridge's <c>VBufferAccessor</c> reads a
    ///     private field the same way).</para>
    ///     <para>A version without the field gets no check (the getter is null): the guard can only remove crashes, never
    ///     refuse a live object. That is what lets the package's ONNX Runtime dependency stay open above (no upper
    ///     bound): a future ORT that renames the field loses this check, not the call. It does not make a concurrent
    ///     <c>Dispose</c> safe - a session disposed by another thread between the check and the native call still
    ///     crashes, as it does in ONNX Runtime itself.</para>
    /// </remarks>
    internal static class OrtDisposalProbe
    {
        /// <summary>Reads <c>InferenceSession._disposed</c>; null when the loaded ONNX Runtime has no such field.</summary>
        private static readonly Func<InferenceSession, bool> SessionDisposed = DisposedFieldReader<InferenceSession>();

        /// <summary>Reads <c>OrtValue._disposed</c>; null when the loaded ONNX Runtime has no such field.</summary>
        private static readonly Func<OrtValue, bool> ValueDisposed = DisposedFieldReader<OrtValue>();

        /// <summary>Has ONNX Runtime disposed this session? <c>false</c> when it cannot be told.</summary>
        /// <param name="session">A session, not null.</param>
        /// <returns><c>true</c> only when the session's own disposal flag is set.</returns>
        internal static bool IsDisposed(InferenceSession session) => SessionDisposed is not null && SessionDisposed(session);

        /// <summary>Has ONNX Runtime disposed this value? <c>false</c> when it cannot be told.</summary>
        /// <param name="value">A value, not null.</param>
        /// <returns><c>true</c> only when the value's own disposal flag is set.</returns>
        internal static bool IsDisposed(OrtValue value) => ValueDisposed is not null && ValueDisposed(value);

        /// <summary>Can the loaded ONNX Runtime's sessions be asked (it has the field)? Tests skip instead of crashing when not.</summary>
        internal static bool CanTellSessions => SessionDisposed is not null;

        /// <summary>Can the loaded ONNX Runtime's values be asked (it has the field)? Tests skip instead of crashing when not.</summary>
        internal static bool CanTellValues => ValueDisposed is not null;

        /// <summary>
        ///     Throw <see cref="ObjectDisposedException"/> for a disposed session, before any member that reaches native
        ///     code is called on it.
        /// </summary>
        /// <param name="session">The session the caller is about to run.</param>
        /// <param name="verb">The public verb, named in the message.</param>
        /// <exception cref="ObjectDisposedException">The session has been disposed.</exception>
        internal static void ThrowIfDisposed(InferenceSession session, string verb)
        {
            if (IsDisposed(session))
                throw new ObjectDisposedException(nameof(InferenceSession),
                    $"{verb}: the InferenceSession has been disposed. ONNX Runtime has released the native session, and a run on it would " +
                    "end the process (ONNX Runtime does not check this itself). Create a new session, or keep this one alive while it is used.");
        }

        /// <summary>
        ///     Build the compiled getter of <typeparamref name="T"/>'s private <c>bool _disposed</c> field, or null when
        ///     the field is absent or of another type.
        /// </summary>
        /// <remarks>
        ///     <typeparamref name="T"/> carries <see cref="DynamicallyAccessedMembersAttribute"/> for its non-public
        ///     fields, so a trimmed app keeps <c>_disposed</c> for every type this is called with and its publish shows no
        ///     warning. Unannotated, the trimmer reported IL2090 here in every trimmed consumer (the field survived only
        ///     because ONNX Runtime's own <c>Dispose</c> writes it - not a promise).
        /// </remarks>
        /// <typeparam name="T">The ONNX Runtime type.</typeparam>
        /// <returns>The getter, or null.</returns>
        private static Func<T, bool> DisposedFieldReader<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.NonPublicFields)] T>() where T : class
        {
            FieldInfo field = typeof(T).GetField("_disposed", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field is null || field.FieldType != typeof(bool))
                return null;
            ParameterExpression target = Expression.Parameter(typeof(T), "target");
            return Expression.Lambda<Func<T, bool>>(Expression.Field(target, field), target).Compile();
        }
    }
}
