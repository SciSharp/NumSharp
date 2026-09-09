using System;
using System.Linq.Expressions;
using System.Reflection;
using Microsoft.ML.Data;

namespace NumSharp.Interop.MLNet
{
    /// <summary>
    ///     Compiled-expression access to <see cref="VBuffer{T}"/>'s private storage, so a dense buffer's values can
    ///     be shared with a NumSharp view <b>without copying</b>.
    ///
    ///     <para><see cref="VBuffer{T}"/> exposes its values publicly only as a <see cref="ReadOnlySpan{T}"/>
    ///     (<see cref="VBuffer{T}.GetValues"/>) — a span cannot be rooted for the lifetime of an <see cref="NDArray"/>
    ///     view. But a dense <see cref="VBuffer{T}"/> is backed by a managed <c>T[]</c> in the private field
    ///     <c>_values</c> (verified across ML.NET 2.0.0 / 3.0.0 / 4.0.2 — the layout has been stable since the 1.0
    ///     redesign), and that array CAN be pinned and wrapped. This reads it through a delegate compiled ONCE per
    ///     <typeparamref name="T"/> from an expression tree (no per-call reflection, no boxing — a plain field load),
    ///     the same technique the pythonnet bridge uses to reach a foreign buffer.</para>
    ///
    ///     <para>If a future ML.NET changes the field's name or type, <see cref="GetValuesArray"/> is left
    ///     <c>null</c> and the caller falls back to a copy — never a wrong reinterpretation.</para>
    /// </summary>
    internal static class VBufferAccessor<T> where T : unmanaged
    {
        /// <summary>
        ///     Reads the private <c>T[] _values</c> backing array of a dense <see cref="VBuffer{T}"/> (which may be
        ///     LONGER than <see cref="VBuffer{T}.Length"/> — ML.NET pools buffers — so callers must slice to
        ///     <c>Length</c>). <c>null</c> when this ML.NET's VBuffer layout is not the expected managed-array form.
        /// </summary>
        internal static readonly Func<VBuffer<T>, T[]> GetValuesArray = BuildAccessor();

        private static Func<VBuffer<T>, T[]> BuildAccessor()
        {
            try
            {
                FieldInfo field = typeof(VBuffer<T>).GetField("_values", BindingFlags.NonPublic | BindingFlags.Instance);
                if (field is null || field.FieldType != typeof(T[]))
                    return null;   // layout changed (e.g. a future ReadOnlyMemory<T>-backed VBuffer) — caller copies
                ParameterExpression vb = Expression.Parameter(typeof(VBuffer<T>), "vb");
                return Expression.Lambda<Func<VBuffer<T>, T[]>>(Expression.Field(vb, field), vb).Compile();
            }
            catch
            {
                return null;
            }
        }
    }
}
