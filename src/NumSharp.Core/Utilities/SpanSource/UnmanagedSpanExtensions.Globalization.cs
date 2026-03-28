// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace System
{
    public static partial class UnmanagedSpanExtensions
    {
        /// <summary>
        /// Indicates whether the specified span contains only white-space characters.
        /// </summary>
        public static bool IsWhiteSpace(this ReadOnlyUnmanagedSpan<char> span)
        {
            for (long i = 0; i < span.Length; i++)
            {
                if (!char.IsWhiteSpace(span[i]))
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Returns a value indicating whether the specified <paramref name="value"/> occurs within the <paramref name="span"/>.
        /// </summary>
        /// <param name="span">The source span.</param>
        /// <param name="value">The value to seek within the source span.</param>
        /// <param name="comparisonType">One of the enumeration values that determines how the <paramref name="span"/> and <paramref name="value"/> are compared.</param>
        public static bool Contains(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> value, StringComparison comparisonType)
        {
            return IndexOf(span, value, comparisonType) >= 0;
        }

        /// <summary>
        /// Determines whether this <paramref name="span"/> and the specified <paramref name="other"/> span have the same characters
        /// when compared using the specified <paramref name="comparisonType"/> option.
        /// </summary>
        /// <param name="span">The source span.</param>
        /// <param name="other">The value to compare with the source span.</param>
        /// <param name="comparisonType">One of the enumeration values that determines how the <paramref name="span"/> and <paramref name="other"/> are compared.</param>
        [Intrinsic] // Unrolled and vectorized for half-constant input (Ordinal)
        public static bool Equals(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> other, StringComparison comparisonType)
        {
            string.CheckStringComparison(comparisonType);

            switch (comparisonType)
            {
                case StringComparison.CurrentCulture:
                case StringComparison.CurrentCultureIgnoreCase:
                    return CultureInfo.CurrentCulture.CompareInfo.Compare(span, other, string.GetCaseCompareOfComparisonCulture(comparisonType)) == 0;

                case StringComparison.InvariantCulture:
                case StringComparison.InvariantCultureIgnoreCase:
                    return CompareInfo.Invariant.Compare(span, other, string.GetCaseCompareOfComparisonCulture(comparisonType)) == 0;

                case StringComparison.Ordinal:
                    return EqualsOrdinal(span, other);

                default:
                    Debug.Assert(comparisonType == StringComparison.OrdinalIgnoreCase);
                    return EqualsOrdinalIgnoreCase(span, other);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool EqualsOrdinal(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> value)
        {
            if (span.Length != value.Length)
                return false;
            if (value.Length == 0)  // span.Length == value.Length == 0
                return true;
            return span.SequenceEqual(value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool EqualsOrdinalIgnoreCase(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> value)
        {
            if (span.Length != value.Length)
                return false;
            if (value.Length == 0)  // span.Length == value.Length == 0
                return true;
            return Ordinal.EqualsIgnoreCase(ref MemoryMarshal.GetReference(span), ref MemoryMarshal.GetReference(value), span.Length);
        }

        /// <summary>
        /// Compares the specified <paramref name="span"/> and <paramref name="other"/> using the specified <paramref name="comparisonType"/>,
        /// and returns an integer that indicates their relative position in the sort order.
        /// </summary>
        /// <param name="span">The source span.</param>
        /// <param name="other">The value to compare with the source span.</param>
        /// <param name="comparisonType">One of the enumeration values that determines how the <paramref name="span"/> and <paramref name="other"/> are compared.</param>
        public static int CompareTo(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> other, StringComparison comparisonType)
        {
            string.CheckStringComparison(comparisonType);

            switch (comparisonType)
            {
                case StringComparison.CurrentCulture:
                case StringComparison.CurrentCultureIgnoreCase:
                    return CultureInfo.CurrentCulture.CompareInfo.Compare(span, other, string.GetCaseCompareOfComparisonCulture(comparisonType));

                case StringComparison.InvariantCulture:
                case StringComparison.InvariantCultureIgnoreCase:
                    return CompareInfo.Invariant.Compare(span, other, string.GetCaseCompareOfComparisonCulture(comparisonType));

                case StringComparison.Ordinal:
                    if (span.Length == 0 || other.Length == 0)
                        return span.Length - other.Length;
                    return string.CompareOrdinal(span, other);

                default:
                    Debug.Assert(comparisonType == StringComparison.OrdinalIgnoreCase);
                    return Ordinal.CompareStringIgnoreCase(ref MemoryMarshal.GetReference(span), span.Length, ref MemoryMarshal.GetReference(other), other.Length);
            }
        }

        /// <summary>
        /// Reports the zero-based index of the first occurrence of the specified <paramref name="value"/> in the current <paramref name="span"/>.
        /// </summary>
        /// <param name="span">The source span.</param>
        /// <param name="value">The value to seek within the source span.</param>
        /// <param name="comparisonType">One of the enumeration values that determines how the <paramref name="span"/> and <paramref name="value"/> are compared.</param>
        public static long IndexOf(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> value, StringComparison comparisonType)
        {
            string.CheckStringComparison(comparisonType);

            if (comparisonType == StringComparison.Ordinal)
            {
                return UnmanagedSpanHelpers.IndexOf(ref MemoryMarshal.GetReference(span), span.Length, ref MemoryMarshal.GetReference(value), value.Length);
            }

            switch (comparisonType)
            {
                case StringComparison.CurrentCulture:
                case StringComparison.CurrentCultureIgnoreCase:
                    return CultureInfo.CurrentCulture.CompareInfo.IndexOf(span, value, string.GetCaseCompareOfComparisonCulture(comparisonType));

                case StringComparison.InvariantCulture:
                case StringComparison.InvariantCultureIgnoreCase:
                    return CompareInfo.Invariant.IndexOf(span, value, string.GetCaseCompareOfComparisonCulture(comparisonType));

                default:
                    Debug.Assert(comparisonType == StringComparison.OrdinalIgnoreCase);
                    return Ordinal.IndexOfOrdinalIgnoreCase(span, value);
            }
        }

        /// <summary>
        /// Reports the zero-based index of the last occurrence of the specified <paramref name="value"/> in the current <paramref name="span"/>.
        /// </summary>
        /// <param name="span">The source span.</param>
        /// <param name="value">The value to seek within the source span.</param>
        /// <param name="comparisonType">One of the enumeration values that determines how the <paramref name="span"/> and <paramref name="value"/> are compared.</param>
        public static long LastIndexOf(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> value, StringComparison comparisonType)
        {
            string.CheckStringComparison(comparisonType);

            if (comparisonType == StringComparison.Ordinal)
            {
                return UnmanagedSpanHelpers.LastIndexOf(
                    ref MemoryMarshal.GetReference(span),
                    span.Length,
                    ref MemoryMarshal.GetReference(value),
                    value.Length);
            }

            switch (comparisonType)
            {
                case StringComparison.CurrentCulture:
                case StringComparison.CurrentCultureIgnoreCase:
                    return CultureInfo.CurrentCulture.CompareInfo.LastIndexOf(span, value, string.GetCaseCompareOfComparisonCulture(comparisonType));

                case StringComparison.InvariantCulture:
                case StringComparison.InvariantCultureIgnoreCase:
                    return CompareInfo.Invariant.LastIndexOf(span, value, string.GetCaseCompareOfComparisonCulture(comparisonType));

                default:
                    Debug.Assert(comparisonType == StringComparison.OrdinalIgnoreCase);
                    return Ordinal.LastIndexOfOrdinalIgnoreCase(span, value);
            }
        }

        /// <summary>
        /// Copies the characters from the source span into the destination, converting each character to lowercase,
        /// using the casing rules of the specified culture.
        /// </summary>
        /// <param name="source">The source span.</param>
        /// <param name="destination">The destination span which contains the transformed characters.</param>
        /// <param name="culture">An object that supplies culture-specific casing rules.</param>
        /// <remarks>If <paramref name="culture"/> is null, <see cref="CultureInfo.CurrentCulture"/> will be used.</remarks>
        /// <returns>The number of characters written into the destination span. If the destination is too small, returns -1.</returns>
        /// <exception cref="InvalidOperationException">The source and destination buffers overlap.</exception>
        public static int ToLower(this ReadOnlyUnmanagedSpan<char> source, UnmanagedSpan<char> destination, CultureInfo? culture)
        {
            if (source.Overlaps(destination))
                ThrowHelper.ThrowInvalidOperationException(ExceptionResource.InvalidOperation_SpanOverlappedOperation);

            culture ??= CultureInfo.CurrentCulture;

            // Assuming that changing case does not affect length
            if (destination.Length < source.Length)
                return -1;

            if (GlobalizationMode.Invariant)
                InvariantModeCasing.ToLower(source, destination);
            else
                culture.TextInfo.ChangeCaseToLower(source, destination);
            return source.Length;
        }

        /// <summary>
        /// Copies the characters from the source span into the destination, converting each character to lowercase,
        /// using the casing rules of the invariant culture.
        /// </summary>
        /// <param name="source">The source span.</param>
        /// <param name="destination">The destination span which contains the transformed characters.</param>
        /// <returns>The number of characters written into the destination span. If the destination is too small, returns -1.</returns>
        /// <exception cref="InvalidOperationException">The source and destination buffers overlap.</exception>
        public static int ToLowerInvariant(this ReadOnlyUnmanagedSpan<char> source, UnmanagedSpan<char> destination)
        {
            if (source.Overlaps(destination))
                ThrowHelper.ThrowInvalidOperationException(ExceptionResource.InvalidOperation_SpanOverlappedOperation);

            // Assuming that changing case does not affect length
            if (destination.Length < source.Length)
                return -1;

            if (GlobalizationMode.Invariant)
                InvariantModeCasing.ToLower(source, destination);
            else
                TextInfo.Invariant.ChangeCaseToLower(source, destination);
            return source.Length;
        }

        /// <summary>
        /// Copies the characters from the source span into the destination, converting each character to uppercase,
        /// using the casing rules of the specified culture.
        /// </summary>
        /// <param name="source">The source span.</param>
        /// <param name="destination">The destination span which contains the transformed characters.</param>
        /// <param name="culture">An object that supplies culture-specific casing rules.</param>
        /// <remarks>If <paramref name="culture"/> is null, <see cref="CultureInfo.CurrentCulture"/> will be used.</remarks>
        /// <returns>The number of characters written into the destination span. If the destination is too small, returns -1.</returns>
        /// <exception cref="InvalidOperationException">The source and destination buffers overlap.</exception>
        public static int ToUpper(this ReadOnlyUnmanagedSpan<char> source, UnmanagedSpan<char> destination, CultureInfo? culture)
        {
            if (source.Overlaps(destination))
                ThrowHelper.ThrowInvalidOperationException(ExceptionResource.InvalidOperation_SpanOverlappedOperation);

            culture ??= CultureInfo.CurrentCulture;

            // Assuming that changing case does not affect length
            if (destination.Length < source.Length)
                return -1;

            if (GlobalizationMode.Invariant)
                InvariantModeCasing.ToUpper(source, destination);
            else
                culture.TextInfo.ChangeCaseToUpper(source, destination);
            return source.Length;
        }

        /// <summary>
        /// Copies the characters from the source span into the destination, converting each character to uppercase
        /// using the casing rules of the invariant culture.
        /// </summary>
        /// <param name="source">The source span.</param>
        /// <param name="destination">The destination span which contains the transformed characters.</param>
        /// <returns>The number of characters written into the destination span. If the destination is too small, returns -1.</returns>
        /// <exception cref="InvalidOperationException">The source and destination buffers overlap.</exception>
        public static int ToUpperInvariant(this ReadOnlyUnmanagedSpan<char> source, UnmanagedSpan<char> destination)
        {
            if (source.Overlaps(destination))
                ThrowHelper.ThrowInvalidOperationException(ExceptionResource.InvalidOperation_SpanOverlappedOperation);

            // Assuming that changing case does not affect length
            if (destination.Length < source.Length)
                return -1;

            if (GlobalizationMode.Invariant)
                InvariantModeCasing.ToUpper(source, destination);
            else
                TextInfo.Invariant.ChangeCaseToUpper(source, destination);
            return source.Length;
        }

        /// <summary>
        /// Determines whether the end of the <paramref name="span"/> matches the specified <paramref name="value"/> when compared using the specified <paramref name="comparisonType"/> option.
        /// </summary>
        /// <param name="span">The source span.</param>
        /// <param name="value">The sequence to compare to the end of the source span.</param>
        /// <param name="comparisonType">One of the enumeration values that determines how the <paramref name="span"/> and <paramref name="value"/> are compared.</param>
        [Intrinsic] // Unrolled and vectorized for half-constant input (Ordinal)
        public static bool EndsWith(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> value, StringComparison comparisonType)
        {
            string.CheckStringComparison(comparisonType);

            switch (comparisonType)
            {
                case StringComparison.CurrentCulture:
                case StringComparison.CurrentCultureIgnoreCase:
                    return CultureInfo.CurrentCulture.CompareInfo.IsSuffix(span, value, string.GetCaseCompareOfComparisonCulture(comparisonType));

                case StringComparison.InvariantCulture:
                case StringComparison.InvariantCultureIgnoreCase:
                    return CompareInfo.Invariant.IsSuffix(span, value, string.GetCaseCompareOfComparisonCulture(comparisonType));

                case StringComparison.Ordinal:
                    return span.EndsWith(value);

                default:
                    Debug.Assert(comparisonType == StringComparison.OrdinalIgnoreCase);
                    return span.EndsWithOrdinalIgnoreCase(value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool EndsWithOrdinalIgnoreCase(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> value)
            => value.Length <= span.Length
            && Ordinal.EqualsIgnoreCase(
                ref Unsafe.Add(ref MemoryMarshal.GetReference(span), span.Length - value.Length),
                ref MemoryMarshal.GetReference(value),
                value.Length);

        /// <summary>
        /// Determines whether the beginning of the <paramref name="span"/> matches the specified <paramref name="value"/> when compared using the specified <paramref name="comparisonType"/> option.
        /// </summary>
        /// <param name="span">The source span.</param>
        /// <param name="value">The sequence to compare to the beginning of the source span.</param>
        /// <param name="comparisonType">One of the enumeration values that determines how the <paramref name="span"/> and <paramref name="value"/> are compared.</param>
        [Intrinsic] // Unrolled and vectorized for half-constant input (Ordinal)
        public static bool StartsWith(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> value, StringComparison comparisonType)
        {
            string.CheckStringComparison(comparisonType);

            switch (comparisonType)
            {
                case StringComparison.CurrentCulture:
                case StringComparison.CurrentCultureIgnoreCase:
                    return CultureInfo.CurrentCulture.CompareInfo.IsPrefix(span, value, string.GetCaseCompareOfComparisonCulture(comparisonType));

                case StringComparison.InvariantCulture:
                case StringComparison.InvariantCultureIgnoreCase:
                    return CompareInfo.Invariant.IsPrefix(span, value, string.GetCaseCompareOfComparisonCulture(comparisonType));

                case StringComparison.Ordinal:
                    return span.StartsWith(value);

                default:
                    Debug.Assert(comparisonType == StringComparison.OrdinalIgnoreCase);
                    return span.StartsWithOrdinalIgnoreCase(value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool StartsWithOrdinalIgnoreCase(this ReadOnlyUnmanagedSpan<char> span, ReadOnlyUnmanagedSpan<char> value)
            => value.Length <= span.Length
            && Ordinal.EqualsIgnoreCase(ref MemoryMarshal.GetReference(span), ref MemoryMarshal.GetReference(value), value.Length);

        /// <summary>
        /// Returns an enumeration of <see cref="Rune"/> from the provided span.
        /// </summary>
        /// <remarks>
        /// Invalid sequences will be represented in the enumeration by <see cref="Rune.ReplacementChar"/>.
        /// </remarks>
        public static UnmanagedSpanRuneEnumerator EnumerateRunes(this ReadOnlyUnmanagedSpan<char> span)
        {
            return new UnmanagedSpanRuneEnumerator(span);
        }

        /// <summary>
        /// Returns an enumeration of <see cref="Rune"/> from the provided span.
        /// </summary>
        /// <remarks>
        /// Invalid sequences will be represented in the enumeration by <see cref="Rune.ReplacementChar"/>.
        /// </remarks>
        [OverloadResolutionPriority(-1)]
        public static UnmanagedSpanRuneEnumerator EnumerateRunes(this UnmanagedSpan<char> span)
        {
            return new UnmanagedSpanRuneEnumerator(span);
        }

        /// <summary>
        /// Returns an enumeration of lines over the provided span.
        /// </summary>
        /// <remarks>
        /// It is recommended that protocol parsers not utilize this API. See the documentation
        /// for <see cref="string.ReplaceLineEndings"/> for more information on how newline
        /// sequences are detected.
        /// </remarks>
        public static UnmanagedSpanLineEnumerator EnumerateLines(this ReadOnlyUnmanagedSpan<char> span)
        {
            return new UnmanagedSpanLineEnumerator(span);
        }

        /// <summary>
        /// Returns an enumeration of lines over the provided span.
        /// </summary>
        /// <remarks>
        /// It is recommended that protocol parsers not utilize this API. See the documentation
        /// for <see cref="string.ReplaceLineEndings"/> for more information on how newline
        /// sequences are detected.
        /// </remarks>
        [OverloadResolutionPriority(-1)]
        public static UnmanagedSpanLineEnumerator EnumerateLines(this UnmanagedSpan<char> span)
        {
            return new UnmanagedSpanLineEnumerator(span);
        }
    }
}
