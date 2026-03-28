using System;
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics.CodeAnalysis;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Minimal ThrowHelper for UnmanagedSpan operations.
    /// </summary>
    internal static class ThrowHelper
    {
        [DoesNotReturn]
        public static void ThrowArgumentOutOfRangeException()
        {
            throw new ArgumentOutOfRangeException();
        }

        [DoesNotReturn]
        public static void ThrowArgumentOutOfRangeException(ExceptionArgument argument)
        {
            throw new ArgumentOutOfRangeException(GetArgumentName(argument));
        }

        [DoesNotReturn]
        public static void ThrowArgumentNullException(ExceptionArgument argument)
        {
            throw new ArgumentNullException(GetArgumentName(argument));
        }

        [DoesNotReturn]
        public static void ThrowArgumentException()
        {
            throw new ArgumentException();
        }

        [DoesNotReturn]
        public static void ThrowArgumentException_DestinationTooShort()
        {
            throw new ArgumentException("Destination is too short.", "destination");
        }

        [DoesNotReturn]
        public static void ThrowArgumentException_OverlapAlignmentMismatch()
        {
            throw new ArgumentException("Overlap alignment mismatch.");
        }

        [DoesNotReturn]
        public static void ThrowArrayTypeMismatchException()
        {
            throw new ArrayTypeMismatchException();
        }

        [DoesNotReturn]
        public static void ThrowIndexOutOfRangeException()
        {
            throw new IndexOutOfRangeException();
        }

        [DoesNotReturn]
        public static void ThrowInvalidOperationException()
        {
            throw new InvalidOperationException();
        }

        [DoesNotReturn]
        public static void ThrowArgument_TypeContainsReferences(Type type)
        {
            throw new ArgumentException($"Type '{type}' contains references and cannot be used with unmanaged memory.");
        }

        private static string GetArgumentName(ExceptionArgument argument)
        {
            return argument switch
            {
                ExceptionArgument.destination => "destination",
                ExceptionArgument.source => "source",
                ExceptionArgument.start => "start",
                ExceptionArgument.length => "length",
                ExceptionArgument.index => "index",
                ExceptionArgument.array => "array",
                ExceptionArgument.value => "value",
                ExceptionArgument.count => "count",
                ExceptionArgument.sourceBytesToCopy => "sourceBytesToCopy",
                _ => argument.ToString()
            };
        }
    }

    /// <summary>
    /// Argument names for exception messages.
    /// </summary>
    internal enum ExceptionArgument
    {
        destination,
        source,
        start,
        length,
        index,
        array,
        value,
        count,
        sourceBytesToCopy
    }

    /// <summary>
    /// String resources for exception messages.
    /// </summary>
    internal static class SR
    {
        public const string Arg_MustBePrimArray = "Array must be a primitive array.";
        public const string Arg_MustBeNullTerminatedString = "String must be null-terminated.";
        public const string Argument_InvalidOffLen = "Invalid offset or length.";
        public const string NotSupported_CannotCallEqualsOnSpan = "Equals() on Span will always throw. Use operator== instead.";
        public const string NotSupported_CannotCallGetHashCodeOnSpan = "GetHashCode() on Span will always throw.";
    }
}
