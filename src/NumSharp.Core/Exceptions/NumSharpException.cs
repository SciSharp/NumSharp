using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    public interface INumSharpException { }

    public class NumSharpException : Exception, INumSharpException
    {
        /// <summary>Initializes a new instance of the <see cref="T:System.Exception"></see> class.</summary>
        public NumSharpException()
        { }

        /// <summary>Initializes a new instance of the <see cref="T:System.Exception"></see> class with a specified error message.</summary>
        /// <param name="message">The message that describes the error.</param>
        public NumSharpException(string message) : base(message)
        { }

        /// <summary>Initializes a new instance of the <see cref="T:System.Exception"></see> class with a specified error message and a reference to the inner exception that is the cause of this exception.</summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="innerException">The exception that is the cause of the current exception, or a null reference (Nothing in Visual Basic) if no inner exception is specified.</param>
        public NumSharpException(string message, Exception innerException) : base(message, innerException)
        { }

        /// <summary>
        ///     Throws if the shape is not writeable (e.g., broadcast arrays).
        ///     Equivalent to NumPy's PyArray_FailUnlessWriteable.
        /// </summary>
        /// <param name="shape">The shape to check.</param>
        /// <param name="name">Name for the array in the error message (default: "assignment destination").</param>
        /// <exception cref="NumSharpException">If the shape is not writeable.</exception>
        /// <remarks>
        ///     NumPy raises: ValueError: assignment destination is read-only
        ///     NumSharp raises: NumSharpException: assignment destination is read-only
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ThrowIfNotWriteable(in Shape shape, string name = "assignment destination")
        {
            if (!shape.IsWriteable)
                ThrowReadOnly(name);
        }

        /// <summary>
        ///     Throws the read-only exception with the standard NumPy error format.
        /// </summary>
        /// <param name="name">Name for the array in the error message.</param>
        [DoesNotReturn]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void ThrowReadOnly(string name = "assignment destination")
        {
            throw new NumSharpException($"{name} is read-only");
        }
    }
}
