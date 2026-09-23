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
        ///     The single writeability gate for every in-place write in the library — the indexing
        ///     setters, <see cref="NDArray"/>.fill, put / place / putmask / put_along_axis, the ufunc
        ///     <c>out=</c> routes, copyto, the matmul / clip / cumsum / fft <c>out=</c> paths, the
        ///     real/imag setters and shuffle all funnel through here (NumPy's
        ///     <c>PyArray_FailUnlessWriteable</c>). It raises when the destination is NOT writeable —
        ///     most importantly a broadcast view (any stride-0 axis reports
        ///     <see cref="Shape.IsWriteable"/> == <c>false</c>), so an attempted write through shared
        ///     broadcast memory is REFUSED here rather than silently corrupting every element that
        ///     aliases the same buffer.
        /// </summary>
        /// <param name="shape">The destination shape to check for writeability.</param>
        /// <param name="name">
        ///     The array's name in the error message (default: "assignment destination"). This is
        ///     load-bearing, not cosmetic: NumPy words the read-only message per operation
        ///     ("output array", "put: output array", "WRITEBACKIFCOPY base", "array", …) and the
        ///     differential-fuzz error-parity gate compares the message VERBATIM, so each caller
        ///     must pass the exact spelling NumPy uses for that operation.
        /// </param>
        /// <exception cref="ValueError">If the shape is not writeable.</exception>
        /// <remarks>
        ///     Raises NumPy's exact exception — type AND message — <c>ValueError: {name} is
        ///     read-only</c>, so Python code ported as <c>except ValueError</c> catches it and the
        ///     oracle's error-parity tier passes without an excuse. <see cref="ValueError"/> derives
        ///     from <see cref="System.ArgumentException"/> and implements
        ///     <see cref="INumSharpException"/>, so a handler catching either of those still works;
        ///     a handler that caught <see cref="NumSharpException"/> specifically for a read-only
        ///     write must switch to <see cref="ValueError"/>.
        /// </remarks>
        [MethodImpl(Inline)]
        public static void ThrowIfNotWriteable(Shape shape, string name = "assignment destination")
        {
            if (!shape.IsWriteable)
                ThrowReadOnly(name);
        }

        /// <summary>
        ///     Throws the read-only failure unconditionally, in NumPy's exact
        ///     <c>ValueError: {name} is read-only</c> form. Split out from
        ///     <see cref="ThrowIfNotWriteable"/> and marked non-inlining so the common
        ///     writeable-fast-path stays inlineable while the throw (cold) does not bloat callers.
        /// </summary>
        /// <param name="name">
        ///     The array's name in the message; see <see cref="ThrowIfNotWriteable"/> for why the
        ///     spelling must match NumPy's per-operation wording.
        /// </param>
        /// <exception cref="ValueError">Always — this method never returns.</exception>
        [DoesNotReturn]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void ThrowReadOnly(string name = "assignment destination")
        {
            // NumPy raises ValueError for every read-only write (probed 2.4.2 across the whole
            // surface: assignment / fill / put / place / putmask / copyto / ufunc-out / matmul-out /
            // fft-out / shuffle / real-setter — all "ValueError: … is read-only"). ValueError is the
            // type the rest of the codebase's read-only guards already use (np.choose, np.nditer,
            // AxisPartition, byteswap, getfield), so this unifies the last NumSharpException hold-out
            // onto NumPy's type and clears the (K10) differential-fuzz excuse.
            throw new ValueError($"{name} is read-only");
        }
    }
}
