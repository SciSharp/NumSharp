using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDVStringMethods.Properties.cs — NumPy's StringDType predicates (strings phase 2, P2.6)
// =============================================================================
//
// NUMPY'S LOOP (stringdtype_ufuncs.cpp string_bool_output_unary_strided_loop)
//   Per element: NpyString_load; an NA is False under a NaN-like NA object ("NA values are always falsy"), the descriptor's
//   default string under a string NA object, and otherwise ValueError "Cannot use the isalpha function with a null that is
//   not a nan-like value" (also without an NA object: the loop tests has_string_na only after has_nan_na); then the
//   Buffer<ENCODING::UTF8> predicate of the text (NDTextElements.IsKindUtf8): code points without the trailing NUL bytes,
//   Python's Unicode rules per code point. The output is bool (resolve_descriptors: NPY_NO_CASTING, no other loop).
//
// HOW AN ELEMENT IS COMPUTED HERE
//   A short cell (text in place, at most 15 bytes) is one 16-byte vector masked to its size: ASCII text (every byte below
//   0x80) takes the byte class masks with Python's whitespace rule, other text is decoded. An arena / heap text goes to
//   IsKindUtf8 (ASCII chunks, else decoded). An NA follows the rule above, recorded in the call's VStringMethodAux and
//   raised by the driver.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDVStringMethods
    {
        /// <summary>
        ///     NumPy's <c>isalpha</c> ... <c>isnumeric</c> over <c>StringDType</c>: the bool result (or <paramref name="out"/>)
        ///     of the predicate per element, with the NA rule of the operand's descriptor (see the file header).
        /// </summary>
        /// <param name="kind">The predicate (<see cref="NDTextElements.IsAlphaKind"/> ...).</param>
        /// <param name="x">The <c>StringDType</c> operand.</param>
        /// <param name="out">The output, or null (a fresh bool array in NumPy's layout); an output of another dtype receives
        ///     the bool loop's result through NumPy's output cast.</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <returns><paramref name="out"/>, or the fresh result.</returns>
        /// <exception cref="ValueError">A selected NA that the descriptor's rule refuses (<c>Cannot use the isalpha function
        ///     with a null that is not a nan-like value</c>).</exception>
        /// <remarks>
        ///     A mask runs NDIter's per-run calls (never the masked blend of the fixed-width loops): an NA in a masked-off
        ///     position must not raise, as NumPy's masked loop never visits it. For the same reason an output of another
        ///     dtype is filled from a temporary computed under the mask.
        /// </remarks>
        internal static NDArray Predicate(int kind, NDArray x, NDArray @out, NDArray where)
        {
            if (@out is not null && @out.typecode != NPTypeCode.Boolean)
                return NDStringArithmetic.CastInto(@out, where, () => Predicate(kind, x, null, where));
            var dims = NDTextLoop.BroadcastDims(x, @out, where);
            NDArray result = @out ?? (where is null
                ? NDArray.ForOverwrite(DType.Boolean, NDStringMethods.UfuncLayout(dims, x))
                : new NDArray(DType.Boolean, NDStringMethods.UfuncLayout(dims, x, where), true));
            try
            {
                RunLocked(new[] { x }, Array.Empty<NDArray>(), x.dtype, aux =>
                {
                    var key = new TextMethodKey(TextMethodOp.VIsKind, 1, Cell, Extra: kind);
                    NDTextLoop.Execute(key, new[] { x, result }, 1, where, (void*)aux, callerOut: @out is not null);
                }, NDTextElements.PredicateName(kind));
                return result;
            }
            catch
            {
                if (@out is null)
                    result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     One element of <c>string_bool_output_unary_strided_loop</c>: the NA rule of the operand's descriptor, then the
        ///     UTF-8 predicate of the text (see the file header).
        /// </summary>
        /// <param name="cell">The element's cell.</param>
        /// <param name="kind">The predicate (a constant in the IL shell).</param>
        /// <param name="aux">The call's state (its NA rule, default string, arena base; the recorded error).</param>
        /// <returns>1 when the predicate holds, 0 otherwise (also for an NA, refused or not).</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte VIsKind(byte* cell, int kind, VStringMethodAux* aux)
        {
            if (IsShort(cell))
                return ShortIsKind(cell, kind, aux->Flags) ? (byte)1 : (byte)0;
            if (Load(cell, aux->ArenaA, out byte* buf, out long size, aux))
            {
                // NumPy's order: a NaN-like NA is always falsy; without a string NA the null is refused; a string NA reads
                // as the descriptor's default string.
                if (aux->Rule == VStringNARule.NaN)
                    return 0;
                if (aux->Rule != VStringNARule.String)
                {
                    Fail(aux, VStringMethodError.PredicateNull);
                    return 0;
                }
                Default(aux, out buf, out size);
            }
            return NDTextElements.IsKindUtf8(buf, size, kind, aux->Flags) ? (byte)1 : (byte)0;
        }

        /// <summary>
        ///     The predicate of a short cell: one 16-byte vector masked to the text's size; ASCII text through the byte class
        ///     masks (Python's whitespace rule), other text through one UTF-8 window
        ///     (<see cref="NDTextElements.IsKindUtf8Window"/>).
        /// </summary>
        /// <param name="cell">A short cell (never an NA).</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="flags">The BMP flags table (the call's <see cref="VStringMethodAux.Flags"/>).</param>
        /// <returns>The predicate's value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShortIsKind(byte* cell, int kind, uint* flags)
        {
            int size = ShortSize(cell);
            if (!Avx2.IsSupported)
                return NDTextElements.IsKindUtf8(cell, size, kind, flags);
            // end: one past the last nonzero byte (num_codepoints ignores the trailing NULs; an embedded one counts).
            var v = ShortText(cell, size, out int end, out _);
            return NDTextElements.IsKindUtf8Window(v, end, kind, flags);
        }
    }
}
