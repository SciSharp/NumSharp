using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Construct an array from an index array and a sequence of arrays to choose from. For every
        ///     position <c>I</c> of the (broadcast) result, the output is
        ///     <c>choices[a[I]][I]</c> — i.e. the value of the <c>a[I]</c>-th choice array at that same
        ///     position. <paramref name="a"/> and every choice are first broadcast to a common shape.
        /// </summary>
        /// <param name="a">
        ///     The index array. Converted to <c>int64</c> under the <c>"safe"</c> casting rule (bool and
        ///     the signed/unsigned integers up to <c>uint32</c> are accepted; <c>uint64</c>, float,
        ///     complex and decimal are rejected with a <see cref="TypeError"/>).
        /// </param>
        /// <param name="choices">
        ///     The choice arrays. Each entry is either an <see cref="NDArray"/> (strong dtype) or a boxed
        ///     C# scalar; as in NumPy (NEP50) an <c>int</c>/<c>float</c>/<c>double</c>/<see cref="Complex"/>
        ///     literal is a weak scalar that adopts the other choices' dtype, while <c>bool</c>,
        ///     <c>char</c>, <see cref="Half"/>, <c>decimal</c> and every <see cref="NDArray"/> are strong.
        ///     All choices are broadcast against each other and against <paramref name="a"/>.
        /// </param>
        /// <param name="out">
        ///     Optional destination. Its shape must EQUAL the broadcast result shape (not merely
        ///     broadcast to it); values are cast into it with unsafe casting and the method returns
        ///     <paramref name="out"/> itself. When <c>null</c> (default) a fresh array is allocated with
        ///     the choices' common dtype.
        /// </param>
        /// <param name="mode">
        ///     How indices outside <c>[0, n-1]</c> are treated: <c>"raise"</c> (default — throw),
        ///     <c>"wrap"</c> (modulo with sign correction) or <c>"clip"</c> (values &lt; 0 → 0,
        ///     values &gt; n-1 → n-1). Case-sensitive, matching NumPy's clip-mode parser.
        /// </param>
        /// <returns>
        ///     A fresh C-contiguous array (or <paramref name="out"/>) whose dtype is
        ///     <see cref="result_type"/> of the choices (NEP50) and whose shape is the broadcast of the
        ///     choices against <paramref name="a"/>.
        /// </returns>
        /// <exception cref="ValueError">
        ///     <paramref name="choices"/> is empty (<c>"0-length sequence."</c>), an index is out of
        ///     range under <c>mode="raise"</c> (<c>"invalid entry in choice array"</c>), or
        ///     <paramref name="mode"/> is not a known clip mode (NumPy's two clip-mode texts).
        /// </exception>
        /// <exception cref="TypeError">
        ///     <paramref name="a"/> cannot be safe-cast to <c>int64</c>, or <paramref name="out"/> has the
        ///     wrong shape (<c>"choose: invalid shape for output array."</c>).
        /// </exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, <paramref name="choices"/> or a choice entry is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.choose.html
        ///     <para>Scoped by <c>[NDScoped]</c>: the int64 copy of a non-int64 index array, the 0-d arrays of
        ///     scalar choices, the converted copies of collection choices, the dtype casts of the choices, their
        ///     broadcast views and, when <paramref name="out"/> is given, the computed result that is copied into
        ///     it are all released on return, exceptions included. The result is yielded when <paramref name="out"/>
        ///     is null. Before the scope, only the index copy was released by hand: the casts, the scalar arrays and
        ///     the <c>out=</c> temporary were left to the finalizer, and a broadcast view of a cast choice kept that
        ///     cast's buffer out of the pool. The caller's arrays were built before the scope opened, so the scope
        ///     never disposes them.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray choose(NDArray a, object[] choices, NDArray @out = null, string mode = "raise")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, choices, @out, mode), static s => choose(s.a, s.choices, s.@out, s.mode), @out);
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (choices is null) throw new ArgumentNullException(nameof(choices));
            return ChooseCore(a, choices, @out, mode);
        }

        /// <summary>
        ///     <see cref="choose(NDArray,object[],NDArray,string)"/> for a strongly-typed array of
        ///     choices (the common case — every choice is an <see cref="NDArray"/>). The result dtype is
        ///     <see cref="result_type"/> of the choices.
        /// </summary>
        /// <param name="a">The index array (see <see cref="choose(NDArray,object[],NDArray,string)"/>).</param>
        /// <param name="choices">The choice arrays.</param>
        /// <param name="out">Optional destination whose shape must equal the broadcast result shape.</param>
        /// <param name="mode"><c>"raise"</c> (default), <c>"wrap"</c> or <c>"clip"</c>.</param>
        /// <returns>A fresh C-contiguous array, or <paramref name="out"/> when given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, <paramref name="choices"/> or a choice is null.</exception>
        /// <exception cref="ValueError"><paramref name="choices"/> is empty, an index is out of range under
        ///     <c>mode="raise"</c>, or <paramref name="mode"/> is not a known clip mode.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> cannot be safe-cast to int64, or <paramref name="out"/>
        ///     has the wrong shape.</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>, with the same reclaimed set as the <see cref="object"/>[] overload.</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray choose(NDArray a, NDArray[] choices, NDArray @out = null, string mode = "raise")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, choices, @out, mode), static s => choose(s.a, s.choices, s.@out, s.mode), @out);
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (choices is null) throw new ArgumentNullException(nameof(choices));
            var objs = new object[choices.Length];
            for (int i = 0; i < choices.Length; i++) objs[i] = choices[i];
            return ChooseCore(a, objs, @out, mode);
        }

        /// <summary>
        ///     <see cref="choose(NDArray,object[],NDArray,string)"/> where <paramref name="choices"/> is a
        ///     single array whose OUTERMOST dimension is taken as the sequence (NumPy's "not recommended"
        ///     abuse — <c>choices.shape[0]</c> choice sub-arrays). A 0-d <paramref name="choices"/> is a
        ///     <see cref="TypeError"/> (<c>"iteration over a 0-d array"</c>), exactly as NumPy raises.
        /// </summary>
        /// <param name="a">The index array (see <see cref="choose(NDArray,object[],NDArray,string)"/>).</param>
        /// <param name="choices">An array whose outermost dimension enumerates the choices.</param>
        /// <param name="out">Optional destination whose shape must equal the broadcast result shape.</param>
        /// <param name="mode"><c>"raise"</c> (default), <c>"wrap"</c> or <c>"clip"</c>.</param>
        /// <returns>A fresh C-contiguous array, or <paramref name="out"/> when given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="choices"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="choices"/> is 0-d, <paramref name="a"/> cannot be safe-cast
        ///     to int64, or <paramref name="out"/> has the wrong shape.</exception>
        /// <exception cref="ValueError"><paramref name="choices"/> has length 0 along its first axis, an index is out
        ///     of range under <c>mode="raise"</c>, or <paramref name="mode"/> is not a known clip mode.</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>, with the same reclaimed set as the <see cref="object"/>[] overload,
        ///     plus the views along axis 0 this overload builds (each held a reference on the caller's buffer until
        ///     the finalizer ran).</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray choose(NDArray a, NDArray choices, NDArray @out = null, string mode = "raise")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, choices, @out, mode), static s => choose(s.a, s.choices, s.@out, s.mode), @out);
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (choices is null) throw new ArgumentNullException(nameof(choices));
            if (choices.ndim == 0)
                throw new TypeError("iteration over a 0-d array");
            int n = (int)choices.Shape.dimensions[0];
            var objs = new object[n];
            for (int i = 0; i < n; i++)
                objs[i] = choices[i];   // view along axis 0
            return ChooseCore(a, objs, @out, mode);
        }

        /// <summary>
        ///     The shared body of the three <c>choose</c> overloads: classifies each choice (NEP50 weak scalar or
        ///     strong array), resolves the common dtype, casts the choices and the index array, broadcasts them to
        ///     the result shape and runs the IL gather kernel.
        /// </summary>
        /// <param name="a">The index array.</param>
        /// <param name="choices">The choices: <see cref="NDArray"/>s, boxed C# scalars, or array-likes.</param>
        /// <param name="out">Optional destination (shape must equal the result shape).</param>
        /// <param name="mode">The clip mode string, parsed by NumPy's rules.</param>
        /// <returns>The fresh result, or <paramref name="out"/> after the result is copied into it.</returns>
        /// <exception cref="ArgumentNullException">A choice entry is null.</exception>
        /// <exception cref="ValueError">No choices, an out-of-range index under raise, a bad clip mode, or a
        ///     read-only <paramref name="out"/>.</exception>
        /// <exception cref="TypeError">The index array cannot be safe-cast to int64, or <paramref name="out"/> has
        ///     the wrong shape.</exception>
        /// <remarks>
        ///     Covered by <c>[NDScopedCovered]</c>: its only callers are the three <c>[NDScoped]</c> <c>choose</c>
        ///     overloads, so every array built here except the returned result is released by the caller's scope,
        ///     exceptions included. The scope also keeps the int64 index copy reachable while the kernel reads it
        ///     through raw pointers.
        /// </remarks>
        [NDScopedCovered]
        private static unsafe NDArray ChooseCore(NDArray a, object[] choices, NDArray @out, string mode)
        {
            int modeInt = ParseChooseMode(mode);

            int n = choices.Length;
            if (n == 0)
                throw new ValueError("0-length sequence.");

            // ── 1) classify each choice + resolve the NEP50 result dtype ──────────────────
            // Mirrors NumPy's ConvertToCommonType: mark python scalars weak, then result_type.
            var mats = new NDArray[n];
            bool anyStrong = false;
            bool anyBeyondInt64 = false;
            bool anyString = false; // a fixed-width string choice: the common dtype is resolved by DESCRIPTOR below
            int weakRank = 0; // 0 none, 1 int, 2 float, 3 complex
            NPTypeCode strong = NPTypeCode.Empty;

            for (int i = 0; i < n; i++)
            {
                object item = choices[i];
                NDArray mat;
                switch (item)
                {
                    case null:
                        // NumPy would build an object array (a dtype NumSharp lacks); a null choice is a bug.
                        throw new ArgumentNullException($"choices[{i}]", "choices entries must not be null.");

                    case NDArray arr:
                        mat = arr;
                        if (arr.typecode == NPTypeCode.String)
                            anyString = true;
                        else if (!anyString)
                            strong = anyStrong ? NDExprTypeRules.PromoteStrong(strong, arr.GetTypeCode) : arr.GetTypeCode;
                        anyStrong = true;
                        break;

                    // Weak (NEP50 "python literal") scalars — the eight C# integer primitives, the two
                    // binary floats and Complex. They adopt the other choices' dtype rather than forcing
                    // promotion by their own width (int8 choice + 1000 stays int8, wrapping 1000 → -24).
                    case sbyte or byte or short or ushort or int or uint or long or ulong:
                        mat = asanyarray(item);
                        weakRank = Math.Max(weakRank, 1);
                        if (item is ulong u && u > long.MaxValue) anyBeyondInt64 = true;
                        break;

                    case float or double:
                        mat = asanyarray(item);
                        weakRank = Math.Max(weakRank, 2);
                        break;

                    case Complex:
                        mat = asanyarray(item);
                        weakRank = Math.Max(weakRank, 3);
                        break;

                    default:
                        // Strong by NumPy's rule `type(x) in (int,float,complex) else asarray(x)`:
                        // bool, char, Half, decimal, C# arrays and collections take the asarray branch.
                        mat = asanyarray(item);
                        if (mat.typecode == NPTypeCode.String)
                            anyString = true;
                        else if (!anyString)
                            strong = anyStrong ? NDExprTypeRules.PromoteStrong(strong, mat.GetTypeCode) : mat.GetTypeCode;
                        anyStrong = true;
                        break;
                }

                mats[i] = mat;
            }

            // Fixed-width strings: NumPy's ConvertToCommonType over DESCRIPTORS — the widest choice's width, S + U
            // giving U; a Python number next to a string has no common dtype (TypeError). A numeric choice array is
            // FORMATTED into that descriptor, which cannot fail — so NumPy's later checks (the index cast, the broadcast,
            // out=, the index range) still decide, and the formatting is the computational phase only where the result
            // reads a formatted element (see step 6). The gather kernels copy elemBytes bytes, any width.
            DType stringDescr = null;
            if (anyString)
            {
                // Mixed text and numbers promote to a string descriptor (a numeric choice widens it to its dtype's text
                // width); the numeric choices are formatted into it below.
                var operands = new object[n];
                for (int i = 0; i < n; i++)
                    operands[i] = choices[i] is NDArray ? (object)mats[i] : choices[i] is sbyte or byte or short or ushort or int or uint or long or ulong or float or double or Complex ? choices[i] : mats[i];
                stringDescr = DTypePromotion.ResultType(operands);
            }

            NPTypeCode dtype = anyString ? NPTypeCode.String : ResolveChooseDtype(anyStrong, strong, weakRank, anyBeyondInt64);
            int elemBytes = anyString ? (int)stringDescr.itemsize : InfoOf.GetSize(dtype);

            // ── 2) cast each choice to the common dtype (copy only when needed) ───────────
            // A numeric choice of a text result is formatted here (each number's str(), truncated to the result width):
            // NumPy's PyArray_ConvertToCommonType converts every choice before the gather, and formatting cannot fail, so
            // when it happens is unobservable.
            var castChoices = new NDArray[n];
            for (int i = 0; i < n; i++)
            {
                if (anyString)
                    castChoices[i] = mats[i].dtype.Meta == stringDescr.Meta && mats[i].dtype.itemsize == stringDescr.itemsize
                        ? mats[i]
                        : mats[i].astype(stringDescr, copy: false);
                else
                    castChoices[i] = mats[i].GetTypeCode == dtype ? mats[i] : mats[i].astype(dtype, copy: false);
            }

            // ── 3) index → int64 under the 'safe' rule ────────────────────────────────────
            // idx64 is EITHER `a` itself (already int64 — the caller's, built before any scope opened, so never
            // disposed) OR a fresh int64 astype built under the caller's scope, which releases it — and its
            // broadcast view below — on return or on the out-of-range throw.
            NDArray idx64 = CastChooseIndexToInt64(a);

            // ── 4) common broadcast shape of (all choices + index) ────────────────────────
            // (a numeric choice of a text result broadcasts with its OWN shape — its placeholder is 0-d)
            var shapes = new Shape[n + 1];
            for (int i = 0; i < n; i++) shapes[i] = castChoices[i].Shape;
            shapes[n] = idx64.Shape;
            Shape common = Shape.Broadcast(shapes)[0];
            long[] resultDims = common.dimensions;
            int ndim = resultDims.Length;

            // ── 5) validate `out` (shape must EQUAL the result shape; must be writeable) ───
            if (@out is not null)
            {
                if (!ChooseDimsEqual(@out.Shape, resultDims))
                    throw new TypeError("choose: invalid shape for output array.");
                if (!@out.Shape.IsWriteable)
                    throw new ValueError("output array is read-only");
            }

            // ── 6) allocate the fresh C-contiguous result (common dtype) ──────────────────
            var result = new NDArray(stringDescr ?? (DType)dtype, Shape.Adopt((long[])resultDims.Clone()), false);
            long totalSize = result.size;

            if (totalSize == 0)
                return FinishChoose(result, @out);   // empty — nothing to gather

            // ── 7) broadcast every choice and the index to the common shape ───────────────
            var chViews = new NDArray[n];
            bool allContig = true;
            for (int i = 0; i < n; i++)
            {
                var v = broadcast_to(castChoices[i], common);
                chViews[i] = v;
                if (!v.Shape.IsContiguous) allContig = false;
            }
            var idxView = broadcast_to(idx64, common);
            if (!idxView.Shape.IsContiguous) allContig = false;

            // ── 8) run the IL kernel (flat when every operand is C-contiguous, else strided) ─
            long status = allContig
                ? RunChooseFlat(chViews, idxView, result, elemBytes, modeInt, n, totalSize)
                : RunChooseStrided(chViews, idxView, result, elemBytes, modeInt, n, totalSize, resultDims, ndim);

            if (status < totalSize)
                throw new ValueError("invalid entry in choice array");

            return FinishChoose(result, @out);
        }

        /// <summary>
        ///     Run the flat choose kernel: every operand is C-contiguous at the result shape, so element
        ///     <c>flat</c> is at <c>base + flat*elem</c> in every choice and <c>idx[flat]</c> in the index.
        /// </summary>
        private static unsafe long RunChooseFlat(NDArray[] chViews, NDArray idxView, NDArray result, int elemBytes, int modeInt, int n, long totalSize)
        {
            var kernel = DirectILKernelGenerator.GetChooseFlatKernel(elemBytes, modeInt);
            if (kernel is null)
                throw new NotSupportedException("np.choose: IL kernel unavailable");

            byte** choiceBases = stackalloc byte*[n];
            for (int i = 0; i < n; i++)
                choiceBases[i] = (byte*)chViews[i].Storage.Address + chViews[i].Shape.offset * elemBytes;

            byte* idxBase = (byte*)idxView.Storage.Address + idxView.Shape.offset * 8;
            byte* dstBase = (byte*)result.Storage.Address;

            return kernel(choiceBases, idxBase, dstBase, totalSize, n);
        }

        /// <summary>
        ///     Run the strided choose kernel: each operand is read through its own per-dimension BYTE
        ///     strides (0 for a broadcast dim), so any layout — broadcast/scalar/strided/transposed/
        ///     negative-stride/sliced — is gathered in place with no materialisation.
        /// </summary>
        private static unsafe long RunChooseStrided(NDArray[] chViews, NDArray idxView, NDArray result, int elemBytes, int modeInt, int n, long totalSize, long[] resultDims, int ndim)
        {
            var kernel = DirectILKernelGenerator.GetChooseStridedKernel(elemBytes, modeInt);
            if (kernel is null)
                throw new NotSupportedException("np.choose: IL kernel unavailable");

            var choiceStrides = new long[n * ndim];
            for (int i = 0; i < n; i++)
            {
                var st = chViews[i].Shape.strides;
                for (int d = 0; d < ndim; d++)
                    choiceStrides[i * ndim + d] = st[d] * elemBytes;
            }

            var idxStrides = new long[ndim];
            var idxSt = idxView.Shape.strides;
            for (int d = 0; d < ndim; d++)
                idxStrides[d] = idxSt[d] * 8;

            byte** choiceBases = stackalloc byte*[n];
            for (int i = 0; i < n; i++)
                choiceBases[i] = (byte*)chViews[i].Storage.Address + chViews[i].Shape.offset * elemBytes;

            byte* idxBase = (byte*)idxView.Storage.Address + idxView.Shape.offset * 8;
            byte* dstBase = (byte*)result.Storage.Address;

            fixed (long* pChoiceStrides = choiceStrides)
            fixed (long* pIdxStrides = idxStrides)
            fixed (long* pShape = resultDims)
            {
                return kernel(choiceBases, pChoiceStrides, idxBase, pIdxStrides, dstBase, pShape, ndim, totalSize, n);
            }
        }

        /// <summary>
        ///     Deliver the computed result: when <paramref name="out"/> is supplied, unsafe-cast the fresh
        ///     result into it (NumPy's <c>PyArray_CopyInto</c>) and return <paramref name="out"/>;
        ///     otherwise return the fresh result. Computing into a temp first is what gives NumPy's
        ///     guarantee that a <c>mode="raise"</c> failure never leaves <paramref name="out"/> partially
        ///     written.
        /// </summary>
        private static NDArray FinishChoose(NDArray result, NDArray @out)
        {
            if (@out is null)
                return result;
            copyto(@out, result, casting: "unsafe");
            return @out;
        }

        /// <summary>True when <paramref name="shape"/>'s dimensions equal <paramref name="dims"/> exactly.</summary>
        private static bool ChooseDimsEqual(Shape shape, long[] dims)
        {
            var sd = shape.dimensions;
            if (sd.Length != dims.Length) return false;
            for (int i = 0; i < dims.Length; i++)
                if (sd[i] != dims[i]) return false;
            return true;
        }

        /// <summary>
        ///     NumPy's <c>result_type(*choices)</c> folded with NEP50 weak-scalar rules — the same
        ///     resolution <see cref="select"/> uses for its choices (minus the default slot).
        /// </summary>
        private static NPTypeCode ResolveChooseDtype(bool anyStrong, NPTypeCode strong, int weakRank, bool anyBeyondInt64)
        {
            if (!anyStrong)
            {
                // All weak literals — the NEP50 defaults; a weak int past int64 lifts to uint64.
                return weakRank switch
                {
                    3 => NPTypeCode.Complex,
                    2 => NPTypeCode.Double,
                    _ => anyBeyondInt64 ? NPTypeCode.UInt64 : NPTypeCode.Int64,
                };
            }

            return weakRank switch
            {
                0 => strong,
                1 => strong == NPTypeCode.Boolean ? NPTypeCode.Int64 : strong,
                2 => NDExprTypeRules.IsFloatKind(strong) || strong == NPTypeCode.Decimal || strong == NPTypeCode.Complex
                    ? strong
                    : NPTypeCode.Double,
                _ => NPTypeCode.Complex,
            };
        }

        /// <summary>
        ///     Convert the index array to <c>int64</c> (NumPy's <c>PyArray_FROM_OT(ip, NPY_INTP)</c>) under
        ///     the <c>"safe"</c> casting rule. Integer and boolean indices pass; <c>uint64</c>, float,
        ///     complex and decimal are rejected with NumPy's verbatim message. An already-<c>int64</c>
        ///     array keeps its layout (the kernel reads it through its strides).
        /// </summary>
        private static NDArray CastChooseIndexToInt64(NDArray a)
        {
            var tc = a.GetTypeCode;
            if (tc == NPTypeCode.Int64)
                return a;

            // Text never reaches int64 under 'safe'; the descriptor names its width.
            if (tc == NPTypeCode.String)
                throw new TypeError(
                    $"Cannot cast array data from {a.dtype.ToString(true)} to dtype('int64') according to the rule 'safe'");
            if (!np.can_cast(tc, NPTypeCode.Int64, "safe"))
                throw new TypeError(
                    $"Cannot cast array data from dtype('{tc.AsNumpyDtypeName()}') to dtype('int64') according to the rule 'safe'");

            return a.astype(NPTypeCode.Int64);
        }

        /// <summary>
        ///     Parse the clip mode with NumPy's <c>PyArray_ClipmodeConverter</c> (<see cref="ClipModeConverter"/>, the
        ///     parser <c>np.take</c>, <c>np.put</c> and <c>np.ravel_multi_index</c> share) and map it to the choose
        ///     kernel's constants.
        /// </summary>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c>, <c>"clip"</c>, or <c>null</c> (NumPy's <c>None</c>:
        ///     raise).</param>
        /// <returns>The <see cref="DirectILKernelGenerator"/> <c>ChooseMode*</c> constant.</returns>
        /// <exception cref="ValueError">See <see cref="ClipModeConverter"/>: a misspelled mode (<c>Use one of 'clip',
        ///     'raise', or 'wrap' for clip mode</c>) or any other string (<c>clipmode must be one of 'clip', 'raise', or
        ///     'wrap' (got 'bogus')</c>).</exception>
        /// <remarks>The mapping is spelled out, not assumed: the shared numbering and the choose kernel's are equal
        ///     today, and a switch keeps a later renumbering of either from silently selecting the wrong mode.</remarks>
        private static int ParseChooseMode(string mode) => ClipModeConverter(mode) switch
        {
            ClipModeWrap => DirectILKernelGenerator.ChooseModeWrap,
            ClipModeClip => DirectILKernelGenerator.ChooseModeClip,
            _ => DirectILKernelGenerator.ChooseModeRaise,
        };
    }
}
