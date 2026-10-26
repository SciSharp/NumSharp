using System;
using System.Linq;

namespace NumSharp.Backends
{
    /// <summary>
    ///     NumPy's parsed core signature of a generalized ufunc: the core fields of <c>PyUFuncObject</c> that
    ///     <c>_parse_signature</c> fills (<c>core_num_dims</c>, <c>core_dim_ixs</c>, <c>core_num_dim_ix</c>,
    ///     <c>core_dim_flags</c>, <c>core_signature</c>), for the product family NumSharp implements as engine members.
    /// </summary>
    /// <remarks>
    ///     The gufunc checks (<see cref="GufuncGuard.CheckCoreDimensions"/>) read the signature through these fields, so
    ///     one port of NumPy's checks serves every member of the family, as one C function serves every gufunc. A
    ///     core dimension's index is its position in the signature's list of DISTINCT names: <c>(n?,k),(k,m?)->(n?,m?)</c>
    ///     has <c>n</c> = 0, <c>k</c> = 1, <c>m</c> = 2.
    /// </remarks>
    internal sealed class GufuncSignature
    {
        /// <summary><c>np.matmul</c>: <c>(n?,k),(k,m?)-&gt;(n?,m?)</c>, whose <c>n</c> and <c>m</c> are flexible (a 1-D
        ///     operand drops them).</summary>
        internal static readonly GufuncSignature Matmul = new("matmul", "(n?,k),(k,m?)->(n?,m?)", nin: 2,
            coreDimIxs: new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 0, 2 } }, canIgnore: new[] { true, false, true });

        /// <summary><c>np.vecdot</c>: <c>(n),(n)-&gt;()</c>.</summary>
        internal static readonly GufuncSignature Vecdot = new("vecdot", "(n),(n)->()", nin: 2,
            coreDimIxs: new[] { new[] { 0 }, new[] { 0 }, Array.Empty<int>() }, canIgnore: new[] { false });

        /// <summary><c>np.matvec</c>: <c>(m,n),(n)-&gt;(m)</c>.</summary>
        internal static readonly GufuncSignature Matvec = new("matvec", "(m,n),(n)->(m)", nin: 2,
            coreDimIxs: new[] { new[] { 0, 1 }, new[] { 1 }, new[] { 0 } }, canIgnore: new[] { false, false });

        /// <summary><c>np.vecmat</c>: <c>(n),(n,m)-&gt;(m)</c>.</summary>
        internal static readonly GufuncSignature Vecmat = new("vecmat", "(n),(n,m)->(m)", nin: 2,
            coreDimIxs: new[] { new[] { 0 }, new[] { 0, 1 }, new[] { 1 } }, canIgnore: new[] { false, false });

        /// <summary>Builds a signature from NumPy's fields.</summary>
        /// <param name="name">The gufunc's name as NumPy prints it.</param>
        /// <param name="text">The signature as NumPy prints it (<c>core_signature</c>).</param>
        /// <param name="nin">The number of inputs; every other operand is an output.</param>
        /// <param name="coreDimIxs">Per operand (inputs, then outputs), the index of each of its core dimensions among
        ///     the signature's distinct names, in the operand's core order.</param>
        /// <param name="canIgnore">Per distinct core dimension, whether it is flexible (a <c>?</c>,
        ///     <c>UFUNC_CORE_DIM_CAN_IGNORE</c>).</param>
        private GufuncSignature(string name, string text, int nin, int[][] coreDimIxs, bool[] canIgnore)
        {
            Name = name;
            Text = text;
            Nin = nin;
            CoreDimIxs = coreDimIxs;
            CanIgnore = canIgnore;
        }

        /// <summary>The gufunc's name as NumPy prints it in its messages.</summary>
        internal string Name { get; }

        /// <summary>The signature as NumPy prints it in its messages (<c>core_signature</c>).</summary>
        internal string Text { get; }

        /// <summary>The number of inputs (<c>nin</c>).</summary>
        internal int Nin { get; }

        /// <summary>The number of operands, inputs and outputs (<c>nargs</c>).</summary>
        internal int Nop => CoreDimIxs.Length;

        /// <summary>Per operand, the distinct-dimension index of each of its core dimensions (<c>core_dim_ixs</c>
        ///     split at <c>core_offsets</c>). Do not modify: the instances are shared.</summary>
        internal int[][] CoreDimIxs { get; }

        /// <summary>Per distinct core dimension, whether it is flexible (<c>UFUNC_CORE_DIM_CAN_IGNORE</c>). Do not
        ///     modify: a call works on its own copy.</summary>
        internal bool[] CanIgnore { get; }

        /// <summary>The number of distinct core dimensions (<c>core_num_dim_ix</c>).</summary>
        internal int CoreNumDimIx => CanIgnore.Length;

        /// <summary>Whether any output has core dimensions (NumPy's <c>_has_output_coredims</c>): then <c>axes=</c> needs
        ///     an entry for every operand.</summary>
        internal bool HasOutputCoreDims => CoreDimIxs.Skip(Nin).Any(c => c.Length > 0);

        /// <summary>The signature's core rank of an operand (<c>core_num_dims[i]</c>), before any flexible dimension is
        ///     dropped.</summary>
        /// <param name="i">The operand.</param>
        /// <returns>The number of core dimensions.</returns>
        internal int CoreNumDims(int i) => CoreDimIxs[i].Length;
    }

    /// <summary>
    ///     The core-dimension facts of one gufunc call, as NumPy's <c>PyUFunc_GeneralizedFunctionInternal</c> settles them
    ///     before the iterator: each operand's core rank after the flexible dimensions are dropped, which dimensions went
    ///     missing, the number of loop (broadcast) dimensions, the axis remap of <c>axes=</c> / <c>axis=</c>, and the size
    ///     of every core dimension.
    /// </summary>
    internal sealed class GufuncLayout
    {
        /// <summary>Per operand, its core rank (<c>op_core_num_dims</c>): the signature's, less the flexible dimensions
        ///     marked missing, or the inputs' under keepdims for an output.</summary>
        internal int[] OpCoreNumDims;

        /// <summary>Per distinct core dimension, whether it is missing (<c>UFUNC_CORE_DIM_MISSING</c>): a flexible
        ///     dimension an operand without enough dimensions dropped.</summary>
        internal bool[] Missing;

        /// <summary>The number of loop dimensions: the largest <c>ndim - op_core_num_dims</c> over the given operands
        ///     (<c>broadcast_ndim</c>).</summary>
        internal int BroadcastNdim;

        /// <summary>Per operand, the actual axis behind each DEFAULT position (the loop axes first, then the core axes in
        ///     core order), or null for the identity (<c>remap_axis</c>).</summary>
        internal int[][] RemapAxis;

        /// <summary>Per distinct core dimension, its size (<c>core_dim_sizes</c>).</summary>
        internal long[] CoreDimSizes;

        /// <summary>Whether keepdims is set and true: the outputs keep the inputs' core rank as length-1 axes.</summary>
        internal bool Keepdims;

        /// <summary>
        ///     The actual axes carrying an operand's core dimensions, in core order: the remap of its last
        ///     <c>op_core_num_dims</c> default positions.
        /// </summary>
        /// <param name="i">The operand.</param>
        /// <param name="ndim">The operand's rank: its ndim, or for an output that was not given,
        ///     <see cref="BroadcastNdim"/> plus its core rank.</param>
        /// <returns>The axes (non-negative), one per remaining core dimension.</returns>
        internal int[] CoreAxes(int i, int ndim)
        {
            int ncore = OpCoreNumDims[i];
            var axes = new int[ncore];
            for (int k = 0; k < ncore; k++)
            {
                int position = ndim - ncore + k;
                axes[k] = RemapAxis?[i] is { } remap ? remap[position] : position;
            }
            return axes;
        }
    }
}
