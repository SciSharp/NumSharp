using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Information about the memory layout of the array — the NumSharp analog of NumPy's
        ///     <c>ndarray.flags</c> (an <c>arrayflags</c> object). A fresh <see cref="NDArrayFlags"/> is
        ///     returned per access; it reads LIVE from this array (and its <see cref="NDArrayFlags.writeable"/>
        ///     setter mutates this array), so <c>a.flags.c_contiguous</c>, <c>a.flags["F"]</c> and
        ///     <c>Console.Write(a.flags)</c> all port from NumPy unchanged.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html</remarks>
        public NDArrayFlags flags => new NDArrayFlags(this);
    }

    /// <summary>
    ///     The memory-layout flags of an <see cref="NDArray"/> — a byte-for-byte port of NumPy 2.4.2's
    ///     <c>arrayflags</c> object (<c>numpy/_core/src/multiarray/flagsobject.c</c>): the lowercase
    ///     dotted attributes (<c>c_contiguous</c>, <c>f_contiguous</c>, <c>owndata</c>, <c>writeable</c>,
    ///     <c>aligned</c>, <c>writebackifcopy</c>, and the derived <c>fnc</c>/<c>forc</c>/<c>behaved</c>/
    ///     <c>carray</c>/<c>farray</c>/<c>num</c>), the bracket-key accessor (<c>flags["C"]</c>,
    ///     <c>flags["F_CONTIGUOUS"]</c>, …), the six-line <see cref="ToString"/> repr, and equality by the
    ///     integer <see cref="num"/>. Values are read live from the array's <see cref="Shape"/> and storage,
    ///     verified equal to NumPy across every layout (C/F/strided/transposed/broadcast/0-d/empty).
    ///
    ///     <para>
    ///     ALIGNED is always true (managed allocations are always aligned) and WRITEBACKIFCOPY is always
    ///     false (NumSharp has no copy-on-write handoff), matching what a NumSharp array can be.
    ///     </para>
    /// </summary>
    public sealed class NDArrayFlags : IEquatable<NDArrayFlags>
    {
        // NumPy persistent flag bit values (ndarraytypes.h), shared with NumSharp's ArrayFlags enum.
        private const int NPY_C_CONTIGUOUS = 0x0001;
        private const int NPY_F_CONTIGUOUS = 0x0002;
        private const int NPY_OWNDATA = 0x0004;
        private const int NPY_ALIGNED = 0x0100;
        private const int NPY_WRITEABLE = 0x0400;

        private readonly NDArray _arr;

        internal NDArrayFlags(NDArray arr)
        {
            _arr = arr ?? throw new ArgumentNullException(nameof(arr));
        }

        // ALIGNED is always true for NumSharp's managed allocations (matching np.require's treatment).
        private const bool A = true;
        private bool C => _arr.Shape.IsContiguous;
        private bool F => _arr.Shape.IsFContiguous;
        private bool W => _arr.Shape.IsWriteable;
        private bool O => !_arr.Storage.IsView;

        /// <summary>The data is in a single, C-style contiguous segment (<c>C_CONTIGUOUS</c>).</summary>
        public bool c_contiguous => C;

        /// <summary>The data is in a single, Fortran-style contiguous segment (<c>F_CONTIGUOUS</c>).</summary>
        public bool f_contiguous => F;

        /// <summary>Alias of <see cref="c_contiguous"/> (NumPy's <c>flags.contiguous</c>).</summary>
        public bool contiguous => C;

        /// <summary>Alias of <see cref="f_contiguous"/> (NumPy's <c>flags.fortran</c>).</summary>
        public bool fortran => F;

        /// <summary>The array owns the memory it uses — false for any view (slice/reshape/transpose/broadcast).</summary>
        public bool owndata => O;

        /// <summary>The data and all elements are aligned appropriately for the hardware — always true in NumSharp.</summary>
        public bool aligned => A;

        /// <summary>A copy-on-write handoff is pending — always false in NumSharp (unmodeled).</summary>
        public bool writebackifcopy => false;

        /// <summary>F-contiguous but NOT C-contiguous (NumPy's <c>fnc</c>).</summary>
        public bool fnc => F && !C;

        /// <summary>F-contiguous OR C-contiguous (NumPy's <c>forc</c>).</summary>
        public bool forc => F || C;

        /// <summary>ALIGNED and WRITEABLE (NumPy's <c>behaved</c>).</summary>
        public bool behaved => A && W;

        /// <summary>ALIGNED, WRITEABLE and C-contiguous (NumPy's <c>carray</c>).</summary>
        public bool carray => A && W && C;

        /// <summary>
        ///     NumPy's <c>farray</c>: <c>(ALIGNED | WRITEABLE | F_CONTIGUOUS) != 0</c> AND NOT C-contiguous —
        ///     the hand-written any-bit form (so it is true for e.g. a strided or broadcast non-C view because
        ///     ALIGNED alone satisfies the left side; since NumSharp is always aligned this reduces to
        ///     <c>!c_contiguous</c>, exactly reproducing NumPy across every probed layout).
        /// </summary>
        public bool farray => (A || W || F) && !C;

        /// <summary>
        ///     The persistent flags as a single integer (NumPy's <c>flags.num</c>): the OR of C_CONTIGUOUS
        ///     (1), F_CONTIGUOUS (2), OWNDATA (4), ALIGNED (256) and WRITEABLE (1024) — the same bit values
        ///     NumPy uses, so <c>num</c> is byte-for-byte NumPy's (e.g. 1281 for a writeable C-contiguous view).
        /// </summary>
        public int num
        {
            get
            {
                int n = A ? NPY_ALIGNED : 0;
                if (C) n |= NPY_C_CONTIGUOUS;
                if (F) n |= NPY_F_CONTIGUOUS;
                if (O) n |= NPY_OWNDATA;
                if (W) n |= NPY_WRITEABLE;
                return n;
            }
        }

        /// <summary>
        ///     Whether the array is writeable. Assignable, matching NumPy's <c>a.flags.writeable = …</c>:
        ///     setting FALSE makes the array read-only (in place). Setting TRUE re-enables an array that OWNS
        ///     its data (the round-trip on your own array); re-enabling a non-owning view — which NumPy
        ///     permits when the base is writeable — is refused here for safety (a stride-0 broadcast or a
        ///     read-only memmap must not be made writeable), a documented divergence.
        /// </summary>
        public bool writeable
        {
            get => W;
            set
            {
                if (value)
                {
                    if (W) return;
                    if (!O)
                        throw new ValueError("cannot set WRITEABLE flag to True of this array");
                    _arr.Storage.SetShapeUnsafe(_arr.Shape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE));
                }
                else
                {
                    if (!W) return;
                    _arr.Storage.SetShapeUnsafe(_arr.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
                }
            }
        }

        /// <summary>
        ///     Bracket-key access, matching NumPy's <c>flags["C_CONTIGUOUS"]</c> / <c>flags["C"]</c> mapping
        ///     (short and long spellings both accepted). GET returns the flag; SET is defined only for the
        ///     writeable/aligned/writebackifcopy keys NumPy allows. An unrecognised key raises
        ///     <see cref="KeyError"/>("Unknown flag"), as NumPy does.
        /// </summary>
        public bool this[string key]
        {
            get
            {
                switch (key)
                {
                    case "C":
                    case "CONTIGUOUS":
                    case "C_CONTIGUOUS": return C;
                    case "F":
                    case "FORTRAN":
                    case "F_CONTIGUOUS": return F;
                    case "W":
                    case "WRITEABLE": return W;
                    case "B":
                    case "BEHAVED": return behaved;
                    case "O":
                    case "OWNDATA": return O;
                    case "A":
                    case "ALIGNED": return A;
                    case "X":
                    case "WRITEBACKIFCOPY": return false;
                    case "CA":
                    case "CARRAY": return carray;
                    case "FA":
                    case "FARRAY": return farray;
                    case "FNC": return fnc;
                    case "FORC": return forc;
                    default:
                        throw new KeyError("Unknown flag");
                }
            }
            set
            {
                switch (key)
                {
                    case "W":
                    case "WRITEABLE":
                        writeable = value;
                        return;
                    case "A":
                    case "ALIGNED":
                        aligned_set(value);
                        return;
                    case "X":
                    case "WRITEBACKIFCOPY":
                        writebackifcopy_set(value);
                        return;
                    default:
                        throw new KeyError("Unknown flag");
                }
            }
        }

        // NumSharp's managed allocations are always aligned, so ALIGNED cannot actually change; the setter is
        // accepted (as NumPy accepts it) but is a no-op, a documented divergence from NumPy's mutable flag.
        private static void aligned_set(bool value) { }

        // NumPy raises when asked to set WRITEBACKIFCOPY to True (it is set only internally, by the C API);
        // False is a harmless no-op here since NumSharp never sets it.
        private static void writebackifcopy_set(bool value)
        {
            if (value)
                throw new ValueError("cannot set WRITEBACKIFCOPY flag to True");
        }

        /// <summary>Two flags objects are equal when their <see cref="num"/> integers match (NumPy's <c>__eq__</c>).</summary>
        public bool Equals(NDArrayFlags other) => other is not null && num == other.num;

        /// <inheritdoc/>
        public override bool Equals(object obj) => Equals(obj as NDArrayFlags);

        /// <inheritdoc/>
        public override int GetHashCode() => num;

        public static bool operator ==(NDArrayFlags left, NDArrayFlags right)
            => left is null ? right is null : left.Equals(right);

        public static bool operator !=(NDArrayFlags left, NDArrayFlags right) => !(left == right);

        /// <summary>
        ///     The six-line human-readable listing NumPy's <c>str(a.flags)</c> / <c>repr(a.flags)</c> produce,
        ///     e.g.<br/><c>  C_CONTIGUOUS : True</c> … <c>  WRITEBACKIFCOPY : False</c> (each line
        ///     two-space-indented, Python-cased booleans, trailing newline).
        /// </summary>
        public override string ToString()
        {
            static string B(bool v) => v ? "True" : "False";
            return $"  C_CONTIGUOUS : {B(C)}\n" +
                   $"  F_CONTIGUOUS : {B(F)}\n" +
                   $"  OWNDATA : {B(O)}\n" +
                   $"  WRITEABLE : {B(W)}\n" +
                   $"  ALIGNED : {B(A)}\n" +
                   $"  WRITEBACKIFCOPY : {B(false)}\n";
        }
    }
}
