using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Set array flags WRITEABLE, ALIGNED and WRITEBACKIFCOPY, respectively — the port of NumPy's
        ///     <c>ndarray.setflags(write=None, align=None, uic=None)</c> (<c>array_setflags</c>,
        ///     <c>numpy/_core/src/multiarray/methods.c</c>, NumPy 2.4.2). <c>null</c> leaves a flag
        ///     untouched; flags are processed in NumPy's order — <c>align</c>, then <c>uic</c>, then
        ///     <c>write</c> — and an error ROLLS BACK every change made earlier in the same call (probed:
        ///     <c>setflags(align=False, uic=True)</c> raises with <c>aligned</c> still True).
        /// </summary>
        /// <param name="write">
        ///     Describes whether or not the array can be written to. Turning it ON follows NumPy's
        ///     <c>_IsWriteable</c> rule, evaluated UNCONDITIONALLY (even when the flag is already on —
        ///     probed: a still-writeable view whose base has since been made read-only is refused, and
        ///     stays writeable): an array that owns ordinary memory may always be re-enabled; a view may
        ///     iff its base is writeable (so <c>np.broadcast_to</c> views CAN be re-enabled — writes then
        ///     alias across the stride-0 axes and reach the source, exactly as in NumPy); an array over
        ///     foreign read-only memory (an <c>'r'</c> memmap, a read-only buffer) is refused with
        ///     NumPy's <c>ValueError("cannot set WRITEABLE flag to True of this array")</c>.
        /// </param>
        /// <param name="align">
        ///     Describes whether or not the data is aligned properly for its type. False CLEARS the
        ///     ALIGNED flag (observable in <see cref="flags"/>: <c>aligned</c>/<c>num</c>/<c>behaved</c>/
        ///     <c>carray</c>/<c>farray</c> and the repr all follow); True re-sets it. NumPy's
        ///     "cannot set aligned flag of mis-aligned array to True" is unreachable here: NumSharp
        ///     addresses memory in whole elements (element strides and offsets), so data can never sit
        ///     mis-aligned for its own dtype — True always succeeds. Fresh views and copies of an
        ///     align-cleared array come back ALIGNED, as in NumPy (each new array recomputes the flag).
        /// </param>
        /// <param name="uic">
        ///     (Write-back-if-copy.) True raises NumPy's
        ///     <c>ValueError("cannot set WRITEBACKIFCOPY flag to True")</c> — the flag can only be set by
        ///     NumPy's C-API. False is accepted as a no-op: NumSharp never sets WRITEBACKIFCOPY, and
        ///     NumPy's side effect of ALSO severing the view's base reference (<c>Py_XDECREF(fa-&gt;base)</c>
        ///     — after which <c>v.base</c> is None and the data can dangle if the owner dies) is
        ///     deliberately NOT reproduced: <see cref="Backends.UnmanagedStorage.BaseStorage"/> roots the
        ///     owner for the view's lifetime, and detaching it would be a use-after-free hazard with no
        ///     writeback state to resolve. A documented memory-safety divergence.
        /// </param>
        /// <exception cref="ValueError">
        ///     <c>uic: true</c>, or <c>write: true</c> on an array NumPy's rule refuses (see above).
        /// </exception>
        /// <remarks>
        ///     <para>
        ///     <c>a.flags.writeable = …</c>, <c>a.flags.aligned = …</c> (no-op result aside) and
        ///     <c>a.flags["W"/"A"/"X"] = …</c> all route through this method, exactly as NumPy's
        ///     <c>arrayflags</c> setters call <c>self.arr.setflags(…)</c> (<c>flagsobject.c</c>).
        ///     </para>
        ///     <para>
        ///     One corner inherits NumSharp's flattened base chain: a view whose DIRECT parent is a
        ///     read-only intermediate but whose ultimate owner is writeable is re-enabled (NumPy's own
        ///     array-chain rule does the same — "if ANY base is writeable" — its collapsed <c>.base</c>
        ///     skips read-only intermediates too; only a non-array buffer boundary pins the exporter's
        ///     read-only bit, which NumSharp models with <see cref="Backends.UnmanagedStorage.WriteProtected"/>
        ///     on the boundary storage itself).
        ///     </para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setflags.html
        /// </remarks>
        public void setflags(bool? write = null, bool? align = null, bool? uic = null)
        {
            // NumPy captures the flags at entry (`flagback`) and restores them on EITHER error below, so
            // a same-call align/uic change never survives a failed call (probed on 2.4.2).
            var flagback = Shape;

            if (align.HasValue)
            {
                Storage.SetShapeUnsafe(align.Value
                    ? Shape.WithFlags(flagsToSet: ArrayFlags.ALIGNED)
                    : Shape.WithFlags(flagsToClear: ArrayFlags.ALIGNED));
            }

            if (uic.HasValue && uic.Value)
            {
                Storage.SetShapeUnsafe(flagback);
                throw new ValueError("cannot set WRITEBACKIFCOPY flag to True");
            }
            // uic: false — no-op (no WRITEBACKIFCOPY state; base deliberately NOT severed, see <param>).

            if (write.HasValue)
            {
                if (write.Value)
                {
                    if (!Storage.CanEnableWriteable())
                    {
                        Storage.SetShapeUnsafe(flagback);
                        throw new ValueError("cannot set WRITEABLE flag to True of this array");
                    }

                    Storage.SetShapeUnsafe(Shape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE));
                }
                else
                {
                    Storage.SetShapeUnsafe(Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
                }
            }
        }
    }
}
