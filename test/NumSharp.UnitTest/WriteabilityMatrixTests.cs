using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Pins <c>Shape.IsWriteable</c> across the (source-writeability × view/copy op) matrix, verified
    ///     against NumPy 2.4.2's <c>flags['WRITEABLE']</c>.
    /// </summary>
    /// <remarks>
    ///     The rule: a VIEW inherits its source's writeability; a COPY / computed result is always
    ///     writeable. Read-only sources here are broadcast arrays (the read-only source that needs no
    ///     files); the identical propagation for a read-only <c>'r'</c> memmap — including the
    ///     memory-safety angle (writes must throw, not segfault) — lives in <c>IO/NpyMemmapTests</c>.
    ///
    ///     Two entries differ from NumPy and are NOT propagation bugs — NumSharp returns a different KIND
    ///     of object and its flag is correct for that object; both are pre-existing, memory-safe design
    ///     choices, marked <see cref="MISALIGNED"/> below:
    ///       • <c>a[i,j]</c> → a writeable 0-d VIEW (NumPy returns an immutable scalar, WRITEABLE=False).
    ///       • <c>squeeze</c> of a broadcast → a writeable COPY via reshape (NumPy returns a read-only view).
    /// </remarks>
    [TestClass]
    public class WriteabilityMatrixTests
    {
        private const bool MISALIGNED = true; // marks an intentional, memory-safe divergence from NumPy

        private static NDArray Owned() => np.arange(24).astype(NPTypeCode.Int32).reshape(4, 6);
        private static NDArray Broadcast() => np.broadcast_to(np.arange(6).astype(NPTypeCode.Int32), new Shape(4, 6));

        // op, expected-writeable-off-a-writeable-source, expected-off-a-read-only-source
        private static IEnumerable<(string name, Func<NDArray, NDArray> op, bool onW, bool onRO)> Matrix()
        {
            yield return ("identity",     a => a,                                      true,  false);
            yield return ("slice_row",    a => a["1:3"],                               true,  false);
            yield return ("slice_col",    a => a[":, 1:3"],                            true,  false);
            yield return ("slice_step",   a => a["::2"],                               true,  false);
            yield return ("slice_rev",    a => a["::-1"],                              true,  false);
            yield return ("slice_2d",     a => a["1:3, 1:3"],                          true,  false);
            yield return ("index_row",    a => a["0"],                                 true,  false);
            yield return ("index_elem",   a => a["0, 0"],                              true,  false); // 0-d view (see MISALIGNED note)
            yield return ("T",            a => a.T,                                    true,  false);
            yield return ("swapaxes",     a => np.swapaxes(a, 0, 1),                   true,  false);
            yield return ("moveaxis",     a => np.moveaxis(a, 0, 1),                   true,  false);
            yield return ("expand_dims",  a => np.expand_dims(a, 0),                   true,  false);
            yield return ("diagonal",     a => np.diagonal(a),                         false, false); // read-only view, NumPy parity
            yield return ("broadcast_to", a => np.broadcast_to(a, new Shape(2, 4, 6)), false, false);
            yield return ("view_of_view", a => a["1:3"][":, 1:3"],                     true,  false);
            // reshape / ravel are VIEWS off a contiguous source (inherit) but COPIES off a broadcast
            // (non-contiguous → materialize → writeable) — hence onRO=true here.
            yield return ("reshape",      a => a.reshape(6, 4),                        true,  true);
            yield return ("ravel",        a => a.ravel(),                              true,  true);
            yield return ("flatten",      a => a.flatten(),                            true,  true);  // always copies
            // copies / computed results — always writeable regardless of source
            yield return ("copy",         a => a.copy(),                               true,  true);
            yield return ("astype",       a => a.astype(NPTypeCode.Double),            true,  true);
            yield return ("fancy",        a => a[np.array(new[] { 0, 2 })],            true,  true);
            yield return ("mask",         a => a[a > 3],                               true,  true);
            yield return ("add1",         a => a + 1,                                  true,  true);
            yield return ("neg",          a => -a,                                     true,  true);
            yield return ("view_copy",    a => a["1:3"].copy(),                        true,  true);
            yield return ("copy_view",    a => a.copy()["1:3"],                        true,  true);
        }

        [TestMethod]
        public void Writeable_Source_Views_AreWriteable_CopiesAreWriteable()
        {
            foreach (var (name, op, onW, _) in Matrix())
                Assert.AreEqual(onW, op(Owned()).Shape.IsWriteable, $"owned source, op '{name}'");
        }

        [TestMethod]
        public void ReadOnly_Source_Views_AreReadOnly_CopiesAreWriteable()
        {
            foreach (var (name, op, _, onRO) in Matrix())
                Assert.AreEqual(onRO, op(Broadcast()).Shape.IsWriteable, $"broadcast source, op '{name}'");
        }

        [TestMethod]
        public void ReadOnly_View_Write_Throws()
        {
            // The flag is enforced: writing through any view of a read-only array raises rather than
            // corrupting the shared/backing memory.
            var b = Broadcast();
            Assert.ThrowsException<NumSharpException>(() => b["1:3"][0, 0] = 9);
            Assert.ThrowsException<NumSharpException>(() => b.T[0, 0] = 9);
        }

        [TestMethod]
        public void Diagonal_IsReadOnlyView_EvenOffWriteableSource()
        {
            // NumPy returns a read-only view from diagonal() (since 1.9, forward-compat) — NumSharp matches.
            Assert.IsFalse(np.diagonal(Owned()).Shape.IsWriteable);
        }

        [TestMethod]
        public void Documented_Divergences_AreStable()
        {
            // (1) a[i,j] is a writeable 0-d view (NumPy: immutable scalar, WRITEABLE=False).
            Assert.IsTrue(MISALIGNED && Owned()["1, 2"].Shape.IsWriteable);
            // (2) squeeze of a broadcast is a writeable copy via reshape (NumPy: read-only view).
            Assert.IsTrue(MISALIGNED && np.squeeze(np.expand_dims(Broadcast(), 0)).Shape.IsWriteable);
        }
    }
}
