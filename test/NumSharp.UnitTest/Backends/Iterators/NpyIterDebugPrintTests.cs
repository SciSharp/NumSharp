using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest suite for NpyIter_DebugPrint (nditer_api.c:1402).
    ///
    /// Verifies the dump format contains expected sections and decodes flags correctly.
    /// Format closely matches NumPy's output structure.
    /// </summary>
    [TestClass]
    public class NpyIterDebugPrintTests
    {
        [TestMethod]
        public void DebugPrint_1D_Int32_ContainsExpectedSections()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(a);

            string dump = it.DebugPrintToString();

            StringAssert.Contains(dump, "BEGIN ITERATOR DUMP");
            StringAssert.Contains(dump, "END ITERATOR DUMP");
            StringAssert.Contains(dump, "Iterator Address:");
            StringAssert.Contains(dump, "ItFlags:");
            StringAssert.Contains(dump, "NDim: 1");
            StringAssert.Contains(dump, "NOp: 1");
            StringAssert.Contains(dump, "IterSize: 5");
            StringAssert.Contains(dump, "Perm:");
            StringAssert.Contains(dump, "DTypes:");
            StringAssert.Contains(dump, "OpItFlags:");
            StringAssert.Contains(dump, "AxisData[0]:");
            StringAssert.Contains(dump, "Shape: 5");
        }

        [TestMethod]
        public void DebugPrint_DecodesIDENTPERM()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(a);
            string dump = it.DebugPrintToString();
            StringAssert.Contains(dump, "IDENTPERM");
        }

        [TestMethod]
        public void DebugPrint_DecodesMULTIINDEX()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);
            string dump = it.DebugPrintToString();
            StringAssert.Contains(dump, "HASMULTIINDEX");
        }

        [TestMethod]
        public void DebugPrint_DecodesNEGPERM()
        {
            var a = np.arange(5).astype(np.int32)["::-1"];
            using var it = NpyIterRef.New(a, order: NPY_ORDER.NPY_KEEPORDER);
            string dump = it.DebugPrintToString();
            StringAssert.Contains(dump, "NEGPERM");
        }

        [TestMethod]
        public void DebugPrint_DecodesBUFFER()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.AdvancedNew(
                nop: 1,
                op: new[] { a },
                flags: NpyIterGlobalFlags.BUFFERED,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY },
                opDtypes: new[] { NumSharp.NPTypeCode.Double });

            string dump = it.DebugPrintToString();
            StringAssert.Contains(dump, "BUFFER");
            StringAssert.Contains(dump, "BufferData:");
            StringAssert.Contains(dump, "BufferSize:");
        }

        [TestMethod]
        public void DebugPrint_DecodesHASINDEX()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.C_INDEX);
            string dump = it.DebugPrintToString();
            StringAssert.Contains(dump, "HASINDEX");
            StringAssert.Contains(dump, "FlatIndex:");
        }

        [TestMethod]
        public void DebugPrint_ListsPerm()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);
            string dump = it.DebugPrintToString();
            // Identity perm for 2D is "0 1"
            StringAssert.Contains(dump, "Perm: 0 1");
        }

        [TestMethod]
        public void DebugPrint_MultiOperand_ListsAllOperands()
        {
            var x = np.arange(5).astype(np.int32);
            var y = np.zeros(new int[] { 5 }, np.int64);
            using var it = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { x, y },
                flags: NpyIterGlobalFlags.None,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            string dump = it.DebugPrintToString();
            StringAssert.Contains(dump, "NOp: 2");
            StringAssert.Contains(dump, "Flags[0]:");
            StringAssert.Contains(dump, "Flags[1]:");
            StringAssert.Contains(dump, "READ");
            StringAssert.Contains(dump, "WRITE");
            StringAssert.Contains(dump, "int32");
            StringAssert.Contains(dump, "int64");
        }

        [TestMethod]
        public void DebugPrint_WritesToTextWriter()
        {
            var a = np.arange(3).astype(np.int32);
            using var it = NpyIterRef.New(a);
            var sb = new System.Text.StringBuilder();
            var sw = new StringWriter(sb);
            it.DebugPrint(sw);

            Assert.IsTrue(sb.Length > 100, "DebugPrint should produce substantial output");
            StringAssert.Contains(sb.ToString(), "BEGIN ITERATOR DUMP");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void DebugPrint_NullWriter_Throws()
        {
            var a = np.arange(3).astype(np.int32);
            using var it = NpyIterRef.New(a);
            it.DebugPrint(null);
        }

        [TestMethod]
        public void DebugPrint_AxisData_ListsShapeAndStrides()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);
            string dump = it.DebugPrintToString();

            StringAssert.Contains(dump, "AxisData[0]:");
            StringAssert.Contains(dump, "AxisData[1]:");
            StringAssert.Contains(dump, "Shape: 2");
            StringAssert.Contains(dump, "Shape: 3");
            StringAssert.Contains(dump, "Strides:");
        }

        [TestMethod]
        public void DebugPrint_NoCrashOnReducedIterator()
        {
            // Reduction iterator: op_axes with -1 entries
            var x = np.arange(12).reshape(3, 4).astype(np.int32);
            var y = np.zeros(new int[] { 4 }, np.int32);

            using var it = NpyIterRef.AdvancedNew(
                nop: 2,
                op: new[] { x, y },
                flags: NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.REDUCE_OK,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
                opDtypes: null,
                opAxesNDim: 2,
                opAxes: new[] { new[] { 0, 1 }, new[] { -1, 0 } });

            string dump = it.DebugPrintToString();
            StringAssert.Contains(dump, "REDUCE");
        }
    }
}
