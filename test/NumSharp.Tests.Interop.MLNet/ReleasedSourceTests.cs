using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     A source whose buffer has been released is refused by every export verb, whatever its layout and dtype; and an
    ///     array too large for ML.NET's buffers is refused at export, before anything is allocated.
    ///
    ///     <para><b>The regression.</b> The ARC reference <c>Pin</c> takes is the only check that a source's buffer is still
    ///     alive (<c>TryAddRef</c> fails on a released buffer), and three paths read the source before it:</para>
    ///     <list type="bullet">
    ///         <item><description><c>ToVBuffer</c> densified a strided source with <c>copy()</c>;</description></item>
    ///         <item><description>every verb converted a Half or Decimal source with <c>astype</c> (ML.NET has no half or
    ///         decimal column type);</description></item>
    ///         <item><description><c>ToDataView</c> snapshot-copied the source and never pinned it at all, so even a
    ///         contiguous source was read.</description></item>
    ///     </list>
    ///     <para>The pin then ran on the fresh copy and succeeded, so a released buffer was read silently: once the pool had
    ///     handed it to another array, the export carried that array's values (a probe read another array's 777s through
    ///     <c>ToVBuffer</c> and <c>ToDataView</c>). <see cref="LifetimeTests.DisposedSource_CannotBeExported"/> only covered a
    ///     contiguous float32 source, the one path that pinned first. The same bug was fixed in the System.Numerics.Tensors
    ///     bridge (<c>c8f1dffa</c>).</para>
    /// </summary>
    [TestClass]
    public class ReleasedSourceTests : MLNetTestBase
    {
        /// <summary>The layouts <see cref="Released"/> builds, all 1-D or 2-D as the data views require.</summary>
        private static readonly string[] Layouts = { "contiguous", "contiguous slice", "transposed", "stepped", "broadcast", "empty" };

        /// <summary>
        ///     Every export verb, for a source that is direct (float64), converted losslessly (Half → Single) or converted
        ///     lossily (Decimal → Double), in every layout. The data views read any layout, so a released source can only be
        ///     refused for being released.
        /// </summary>
        [TestMethod]
        public void ExportVerbs_ReleasedSource_Throw_EveryLayoutAndConversion()
        {
            foreach (NPTypeCode code in new[] { NPTypeCode.Double, NPTypeCode.Half, NPTypeCode.Decimal })
            foreach (string layout in Layouts)
            {
                NDArray released = Released(layout, code);
                string what = $"a released {code} {layout} array";
                string[] names = ColumnNames(released);

                Assert.ThrowsException<ObjectDisposedException>(() => released.AsDataView("F"), $"AsDataView(column) of {what}");
                Assert.ThrowsException<ObjectDisposedException>(() => released.AsDataView(names), $"AsDataView(columns) of {what}");
                Assert.ThrowsException<ObjectDisposedException>(() => released.ToDataView("F"), $"ToDataView(column) of {what}");
                Assert.ThrowsException<ObjectDisposedException>(() => released.ToDataView(names), $"ToDataView(columns) of {what}");
                if (code == NPTypeCode.Half)
                    Assert.ThrowsException<ObjectDisposedException>(() => released.ToVBuffer<float>(), $"ToVBuffer of {what}");
                else
                    Assert.ThrowsException<ObjectDisposedException>(() => released.ToVBuffer<double>(), $"ToVBuffer of {what}");
            }
        }

        /// <summary>
        ///     ML.NET's buffers hold at most <see cref="NDArrayMLNetInterop.MaxBufferLength"/> (2,146,435,071) elements,
        ///     a little below a <c>T[]</c>'s own <see cref="Array.MaxLength"/> (2,147,483,591). The guard compared against
        ///     <see cref="int.MaxValue"/>: past <see cref="Array.MaxLength"/> the copy failed with OutOfMemoryException
        ///     ("Array dimensions exceeded supported range") instead of the refusal, and between the two limits it built a
        ///     VBuffer ML.NET cannot copy (see <see cref="MaxBufferLength_IsMLNetsOwnBufferCap"/>). Broadcast views give the
        ///     element counts without the memory, and the refusal comes before anything is allocated.
        /// </summary>
        [TestMethod]
        public void ToVBuffer_PastMLNetsBufferLimit_IsRefused_NotOutOfMemory()
        {
            using NDArray one = NDArray.Scalar(true);
            foreach (long size in new[] { (long)NDArrayMLNetInterop.MaxBufferLength + 1, (long)Array.MaxLength + 1, int.MaxValue, (long)int.MaxValue + 1 })
            {
                using NDArray huge = np.broadcast_to(one, new Shape(size));
                var ex = Assert.ThrowsException<NotSupportedException>(() => huge.ToVBuffer<bool>(), $"{size} elements");
                StringAssert.Contains(ex.Message, NDArrayMLNetInterop.MaxBufferLength.ToString());
                StringAssert.Contains(ex.Message, "ArrayMaxSize");
            }
        }

        /// <summary>
        ///     A vector column hands ML.NET one dense <c>VBuffer&lt;T&gt;</c> per row, which the cursor fills through
        ///     <c>VBufferEditor.Create</c>, and that refuses any length past <see cref="NDArrayMLNetInterop.MaxBufferLength"/>.
        ///     The guard compared against <see cref="int.MaxValue"/>: the view was created, and every row read then failed
        ///     inside the cursor (ArgumentOutOfRangeException "min must not exceed max", from ML.NET). It is refused at export
        ///     now, and <c>ToDataView</c> refuses before snapshot-copying the source (a 2 GB copy just to be refused).
        /// </summary>
        [TestMethod]
        public void VectorColumn_WiderThanMLNetsBufferLimit_IsRefused_AtExport()
        {
            using NDArray one = NDArray.Scalar(true);
            foreach (long width in new[] { (long)NDArrayMLNetInterop.MaxBufferLength + 1, (long)Array.MaxLength + 1 })
            foreach (Shape shape in new[] { new Shape(width), new Shape(1, width) })
            {
                using NDArray wide = np.broadcast_to(one, shape);
                var ex = Assert.ThrowsException<NotSupportedException>(() => wide.AsDataView("F"), $"AsDataView of shape {shape}");
                StringAssert.Contains(ex.Message, NDArrayMLNetInterop.MaxBufferLength.ToString());
                ex = Assert.ThrowsException<NotSupportedException>(() => wide.ToDataView("F"), $"ToDataView of shape {shape}");
                StringAssert.Contains(ex.Message, NDArrayMLNetInterop.MaxBufferLength.ToString());
            }
        }

        /// <summary>
        ///     The bridge restates ML.NET's buffer cap because ML.NET keeps it internal
        ///     (<c>Microsoft.ML.Internal.Utilities.ArrayUtils.ArrayMaxSize</c>). This pins the restated value to ML.NET's
        ///     own, read by reflection from whichever ML.NET the suite runs against (CI runs the floor, 2.0.1, and the
        ///     current release), and pins the reason the cap matters: <c>VBufferEditor.Create</c> — the path of every cursor
        ///     getter and of <c>VBuffer.CopyTo</c> — refuses one element more (refused up front, so nothing is allocated).
        /// </summary>
        [TestMethod]
        public void MaxBufferLength_IsMLNetsOwnBufferCap()
        {
            Type arrayUtils = Array.Find(typeof(VBuffer<int>).Assembly.GetTypes(), t => t.FullName == "Microsoft.ML.Internal.Utilities.ArrayUtils");
            Assert.IsNotNull(arrayUtils, "ML.NET's ArrayUtils moved; find its buffer cap again and update NDArrayMLNetInterop.MaxBufferLength.");
            FieldInfo cap = arrayUtils.GetField("ArrayMaxSize", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
            Assert.IsNotNull(cap, "ML.NET's ArrayUtils.ArrayMaxSize is gone; find its buffer cap again.");
            Assert.AreEqual(NDArrayMLNetInterop.MaxBufferLength, (int)cap.GetValue(null));

            var buffer = default(VBuffer<bool>);
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => VBufferEditor.Create(ref buffer, NDArrayMLNetInterop.MaxBufferLength + 1));
        }

        /// <summary>
        ///     The refusals decided by the dtype, the shape and the names alone come BEFORE the liveness check, so they hold
        ///     for a released array too: an array that could never cross is refused for that, without anything being read,
        ///     converted, snapshotted or allocated to find out. Pins the order the fix put the checks in.
        /// </summary>
        [TestMethod]
        public void ShapeDtypeAndSizeRefusals_ComeBeforeTheLivenessCheck()
        {
            NDArray cube = Released("cube", NPTypeCode.Half);
            Assert.ThrowsException<ArgumentException>(() => cube.AsDataView("F"), "a 3-D array is no vector column");
            Assert.ThrowsException<ArgumentException>(() => cube.ToDataView("F"), "a 3-D array is no vector column");
            Assert.ThrowsException<ArgumentException>(() => cube.AsDataView(new[] { "a" }), "a 3-D array has no scalar columns");
            Assert.ThrowsException<ArgumentException>(() => cube.ToDataView(new[] { "a" }), "a 3-D array has no scalar columns");

            NDArray matrix = Released("transposed", NPTypeCode.Decimal);   // (3, 2): two columns
            Assert.ThrowsException<ArgumentException>(() => matrix.AsDataView(new[] { "a" }), "one name for two columns");
            Assert.ThrowsException<ArgumentException>(() => matrix.ToDataView(new[] { "a", "" }), "an empty column name");
            Assert.ThrowsException<ArgumentException>(() => matrix.ToVBuffer<decimal>(), "Decimal crosses as VBuffer<double>");

            NDArray complex = Released("contiguous", NPTypeCode.Complex);
            Assert.ThrowsException<NotSupportedException>(() => complex.AsDataView("F"), "Complex has no ML.NET column type");
            Assert.ThrowsException<NotSupportedException>(() => complex.ToDataView(new[] { "a" }), "Complex has no ML.NET column type");

            NDArray huge = ReleasedBroadcast((long)NDArrayMLNetInterop.MaxBufferLength + 1);
            Assert.ThrowsException<NotSupportedException>(() => huge.ToVBuffer<bool>(), "past ML.NET's buffer limit");
            Assert.ThrowsException<NotSupportedException>(() => huge.ToDataView("F"), "a row past ML.NET's buffer limit");
        }

        /// <summary>A broadcast view of <paramref name="size"/> booleans whose buffer has been released (see <see cref="Released"/>).</summary>
        /// <param name="size">The element count of the view; a broadcast needs no memory for it.</param>
        /// <returns>The disposed view.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray ReleasedBroadcast(long size)
        {
            NDArray one = NDArray.Scalar(true);
            NDArray view = np.broadcast_to(one, new Shape(size));
            view.Dispose();
            one.Dispose();
            return view;
        }

        /// <summary>One scalar-column name per column of a 1-D or 2-D array, as <c>AsDataView(string[])</c> requires.</summary>
        /// <param name="nd">The array the names are for.</param>
        /// <returns><c>c0</c>, <c>c1</c>, … — one name for a 1-D array, <c>shape[1]</c> names for a 2-D one.</returns>
        private static string[] ColumnNames(NDArray nd)
        {
            int count = nd.ndim == 1 ? 1 : (int)nd.shape[1];
            var names = new string[count];
            for (int i = 0; i < count; i++)
                names[i] = "c" + i;
            return names;
        }

        /// <summary>
        ///     Builds an array of the requested layout and dtype, then disposes it together with every array it was derived
        ///     from, so no reference is left and its buffer goes back to the pool. A helper of its own (no inlining) keeps
        ///     the JIT from holding any of them alive in the caller's frame.
        /// </summary>
        /// <param name="layout">
        ///     One of "contiguous", "contiguous slice", "transposed", "stepped", "broadcast", "empty" (a <c>(0, 3)</c> array:
        ///     zero rows, three columns, so the scalar-column overload still has names to take) or "cube" (a <c>(1, 2, 3)</c>
        ///     array, which no data view accepts; not in <see cref="Layouts"/>).
        /// </param>
        /// <param name="code">The dtype of the array; Half and Decimal exercise the conversions.</param>
        /// <returns>The disposed array, whose buffer has been released.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="layout"/> is not one of the names above.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray Released(string layout, NPTypeCode code)
        {
            var made = new List<NDArray>();
            NDArray Keep(NDArray a)
            {
                made.Add(a);
                return a;
            }

            NDArray owner = Keep(layout == "empty" ? np.zeros(new Shape(0, 3), code) : Arange(code, 6));
            NDArray view = layout switch
            {
                "contiguous" => owner,
                "contiguous slice" => Keep(owner["1:5"]),
                "transposed" => Keep(Keep(owner.reshape(2, 3)).T),
                "stepped" => Keep(owner["::2"]),
                "broadcast" => Keep(np.broadcast_to(Keep(owner["0:3"]), new Shape(2, 3))),
                "empty" => owner,
                "cube" => Keep(owner.reshape(1, 2, 3)),
                _ => throw new ArgumentOutOfRangeException(nameof(layout), layout, "unknown layout"),
            };

            // Dispose the view and every array it was derived from: only then is the buffer's last reference gone.
            foreach (NDArray a in made)
                a.Dispose();
            return view;
        }
    }
}
