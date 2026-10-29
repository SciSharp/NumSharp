using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     The cursor's state machine, checked differentially against ML.NET's OWN cursors (an in-memory view, its cache,
    ///     a transform, a row filter): whatever ML.NET's cursors do before the first row, after the last one and after
    ///     <c>Dispose</c>, the NumSharp cursor must do too.
    ///
    ///     <para><b>The regressions.</b> ML.NET's cursors throw <see cref="InvalidOperationException"/> ("Values cannot be
    ///     fetched at this time…") when a getter is called outside a row, report <c>Position</c> -1 before the first row,
    ///     after the end and after <c>Dispose</c>, and keep answering <c>Schema</c> for as long as they live. The NumSharp
    ///     cursor instead computed the element address from position -1 or <c>rowCount</c> and READ OUTSIDE THE ARRAY (a
    ///     probe read <c>-3.86E-34</c> before the first row and <c>-3.95E-31</c> after the last), kept reading after
    ///     <c>Dispose</c> released its pin, parked <c>Position</c> at <c>rowCount</c> after the end, and threw
    ///     <see cref="ObjectDisposedException"/> from <c>Schema</c> once its view was disposed, although the docs promise
    ///     a cursor survives its view.</para>
    /// </summary>
    [TestClass]
    public class CursorStateTests : MLNetTestBase
    {
        /// <summary>ML.NET's own text for a getter called outside a row (RowCursorUtils.FetchValueStateError).</summary>
        private const string FetchValueStateError = "Values cannot be fetched at this time";

        /// <summary>A POCO row for ML.NET's own in-memory view, the reference implementation of a cursor.</summary>
        private sealed class Row
        {
            /// <summary>The single float column.</summary>
            public float X { get; set; }
        }

        /// <summary>
        ///     The cursor kinds whose state contract is checked: the NumSharp scalar-column and vector-column views over
        ///     strided layouts (reversed rows, a transposed matrix, a 1-D array), and — as the reference the NumSharp ones
        ///     must agree with — ML.NET's own in-memory view, its cache and a row filter.
        /// </summary>
        /// <returns>Each kind's name, a factory for its view, and a getter for the column it reads.</returns>
        private static IEnumerable<(string Name, Func<IDataView> View, Func<DataViewRowCursor, IDataView, Action> Read)> Kinds()
        {
            NDArray m = Arange(NPTypeCode.Single, 3, 2);   // owned by the test process; the views pin what they read
            Action ScalarRead(DataViewRowCursor c, IDataView v)
            {
                ValueGetter<float> g = c.GetGetter<float>(v.Schema[v.Schema.Count - 1]);
                return () => { float x = 0; g(ref x); };
            }
            Action VectorRead(DataViewRowCursor c, IDataView v)
            {
                ValueGetter<VBuffer<float>> g = c.GetGetter<VBuffer<float>>(v.Schema["F"]);
                return () => { var x = default(VBuffer<float>); g(ref x); };
            }

            IDataView reference = Ml.Data.LoadFromEnumerable(new[] { new Row { X = 1 }, new Row { X = 2 }, new Row { X = 3 } });
            yield return ("ML.NET Cache (reference)", () => Ml.Data.Cache(reference), ScalarRead);
            yield return ("ML.NET FilterRowsByColumn (reference)", () => Ml.Data.FilterRowsByColumn(reference, "X", 0, 10), ScalarRead);
            yield return ("NumSharp scalar columns", () => m.AsDataView(new[] { "a", "b" }), ScalarRead);
            yield return ("NumSharp scalar columns, reversed rows", () => m["::-1"].AsDataView(new[] { "a", "b" }), ScalarRead);
            yield return ("NumSharp vector column", () => m.AsDataView("F"), VectorRead);
            yield return ("NumSharp vector column, transposed", () => m.T.AsDataView("F"), VectorRead);
            yield return ("NumSharp vector column, 1-D", () => m[":, 0"].AsDataView("F"), VectorRead);
            yield return ("NumSharp snapshot (ToDataView)", () => m["::-1"].ToDataView("F"), VectorRead);
        }

        /// <summary>
        ///     A value getter called before the first <c>MoveNext</c>, after <c>MoveNext</c> returned false, or after the
        ///     cursor was disposed throws ML.NET's own <see cref="InvalidOperationException"/> — for every NumSharp cursor kind,
        ///     exactly as ML.NET's reference cursors do — instead of reading memory outside the array.
        /// </summary>
        [TestMethod]
        public void ValueGetters_OutsideARow_Throw_LikeMLNetsOwnCursors()
        {
            foreach (var (name, makeView, read) in Kinds())
            {
                IDataView view = makeView();
                try
                {
                    using (DataViewRowCursor c = view.GetRowCursor(view.Schema))
                    {
                        Action get = read(c, view);
                        AssertFetchRefused(get, $"{name}: getter before the first MoveNext");
                        while (c.MoveNext())
                            get();   // inside a row: must read
                        AssertFetchRefused(get, $"{name}: getter after MoveNext returned false");
                    }

                    DataViewRowCursor disposed = view.GetRowCursor(view.Schema);
                    Action afterDispose = read(disposed, view);
                    Assert.IsTrue(disposed.MoveNext(), $"{name}: the first row");
                    disposed.Dispose();
                    AssertFetchRefused(afterDispose, $"{name}: getter after the cursor was disposed (its pin is gone)");
                }
                finally
                {
                    (view as IDisposable)?.Dispose();
                }
            }
        }

        /// <summary>
        ///     The row-id getter follows the same rule as the value getters: ML.NET's reference cursors refuse it outside a
        ///     row, and so must the NumSharp cursor (it used to return the id of position -1 as a huge unsigned number).
        /// </summary>
        [TestMethod]
        public void IdGetter_OutsideARow_Throws_LikeMLNetsOwnCursors()
        {
            foreach (var (name, makeView, _) in Kinds())
            {
                IDataView view = makeView();
                try
                {
                    using DataViewRowCursor c = view.GetRowCursor(view.Schema);
                    ValueGetter<DataViewRowId> id = c.GetIdGetter();
                    var value = default(DataViewRowId);
                    Assert.ThrowsException<InvalidOperationException>(() => id(ref value), $"{name}: id before the first MoveNext");
                    var seen = new HashSet<DataViewRowId>();
                    while (c.MoveNext())
                    {
                        id(ref value);
                        Assert.IsTrue(seen.Add(value), $"{name}: row ids must be unique, {value} repeated");
                    }
                    Assert.ThrowsException<InvalidOperationException>(() => id(ref value), $"{name}: id after the end");
                }
                finally
                {
                    (view as IDisposable)?.Dispose();
                }
            }
        }

        /// <summary>
        ///     <c>Position</c> is -1 before the first row, counts 0, 1, … while on a row, is -1 again once <c>MoveNext</c>
        ///     returned false (and stays false), and is -1 after <c>Dispose</c> — the reference cursors' exact sequence. The
        ///     NumSharp cursor used to park <c>Position</c> at <c>rowCount</c> after the end.
        /// </summary>
        [TestMethod]
        public void Position_MatchesMLNetsOwnSequence()
        {
            foreach (var (name, makeView, _) in Kinds())
            {
                IDataView view = makeView();
                try
                {
                    var sequence = new List<long>();
                    DataViewRowCursor c = view.GetRowCursor(view.Schema);
                    sequence.Add(c.Position);
                    while (c.MoveNext())
                        sequence.Add(c.Position);
                    sequence.Add(c.Position);
                    Assert.IsFalse(c.MoveNext(), $"{name}: MoveNext after the end");
                    sequence.Add(c.Position);
                    c.Dispose();
                    sequence.Add(c.Position);
                    Assert.IsFalse(c.MoveNext(), $"{name}: MoveNext after Dispose");

                    long rows = view.GetRowCount() ?? (sequence.Count - 4);
                    var want = new List<long> { -1 };
                    for (long r = 0; r < rows; r++)
                        want.Add(r);
                    want.AddRange(new[] { -1L, -1L, -1L });
                    CollectionAssert.AreEqual(want, sequence, $"{name}: Position sequence {string.Join(",", sequence)}");
                }
                finally
                {
                    (view as IDisposable)?.Dispose();
                }
            }
        }

        /// <summary>
        ///     Asserts that a getter call is refused the way ML.NET's own cursors refuse it outside a row: an
        ///     <see cref="InvalidOperationException"/> carrying ML.NET's text, and no value read.
        /// </summary>
        /// <param name="get">The getter call.</param>
        /// <param name="what">What the call is, for the failure message.</param>
        /// <exception cref="AssertFailedException">The call returned, or threw something else.</exception>
        private static void AssertFetchRefused(Action get, string what)
        {
            var e = Assert.ThrowsException<InvalidOperationException>(get, what);
            StringAssert.Contains(e.Message, FetchValueStateError, what);
        }

        /// <summary>
        ///     A cursor survives its view: after the view is disposed the cursor still answers <c>Schema</c> and
        ///     <c>IsColumnActive</c>, still hands out getters, and still reads the right values (its own pin keeps the
        ///     buffer alive) — and both pins are released once the cursor is disposed too.
        /// </summary>
        [TestMethod]
        public void Cursor_OutlivesItsView_SchemaGettersAndValues()
        {
            using NDArray m = Arange(NPTypeCode.Double, 4, 3);
            foreach (string kind in new[] { "vector", "scalar" })
            {
                NDArrayDataView view = kind == "vector" ? m.T.AsDataView("F") : m.T.AsDataView(new[] { "a", "b", "c", "d" });
                DataViewRowCursor c = view.GetRowCursor(view.Schema);
                view.Dispose();

                Assert.AreEqual(kind == "vector" ? 1 : 4, c.Schema.Count, $"{kind}: Schema after the view was disposed");
                Assert.IsTrue(c.IsColumnActive(c.Schema[0]), $"{kind}: IsColumnActive after the view was disposed");
                var rows = new List<double[]>();
                if (kind == "vector")
                {
                    ValueGetter<VBuffer<double>> g = c.GetGetter<VBuffer<double>>(c.Schema["F"]);
                    var buf = default(VBuffer<double>);
                    while (c.MoveNext())
                    {
                        g(ref buf);
                        rows.Add(buf.DenseValues().ToArrayCopy());
                    }
                }
                else
                {
                    var getters = new ValueGetter<double>[4];
                    for (int j = 0; j < 4; j++)
                        getters[j] = c.GetGetter<double>(c.Schema[j]);
                    while (c.MoveNext())
                    {
                        var row = new double[4];
                        for (int j = 0; j < 4; j++)
                            getters[j](ref row[j]);
                        rows.Add(row);
                    }
                }
                c.Dispose();

                // m.T is (3, 4): row r holds m[:, r] = r, 3 + r, 6 + r, 9 + r.
                Assert.AreEqual(3, rows.Count, $"{kind}: rows read after the view was disposed");
                for (int r = 0; r < 3; r++)
                    CollectionAssert.AreEqual(new[] { r, 3.0 + r, 6.0 + r, 9.0 + r }, rows[r], $"{kind}: row {r}");
            }
        }
    }

    /// <summary>Small helpers for the cursor tests.</summary>
    internal static class CursorTestExtensions
    {
        /// <summary>Copies a sequence of dense values into a new array (the VBuffer's own array is reused per row).</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="values">The values to copy.</param>
        /// <returns>A new array holding them.</returns>
        internal static T[] ToArrayCopy<T>(this IEnumerable<T> values) => new List<T>(values).ToArray();
    }
}
