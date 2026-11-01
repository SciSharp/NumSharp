using System;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     <see cref="TensorEngine.Default"/> (the np-layer dtype plan's P9): the settable engine an array binds when nothing
    ///     else decides. An array binds its engine once, at creation, so assigning the default reaches the arrays created
    ///     afterwards (through chained operations, views, reductions and the functions without an operand), while the arrays
    ///     that already exist keep theirs; <see cref="BackendFactory.GetEngine(BackendType)"/> forwards to it, and null
    ///     restores the built-in <see cref="DefaultEngine"/>.
    /// </summary>
    [TestClass]
    [DoNotParallelize /* assigns the process-wide TensorEngine.Default, which every array created meanwhile would bind */]
    public class TensorEngineDefaultTests : TestClass
    {
        /// <summary>
        ///     A <see cref="DefaultEngine"/> that counts the members these tests drive and otherwise computes as the built-in
        ///     engine does, so a count proves which engine an operation ran on.
        /// </summary>
        private sealed class CountingEngine : DefaultEngine
        {
            /// <summary>Calls of <see cref="Add"/>.</summary>
            public int Adds;

            /// <summary>Calls of <see cref="Sqrt"/>.</summary>
            public int Sqrts;

            /// <summary>Calls of <see cref="Sum"/>.</summary>
            public int Sums;

            /// <summary>Counts the call, then adds as the built-in engine does.</summary>
            /// <param name="lhs">The first operand.</param>
            /// <param name="rhs">The second operand.</param>
            /// <param name="dtype">The loop dtype, or null.</param>
            /// <param name="out">The output, or null.</param>
            /// <param name="where">The mask, or null.</param>
            /// <returns>The sum.</returns>
            public override NDArray Add(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
            {
                Adds++;
                return base.Add(lhs, rhs, dtype, @out, where);
            }

            /// <summary>Counts the call, then takes the square root as the built-in engine does.</summary>
            /// <param name="nd">The operand.</param>
            /// <param name="dtype">The loop dtype, or null.</param>
            /// <param name="out">The output, or null.</param>
            /// <param name="where">The mask, or null.</param>
            /// <returns>The square roots.</returns>
            public override NDArray Sqrt(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
            {
                Sqrts++;
                return base.Sqrt(nd, dtype, @out, where);
            }

            /// <summary>Counts the call, then sums as the built-in engine does.</summary>
            /// <param name="nd">The operand.</param>
            /// <param name="axis">The axis, or null for every axis.</param>
            /// <param name="dtype">The accumulator dtype, or null.</param>
            /// <param name="keepdims">Whether the reduced axis stays with length one.</param>
            /// <returns>The sum.</returns>
            public override NDArray Sum(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false)
            {
                Sums++;
                return base.Sum(nd, axis, dtype, keepdims);
            }
        }

        /// <summary>
        ///     Unassigned, the default is the cached built-in <see cref="DefaultEngine"/>, the shipped factory returns it, and
        ///     a fresh array binds it.
        /// </summary>
        [TestMethod]
        public void Default_IsTheBuiltInEngine_AndBackendFactoryForwardsToIt()
        {
            TensorEngine.Default.Should().BeSameAs(BackendFactory.GetEngine<DefaultEngine>());
            BackendFactory.GetEngine().Should().BeSameAs(TensorEngine.Default);
            np.zeros(3).TensorEngine.Should().BeSameAs(TensorEngine.Default);
        }

        /// <summary>
        ///     An assigned engine is what the property and the shipped factory return; assigning null brings the built-in
        ///     engine back (the rule <see cref="TensorEngine.Linalg"/> and <see cref="TensorEngine.Fft"/> follow).
        /// </summary>
        [TestMethod]
        public void AssigningNull_RestoresTheBuiltInEngine()
        {
            var counting = new CountingEngine();
            try
            {
                TensorEngine.Default = counting;
                TensorEngine.Default.Should().BeSameAs(counting);
                BackendFactory.GetEngine().Should().BeSameAs(counting, "the shipped factory forwards to the property");
            }
            finally
            {
                TensorEngine.Default = null;
            }

            TensorEngine.Default.Should().BeSameAs(BackendFactory.GetEngine<DefaultEngine>());
        }

        /// <summary>
        ///     The plan's gate: after the assignment, the arrays created from then on compute on the new engine, through
        ///     chained operations, views, reductions and np.zeros / the functions with no operand.
        /// </summary>
        [TestMethod]
        public void ArraysCreatedAfterTheAssignment_ComputeOnTheNewDefault()
        {
            var counting = new CountingEngine();
            try
            {
                TensorEngine.Default = counting;

                var zeros = np.zeros(new Shape(2, 3));
                zeros.TensorEngine.Should().BeSameAs(counting, "np.zeros binds the default engine");
                np.linspace(0.0, 1.0, 5).TensorEngine.Should().BeSameAs(counting, "a function with no operand computes on the default");

                var a = np.arange(12.0).reshape(3, 4);
                a.TensorEngine.Should().BeSameAs(counting);

                // chained operations: each result computes the next step on the same engine
                var chained = np.sqrt(a + a);
                counting.Adds.Should().Be(1);
                counting.Sqrts.Should().Be(1);
                chained.TensorEngine.Should().BeSameAs(counting);

                // views keep the engine, and an operation on one runs there
                var rows = a["1:3"];
                rows.TensorEngine.Should().BeSameAs(counting);
                var row = a[1];
                row.TensorEngine.Should().BeSameAs(counting);
                (rows + rows).TensorEngine.Should().BeSameAs(counting);
                counting.Adds.Should().Be(2);
                a.T.TensorEngine.Should().BeSameAs(counting);

                // reductions
                var colSums = np.sum(a, 0);
                counting.Sums.Should().Be(1);
                colSums.TensorEngine.Should().BeSameAs(counting);
                np.sum(colSums).TensorEngine.Should().BeSameAs(counting);
                counting.Sums.Should().Be(2);

                // the values are the built-in engine's
                chained.ToArray<double>().Should().Equal(np.sqrt(np.arange(12.0) * 2.0).ToArray<double>());
                colSums.ToArray<double>().Should().Equal(12.0, 15.0, 18.0, 21.0);
            }
            finally
            {
                TensorEngine.Default = null;
            }
        }

        /// <summary>
        ///     An array bound before the assignment keeps its engine: its operations do not reach the new default.
        /// </summary>
        [TestMethod]
        public void ArraysThatExist_KeepTheirEngine_WhenTheDefaultChanges()
        {
            var before = np.arange(4.0);
            var builtIn = before.TensorEngine;
            var counting = new CountingEngine();
            try
            {
                TensorEngine.Default = counting;
                before.TensorEngine.Should().BeSameAs(builtIn, "an array binds its engine once, when it is created");
                using var sum = before + before;
                counting.Adds.Should().Be(0, "the existing array's own engine computed the addition");
            }
            finally
            {
                TensorEngine.Default = null;
            }
        }

        /// <summary>
        ///     <c>a.TensorEngine = null</c> binds <see cref="TensorEngine.Default"/> (it used to keep the engine the storage
        ///     last saw), and the storage object the array holds moves with the array.
        /// </summary>
        [TestMethod]
        public void AssigningNullToAnArraysEngine_BindsTheCurrentDefault()
        {
            var counting = new CountingEngine();
            var a = np.arange(3.0);
            a.TensorEngine = counting;
            a.TensorEngine.Should().BeSameAs(counting);
            a.Storage.Engine.Should().BeSameAs(counting, "the storage object the array holds moves with it");

            a.TensorEngine = null;
            a.TensorEngine.Should().BeSameAs(TensorEngine.Default);
            a.Storage.Engine.Should().BeSameAs(TensorEngine.Default);
        }

        /// <summary>
        ///     The facades that dispatch on an operand refuse a null one at their entry (np.EngineOf, which sent a null
        ///     operand to the default engine, is gone), while a sequence facade with no first operand still reaches
        ///     TensorEngine.Default's member and NumPy's error for it.
        /// </summary>
        [TestMethod]
        public void NullOperand_IsRefusedAtTheFacade_AnEmptySequenceStillMeetsNumPysError()
        {
            new Action(() => np.fft.fft(null)).Should().Throw<ArgumentNullException>().Which.ParamName.Should().Be("a");
            new Action(() => np.linalg.inv(null)).Should().Throw<ArgumentNullException>().Which.ParamName.Should().Be("a");
            new Action(() => np.kron(null, np.arange(2.0))).Should().Throw<ArgumentNullException>().Which.ParamName.Should().Be("a");
            new Action(() => np.interp(null, np.arange(2.0), np.arange(2.0))).Should().Throw<ArgumentNullException>()
                .Which.ParamName.Should().Be("x");

            new Action(() => np.lexsort(new NDArray[0])).Should().Throw<TypeError>()
                .WithMessage("need sequence of keys with len > 0 in lexsort");
            new Action(() => np.linalg.multi_dot(new NDArray[0])).Should().Throw<ValueError>()
                .WithMessage("Expecting at least two arrays.");
            new Action(() => np.ravel_multi_index(new NDArray[] { null }, new[] { 4, 4 })).Should().Throw<ValueError>()
                .WithMessage("parameter multi_index must be a sequence of length 2");
        }
    }
}
