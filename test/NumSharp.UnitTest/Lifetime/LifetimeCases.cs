using System;
using System.Collections.Generic;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Lifetime
{
    /// <summary>
    ///     One operation under lifetime test: how to build its operands, and how to run it.
    /// </summary>
    /// <remarks>
    ///     Operands come from a factory rather than shared statics so each sweep gets its own live
    ///     arrays — the over-dispose sweep disposes them on purpose.
    /// </remarks>
    internal sealed class LifetimeCase
    {
        public string Name { get; }

        /// <summary>Builds the operands. Called OUTSIDE any measured window.</summary>
        public Func<NDArray[]> MakeOperands { get; }

        /// <summary>Runs the operation against those operands and returns whatever it produced.</summary>
        public Func<NDArray[], object> Run { get; }

        public LifetimeCase(string name, Func<NDArray[]> makeOperands, Func<NDArray[], object> run)
        {
            Name = name;
            MakeOperands = makeOperands;
            Run = run;
        }

        /// <summary>
        ///     Releases whatever <see cref="Run"/> handed back. Ops return an NDArray, an array of
        ///     them, or a plain scalar; only the first two own memory.
        /// </summary>
        public static void DisposeResult(object result)
        {
            switch (result)
            {
                case NDArray nd:
                    nd.Dispose();
                    break;
                case NDArray[] many:
                    foreach (var n in many)
                    {
                        // NOT `n != null` — NDArray overloads != into an elementwise comparison
                        // that returns an NDArray<bool>, so the null check would build an array.
                        if (n is not null)
                            n.Dispose();
                    }
                    break;
            }
        }
    }

    /// <summary>
    ///     The catalogue swept by the lifetime tests, covering the np.* families that allocate
    ///     intermediates: creation, elementwise, reductions, manipulation, indexing, linear
    ///     algebra, sorting, casting and logic.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Every operand must be built in <c>MakeOperands</c>, never inside <c>Run</c>.</b> An
    ///     array allocated inside Run is an allocation the sweep attributes to the operation — the
    ///     harness would report its own garbage as a library defect. This bit the first draft: a
    ///     <c>convolve</c> case that built its kernel inline read as 3 stranded buffers per call
    ///     when the true figure was 1.
    ///     </para>
    ///     <para>
    ///     Sizes stay small and inside the pool's window
    ///     (<c>SizeBucketedBufferPool.MinPoolableBytes</c>..<c>MaxPoolableBytes</c>, 1 B..64 MiB) so
    ///     every acquire and release passes the counters the release sweep reads.
    ///     </para>
    /// </remarks>
    internal static class LifetimeCases
    {
        private static NDArray D1k() => np.arange(1000).astype(NPTypeCode.Double);
        private static NDArray I2d() => np.arange(2000).reshape(50, 40).astype(NPTypeCode.Int32);
        private static NDArray D2d() => np.arange(2000).reshape(50, 40).astype(NPTypeCode.Double);
        private static NDArray Sq() => np.arange(400).reshape(20, 20).astype(NPTypeCode.Double);

        private static NDArray RowMask(int rows)
        {
            var bits = new bool[rows];
            for (int i = 0; i < rows; i++) bits[i] = (i % 2) == 0;
            return new NDArray(bits);
        }

        public static IEnumerable<LifetimeCase> All()
        {
            // ---------------------------------------------------------------- creation
            yield return new("np.zeros", () => new NDArray[0], _ => np.zeros(new Shape(1000), NPTypeCode.Double));
            yield return new("np.ones", () => new NDArray[0], _ => np.ones(new Shape(1000), NPTypeCode.Double));
            yield return new("np.full", () => new NDArray[0], _ => np.full(new Shape(1000), 3.5, NPTypeCode.Double));
            yield return new("np.arange", () => new NDArray[0], _ => np.arange(1000));
            yield return new("np.linspace", () => new NDArray[0], _ => np.linspace(0, 1, 1000));
            yield return new("np.eye", () => new NDArray[0], _ => np.eye(50));
            yield return new("np.identity", () => new NDArray[0], _ => np.identity(50));
            yield return new("np.zeros_like", () => new[] { D1k() }, o => np.zeros_like(o[0]));
            yield return new("np.ones_like", () => new[] { D1k() }, o => np.ones_like(o[0]));
            yield return new("np.copy", () => new[] { D1k() }, o => np.copy(o[0]));

            // ------------------------------------------------------------- elementwise
            yield return new("add", () => new[] { D1k(), D1k() }, o => o[0] + o[1]);
            yield return new("subtract", () => new[] { D1k(), D1k() }, o => o[0] - o[1]);
            yield return new("multiply", () => new[] { D1k(), D1k() }, o => o[0] * o[1]);
            yield return new("divide", () => new[] { D1k(), D1k() }, o => o[0] / o[1]);
            yield return new("np.power", () => new[] { D1k(), D1k() }, o => np.power(o[0], o[1]));
            yield return new("np.mod", () => new[] { D1k(), D1k() }, o => np.mod(o[0], o[1]));
            yield return new("np.floor_divide", () => new[] { D1k(), D1k() }, o => np.floor_divide(o[0], o[1]));
            yield return new("np.arctan2", () => new[] { D1k(), D1k() }, o => np.arctan2(o[0], o[1]));
            yield return new("np.maximum", () => new[] { D1k(), D1k() }, o => np.maximum(o[0], o[1]));
            yield return new("np.minimum", () => new[] { D1k(), D1k() }, o => np.minimum(o[0], o[1]));
            yield return new("np.sqrt", () => new[] { D1k() }, o => np.sqrt(o[0]));
            yield return new("np.exp", () => new[] { D1k() }, o => np.exp(o[0]));
            yield return new("np.log", () => new[] { D1k() }, o => np.log(o[0]));
            yield return new("np.sin", () => new[] { D1k() }, o => np.sin(o[0]));
            yield return new("np.cos", () => new[] { D1k() }, o => np.cos(o[0]));
            yield return new("np.tanh", () => new[] { D1k() }, o => np.tanh(o[0]));
            yield return new("np.abs", () => new[] { D1k() }, o => np.abs(o[0]));
            yield return new("np.negative", () => new[] { D1k() }, o => np.negative(o[0]));
            yield return new("np.square", () => new[] { D1k() }, o => np.square(o[0]));
            yield return new("np.sign", () => new[] { D1k() }, o => np.sign(o[0]));
            yield return new("np.floor", () => new[] { D1k() }, o => np.floor(o[0]));
            yield return new("np.ceil", () => new[] { D1k() }, o => np.ceil(o[0]));
            yield return new("np.rint", () => new[] { D1k() }, o => np.rint(o[0]));
            yield return new("np.clip", () => new[] { D1k() }, o => np.clip(o[0], 100, 900));
            yield return new("np.reciprocal", () => new[] { D1k() }, o => np.reciprocal(o[0]));

            // -------------------------------------------------------------- comparison
            yield return new("equal", () => new[] { D1k(), D1k() }, o => np.equal(o[0], o[1]));
            yield return new("greater", () => new[] { D1k(), D1k() }, o => np.greater(o[0], o[1]));
            yield return new("less_equal", () => new[] { D1k(), D1k() }, o => np.less_equal(o[0], o[1]));
            yield return new("logical_and", () => new[] { D1k(), D1k() }, o => np.logical_and(o[0], o[1]));
            yield return new("np.isnan", () => new[] { D1k() }, o => np.isnan(o[0]));
            yield return new("np.isfinite", () => new[] { D1k() }, o => np.isfinite(o[0]));

            // -------------------------------------------------------------- reductions
            yield return new("np.sum", () => new[] { D1k() }, o => np.sum(o[0]));
            yield return new("np.sum axis", () => new[] { D2d() }, o => np.sum(o[0], axis: 1));
            yield return new("np.prod", () => new[] { D1k() }, o => np.prod(o[0]));
            yield return new("np.mean", () => new[] { D1k() }, o => np.mean(o[0]));
            yield return new("np.mean axis", () => new[] { D2d() }, o => np.mean(o[0], axis: 0));
            yield return new("np.std", () => new[] { D1k() }, o => np.std(o[0]));
            yield return new("np.var", () => new[] { D1k() }, o => np.var(o[0]));
            yield return new("np.amax", () => new[] { D1k() }, o => np.amax(o[0]));
            yield return new("np.amin", () => new[] { D1k() }, o => np.amin(o[0]));
            yield return new("np.median", () => new[] { D1k() }, o => np.median(o[0]));
            yield return new("np.percentile", () => new[] { D1k() }, o => np.percentile(o[0], 75));
            yield return new("np.argmax", () => new[] { D1k() }, o => np.argmax(o[0]));
            yield return new("np.argmax axis", () => new[] { I2d() }, o => np.argmax(o[0], axis: 1));
            yield return new("np.argmin axis", () => new[] { I2d() }, o => np.argmin(o[0], axis: 1));
            yield return new("np.cumsum", () => new[] { D1k() }, o => np.cumsum(o[0]));
            yield return new("np.cumprod", () => new[] { D1k() }, o => np.cumprod(o[0]));
            yield return new("np.diff", () => new[] { D1k() }, o => np.diff(o[0]));
            yield return new("np.ptp", () => new[] { D1k() }, o => np.ptp(o[0]));
            yield return new("np.count_nonzero", () => new[] { D1k() }, o => np.count_nonzero(o[0]));
            yield return new("np.average", () => new[] { D1k() }, o => np.average(o[0]));
            yield return new("np.nansum", () => new[] { D1k() }, o => np.nansum(o[0]));
            yield return new("np.nanmean", () => new[] { D1k() }, o => np.nanmean(o[0]));
            yield return new("np.nanmax", () => new[] { D1k() }, o => np.nanmax(o[0]));

            // ------------------------------------------------------------ manipulation
            yield return new("reshape", () => new[] { D1k() }, o => o[0].reshape(25, 40));
            yield return new("transpose", () => new[] { D2d() }, o => o[0].T);
            yield return new("np.ravel", () => new[] { D2d() }, o => np.ravel(o[0]));
            yield return new("flatten C", () => new[] { D2d() }, o => o[0].flatten());
            yield return new("flatten F", () => new[] { D2d() }, o => o[0].flatten('F'));
            yield return new("np.concatenate", () => new[] { I2d(), I2d() }, o => np.concatenate(new[] { o[0], o[1] }, axis: 0));
            yield return new("np.concatenate null", () => new[] { I2d(), I2d() }, o => np.concatenate(new[] { o[0], o[1] }, axis: (int?)null));
            yield return new("np.stack", () => new[] { D1k(), D1k() }, o => np.stack(new[] { o[0], o[1] }));
            yield return new("np.split", () => new[] { D1k() }, o => np.split(o[0], 2));
            yield return new("np.tile", () => new[] { I2d() }, o => np.tile(o[0], new long[] { 2, 2 }));
            yield return new("np.repeat", () => new[] { D1k() }, o => np.repeat(o[0], 2));
            yield return new("np.roll", () => new[] { D1k() }, o => np.roll(o[0], 3));
            yield return new("np.flip", () => new[] { D2d() }, o => np.flip(o[0], 0));
            yield return new("np.expand_dims", () => new[] { D1k() }, o => np.expand_dims(o[0], 0));
            yield return new("np.moveaxis", () => new[] { D2d() }, o => np.moveaxis(o[0], 0, 1));
            yield return new("np.swapaxes", () => new[] { D2d() }, o => np.swapaxes(o[0], 0, 1));
            yield return new("np.rot90", () => new[] { D2d() }, o => np.rot90(o[0]));
            yield return new("np.broadcast_to", () => new[] { D1k() }, o => np.broadcast_to(o[0], new Shape(4, 1000)));
            yield return new("np.unique", () => new[] { I2d() }, o => np.unique(o[0]));
            yield return new("np.append", () => new[] { D1k(), D1k() }, o => np.append(o[0], o[1]));

            // ---------------------------------------------------------------- indexing
            yield return new("slice view", () => new[] { D2d() }, o => o[0]["1:10"]);
            yield return new("boolean mask", () => new[] { I2d(), RowMask(50) }, o => o[0][o[1].MakeGeneric<bool>()]);
            yield return new("fancy index", () => new[] { I2d(), np.array(new int[] { 0, 5, 10, 20 }) }, o => o[0][o[1]]);
            yield return new("np.where 3-arg", () => new[] { I2d(), I2d() }, o => np.where(o[0] > 100, o[0], o[1]));
            yield return new("np.nonzero", () => new[] { I2d() }, o => np.nonzero(o[0]));
            yield return new("np.argwhere", () => new[] { I2d() }, o => np.argwhere(o[0]));
            yield return new("np.flatnonzero", () => new[] { I2d() }, o => np.flatnonzero(o[0]));
            yield return new("np.take", () => new[] { D1k(), np.array(new int[] { 1, 3, 5 }) }, o => np.take(o[0], o[1]));
            yield return new("np.extract", () => new[] { I2d() }, o => np.extract(o[0] > 100, o[0]));

            // ---------------------------------------------------------- linear algebra
            yield return new("np.matmul", () => new[] { Sq(), Sq() }, o => np.matmul(o[0], o[1]));
            yield return new("np.dot", () => new[] { D1k(), D1k() }, o => np.dot(o[0], o[1]));
            yield return new("np.outer", () => new[] { Sq(), Sq() }, o => np.outer(o[0], o[1]));
            yield return new("np.trace", () => new[] { Sq() }, o => np.trace(o[0]));
            yield return new("np.diagonal", () => new[] { Sq() }, o => np.diagonal(o[0]));

            // ----------------------------------------------------------------- sorting
            yield return new("np.sort", () => new[] { D1k() }, o => np.sort(o[0]));
            yield return new("np.argsort", () => new[] { D1k() }, o => np.argsort(o[0]));

            // ----------------------------------------------------------------- casting
            yield return new("astype double", () => new[] { I2d() }, o => o[0].astype(NPTypeCode.Double));
            yield return new("astype int16", () => new[] { D2d() }, o => o[0].astype(NPTypeCode.Int16));
            yield return new("astype bool", () => new[] { I2d() }, o => o[0].astype(NPTypeCode.Boolean));

            // ------------------------------------------------------------------- logic
            yield return new("np.all", () => new[] { D1k() }, o => np.all(o[0]));
            yield return new("np.any", () => new[] { D1k() }, o => np.any(o[0]));
            yield return new("np.allclose", () => new[] { D1k(), D1k() }, o => np.allclose(o[0], o[1]));
            yield return new("np.isclose", () => new[] { D1k(), D1k() }, o => np.isclose(o[0], o[1]));
            yield return new("np.array_equal", () => new[] { D1k(), D1k() }, o => np.array_equal(o[0], o[1]));

            // ---------------------------------------------------------------- convolve
            yield return new("convolve full", () => new[] { D1k(), np.arange(16).astype(NPTypeCode.Double) }, o => o[0].convolve(o[1], "full"));
            yield return new("convolve same", () => new[] { D1k(), np.arange(16).astype(NPTypeCode.Double) }, o => o[0].convolve(o[1], "same"));
            yield return new("convolve valid", () => new[] { D1k(), np.arange(16).astype(NPTypeCode.Double) }, o => o[0].convolve(o[1], "valid"));
        }
    }
}
