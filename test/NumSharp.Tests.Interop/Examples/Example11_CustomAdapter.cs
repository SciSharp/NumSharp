using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/11-custom-adapter.cs</c>, section by section.</summary>
    [TestClass]
    public class Example11_CustomAdapter : ExampleTestBase
    {
        [TestInitialize]
        public void DefineTheTinyImageLibrary()
        {
            PyExec("class Tile:\n" +
                   "    def __init__(self, pixels, name):\n" +
                   "        self.pixels = pixels\n" +
                   "        self.name = name\n" +
                   "\n" +
                   "class LazyTile:\n" +
                   "    def __init__(self, n):\n" +
                   "        self.n = n\n" +
                   "    def materialize(self):\n" +
                   "        return np.arange(self.n, dtype='f8') ** 2\n" +
                   "\n" +
                   "tile = Tile(np.arange(6, dtype='f8').reshape(2, 3), 'north')\n" +
                   "lazy = LazyTile(4)");
            PythonArrayAdapterRegistry.Register(new TileAdapter());     // process-wide and sticky: whichever test runs first adds it
        }

        [TestMethod]
        public void Registration_ProcessWide_ThreadSafe_IdempotentByName()
        {
            NDArrayPythonInterop.RegisterArrayAdapter(new TileAdapter()).Should().BeFalse("the same Name again");
            PythonArrayAdapterRegistry.Register(new TileAdapter()).Should().BeFalse("RegisterArrayAdapter is the same call");
            PythonArrayAdapterRegistry.Register(TorchPythonArrayAdapter.Instance).Should().BeFalse("'torch.Tensor' is built in");
            PythonArrayAdapterRegistry.Register(PandasPythonArrayAdapter.Instance).Should().BeFalse("'pandas' is built in");
            Throws(() => PythonArrayAdapterRegistry.Register(new Nameless())).Should().BeOfType<ArgumentException>("an adapter must have a non-empty Name");
        }

        [TestMethod]
        public void TheDirectVerbs_TheAdapterSelectsPixels_TheBridgeDoesTheRest()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject tile = py.tile;
                NDArray view = tile.AsNDArray();
                view.shape.Should().Equal(2, 3);
                view.typecode.Should().Be(NPTypeCode.Double);
                view.Shape.IsWriteable.Should().BeTrue();
                view[1, 2] = 71.0;
                PyFloat("tile.pixels[1, 2]").Should().Be(71.0, "the adapter chose the object, the bridge shares its memory");
                NDArray copy = tile.ToNDArray();
                copy[0, 0] = -8.0;
                PyFloat("tile.pixels[0, 0]").Should().Be(0.0, "an owning copy is detached");
            }
        }

        [TestMethod]
        public void Declining_NullFromAdaptMeansNotThisOne()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject lazy = py.lazy;
                Throws(() => lazy.AsNDArray()).Should().BeOfType<NotSupportedException>().Which.Message.Should().Contain("IPythonArrayAdapter");
                NDArray materialized = lazy.ToNDArray();
                materialized.size.Should().Be(4);
                materialized.GetDouble(3).Should().Be(9.0, "allowCopy: true -> materialize()");
            }
        }

        [TestMethod]
        public void TheRegisteredCodec_ConsultsTheSameRegistry()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                NDArray auto = py.tile;
                auto[0, 1] = 5.5;
                PyFloat("tile.pixels[0, 1]").Should().Be(5.5, "Auto: a shared view of tile.pixels");
                NDArray lazyAuto = py.lazy;
                lazyAuto.GetDouble(2).Should().Be(4.0, "Auto: View declined -> the Copy route");
                lazyAuto.Shape.IsWriteable.Should().BeTrue();

                var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View });
                using PyObject lazy = py.lazy;
                viewOnly.TryDecode(lazy, out NDArray none).Should().BeFalse("a View-mode codec declines LazyTile");
                none.Should().BeNull();

                var noAdapters = new NumpyCodec(new NumpyCodecOptions { DecodeArrayAdapters = false });
                using PyObject tile = py.tile;
                using PyObject t = tile.GetPythonType();
                using var tileType = new PyType(t);
                noAdapters.CanDecode(tileType, typeof(NDArray)).Should().BeFalse("Tile has no buffer of its own");
            }
        }

        [TestMethod]
        public void Robustness_ABrokenAdapterCannotBreakThePipeline()
        {
            PythonArrayAdapterRegistry.Register(new ThrowingAdapter());   // registered like any other (sticky; the result depends on test order)
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("tile.pixels[0, 1] = 5.5");
                NDArray still = py.tile;
                still.GetDouble(0, 1).Should().Be(5.5, "the registry swallows the broken adapter's exception and TileAdapter still serves");
            }
        }

        /// <summary>Recognize the type, hand back the library's canonical array object, or decline.</summary>
        private sealed class TileAdapter : IPythonArrayAdapter
        {
            public string Name => "examples.Tile";

            public bool CanAdapt(PyType objectType)
            {
                try
                {
                    using PyObject name = objectType.GetAttr("__name__");
                    string n = name.As<string>();
                    return n == "Tile" || n == "LazyTile";
                }
                catch { return false; }
            }

            public PyObject Adapt(PyObject source, bool allowCopy)
            {
                using PyObject typeObj = source.GetPythonType();
                using var type = new PyType(typeObj);
                if (type.Name == "Tile")
                    return source.GetAttr("pixels");
                return allowCopy ? source.InvokeMethod("materialize") : null;
            }
        }

        private sealed class ThrowingAdapter : IPythonArrayAdapter
        {
            public string Name => "examples.Throwing";
            public bool CanAdapt(PyType objectType) => throw new InvalidOperationException("a third-party probe that fails");
            public PyObject Adapt(PyObject source, bool allowCopy) => null;
        }

        private sealed class Nameless : IPythonArrayAdapter
        {
            public string Name => "";
            public bool CanAdapt(PyType objectType) => false;
            public PyObject Adapt(PyObject source, bool allowCopy) => null;
        }
    }
}
