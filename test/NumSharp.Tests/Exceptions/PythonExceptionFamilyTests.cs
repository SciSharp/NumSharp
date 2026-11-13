using System;
using System.Reflection;

namespace NumSharp.Tests.Exceptions
{
    /// <summary>
    ///     The Python-named exception family against CPython 3.12.12 (the interpreter the NumPy 2.4.2 oracle runs): a bare
    ///     form reads Python's <c>str(X())</c>, every class takes a cause (Python's <c>raise X(msg) from e</c>), Python's
    ///     hierarchy holds where NumSharp follows it, and NumSharp's raise sites that mirror NumPy's <c>NotImplementedError</c>
    ///     raise that class with NumPy's text.
    /// </summary>
    /// <remarks>
    ///     A port that compares its error texts with Python's (Gym.NET's parity suite does) depends on every row here: a bare
    ///     <c>assert</c> is <c>AssertionError()</c>, whose text is empty, and a base class's <c>raise NotImplementedError</c>
    ///     is the empty <c>NotImplementedError()</c>.
    /// </remarks>
    [TestClass]
    public class PythonExceptionFamilyTests
    {
        /// <summary>
        ///     <c>str(X())</c> in CPython 3.12.12 for every Python-named class NumSharp defines: empty, except
        ///     <c>SyntaxError</c>'s <c>'None'</c> (its <c>__str__</c> prints the <c>msg</c> attribute). NumPy's
        ///     <c>AxisError</c> has no bare form (it requires an axis), and NumSharp's reads empty like the rest.
        /// </summary>
        /// <param name="type">The NumSharp class.</param>
        /// <param name="python">CPython's <c>str()</c> of the bare exception.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        [DataRow(typeof(AssertionError), "")]
        [DataRow(typeof(AttributeError), "")]
        [DataRow(typeof(AxisError), "")]
        [DataRow(typeof(BufferError), "")]
        [DataRow(typeof(DTypePromotionError), "")]
        [DataRow(typeof(ImportError), "")]
        [DataRow(typeof(IndexError), "")]
        [DataRow(typeof(KeyError), "")]
        [DataRow(typeof(LinAlgError), "")]
        [DataRow(typeof(LookupError), "")]
        [DataRow(typeof(MAError), "")]
        [DataRow(typeof(MaskError), "")]
        [DataRow(typeof(ModuleNotFoundError), "")]
        [DataRow(typeof(NameError), "")]
        [DataRow(typeof(NotImplementedError), "")]
        [DataRow(typeof(OSError), "")]
        [DataRow(typeof(RuntimeError), "")]
        [DataRow(typeof(SyntaxError), "None")]
        [DataRow(typeof(SystemError), "")]
        [DataRow(typeof(TooHardError), "")]
        [DataRow(typeof(TypeError), "")]
        [DataRow(typeof(UnicodeError), "")]
        [DataRow(typeof(ValueError), "")]
        public void Parameterless_ReadsPythonsStr(Type type, string python)
        {
            var error = (Exception)Activator.CreateInstance(type);
            error.Message.Should().Be(python, $"str({type.Name}()) is {python}");
        }

        /// <summary>
        ///     A bare error prints as Python's traceback prints it: the class name alone, no <c>": "</c> after it (an empty
        ///     message adds nothing to <see cref="Exception.ToString"/>).
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence.</exception>
        [TestMethod]
        public void BareError_PrintsTheClassNameAlone()
        {
            new AssertionError().ToString().Should().Be("NumSharp.AssertionError");
            new NotImplementedError().ToString().Should().Be("NumSharp.NotImplementedError");
            new KeyError().ToString().Should().Be("NumSharp.KeyError");
        }

        /// <summary>
        ///     Every class takes a message with the exception that caused it, through a public <c>(string, Exception)</c>
        ///     constructor: C#'s spelling of Python's <c>raise X(msg) from e</c>, the cause kept in
        ///     <see cref="Exception.InnerException"/> and the message verbatim.
        /// </summary>
        /// <param name="type">The NumSharp class.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, on a missing constructor or a lost
        ///     message or cause.</exception>
        [TestMethod]
        [DataRow(typeof(AssertionError))]
        [DataRow(typeof(AttributeError))]
        [DataRow(typeof(AxisError))]
        [DataRow(typeof(BufferError))]
        [DataRow(typeof(DTypePromotionError))]
        [DataRow(typeof(ImportError))]
        [DataRow(typeof(IndexError))]
        [DataRow(typeof(KeyError))]
        [DataRow(typeof(LinAlgError))]
        [DataRow(typeof(LookupError))]
        [DataRow(typeof(MAError))]
        [DataRow(typeof(MaskError))]
        [DataRow(typeof(ModuleNotFoundError))]
        [DataRow(typeof(NameError))]
        [DataRow(typeof(NotImplementedError))]
        [DataRow(typeof(OSError))]
        [DataRow(typeof(RuntimeError))]
        [DataRow(typeof(SyntaxError))]
        [DataRow(typeof(SystemError))]
        [DataRow(typeof(TooHardError))]
        [DataRow(typeof(TypeError))]
        [DataRow(typeof(UnicodeDecodeError))]
        [DataRow(typeof(UnicodeEncodeError))]
        [DataRow(typeof(UnicodeError))]
        [DataRow(typeof(ValueError))]
        public void MessageAndCause_KeepsBoth(Type type)
        {
            ConstructorInfo ctor = type.GetConstructor(new[] { typeof(string), typeof(Exception) });
            ctor.Should().NotBeNull($"NumSharp.{type.Name} takes (message, innerException)");
            ctor.IsPublic.Should().BeTrue();
            var cause = new InvalidOperationException("cause");
            var error = (Exception)ctor.Invoke(new object[] { "outer", cause });
            error.Message.Should().Be("outer");
            error.InnerException.Should().BeSameAs(cause);
        }

        /// <summary>
        ///     <c>NotImplementedError</c> sits where Python puts it, under <c>RuntimeError</c>: <c>catch (RuntimeError)</c>
        ///     catches it as <c>except RuntimeError</c> does, and it is no .NET <see cref="NotImplementedException"/>.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from Python's MRO.</exception>
        /// <exception cref="NotImplementedError">Never escapes: the probe's own throw, which the
        ///     <c>Throw&lt;RuntimeError&gt;</c> assertion catches to prove the hierarchy.</exception>
        [TestMethod]
        public void NotImplementedError_FollowsPythonsHierarchy()
        {
            typeof(NotImplementedError).BaseType.Should().Be(typeof(RuntimeError));
            typeof(INumSharpException).IsAssignableFrom(typeof(NotImplementedError)).Should().BeTrue();
            typeof(NotImplementedException).IsAssignableFrom(typeof(NotImplementedError)).Should().BeFalse();

            Action raise = () => throw new NotImplementedError("abstract");
            raise.Should().Throw<RuntimeError>().WithMessage("abstract");
        }

        /// <summary>
        ///     <c>ImportError</c> carries CPython's <c>msg</c> / <c>name</c> / <c>path</c> for each of its constructor forms
        ///     (<c>ImportError()</c>, <c>ImportError('m')</c>, <c>ImportError('m', name='x', path='/p')</c> and
        ///     <c>ImportError(name='x')</c>, whose <c>str()</c> is <c>''</c>).
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void ImportError_CarriesPythonsAttributes()
        {
            var bare = new ImportError();
            (bare.Message, bare.msg, bare.name, bare.path).Should().Be(("", null, null, null));

            var plain = new ImportError("m");
            (plain.Message, plain.msg, plain.name, plain.path).Should().Be(("m", "m", null, null));

            var keywords = new ImportError("m", name: "x", path: "/p");
            (keywords.Message, keywords.msg, keywords.name, keywords.path).Should().Be(("m", "m", "x", "/p"));

            var nameOnly = new ImportError(null, name: "x", path: null);
            (nameOnly.Message, nameOnly.msg, nameOnly.name, nameOnly.path).Should().Be(("", null, "x", null));

            var cause = new InvalidOperationException("cause");
            var chained = new ImportError("m", "x", null, cause);
            (chained.Message, chained.name, chained.InnerException).Should().Be(("m", "x", (Exception)cause));
        }

        /// <summary>
        ///     <c>ModuleNotFoundError</c> is an <c>ImportError</c>, as in Python, and takes the import system's
        ///     <c>name</c>: CPython's <c>import nosuchmodule</c> raises <c>No module named 'nosuchmodule'</c> with that name.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void ModuleNotFoundError_IsAnImportError()
        {
            typeof(ModuleNotFoundError).BaseType.Should().Be(typeof(ImportError));
            typeof(ImportError).BaseType.Should().Be(typeof(NumSharpException));

            var missing = new ModuleNotFoundError("No module named 'nosuchmodule'", "nosuchmodule", null);
            missing.Message.Should().Be("No module named 'nosuchmodule'");
            missing.name.Should().Be("nosuchmodule");
            Action raise = () => throw missing;
            raise.Should().Throw<ImportError>();
        }

        /// <summary>
        ///     NumSharp's raise sites that mirror NumPy's <c>NotImplementedError</c> raise that class with NumPy 2.4.2's text
        ///     (probed): numpy.ma's 2-D-only functions on a 1-D or 3-D array, a masked array's <c>tofile</c>, and a seedless
        ///     <c>SeedSequence</c> asked for state directly or by a bit generator.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void NumPysNotImplementedSites_RaiseNotImplementedError()
        {
            using var a1 = np.arange(3);
            using var a3 = np.arange(8).reshape(2, 2, 2);
            var m1 = np.ma.array(a1);
            var m3 = np.ma.array(a3);

            Action tofile = () => np.ma.array(np.array(new[] { 1.0 })).tofile("x.bin");
            tofile.Should().ThrowExactly<NotImplementedError>().WithMessage("MaskedArray.tofile() not implemented yet.");
            Action rows = () => np.ma.compress_rows(m1);
            rows.Should().ThrowExactly<NotImplementedError>().WithMessage("compress_rows works for 2D arrays only.");
            Action cols = () => np.ma.compress_cols(m1);
            cols.Should().ThrowExactly<NotImplementedError>().WithMessage("compress_cols works for 2D arrays only.");
            Action rowcols = () => np.ma.compress_rowcols(m3);
            rowcols.Should().ThrowExactly<NotImplementedError>().WithMessage("compress_rowcols works for 2D arrays only.");
            Action mask = () => np.ma.mask_rowcols(m3);
            mask.Should().ThrowExactly<NotImplementedError>().WithMessage("mask_rowcols works for 2D arrays only.");
            Action maskRows = () => np.ma.mask_rows(m3);
            maskRows.Should().ThrowExactly<NotImplementedError>().WithMessage("mask_rowcols works for 2D arrays only.");
            Action maskCols = () => np.ma.mask_cols(m3);
            maskCols.Should().ThrowExactly<NotImplementedError>().WithMessage("mask_rowcols works for 2D arrays only.");
            Action contiguous = () => np.ma.notmasked_contiguous(m3, 0);
            contiguous.Should().ThrowExactly<NotImplementedError>().WithMessage("Currently limited to at most 2D array.");

            var seedless = new SeedlessSeedSequence();
            Action state = () => seedless.generate_state(2);
            state.Should().ThrowExactly<NotImplementedError>().WithMessage("seedless SeedSequences cannot generate state");
            Action pcg = () => new PCG64(seedless);
            pcg.Should().ThrowExactly<NotImplementedError>().WithMessage("seedless SeedSequences cannot generate state");
        }
    }
}
