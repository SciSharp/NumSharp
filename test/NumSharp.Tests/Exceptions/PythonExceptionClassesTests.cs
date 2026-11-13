using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using NumSharp.Backends;
using NumSharp.IO;

namespace NumSharp.Tests.Exceptions
{
    /// <summary>
    ///     The Python and NumPy exception classes NumSharp provides in full (CPython 3.12.12's builtins, NumPy 2.4.2's
    ///     <c>LapackError</c>, <c>ConverterError</c>, <c>ConverterLockError</c>, <c>_ArrayMemoryError</c> and the
    ///     <c>UFuncTypeError</c> family): each row is a CPython / NumPy probe of the same call, on Windows and Linux (the
    ///     platform-dependent rows check the platform the test runs on).
    /// </summary>
    /// <remarks>
    ///     The classes follow the owner's 2026-11-11 decisions: a class directly under Python's <c>Exception</c> takes the
    ///     .NET exception with the same meaning as its base, every child follows Python's MRO, NumSharp's raise sites that
    ///     mirror NumPy / CPython raise the Python class, and the classes carry Python's attributes.
    /// </remarks>
    [TestClass]
    public class PythonExceptionClassesTests
    {
        /// <summary><c>str(X())</c> in CPython 3.12.12 for every class with a bare form: empty, except the
        ///     <c>SyntaxError</c> family's <c>'None'</c>.</summary>
        /// <param name="type">The NumSharp class.</param>
        /// <param name="python">CPython's <c>str()</c> of the bare exception.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        [DataRow(typeof(ArithmeticError), "")]
        [DataRow(typeof(FloatingPointError), "")]
        [DataRow(typeof(OverflowError), "")]
        [DataRow(typeof(ZeroDivisionError), "")]
        [DataRow(typeof(MemoryError), "")]
        [DataRow(typeof(EOFError), "")]
        [DataRow(typeof(ReferenceError), "")]
        [DataRow(typeof(StopIteration), "")]
        [DataRow(typeof(StopAsyncIteration), "")]
        [DataRow(typeof(RecursionError), "")]
        [DataRow(typeof(UnboundLocalError), "")]
        [DataRow(typeof(IndentationError), "None")]
        [DataRow(typeof(TabError), "None")]
        [DataRow(typeof(KeyboardInterrupt), "")]
        [DataRow(typeof(SystemExit), "")]
        [DataRow(typeof(GeneratorExit), "")]
        [DataRow(typeof(LapackError), "")]
        [DataRow(typeof(ConverterError), "")]
        [DataRow(typeof(ConverterLockError), "")]
        [DataRow(typeof(BlockingIOError), "")]
        [DataRow(typeof(ChildProcessError), "")]
        [DataRow(typeof(ConnectionError), "")]
        [DataRow(typeof(BrokenPipeError), "")]
        [DataRow(typeof(ConnectionAbortedError), "")]
        [DataRow(typeof(ConnectionRefusedError), "")]
        [DataRow(typeof(ConnectionResetError), "")]
        [DataRow(typeof(FileExistsError), "")]
        [DataRow(typeof(FileNotFoundError), "")]
        [DataRow(typeof(InterruptedError), "")]
        [DataRow(typeof(IsADirectoryError), "")]
        [DataRow(typeof(NotADirectoryError), "")]
        [DataRow(typeof(PermissionError), "")]
        [DataRow(typeof(ProcessLookupError), "")]
        [DataRow(typeof(TimeoutError), "")]
        public void Parameterless_ReadsPythonsStr(Type type, string python)
        {
            var error = (Exception)Activator.CreateInstance(type);
            error.Message.Should().Be(python, $"str({type.Name}()) is {python}");
        }

        /// <summary>Every class with Python's one-argument form takes a message with a cause: C#'s spelling of
        ///     <c>raise X(msg) from e</c>.</summary>
        /// <param name="type">The NumSharp class.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, on a missing constructor or a lost
        ///     message or cause.</exception>
        [TestMethod]
        [DataRow(typeof(ArithmeticError))]
        [DataRow(typeof(FloatingPointError))]
        [DataRow(typeof(OverflowError))]
        [DataRow(typeof(ZeroDivisionError))]
        [DataRow(typeof(MemoryError))]
        [DataRow(typeof(EOFError))]
        [DataRow(typeof(ReferenceError))]
        [DataRow(typeof(StopIteration))]
        [DataRow(typeof(StopAsyncIteration))]
        [DataRow(typeof(RecursionError))]
        [DataRow(typeof(UnboundLocalError))]
        [DataRow(typeof(IndentationError))]
        [DataRow(typeof(TabError))]
        [DataRow(typeof(KeyboardInterrupt))]
        [DataRow(typeof(SystemExit))]
        [DataRow(typeof(GeneratorExit))]
        [DataRow(typeof(LapackError))]
        [DataRow(typeof(ConverterError))]
        [DataRow(typeof(ConverterLockError))]
        [DataRow(typeof(FileNotFoundError))]
        [DataRow(typeof(PermissionError))]
        [DataRow(typeof(BrokenPipeError))]
        [DataRow(typeof(TimeoutError))]
        [DataRow(typeof(UFuncTypeError))]
        [DataRow(typeof(UFuncNoLoopError))]
        [DataRow(typeof(UFuncBinaryResolutionError))]
        [DataRow(typeof(UFuncCastingError))]
        [DataRow(typeof(UFuncInputCastingError))]
        [DataRow(typeof(UFuncOutputCastingError))]
        [DataRow(typeof(ArrayMemoryError))]
        [DataRow(typeof(UnicodeTranslateError))]
        [DataRow(typeof(IncorrectSizeException))]
        [DataRow(typeof(IncorrectTypeException))]
        [DataRow(typeof(StringComputationNotSupportedException))]
        public void MessageAndCause_KeepsBoth(Type type)
        {
            ConstructorInfo ctor = type.GetConstructor(new[] { typeof(string), typeof(Exception) });
            ctor.Should().NotBeNull($"NumSharp.{type.Name} takes (message, innerException)");
            var cause = new InvalidOperationException("cause");
            var error = (Exception)ctor.Invoke(new object[] { "outer", cause });
            error.Message.Should().Be("outer");
            error.InnerException.Should().BeSameAs(cause);
        }

        /// <summary>
        ///     The gate: every public exception class NumSharp declares has .NET's standard <c>(string message, Exception
        ///     innerException)</c> constructor, and it keeps both (a group keeps the exception as its one member, which
        ///     <see cref="AggregateException"/> reports as <see cref="Exception.InnerException"/>). A new class without it
        ///     fails here, whatever its Python constructor is.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, naming every class without the
        ///     constructor or whose constructor loses the message or the cause.</exception>
        [TestMethod]
        public void EveryPublicException_TakesAMessageAndItsCause()
        {
            var types = typeof(np).Assembly.GetExportedTypes()
                .Where(t => typeof(Exception).IsAssignableFrom(t) && !t.IsAbstract && !t.ContainsGenericParameters)
                .OrderBy(t => t.FullName, StringComparer.Ordinal)
                .ToArray();
            types.Should().Contain(new[] { typeof(ValueError), typeof(UFuncNoLoopError), typeof(ExceptionGroup) },
                "the scan reaches the exported exception classes");

            var failures = new List<string>();
            var cause = new InvalidOperationException("cause");
            foreach (Type type in types)
            {
                ConstructorInfo ctor = type.GetConstructor(new[] { typeof(string), typeof(Exception) });
                if (ctor is null)
                {
                    failures.Add($"{type.FullName}: no public (string, Exception) constructor");
                    continue;
                }
                var error = (Exception)ctor.Invoke(new object[] { "outer", cause });
                // A group appends its member count ("outer (1 sub-exception)"); every other class keeps the text verbatim.
                if (!error.Message.StartsWith("outer", StringComparison.Ordinal))
                    failures.Add($"{type.FullName}: Message is \"{error.Message}\"");
                if (!ReferenceEquals(error.InnerException, cause))
                    failures.Add($"{type.FullName}: InnerException is not the cause");
            }
            failures.Should().BeEmpty();
        }

        /// <summary>
        ///     What the standard constructor leaves in Python's attributes for the classes whose Python constructor takes
        ///     other arguments: null (no ufunc, no dtypes, no text), 0 for the indices and positions, as
        ///     <see cref="UnicodeEncodeError"/>'s message form always did; a group gets the exception as its one member.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence.</exception>
        [TestMethod]
        public void MessageForms_LeaveThePythonAttributesEmpty()
        {
            var cause = new InvalidOperationException("cause");

            // The one-string UFuncTypeError is NumPy's UFuncTypeError(ufunc); the two-argument form is the message.
            new UFuncTypeError("add").Message.Should().Be("<ufunc 'add'>");
            new UFuncTypeError("add").ufunc.Should().Be("add");
            new UFuncTypeError("add", null).Message.Should().Be("add");
            new UFuncTypeError("add", null).ufunc.Should().BeNull();

            var noLoop = new UFuncNoLoopError("m", cause);
            noLoop.ufunc.Should().BeNull();
            noLoop.dtypes.Should().BeEmpty();
            var binary = new UFuncBinaryResolutionError("m", cause);
            binary.dtypes.Should().BeEmpty("NumPy's two-dtype assertion is for its own constructor");
            var cast = new UFuncCastingError("m", cause);
            cast.casting.Should().BeNull();
            cast.from_.Should().BeNull();
            cast.to.Should().BeNull();
            new UFuncInputCastingError("m", cause).in_i.Should().Be(0);
            new UFuncOutputCastingError("m", cause).out_i.Should().Be(0);
            new UFuncOutputCastingError("m", cause).Should().BeAssignableTo<TypeError>();

            var memory = new ArrayMemoryError("m", cause);
            memory.shape.Should().BeNull();
            memory.dtype.Should().BeNull();
            memory.Should().BeAssignableTo<OutOfMemoryException>();

            foreach (var translate in new[] { new UnicodeTranslateError("m"), new UnicodeTranslateError("m", cause) })
            {
                translate.Message.Should().Be("m");
                translate.@object.Should().BeNull();
                translate.reason.Should().BeNull();
                translate.encoding.Should().BeNull();
                translate.start.Should().Be(0);
                translate.end.Should().Be(0);
            }

            // A group has no separate cause: the exception is its one member, as AggregateException(message, inner) has it.
            var member = new ValueError("v");
            var group = new ExceptionGroup("m", member);
            group.Message.Should().Be("m (1 sub-exception)");
            group.message.Should().Be("m");
            group.exceptions.Should().Equal(member);
            group.InnerException.Should().BeSameAs(member);
            // The member is matched like any member (CPython derives a new group around it: only a group that matches as a
            // whole is returned itself).
            group.subgroup(typeof(ValueError)).exceptions.Should().Equal(member);
            new BaseExceptionGroup("m", new KeyboardInterrupt()).exceptions.Should().ContainSingle();
            Action nest = () => _ = new ExceptionGroup("m", new KeyboardInterrupt());
            nest.Should().ThrowExactly<TypeError>().WithMessage("Cannot nest BaseExceptions in an ExceptionGroup");
            Action nullMember = () => _ = new BaseExceptionGroup("m", (Exception)null);
            nullMember.Should().ThrowExactly<ValueError>().WithMessage("Item 0 of second argument (exceptions) is not an exception");
            Action nullMessage = () => _ = new ExceptionGroup(null, member);
            nullMessage.Should().ThrowExactly<TypeError>().WithMessage("BaseExceptionGroup.__new__() argument 1 must be str, not None");
        }

        /// <summary>
        ///     The bases: a class directly under Python's <c>Exception</c> takes the .NET exception with the same meaning,
        ///     every child its Python parent; every class is an <see cref="INumSharpException"/>.
        /// </summary>
        /// <param name="type">The NumSharp class.</param>
        /// <param name="baseType">Its direct base.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence.</exception>
        [TestMethod]
        [DataRow(typeof(ArithmeticError), typeof(ArithmeticException))]
        [DataRow(typeof(OverflowError), typeof(ArithmeticError))]
        [DataRow(typeof(ZeroDivisionError), typeof(ArithmeticError))]
        [DataRow(typeof(FloatingPointError), typeof(ArithmeticError))]
        [DataRow(typeof(MemoryError), typeof(OutOfMemoryException))]
        [DataRow(typeof(ArrayMemoryError), typeof(MemoryError))]
        [DataRow(typeof(EOFError), typeof(EndOfStreamException))]
        [DataRow(typeof(BaseExceptionGroup), typeof(AggregateException))]
        [DataRow(typeof(ExceptionGroup), typeof(BaseExceptionGroup))]
        [DataRow(typeof(ReferenceError), typeof(NumSharpException))]
        [DataRow(typeof(StopIteration), typeof(NumSharpException))]
        [DataRow(typeof(StopAsyncIteration), typeof(NumSharpException))]
        [DataRow(typeof(KeyboardInterrupt), typeof(NumSharpException))]
        [DataRow(typeof(SystemExit), typeof(NumSharpException))]
        [DataRow(typeof(GeneratorExit), typeof(NumSharpException))]
        [DataRow(typeof(LapackError), typeof(NumSharpException))]
        [DataRow(typeof(ConverterError), typeof(NumSharpException))]
        [DataRow(typeof(ConverterLockError), typeof(ConverterError))]
        [DataRow(typeof(RecursionError), typeof(RuntimeError))]
        [DataRow(typeof(UnboundLocalError), typeof(NameError))]
        [DataRow(typeof(IndentationError), typeof(SyntaxError))]
        [DataRow(typeof(TabError), typeof(IndentationError))]
        [DataRow(typeof(UnicodeTranslateError), typeof(UnicodeError))]
        [DataRow(typeof(BlockingIOError), typeof(OSError))]
        [DataRow(typeof(ChildProcessError), typeof(OSError))]
        [DataRow(typeof(ConnectionError), typeof(OSError))]
        [DataRow(typeof(BrokenPipeError), typeof(ConnectionError))]
        [DataRow(typeof(ConnectionAbortedError), typeof(ConnectionError))]
        [DataRow(typeof(ConnectionRefusedError), typeof(ConnectionError))]
        [DataRow(typeof(ConnectionResetError), typeof(ConnectionError))]
        [DataRow(typeof(FileExistsError), typeof(OSError))]
        [DataRow(typeof(FileNotFoundError), typeof(OSError))]
        [DataRow(typeof(InterruptedError), typeof(OSError))]
        [DataRow(typeof(IsADirectoryError), typeof(OSError))]
        [DataRow(typeof(NotADirectoryError), typeof(OSError))]
        [DataRow(typeof(PermissionError), typeof(OSError))]
        [DataRow(typeof(ProcessLookupError), typeof(OSError))]
        [DataRow(typeof(TimeoutError), typeof(OSError))]
        [DataRow(typeof(UFuncTypeError), typeof(TypeError))]
        [DataRow(typeof(UFuncNoLoopError), typeof(UFuncTypeError))]
        [DataRow(typeof(UFuncBinaryResolutionError), typeof(UFuncNoLoopError))]
        [DataRow(typeof(UFuncCastingError), typeof(UFuncTypeError))]
        [DataRow(typeof(UFuncInputCastingError), typeof(UFuncCastingError))]
        [DataRow(typeof(UFuncOutputCastingError), typeof(UFuncCastingError))]
        public void Bases_FollowTheDotNetTwinThenPythonsMro(Type type, Type baseType)
        {
            type.BaseType.Should().Be(baseType);
            typeof(INumSharpException).IsAssignableFrom(type).Should().BeTrue();
        }

        /// <summary>
        ///     The catches the bases buy: <c>catch (ArithmeticException)</c> sees every arithmetic error,
        ///     <c>catch (IOException)</c> every OSError and EOFError, and EOFError is no OSError (as in Python); the
        ///     Python errors are NOT the .NET siblings a pre-switch <c>catch</c> named.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence.</exception>
        [TestMethod]
        public void Catches_SeeWhatPythonsExceptSees()
        {
            new OverflowError().Should().BeAssignableTo<ArithmeticException>().And.NotBeAssignableTo<OverflowException>();
            new ZeroDivisionError().Should().BeAssignableTo<ArithmeticException>().And.NotBeAssignableTo<DivideByZeroException>();
            new MemoryError().Should().BeAssignableTo<OutOfMemoryException>();
            new EOFError().Should().BeAssignableTo<IOException>().And.NotBeAssignableTo<OSError>();
            new FileNotFoundError().Should().BeAssignableTo<IOException>().And.NotBeAssignableTo<FileNotFoundException>();
            new TimeoutError().Should().NotBeAssignableTo<TimeoutException>();
            new UFuncOutputCastingError("add", 1, "same_kind", DType.Double, DType.Int64, 2).Should().BeAssignableTo<TypeError>();
        }

        /// <summary>
        ///     <c>OSError</c>'s CPython forms on the running platform (probed on Windows and Linux): the
        ///     <c>[Errno N] strerror: 'file'</c> text, the Windows-only <c>[WinError N]</c> forms of a four- or five-argument
        ///     call, the filename as Python's <c>repr()</c>, and <c>filename2</c> kept only beside a filename.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void OSError_ReadsCPythonsText()
        {
            bool windows = OperatingSystem.IsWindows();
            var two = new OSError(2, "No such file or directory");
            (two.Message, two.errno, two.strerror, two.filename, two.winerror)
                .Should().Be(("[Errno 2] No such file or directory", (int?)2, "No such file or directory", (string)null, (int?)null));
            new OSError(2, "No such file or directory", "x.npy").Message.Should().Be("[Errno 2] No such file or directory: 'x.npy'");
            new OSError(2, "msg", "a", null).Message.Should().Be(windows ? "[WinError None] msg: 'a'" : "[Errno 2] msg: 'a'");
            var both = new OSError(2, "msg", "a", null, "b");
            both.Message.Should().Be(windows ? "[WinError None] msg: 'a' -> 'b'" : "[Errno 2] msg: 'a' -> 'b'");
            both.filename2.Should().Be("b");
            var noFile = new OSError(2, "x", null, null, "g");
            noFile.filename2.Should().BeNull("CPython keeps filename2 only beside a filename");
            noFile.Message.Should().Be(windows ? "[WinError None] x" : "[Errno 2] x");
            new OSError(2, null).Message.Should().Be("[Errno 2] None");
            new OSError(null, null, "f").Message.Should().Be("[Errno None] None: 'f'");
            new OSError(2, "x", "it's").Message.Should().Be("[Errno 2] x: \"it's\"");
            new OSError(2, "x", "a'b\"c").Message.Should().Be("[Errno 2] x: 'a\\'b\"c'");
            new OSError(2, "x", "C:\\a\\b").Message.Should().Be("[Errno 2] x: 'C:\\\\a\\\\b'");
            new OSError(2, "x", "").Message.Should().Be("[Errno 2] x: ''");
            new OSError("m").Message.Should().Be("m");
            new OSError("m").errno.Should().BeNull();

            var denied = new OSError(null, "Access is denied", "f", 5);
            if (windows)
            {
                (denied.errno, denied.winerror, denied.Message).Should().Be(((int?)13, (int?)5, "[WinError 5] Access is denied: 'f'"));
                new OSError(null, "Access is denied", null, 5).Message.Should().Be("[WinError 5] Access is denied");
            }
            else
            {
                (denied.errno, denied.winerror, denied.Message).Should().Be(((int?)null, (int?)null, "[Errno None] Access is denied: 'f'"));
                new OSError(null, "Access is denied", null, 5).Message.Should().Be("[Errno None] Access is denied");
            }
        }

        /// <summary>
        ///     <see cref="OSError.Create(int?, string)"/> is Python's <c>OSError(errno, ...)</c> call: the subclass the errno
        ///     names on this operating system, while a subclass's own constructor keeps its class whatever the errno.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void OSErrorCreate_PicksTheErrnoSubclass()
        {
            OSError.Create(2, "x").Should().BeOfType<FileNotFoundError>();
            OSError.Create(13, "x").Should().BeOfType<PermissionError>();
            OSError.Create(1, "x").Should().BeOfType<PermissionError>();
            OSError.Create(17, "x").Should().BeOfType<FileExistsError>();
            OSError.Create(20, "x").Should().BeOfType<NotADirectoryError>();
            OSError.Create(21, "x").Should().BeOfType<IsADirectoryError>();
            OSError.Create(4, "x").Should().BeOfType<InterruptedError>();
            OSError.Create(3, "x").Should().BeOfType<ProcessLookupError>();
            OSError.Create(10, "x").Should().BeOfType<ChildProcessError>();
            OSError.Create(32, "x").Should().BeOfType<BrokenPipeError>();
            OSError.Create(-1, "x").Should().BeOfType<OSError>();
            OSError.Create(null, "x").Should().BeOfType<OSError>();
            if (OperatingSystem.IsWindows())
            {
                OSError.Create(10054, "x").Should().BeOfType<ConnectionResetError>();
                OSError.Create(10060, "x").Should().BeOfType<TimeoutError>();
                OSError.Create(10035, "x").Should().BeOfType<BlockingIOError>();
                OSError.Create(null, "x", null, 5).Should().BeOfType<PermissionError>();
                // An HRESULT carrying ERROR_FILE_NOT_FOUND, as .NET's IOException.HResult holds it.
                var hresult = OSError.Create(null, "x", null, unchecked((int)0x80070002));
                hresult.Should().BeOfType<FileNotFoundError>();
                hresult.errno.Should().Be(2);
                hresult.winerror.Should().Be(unchecked((int)0x80070002));
            }
            else if (OperatingSystem.IsLinux())
            {
                OSError.Create(104, "x").Should().BeOfType<ConnectionResetError>();
                OSError.Create(110, "x").Should().BeOfType<TimeoutError>();
                OSError.Create(11, "x").Should().BeOfType<BlockingIOError>();
                OSError.Create(null, "x", null, 5).Should().BeOfType<OSError>("winerror is ignored off Windows");
            }
            new FileNotFoundError(13, "x").Should().BeOfType<FileNotFoundError>();
        }

        /// <summary>
        ///     CPython's <c>winerror_to_errno</c> over a sample of its table (probed: every code 0..19999 of CPython 3.12.12
        ///     on Windows).
        /// </summary>
        /// <param name="winerror">The Windows error code.</param>
        /// <param name="errno">The errno CPython maps it to.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        [DataRow(2, 2)]
        [DataRow(3, 2)]
        [DataRow(206, 2)]
        [DataRow(10, 7)]
        [DataRow(193, 8)]
        [DataRow(6, 9)]
        [DataRow(128, 10)]
        [DataRow(215, 11)]
        [DataRow(1816, 12)]
        [DataRow(5, 13)]
        [DataRow(32, 13)]
        [DataRow(35, 13)]
        [DataRow(183, 17)]
        [DataRow(17, 18)]
        [DataRow(267, 20)]
        [DataRow(4, 24)]
        [DataRow(112, 28)]
        [DataRow(232, 32)]
        [DataRow(145, 41)]
        [DataRow(1113, 42)]
        [DataRow(10004, 4)]
        [DataRow(10022, 22)]
        [DataRow(10024, 24)]
        [DataRow(10035, 10035)]
        [DataRow(11999, 11999)]
        [DataRow(87, 22)]
        [DataRow(0, 22)]
        [DataRow(-1, 22)]
        [DataRow(99999, 22)]
        public void WinErrorToErrno_MatchesCPythonsTable(int winerror, int errno)
            => OSErrorArgs.WinErrorToErrno(winerror).Should().Be(errno);

        /// <summary><c>BlockingIOError(errno, strerror, n)</c>: a number in the third place is
        ///     <c>characters_written</c>, not a file name.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void BlockingIOError_TakesCharactersWritten()
        {
            var e = new BlockingIOError(11, "x", 3);
            (e.Message, e.characters_written, e.filename).Should().Be(("[Errno 11] x", (long?)3, (string)null));
            new BlockingIOError(11, "x", "f").characters_written.Should().BeNull();
            new OSError(2, "x").characters_written.Should().BeNull();
        }

        /// <summary>
        ///     <c>SyntaxError(msg, (filename, lineno, offset, text[, end_lineno, end_offset]))</c>: the location attributes
        ///     and CPython's text, whose file is the base name at the running platform's separator.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void SyntaxError_CarriesTheLocation()
        {
            bool windows = OperatingSystem.IsWindows();
            var full = new SyntaxError("invalid syntax", "a/b/c.py", 3, 5, "x = (\n", 3, 6);
            full.Message.Should().Be(windows ? "invalid syntax (a/b/c.py, line 3)" : "invalid syntax (c.py, line 3)");
            (full.msg, full.filename, full.lineno, full.offset, full.text, full.end_lineno, full.end_offset)
                .Should().Be(("invalid syntax", "a/b/c.py", (int?)3, (int?)5, "x = (\n", (int?)3, (int?)6));
            new SyntaxError("invalid syntax", "C:\\a\\b\\c.py", 3, 5, "x\n").Message
                .Should().Be(windows ? "invalid syntax (c.py, line 3)" : "invalid syntax (C:\\a\\b\\c.py, line 3)");
            new SyntaxError("m", "f.py", null, null, null).Message.Should().Be("m (f.py)");
            new SyntaxError("m", null, 7, null, null).Message.Should().Be("m (line 7)");
            new SyntaxError("invalid syntax", "<unknown>", 1, 3, "1 + 2").Message.Should().Be("invalid syntax (<unknown>, line 1)");
            new SyntaxError(null, "f.py", 2, 1, "x").Message.Should().Be("None (f.py, line 2)");
            new SyntaxError("m", "", 3, 2, "t").Message.Should().Be("m (, line 3)");
            new IndentationError("unexpected indent", "f.py", 2, 4, "    x\n").Message.Should().Be("unexpected indent (f.py, line 2)");
            new TabError("m").msg.Should().Be("m");
            new SyntaxError().msg.Should().BeNull();
            Action noEnd = () => _ = new SyntaxError("m", "f", 1, 2, "t", 3);
            noEnd.Should().ThrowExactly<TypeError>().WithMessage("end_offset must be provided when end_lineno is provided");
        }

        /// <summary><c>StopIteration.value</c> and <c>SystemExit.code</c>: the argument, and Python's <c>str()</c> of it as
        ///     the text.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void StopIterationAndSystemExit_KeepTheirValue()
        {
            var bare = new StopIteration();
            (bare.Message, bare.value).Should().Be(("", (object)null));
            new StopIteration(5).Message.Should().Be("5");
            new StopIteration(5).value.Should().Be(5);
            new StopIteration("x").value.Should().Be("x");
            new StopIteration(1.5).Message.Should().Be("1.5");
            new StopIteration(1e20).Message.Should().Be("1e+20");
            new StopIteration(true).Message.Should().Be("True");
            new StopIteration((object)null).Message.Should().Be("None");
            new StopIteration((string)null).Message.Should().Be("None");

            var exit = new SystemExit();
            (exit.Message, exit.code).Should().Be(("", (object)null));
            new SystemExit(3).Message.Should().Be("3");
            new SystemExit(3).code.Should().Be(3);
            new SystemExit("bye").code.Should().Be("bye");
            new SystemExit((object)null).Message.Should().Be("None");
        }

        /// <summary><c>NameError.name</c> and <c>AttributeError.name</c> / <c>.obj</c> (keyword-only in Python, positional
        ///     here).</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void NameAndAttributeError_CarryTheName()
        {
            var name = new NameError("name 'x' is not defined", "x");
            (name.Message, name.name).Should().Be(("name 'x' is not defined", "x"));
            new NameError("m").name.Should().BeNull();
            new UnboundLocalError("cannot access local variable 'x'", "x").name.Should().Be("x");

            var attr = new AttributeError("no attr", "a", 5);
            (attr.Message, attr.name, attr.obj).Should().Be(("no attr", "a", (object)5));
            new AttributeError("m").obj.Should().BeNull();

            // A raise site that stands for CPython's attribute lookup marks the error as CPython 3.12 does.
            Action lookup = () => _ = DType.Int64.na_object;
            var raised = lookup.Should().ThrowExactly<AttributeError>().Which;
            raised.Message.Should().Be("'numpy.dtypes.Int64DType' object has no attribute 'na_object'");
            (raised.name, raised.obj).Should().Be(("na_object", (object)DType.Int64));
        }

        /// <summary>CPython's <c>UnicodeTranslateError.__str__</c>: one character escaped, a range by its bounds, the
        ///     encoding always None.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void UnicodeTranslateError_ReadsCPythonsText()
        {
            new UnicodeTranslateError("a\u00e9b", 1, 2, "reason").Message.Should().Be("can't translate character '\\xe9' in position 1: reason");
            new UnicodeTranslateError("abc", 0, 2, "reason").Message.Should().Be("can't translate characters in position 0-1: reason");
            new UnicodeTranslateError("\U0001F600", 0, 1, "r").Message.Should().Be("can't translate character '\\U0001f600' in position 0: r");
            new UnicodeTranslateError("\u4e2d", 0, 1, "r").Message.Should().Be("can't translate character '\\u4e2d' in position 0: r");
            new UnicodeTranslateError("ab", 5, 6, "r").Message.Should().Be("can't translate characters in position 5-5: r");
            new UnicodeTranslateError("abc", 2, 1, "r").Message.Should().Be("can't translate characters in position 2-0: r");
            var e = new UnicodeTranslateError("ab", 0, 1, "r");
            (e.encoding, e.@object, e.start, e.end, e.reason).Should().Be(((string)null, "ab", 0L, 1L, "r"));
            Action none = () => _ = new UnicodeTranslateError(null, 0, 1, "r");
            none.Should().ThrowExactly<TypeError>().WithMessage("argument 1 must be str, not None");
        }

        /// <summary>
        ///     <c>ExceptionGroup</c>: the <c>(n sub-exceptions)</c> text, CPython's constructor checks, and
        ///     <c>subgroup</c> / <c>split</c> / <c>derive</c> over a nested group (probed CPython 3.12.12).
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void ExceptionGroup_FollowsCPython()
        {
            new ExceptionGroup("m", new Exception[] { new ValueError() }).Message.Should().Be("m (1 sub-exception)");
            new ExceptionGroup("m", new Exception[] { new ValueError(), new TypeError() }).Message.Should().Be("m (2 sub-exceptions)");
            new ExceptionGroup("", new Exception[] { new ValueError() }).Message.Should().Be(" (1 sub-exception)");

            BaseExceptionGroup.Create("m", new Exception[] { new ValueError() }).Should().BeOfType<ExceptionGroup>();
            BaseExceptionGroup.Create("m", new Exception[] { new KeyboardInterrupt() }).Should().BeOfType<BaseExceptionGroup>();
            Action nest = () => _ = new ExceptionGroup("m", new Exception[] { new KeyboardInterrupt() });
            nest.Should().ThrowExactly<TypeError>().WithMessage("Cannot nest BaseExceptions in an ExceptionGroup");
            Action nestGroup = () => _ = new ExceptionGroup("m", new Exception[] { new BaseExceptionGroup("i", new Exception[] { new SystemExit() }) });
            nestGroup.Should().ThrowExactly<TypeError>().WithMessage("Cannot nest BaseExceptions in an ExceptionGroup");
            Action empty = () => _ = new ExceptionGroup("m", Array.Empty<Exception>());
            empty.Should().ThrowExactly<ValueError>().WithMessage("second argument (exceptions) must be a non-empty sequence");
            Action nullItem = () => _ = new ExceptionGroup("m", new Exception[] { new ValueError(), null });
            nullItem.Should().ThrowExactly<ValueError>().WithMessage("Item 1 of second argument (exceptions) is not an exception");
            Action noMessage = () => _ = new ExceptionGroup(null, new Exception[] { new ValueError() });
            noMessage.Should().ThrowExactly<TypeError>().WithMessage("BaseExceptionGroup.__new__() argument 1 must be str, not None");
            // The cast picks the sequence constructor: a bare null would also fit the (message, innerException) one.
            Action noSequence = () => _ = new ExceptionGroup("m", (IEnumerable<Exception>)null);
            noSequence.Should().ThrowExactly<TypeError>().WithMessage("second argument (exceptions) must be a sequence");

            var ve = new ValueError("v");
            var te = new TypeError("t");
            var ke = new KeyError("k");
            var inner = new ExceptionGroup("inner", new Exception[] { te, ke });
            var eg = new ExceptionGroup("outer", new Exception[] { ve, inner });
            eg.message.Should().Be("outer");
            eg.exceptions.Should().Equal(ve, inner);
            eg.Message.Should().Be("outer (2 sub-exceptions)");

            eg.subgroup(typeof(Exception)).Should().BeSameAs(eg);
            eg.subgroup(typeof(OSError)).Should().BeNull();
            var onlyT = eg.subgroup(typeof(TypeError));
            Describe(onlyT).Should().Be("ExceptionGroup('outer', [ExceptionGroup('inner', [t])])");
            Describe(eg.subgroup(typeof(TypeError), typeof(KeyError))).Should().Be("ExceptionGroup('outer', [ExceptionGroup('inner', [t, k])])");
            Describe(eg.subgroup(e => e is ValueError)).Should().Be("ExceptionGroup('outer', [v])");
            Describe(eg.subgroup(e => e is ExceptionGroup g && g.message == "inner"))
                .Should().Be("ExceptionGroup('outer', [ExceptionGroup('inner', [t, k])])");

            var (match, rest) = eg.split(typeof(ValueError));
            Describe(match).Should().Be("ExceptionGroup('outer', [v])");
            Describe(rest).Should().Be("ExceptionGroup('outer', [ExceptionGroup('inner', [t, k])])");
            var (none, all) = eg.split(typeof(OSError));
            none.Should().BeNull();
            Describe(all).Should().Be("ExceptionGroup('outer', [v, ExceptionGroup('inner', [t, k])])");
            var (whole, nothing) = eg.split(typeof(Exception));
            whole.Should().BeSameAs(eg);
            nothing.Should().BeNull();
            Describe(eg.derive(new Exception[] { new ValueError("z") })).Should().Be("ExceptionGroup('outer', [z])");
            new BaseExceptionGroup("m", new Exception[] { new KeyboardInterrupt() }).derive(new Exception[] { new ValueError() })
                .Should().BeOfType<ExceptionGroup>();

            Action badMatcher = () => eg.subgroup(typeof(int));
            badMatcher.Should().ThrowExactly<TypeError>().WithMessage("expected a function, exception type or tuple of exception types");
        }

        /// <summary>A user subclass: its nesting refusal names it, and the default <c>derive</c> returns a plain group (both
        ///     as in CPython).</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from CPython.</exception>
        [TestMethod]
        public void ExceptionGroupSubclass_FollowsCPython()
        {
            Action nest = () => _ = new MyGroup("m", new Exception[] { new KeyboardInterrupt() });
            nest.Should().ThrowExactly<TypeError>().WithMessage("Cannot nest BaseExceptions in 'MyGroup'");
            var group = new MyGroup("m", new Exception[] { new ValueError(), new TypeError() });
            group.derive(new Exception[] { new TypeError() }).Should().BeOfType<ExceptionGroup>();
            var (match, rest) = group.split(typeof(ValueError));
            match.Should().BeOfType<ExceptionGroup>();
            rest.Should().BeOfType<ExceptionGroup>();
        }

        /// <summary>
        ///     NumPy's <c>UFuncTypeError</c> family from its structured arguments (probed NumPy 2.4.2): the one-or-tuple
        ///     unpacking of each side of a no-loop signature, the index printed only beside other operands, the attributes.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void UFuncErrors_ReadNumPysText()
        {
            DType u1 = "<U1", s1 = "S1";
            new UFuncTypeError("add").Message.Should().Be("<ufunc 'add'>");
            new UFuncNoLoopError("modf", 1, u1, null, null).Message
                .Should().Be("ufunc 'modf' did not contain a loop with signature matching types dtype('<U1') -> (None, None)");
            var add = new UFuncNoLoopError("add", 2, u1, DType.Int64, null);
            add.Message.Should().Be("ufunc 'add' did not contain a loop with signature matching types (dtype('<U1'), dtype('int64')) -> None");
            add.dtypes.Should().Equal(u1, DType.Int64, null);
            add.ufunc.Should().Be("add");
            new UFuncNoLoopError("negative", 1, u1, u1).Message
                .Should().Be("ufunc 'negative' did not contain a loop with signature matching types dtype('<U1') -> dtype('<U1')");
            new UFuncNoLoopError("divmod", 2, u1, DType.Int64, null, null).Message
                .Should().Be("ufunc 'divmod' did not contain a loop with signature matching types (dtype('<U1'), dtype('int64')) -> (None, None)");
            new UFuncNoLoopError("str_len", 1, DType.Int64.Meta, null).Message
                .Should().Be("ufunc 'str_len' did not contain a loop with signature matching types <class 'numpy.dtypes.Int64DType'> -> None");

            new UFuncBinaryResolutionError("add", s1, u1).Message
                .Should().Be("ufunc 'add' cannot use operands with types dtype('S1') and dtype('<U1')");
            Action three = () => _ = new UFuncBinaryResolutionError("add", s1, u1, null);
            three.Should().ThrowExactly<AssertionError>();

            new UFuncCastingError("add", "safe", DType.Double, DType.Int64).Message
                .Should().Be("(<ufunc 'add'>, 'safe', dtype('float64'), dtype('int64'))");
            var in1 = new UFuncInputCastingError("add", 2, "safe", DType.Double, DType.Int64, 1);
            in1.Message.Should().Be("Cannot cast ufunc 'add' input 1 from dtype('float64') to dtype('int64') with casting rule 'safe'");
            (in1.casting, in1.from_, in1.to, in1.in_i).Should().Be(("safe", (object)DType.Double, (object)DType.Int64, 1));
            new UFuncInputCastingError("sqrt", 1, "safe", DType.Double, DType.Int64, 0).Message
                .Should().Be("Cannot cast ufunc 'sqrt' input from dtype('float64') to dtype('int64') with casting rule 'safe'");
            new UFuncOutputCastingError("add", 1, "same_kind", DType.Double, DType.Int64, 2).Message
                .Should().Be("Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
            var modf = new UFuncOutputCastingError("modf", 2, "same_kind", DType.Double, DType.Int64, 2);
            modf.Message.Should().Be("Cannot cast ufunc 'modf' output 2 from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
            modf.out_i.Should().Be(2);
        }

        /// <summary>
        ///     NumSharp's own ufunc refusals raise the family with NumPy's text and attributes: a no-loop add of text and
        ///     numbers, a bytes/str add, a unary dtype= input cast, an out= output cast and modf's numbered output.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void UFuncRaiseSites_RaiseTheFamily()
        {
            using var text = np.array(new[] { "a" });
            using var ints = np.array(new long[] { 1 });
            Action noLoop = () => np.add(text, ints);
            var e = noLoop.Should().ThrowExactly<UFuncNoLoopError>().Which;
            e.Message.Should().Be("ufunc 'add' did not contain a loop with signature matching types (dtype('<U1'), dtype('int64')) -> None");
            e.dtypes.Count.Should().Be(3);
            e.dtypes[2].Should().BeNull();

            using var bytes = np.array(new[] { "a" }, "S");
            Action binres = () => np.add(bytes, text);
            binres.Should().ThrowExactly<UFuncBinaryResolutionError>()
                .Which.Message.Should().Be("ufunc 'add' cannot use operands with types dtype('S1') and dtype('<U1')");

            using var floats = np.array(new[] { 1.5 });
            Action input = () => np.negative(floats, dtype: np.int32);
            var ic = input.Should().ThrowExactly<UFuncInputCastingError>().Which;
            ic.Message.Should().Be("Cannot cast ufunc 'negative' input from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
            (ic.ufunc, ic.in_i).Should().Be(("negative", 0));

            using var intOut = np.empty(new Shape(1), np.int64);
            Action output = () => np.add(floats, floats, @out: intOut);
            var oc = output.Should().ThrowExactly<UFuncOutputCastingError>().Which;
            oc.Message.Should().Be("Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
            (oc.casting, oc.out_i).Should().Be(("same_kind", 2));

            using var fracOut = np.empty(new Shape(1), np.int64);
            Action modf = () => np.modf(floats, (fracOut, null));
            modf.Should().ThrowExactly<UFuncOutputCastingError>()
                .Which.Message.Should().Be("Cannot cast ufunc 'modf' output 1 from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
        }

        /// <summary>
        ///     NumPy's <c>_ArrayMemoryError</c>: the size in binary units (<c>_size_to_string</c>), the shape as a Python
        ///     tuple and the dtype's <c>str()</c> (probed NumPy 2.4.2).
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void ArrayMemoryError_ReadsNumPysText()
        {
            new ArrayMemoryError(new long[] { 3, 4 }, DType.Double).Message
                .Should().Be("Unable to allocate 96 bytes for an array with shape (3, 4) and data type float64");
            new ArrayMemoryError(new long[] { 1000000000 }, "<U3").Message
                .Should().Be("Unable to allocate 11.2 GiB for an array with shape (1000000000,) and data type <U3");
            new ArrayMemoryError(Array.Empty<long>(), DType.SByte).Message
                .Should().Be("Unable to allocate 1 bytes for an array with shape () and data type int8");
            new ArrayMemoryError(new long[] { 1000000000000, 1000000 }, DType.Double).Message
                .Should().Be("Unable to allocate 6.94 EiB for an array with shape (1000000000000, 1000000) and data type float64");
            new ArrayMemoryError(new long[] { 2 }, "S5").Message
                .Should().Be("Unable to allocate 10 bytes for an array with shape (2,) and data type |S5");
            new ArrayMemoryError(new long[] { 0, 3 }, DType.Int32).Message
                .Should().Be("Unable to allocate 0 bytes for an array with shape (0, 3) and data type int32");

            var dims = new long[] { 2, 3 };
            var error = new ArrayMemoryError(dims, DType.Int32);
            dims[0] = 99;
            error.shape.Should().Equal(2, 3);
            error.shape[0] = 7;
            error.shape.Should().Equal(2, 3);
            error.dtype.Should().Be(DType.Int32);
        }

        /// <summary><c>_ArrayMemoryError._size_to_string</c> at its unit edges (probed NumPy 2.4.2).</summary>
        /// <param name="bytes">The byte count, as a decimal string (it may exceed a <see cref="long"/>).</param>
        /// <param name="text">NumPy's text.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        [DataRow("0", "0 bytes")]
        [DataRow("1023", "1023 bytes")]
        [DataRow("1024", "1.00 KiB")]
        [DataRow("10240", "10.0 KiB")]
        [DataRow("1024000", "1000. KiB")]
        [DataRow("1048452", "1.00 MiB")]
        [DataRow("1048575", "1.00 MiB")]
        [DataRow("1022976", "999. KiB")]
        [DataRow("1023487", "999. KiB")]
        [DataRow("1023488", "1000. KiB")]
        [DataRow("123456789", "118. MiB")]
        [DataRow("7999376588", "7.45 GiB")]
        [DataRow("1152921504606846976", "1.00 EiB")]
        [DataRow("9223372036854775808", "8.00 EiB")]
        [DataRow("18446744073709551616", "16.0 EiB")]
        [DataRow("1180591620717411303424", "1024. EiB")]
        [DataRow("1208925819614629174706176", "1048576. EiB")]
        public void ArrayMemoryErrorSizeToString_MatchesNumPy(string bytes, string text)
            => ArrayMemoryError.SizeToString(BigInteger.Parse(bytes, System.Globalization.CultureInfo.InvariantCulture)).Should().Be(text);

        /// <summary>
        ///     NumSharp's raise sites that mirror NumPy raise the Python classes: an integer literal out of the dtype's range
        ///     (<see cref="OverflowError"/>), weights that sum to zero (<see cref="ZeroDivisionError"/>), an entropy pool no
        ///     array can hold (<see cref="ArrayMemoryError"/>), the random generators' range check.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void ArithmeticAndMemorySites_RaiseThePythonClasses()
        {
            Action overflow = () => np.full(new Shape(3), 1000, np.int8);
            overflow.Should().ThrowExactly<OverflowError>().WithMessage("Python integer 1000 out of bounds for int8");

            using var a = np.array(new[] { 1.0, 2.0 });
            using var w = np.array(new[] { 1.0, -1.0 });
            Action zero = () => np.average(a, weights: w);
            zero.Should().ThrowExactly<ZeroDivisionError>().WithMessage("Weights sum to zero, can't be normalized");

            Action pool = () => _ = new SeedSequence(0L, null, (long)int.MaxValue + 10);
            pool.Should().ThrowExactly<ArrayMemoryError>()
                .Which.Message.Should().Be("Unable to allocate 8.00 GiB for an array with shape (2147483657,) and data type uint32");

            // An array within intp but past any machine's memory: NumPy 2.4.2's np.empty(2**60, dtype=np.uint8) text.
            Action huge = () => np.empty(new Shape(1L << 60), np.uint8);
            var allocation = huge.Should().ThrowExactly<ArrayMemoryError>().Which;
            allocation.Message.Should().Be("Unable to allocate 1.00 EiB for an array with shape (1152921504606846976,) and data type uint8");
            allocation.InnerException.Should().BeOfType<OutOfMemoryException>("the allocator's own failure is the cause");
            Action zeros = () => np.zeros(new Shape(1000000000000L, 1000000L), np.float64);
            zeros.Should().ThrowExactly<ArrayMemoryError>().Which.Message
                .Should().Be("Unable to allocate 6.94 EiB for an array with shape (1000000000000, 1000000) and data type float64");
            Action tooBig = () => np.zeros(new Shape(1L << 60), np.float64);
            tooBig.Should().ThrowExactly<ValueError>("a byte count past intp is NumPy's 'array is too big', before any allocation");

            Action range = () => np.random.uniform(-1e308, 1e308);
            range.Should().ThrowExactly<OverflowError>().WithMessage("Range exceeds valid bounds");
        }

        /// <summary>
        ///     NumSharp's file functions fail as CPython's <c>open()</c> does: a missing file is
        ///     <see cref="FileNotFoundError"/> with <c>[Errno 2]</c> and the name as given, a directory is
        ///     <see cref="PermissionError"/> on Windows and <see cref="IsADirectoryError"/> elsewhere, an empty file
        ///     <see cref="EOFError"/>, and an archive's missing member <see cref="KeyError"/> (the indexer) or
        ///     <see cref="AttributeError"/> (dot access).
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void FileSites_RaiseCPythonsOpenErrors()
        {
            string dir = Path.Combine(Path.GetTempPath(), "numsharp-pyexc-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string missing = Path.Combine(dir, "missing.npy");
                Action load = () => np.load(missing);
                var notFound = load.Should().ThrowExactly<FileNotFoundError>().Which;
                notFound.Message.Should().Be("[Errno 2] No such file or directory: " + Repr(missing));
                (notFound.errno, notFound.strerror, notFound.filename).Should().Be(((int?)2, "No such file or directory", missing));
                notFound.InnerException.Should().BeAssignableTo<IOException>();

                Action save = () => np.save(Path.Combine(dir, "nodir", "x.npy"), np.arange(2));
                save.Should().ThrowExactly<FileNotFoundError>();

                Action directory = () => np.load(dir);
                if (OperatingSystem.IsWindows())
                    directory.Should().ThrowExactly<PermissionError>().Which.Message.Should().Be("[Errno 13] Permission denied: " + Repr(dir));
                else
                    directory.Should().ThrowExactly<IsADirectoryError>().Which.Message.Should().Be("[Errno 21] Is a directory: " + Repr(dir));

                string empty = Path.Combine(dir, "empty.npy");
                File.WriteAllBytes(empty, Array.Empty<byte>());
                Action eof = () => np.load(empty);
                eof.Should().ThrowExactly<EOFError>().WithMessage("No data left in file");

                string archive = Path.Combine(dir, "a.npz");
                np.savez(archive, new Dictionary<string, NDArray> { ["x"] = np.arange(3) });
                using var npz = np.load_npz(archive);
                Action key = () => _ = npz["y"];
                key.Should().ThrowExactly<KeyError>().Which.Message.Should().Be("'y is not a file in the archive'");
                Action attribute = () => _ = npz.f.y;
                var attr = attribute.Should().ThrowExactly<AttributeError>().Which;
                (attr.Message, attr.name).Should().Be(("y", "y"));

                Action loadtxt = () => np.loadtxt(Path.Combine(dir, "nothing.txt"));
                var dataSource = loadtxt.Should().ThrowExactly<FileNotFoundError>().Which;
                dataSource.Message.Should().Be(Path.Combine(dir, "nothing.txt") + " not found.");
                dataSource.errno.Should().BeNull("NumPy's DataSource raises FileNotFoundError(message) without an errno");
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }

        /// <summary>Python's <c>repr()</c> of a path without quotes in it: single-quoted, backslashes doubled.</summary>
        /// <param name="path">The path.</param>
        /// <returns>The repr.</returns>
        private static string Repr(string path) => "'" + path.Replace("\\", "\\\\") + "'";

        /// <summary>A group as <c>ExceptionGroup('msg', [members])</c>, each leaf by its message, for structural checks.</summary>
        /// <param name="e">The group or leaf, or null.</param>
        /// <returns>The description.</returns>
        private static string Describe(Exception e)
            => e switch
            {
                null => "None",
                BaseExceptionGroup g => $"{g.GetType().Name}('{g.message}', [{string.Join(", ", g.exceptions.Select(Describe))}])",
                _ => e.Message,
            };

        /// <summary>A user subclass of <see cref="ExceptionGroup"/>, as Python's tests define one.</summary>
        private sealed class MyGroup : ExceptionGroup
        {
            /// <summary>Python's <c>MyGroup(message, exceptions)</c>.</summary>
            /// <param name="message">The message.</param>
            /// <param name="exceptions">The members.</param>
            public MyGroup(string message, IEnumerable<Exception> exceptions) : base(message, exceptions) { }
        }
    }
}
