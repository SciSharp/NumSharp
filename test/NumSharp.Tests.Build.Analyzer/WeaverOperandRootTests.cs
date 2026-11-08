using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Mono.Cecil;
using Mono.Cecil.Cil;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The weaver's OPERAND ROOTS (<c>ScopeWeaver.InjectParameterAnchors</c>), pinned on Release and Debug IL: which
    ///     parameters a woven method roots (every one that can reach an NDArray, and <c>this</c> of a class), that the
    ///     shadows are taken before the body and anchored before every return, which methods get them (an
    ///     <c>[NDScoped]</c> weave, a hand-written <c>NDScope.Open()</c>, an <c>[NDAnchored]</c> entry point — never an
    ///     unmarked unscoped method), the NDW019 rejections, idempotence, and that the woven code still runs.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why roots.</b> A scope roots only the arrays it tracks (those built while it is open); a caller's operand is
    ///     never tracked, and optimized code stops reporting a parameter to the GC at its last use, so a kernel that reads
    ///     an operand's raw address after that point could run on a buffer the GC already reclaimed when the caller passed
    ///     a temporary. The lifetime oracle (<c>test/NumSharp.Tests.Oracle/Fuzz/Lifetime</c>) proves the class at run time
    ///     under a stress thread; these tests pin the IL the fix depends on, deterministically.
    ///     </para>
    ///     <para>
    ///     <b>How a root is recognised.</b> A shadow is a local of type <c>object</c> that is passed to
    ///     <c>NDScope.Anchor</c>; its single store is preceded by the load of the rooted argument (<c>ldarg</c>, possibly
    ///     followed by a <c>box</c>). <see cref="Roots"/> recovers the rooted parameters' names from that pattern
    ///     (<c>"this"</c> for argument 0 of an instance method).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class WeaverOperandRootTests
    {
        private const string Prelude =
            "using System;\nusing System.Collections.Generic;\nusing System.Threading.Tasks;\nusing NumSharp;\nusing NumSharp.Generic;\n";

        // ---------------------------------------------------------------- which parameters are rooted

        /// <summary>
        ///     An <c>[NDScoped]</c> instance method roots <c>this</c> and every NDArray-reaching parameter — an NDArray, an
        ///     <c>object</c>, an <c>NDArray[]</c>, a delegate, a tuple of arrays (boxed) — and leaves numbers, strings, a
        ///     <c>long[]</c> shape, a span and a by-ref parameter alone (<c>[NDScoped]</c> itself refuses a by-ref that
        ///     carries an NDArray, NDW002).
        /// </summary>
        /// <exception cref="AssertFailedException">A root is missing, or a parameter that cannot reach an array is rooted.</exception>
        [TestMethod]
        public void Scoped_RootsEveryNDArrayReachingParameter_AndThis()
        {
            foreach (var run in WeaverTestHarness.CompileAndWeaveBoth(Prelude +
                "public class C {\n" +
                "  [NDScoped] public NDArray F(NDArray a, int k, string s, long[] shape, object o, NDArray[] arr,\n" +
                "      Func<NDArray, NDArray> f, (NDArray, NDArray) t, ReadOnlySpan<long> span, ref int r) {\n" +
                "    r = k; return a + k;\n" +
                "  }\n" +
                "}\n", "RootsScoped"))
            {
                run.AssertWoven(1);
                using var asm = run.ReadCecil();
                var m = WeaveRun.Find(asm.MainModule, "C", "F");
                CollectionAssert.AreEquivalent(new[] { "this", "a", "o", "arr", "f", "t" }, Roots(m),
                    $"rooted parameters ({(run.Fixture.Optimized ? "Release" : "Debug")})");
                Assert.IsTrue(BoxedRoots(m).SequenceEqual(new[] { "t" }), "only the tuple of arrays is boxed (a struct carrying NDArrays)");
            }
        }

        /// <summary>
        ///     A generic parameter is rooted only when its constraint makes it a reference type (its box is then free); an
        ///     unconstrained or struct-constrained <c>T</c> would allocate a box per call and is left alone.
        /// </summary>
        /// <exception cref="AssertFailedException">The generic rooting rule is broken.</exception>
        [TestMethod]
        public void Generic_RootedOnlyWhenReferenceConstrained()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class G {\n" +
                "  [NDScoped] public static NDArray F<TArr, TRef, TAny, TVal>(TArr a, TRef r, TAny any, TVal v)\n" +
                "      where TArr : NDArray where TRef : class where TVal : unmanaged => a.copy();\n" +
                "}\n", "RootsGeneric");
            run.AssertWoven(1);
            using var asm = run.ReadCecil();
            var m = WeaveRun.Find(asm.MainModule, "G", "F");
            CollectionAssert.AreEquivalent(new[] { "a", "r" }, Roots(m), "a T : NDArray and a T : class are rooted; an open T and a T : unmanaged are not");

            var result = (NDArray)run.Load().GetType("G").GetMethod("F").MakeGenericMethod(typeof(NDArray), typeof(string), typeof(int), typeof(long))
                .Invoke(null, new object[] { np.arange(3).astype(np.float64), "x", 7, 9L });
            CollectionAssert.AreEqual(new[] { 0.0, 1.0, 2.0 }, result.ToArray<double>(), "the woven generic method still runs");
        }

        /// <summary>
        ///     A struct's instance method does not root <c>this</c> (a managed pointer into the caller's frame, not a GC
        ///     reference), and a constructor does not root its object (the caller's <c>newobj</c> keeps it alive).
        /// </summary>
        /// <exception cref="AssertFailedException">A struct <c>this</c> or a constructor's object is rooted.</exception>
        [TestMethod]
        public void StructThis_AndConstructorThis_AreNotRooted()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public struct S { public double K; [NDScoped] public NDArray Scale(NDArray a) => a * K; }\n" +
                "public class H { public NDArray Kept; [NDAnchored] public H(NDArray a) { Kept = a.copy(); } }\n",
                "RootsThis");
            run.AssertWoven(2);
            using var asm = run.ReadCecil();
            CollectionAssert.AreEquivalent(new[] { "a" }, Roots(WeaveRun.Find(asm.MainModule, "S", "Scale")), "struct: only the operand");
            CollectionAssert.AreEquivalent(new[] { "a" }, Roots(WeaveRun.Find(asm.MainModule, "H", ".ctor")), "constructor: only the operand");

            var h = run.New("H", np.arange(3).astype(np.float64));
            CollectionAssert.AreEqual(new[] { 0.0, 1.0, 2.0 }, ((NDArray)run.GetField(h, "Kept")).ToArray<double>(), "the woven constructor runs");
        }

        // ---------------------------------------------------------------- where the roots sit

        /// <summary>
        ///     Every <c>ret</c> of a rooted body is preceded by one <c>NDScope.Anchor</c> per shadow, and every shadow is
        ///     stored before the original body runs — so a body that reassigns its parameter (<c>a = a + 1.0</c>) still roots
        ///     the caller's ORIGINAL array, and every return path keeps the shadows live. Checked on a scoped body (one
        ///     epilogue ret) and on an anchored one (one ret per return statement).
        /// </summary>
        /// <exception cref="AssertFailedException">A return without anchors, or a shadow stored after the body started.</exception>
        [TestMethod]
        public void AnchorsPrecedeEveryReturn_AndShadowsAreTakenBeforeTheBody()
        {
            foreach (var run in WeaverTestHarness.CompileAndWeaveBoth(Prelude +
                "public static class R {\n" +
                "  [NDScoped] public static NDArray Scoped(NDArray a, int k) {\n" +
                "    if (k == 0) return a.copy();\n" +
                "    a = a + 1.0;\n" +
                "    if (k == 1) return a;\n" +
                "    for (int i = 0; i < k; i++) a = a * 2.0;\n" +
                "    return a;\n" +
                "  }\n" +
                "  [NDAnchored] public static double Anchored(NDArray a, NDArray b, int k) {\n" +
                "    if (k == 0) return a.GetDouble(0);\n" +
                "    if (k == 1) return b.GetDouble(0);\n" +
                "    a = b;\n" +
                "    return a.GetDouble(1);\n" +
                "  }\n" +
                "}\n", "RootsReturns"))
            {
                run.AssertWoven(2);
                using var asm = run.ReadCecil();
                foreach (var (name, shadows) in new[] { ("Scoped", 1), ("Anchored", 2) })
                {
                    var m = WeaveRun.Find(asm.MainModule, "R", name);
                    var rets = m.Body.Instructions.Where(i => i.OpCode == OpCodes.Ret).ToList();
                    Assert.IsTrue(rets.Count >= 1, name);
                    foreach (var ret in rets)
                    {
                        // Walk back over the anchor calls in front of this ret: each is `ldloc shadow; call Anchor`.
                        int anchors = 0;
                        for (var i = ret.Previous; i != null && IsAnchorCall(i); i = i.Previous?.Previous)
                            anchors++;
                        Assert.AreEqual(shadows, anchors, $"{name}: every ret is preceded by one Anchor per shadow");
                    }

                    // The shadow stores come first: nothing of the original body runs before the last shadow store.
                    var prologue = ShadowPrologueLength(m);
                    Assert.AreEqual(shadows * 2, prologue, $"{name}: the body starts with exactly the shadow stores (ldarg; stloc per root)");
                }

                Assert.AreEqual(0.0, (double)run.Invoke("R", "Anchored", null, np.arange(3).astype(np.float64), np.arange(3).astype(np.float64) + 10.0, 0));
                Assert.AreEqual(10.0, (double)run.Invoke("R", "Anchored", null, np.arange(3).astype(np.float64), np.arange(3).astype(np.float64) + 10.0, 1));
                Assert.AreEqual(11.0, (double)run.Invoke("R", "Anchored", null, np.arange(3).astype(np.float64), np.arange(3).astype(np.float64) + 10.0, 2));
                var r = (NDArray)run.Invoke("R", "Scoped", null, np.arange(3).astype(np.float64), 2);
                CollectionAssert.AreEqual(new[] { 4.0, 8.0, 12.0 }, r.ToArray<double>(), "the scoped body still computes ((a + 1) * 2 * 2)");
            }
        }

        // ---------------------------------------------------------------- which methods get roots

        /// <summary>
        ///     A method that opens an NDScope BY HAND, with no attribute, gets the same roots as a woven one — "a method
        ///     that opens a scope roots its parameters" holds however the scope was written — while an unmarked method that
        ///     opens no scope is not touched. A hand-scoped method carrying <c>[NDScoped]</c> (idempotence: its scope is
        ///     kept) gets the roots too.
        /// </summary>
        /// <exception cref="AssertFailedException">A hand-scoped method went unrooted, or an unmarked unscoped one was rooted.</exception>
        [TestMethod]
        public void HandWrittenScopes_AreRooted_UnmarkedUnscopedMethodsAreNot()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class H {\n" +
                "  public static NDArray Hand(NDArray a) { using var scope = NDScope.Open(); return scope.Returns(a + 1.0); }\n" +
                "  [NDScoped] public static NDArray HandAndAttribute(NDArray a) { using var scope = NDScope.Open(); return scope.Returns(a + 1.0); }\n" +
                "  public static NDArray Plain(NDArray a) => a + 1.0;\n" +
                "}\n", "RootsHand");
            run.AssertWoven(2);
            using var asm = run.ReadCecil();
            CollectionAssert.AreEquivalent(new[] { "a" }, Roots(WeaveRun.Find(asm.MainModule, "H", "Hand")), "a hand-written scope roots its operand");
            CollectionAssert.AreEquivalent(new[] { "a" }, Roots(WeaveRun.Find(asm.MainModule, "H", "HandAndAttribute")), "an attributed hand-scoped body too");
            Assert.AreEqual(1, WeaveRun.CountScopeCalls(WeaveRun.Find(asm.MainModule, "H", "HandAndAttribute"), "Open"), "its own scope is kept, not doubled");
            Assert.AreEqual(0, Roots(WeaveRun.Find(asm.MainModule, "H", "Plain")).Count, "an unmarked method that opens no scope is left alone");
            StringAssert.Contains(run.Stdout, "operand roots of a hand-written NDScope", "the verbose log names the hand-scoped pass");

            var r = (NDArray)run.Invoke("H", "Hand", null, np.arange(3).astype(np.float64));
            CollectionAssert.AreEqual(new[] { 1.0, 2.0, 3.0 }, r.ToArray<double>());
        }

        /// <summary>
        ///     An <c>[NDAnchored]</c> method gets the roots and nothing else: no scope local, no <c>NDScope.Open</c>, no
        ///     finally — its temporaries are not tracked (that is <c>[NDScoped]</c>'s job), and it runs unchanged.
        /// </summary>
        /// <exception cref="AssertFailedException">The anchored method gained a scope, or lost its roots.</exception>
        [TestMethod]
        public void Anchored_RootsWithoutAScope()
        {
            foreach (var run in WeaverTestHarness.CompileAndWeaveBoth(Prelude +
                "public static class A {\n" +
                "  [NDAnchored] public static void Fill(NDArray dst, NDArray src) { var n = src.size; for (long i = 0; i < n; i++) dst.SetDouble(src.GetDouble(i) * 2.0, i); }\n" +
                "}\n", "RootsAnchored"))
            {
                run.AssertWoven(1);
                using var asm = run.ReadCecil();
                var m = WeaveRun.Find(asm.MainModule, "A", "Fill");
                CollectionAssert.AreEquivalent(new[] { "dst", "src" }, Roots(m));
                Assert.IsFalse(WeaveRun.HasScopeLocal(m), "no scope local");
                Assert.AreEqual(0, WeaveRun.CountScopeCalls(m, "Open"), "no NDScope.Open");
                Assert.IsFalse(WeaveRun.HasFinallyDispose(m), "no finally");
                StringAssert.Contains(run.Stdout, "[NDAnchored] operand roots");

                var dst = np.zeros(3);
                run.Invoke("A", "Fill", null, dst, np.arange(3).astype(np.float64));
                CollectionAssert.AreEqual(new[] { 0.0, 2.0, 4.0 }, dst.ToArray<double>(), "the anchored body still runs");
            }
        }

        /// <summary>
        ///     A body that had no locals gets <c>.locals init</c> with its shadows (otherwise the method is unverifiable), and
        ///     a <c>[SkipLocalsInit]</c> body that has locals keeps its choice.
        /// </summary>
        /// <exception cref="AssertFailedException">The init-locals flag was set or cleared against the rule.</exception>
        [TestMethod]
        public void InitLocals_SetOnlyForABodyThatHadNoLocals()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class L {\n" +
                "  [NDAnchored] public static double First(NDArray a) => a.GetDouble(0);\n" +
                "  [NDAnchored, System.Runtime.CompilerServices.SkipLocalsInit] public static unsafe double Sum(NDArray a) {\n" +
                "    double* tmp = stackalloc double[4]; double s = 0; for (int i = 0; i < 3; i++) { tmp[i] = a.GetDouble(i); s += tmp[i]; } return s;\n" +
                "  }\n" +
                "}\n", "RootsInitLocals");
            run.AssertWoven(2);
            using var asm = run.ReadCecil();
            Assert.IsTrue(WeaveRun.Find(asm.MainModule, "L", "First").Body.InitLocals, "a body without locals gains .locals init");
            Assert.IsFalse(WeaveRun.Find(asm.MainModule, "L", "Sum").Body.InitLocals, "[SkipLocalsInit] stays in force");
            Assert.AreEqual(3.0, (double)run.Invoke("L", "Sum", null, np.arange(3).astype(np.float64)));
        }

        // ---------------------------------------------------------------- rejections and idempotence

        /// <summary>
        ///     NDW019: <c>[NDAnchored]</c> on an async method, an iterator, a method with nothing to root, or a method that
        ///     is also <c>[NDScoped]</c> is a build error, and the assembly is left untouched.
        /// </summary>
        /// <param name="member">The offending member's source.</param>
        /// <exception cref="AssertFailedException">The shape was not rejected with NDW019, or the assembly was rewritten.</exception>
        [TestMethod]
        [DataRow("[NDAnchored] public static async Task<NDArray> F(NDArray a) { await Task.Yield(); return a + 1.0; }")]
        [DataRow("[NDAnchored] public static IEnumerable<double> F(NDArray a) { yield return a.GetDouble(0); }")]
        [DataRow("[NDAnchored] public static int F(int a, string s) => a + s.Length;")]
        [DataRow("[NDAnchored, NDScoped] public static NDArray F(NDArray a) => a + 1.0;")]
        public void Anchored_RejectedShapes_AreNDW019(string member)
        {
            var fixture = WeaverTestHarness.Compile(Prelude + "public static class X {\n  " + member + "\n}\n", "RootsRejected");
            WeaverTestHarness.Weave(fixture).AssertRejected("NDW019");
        }

        /// <summary>
        ///     A second weave over a woven assembly adds nothing: a rooted body already calls <c>NDScope.Anchor</c>, so it is
        ///     skipped, and the file is not rewritten.
        /// </summary>
        /// <exception cref="AssertFailedException">The second weave changed the assembly or doubled a root.</exception>
        [TestMethod]
        public void SecondWeave_AddsNothing()
        {
            var fixture = WeaverTestHarness.Compile(Prelude +
                "public static class I {\n" +
                "  [NDScoped] public static NDArray S(NDArray a) => a + 1.0;\n" +
                "  [NDAnchored] public static double A(NDArray a) => a.GetDouble(0);\n" +
                "  public static NDArray H(NDArray a) { using var scope = NDScope.Open(); return scope.Returns(a + 1.0); }\n" +
                "}\n", "RootsTwice");
            WeaverTestHarness.Weave(fixture).AssertWoven(3);
            var second = WeaverTestHarness.Weave(fixture);
            Assert.AreEqual(0, second.Result.Errors, second.Report);
            Assert.AreEqual(0, second.Result.Woven, "nothing is woven twice:\n" + second.Report);
            Assert.IsFalse(second.Rewrote, "the second weave must not rewrite the file:\n" + second.Report);
            using var asm = second.ReadCecil();
            foreach (var name in new[] { "S", "A", "H" })
                CollectionAssert.AreEquivalent(new[] { "a" }, Roots(WeaveRun.Find(asm.MainModule, "I", name)), $"{name}: one root, not two");
        }

        // ---------------------------------------------------------------- helpers

        /// <summary>
        ///     The parameters a woven body roots, recovered from its IL: every <c>object</c> local passed to
        ///     <c>NDScope.Anchor</c> is a shadow, and the argument loaded just before its store (skipping one <c>box</c>) is
        ///     the rooted parameter (<c>"this"</c> for argument 0 of an instance method).
        /// </summary>
        /// <param name="m">The woven method.</param>
        /// <returns>The rooted parameters' names, without duplicates.</returns>
        private static List<string> Roots(MethodDefinition m) => RootLoads(m).Select(r => r.Name).Distinct().ToList();

        /// <summary>The names of the rooted parameters whose shadow store is preceded by a <c>box</c>.</summary>
        /// <param name="m">The woven method.</param>
        /// <returns>The boxed roots' names.</returns>
        private static List<string> BoxedRoots(MethodDefinition m) => RootLoads(m).Where(r => r.Boxed).Select(r => r.Name).Distinct().ToList();

        /// <summary>Each shadow's rooted argument and whether it is boxed before the store.</summary>
        /// <param name="m">The woven method.</param>
        /// <returns>One entry per shadow store.</returns>
        private static List<(string Name, bool Boxed)> RootLoads(MethodDefinition m)
        {
            var result = new List<(string, bool)>();
            if (!m.HasBody)
                return result;
            CurrentBody = m.Body;
            var shadows = new HashSet<VariableDefinition>();
            foreach (var i in m.Body.Instructions)
                if (IsAnchorCall(i) && LoadedLocal(i.Previous) is { } v)
                    shadows.Add(v);

            foreach (var i in m.Body.Instructions)
            {
                if (StoredLocal(i) is not { } v || !shadows.Contains(v))
                    continue;
                var load = i.Previous;
                bool boxed = load != null && load.OpCode == OpCodes.Box;
                if (boxed)
                    load = load.Previous;
                result.Add((ArgumentName(m, load), boxed));
            }

            return result;
        }

        /// <summary>
        ///     The number of instructions at the start of the body that store shadows: consecutive groups of an argument load,
        ///     an optional <c>box</c>, and a store into a local that is passed to <c>NDScope.Anchor</c>. The count stops at the
        ///     first instruction that does not continue such a group, so an original body that itself starts with an argument
        ///     load is not counted.
        /// </summary>
        /// <param name="m">The woven method.</param>
        /// <returns>The prologue length.</returns>
        private static int ShadowPrologueLength(MethodDefinition m)
        {
            CurrentBody = m.Body;
            var shadows = new HashSet<VariableDefinition>();
            foreach (var i in m.Body.Instructions)
                if (IsAnchorCall(i) && LoadedLocal(i.Previous) is { } v)
                    shadows.Add(v);

            int n = 0;
            for (var i = m.Body.Instructions.FirstOrDefault(); i != null;)
            {
                var code = i.OpCode.Code;
                if (code is not (Code.Ldarg or Code.Ldarg_S or Code.Ldarg_0 or Code.Ldarg_1 or Code.Ldarg_2 or Code.Ldarg_3))
                    break;
                var next = i.Next;
                int group = 1;
                if (next != null && next.OpCode == OpCodes.Box)
                {
                    next = next.Next;
                    group++;
                }

                if (StoredLocal(next) is not { } stored || !shadows.Contains(stored))
                    break;
                n += group + 1;
                i = next.Next;
            }

            return n;
        }

        /// <summary>Whether <paramref name="i"/> is a call to <c>NDScope.Anchor</c>.</summary>
        /// <param name="i">An instruction.</param>
        /// <returns>True for the anchor call.</returns>
        private static bool IsAnchorCall(Instruction i)
            => i != null && i.OpCode.FlowControl == FlowControl.Call && i.Operand is MethodReference mr &&
               mr.Name == "Anchor" && mr.DeclaringType.FullName == "NumSharp.NDScope";

        /// <summary>
        ///     The local an <c>ldloc</c> (any form) reads, or null. The macro forms <c>ldloc.0</c>–<c>ldloc.3</c> carry NO operand
        ///     in Cecil (the index is the opcode), so the variable is looked up in the body by index.
        /// </summary>
        /// <param name="i">An instruction.</param>
        /// <returns>The local, or null.</returns>
        private static VariableDefinition LoadedLocal(Instruction i)
        {
            if (i == null)
                return null;
            return i.OpCode.Code switch
            {
                Code.Ldloc or Code.Ldloc_S => (VariableDefinition)i.Operand,
                Code.Ldloc_0 => LocalAt(i, 0),
                Code.Ldloc_1 => LocalAt(i, 1),
                Code.Ldloc_2 => LocalAt(i, 2),
                Code.Ldloc_3 => LocalAt(i, 3),
                _ => null
            };
        }

        /// <summary>
        ///     The local an <c>stloc</c> (any form) writes, or null; <c>stloc.0</c>–<c>stloc.3</c> are resolved by index, as in
        ///     <see cref="LoadedLocal"/>.
        /// </summary>
        /// <param name="i">An instruction.</param>
        /// <returns>The local, or null.</returns>
        private static VariableDefinition StoredLocal(Instruction i)
        {
            if (i == null)
                return null;
            return i.OpCode.Code switch
            {
                Code.Stloc or Code.Stloc_S => (VariableDefinition)i.Operand,
                Code.Stloc_0 => LocalAt(i, 0),
                Code.Stloc_1 => LocalAt(i, 1),
                Code.Stloc_2 => LocalAt(i, 2),
                Code.Stloc_3 => LocalAt(i, 3),
                _ => null
            };
        }

        /// <summary>The method's local at <paramref name="index"/>, found through the method the instruction belongs to.</summary>
        /// <param name="i">An instruction of the method being inspected (<see cref="CurrentBody"/> is that method's body).</param>
        /// <param name="index">The local's index.</param>
        /// <returns>The local.</returns>
        private static VariableDefinition LocalAt(Instruction i, int index) => CurrentBody.Variables[index];

        /// <summary>
        ///     The body the helpers are reading. An <see cref="Instruction"/> does not know its method, so the macro local forms
        ///     are resolved against the body the public helpers set before they scan it (tests in this class run one at a
        ///     time on their own thread, so a thread-static is enough).
        /// </summary>
        [ThreadStatic] private static Mono.Cecil.Cil.MethodBody CurrentBody;

        /// <summary>The name of the argument an <c>ldarg</c> (any form) loads: <c>"this"</c>, or the parameter's name.</summary>
        /// <param name="m">The method.</param>
        /// <param name="i">The load.</param>
        /// <returns>The argument's name.</returns>
        /// <exception cref="AssertFailedException"><paramref name="i"/> is not an argument load (the shadow store does not
        ///     follow the weaver's pattern).</exception>
        private static string ArgumentName(MethodDefinition m, Instruction i)
        {
            int index = i.OpCode.Code switch
            {
                Code.Ldarg_0 => 0,
                Code.Ldarg_1 => 1,
                Code.Ldarg_2 => 2,
                Code.Ldarg_3 => 3,
                Code.Ldarg or Code.Ldarg_S => ((ParameterDefinition)i.Operand).Sequence,
                _ => throw new AssertFailedException($"a shadow store is not preceded by an argument load: {i}")
            };
            if (m.HasThis)
            {
                if (index == 0)
                    return "this";
                index--;
            }

            return m.Parameters[index].Name;
        }
    }
}
