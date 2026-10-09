using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     The OPERAND ROOTS of the woven NumSharp.Core, checked in its IL — the deterministic half of the fix for the lost
    ///     GC anchors the lifetime oracle found (LT01–LT28, <c>test/NumSharp.Tests.Oracle/Fuzz/Lifetime</c>). Every method
    ///     that opens an <see cref="NDScope"/> and takes an <see cref="NDArray"/> must call <see cref="NDScope.Anchor"/>
    ///     (the weaver roots its operands), every <see cref="NDAnchoredAttribute"/> method must, and so must the entry point
    ///     of each of the 28 fixed sites.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why an IL check.</b> The lifetime oracle proves the class at run time, under a stress thread, in an optimized
    ///     child process — a race it wins with high probability, never with certainty. These tests read the woven IL of the
    ///     assembly the suite runs against, so a weaver regression (roots no longer injected), a site whose entry point lost
    ///     its attribute, or a new hand-written scope the weaver no longer sees fails on every run, in Debug as in Release
    ///     (the weave runs in both configurations).
    ///     </para>
    ///     <para>
    ///     <b>How calls are read.</b> <see cref="CalledMethods"/> walks a method's IL with the <see cref="OpCodes"/> operand
    ///     table and resolves every call operand through <see cref="Module.ResolveMethod(int, Type[], Type[])"/>.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OperandRootsTests
    {
        /// <summary>Every declared method and constructor, any visibility, static or instance.</summary>
        private const BindingFlags Declared = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static |
                                              BindingFlags.Instance | BindingFlags.DeclaredOnly;

        /// <summary>
        ///     The library-wide invariant: a method that opens an NDScope (woven from <see cref="NDScopedAttribute"/>, or a
        ///     hand-written <c>NDScope.Open()</c>) and takes an NDArray parameter roots it — its body calls
        ///     <see cref="NDScope.Anchor"/>. A scope roots only the arrays it tracks, never the caller's operands.
        /// </summary>
        /// <exception cref="AssertFailedException">A scope-opening method with an NDArray parameter has no roots (the weaver
        ///     missed it), or the scan found too few scope-opening methods to mean anything.</exception>
        [TestMethod]
        public void EveryScopeOpeningMethod_RootsItsNDArrayParameters()
        {
            int scoped = 0;
            var unrooted = new List<string>();
            foreach (var m in AllMethods())
            {
                var calls = CalledMethods(m);
                if (!calls.Any(c => c.DeclaringType == typeof(NDScope) && c.Name == nameof(NDScope.Open)))
                    continue;
                scoped++;
                bool takesArray = m.GetParameters().Any(p => !p.ParameterType.IsByRef && typeof(NDArray).IsAssignableFrom(p.ParameterType));
                if (takesArray && !calls.Any(IsAnchor))
                    unrooted.Add(Describe(m));
            }

            // About 600 methods open a scope (2026-10-09); far fewer means the scan stopped seeing them.
            Assert.IsTrue(scoped >= 400, $"only {scoped} scope-opening methods found: is the assembly woven, and the IL scan sound?");
            Assert.AreEqual(0, unrooted.Count, "scope-opening methods whose NDArray operands are not rooted:\n  " +
                                               string.Join("\n  ", unrooted.Take(40)));
        }

        /// <summary>Every <see cref="NDAnchoredAttribute"/> method calls <see cref="NDScope.Anchor"/> (the weaver wove it).</summary>
        /// <exception cref="AssertFailedException">An anchored method without roots, or no anchored method at all.</exception>
        [TestMethod]
        public void EveryAnchoredMethod_Roots()
        {
            var anchored = AllMethods().Where(m => m.IsDefined(typeof(NDAnchoredAttribute), inherit: false)).ToList();
            Assert.IsTrue(anchored.Count >= 10, $"only {anchored.Count} [NDAnchored] methods found");
            var unrooted = anchored.Where(m => !CalledMethods(m).Any(IsAnchor)).Select(Describe).ToList();
            Assert.AreEqual(0, unrooted.Count, "[NDAnchored] methods without roots:\n  " + string.Join("\n  ", unrooted));
        }

        /// <summary>
        ///     The entry point of each of the 28 sites the lifetime oracle found roots its operands: the method that held the
        ///     operand while its kernel ran (or the scoped frame above it) now calls <see cref="NDScope.Anchor"/>.
        /// </summary>
        /// <param name="gap">The fixed gap id (for the failure message).</param>
        /// <param name="owner">The declaring type's name: <c>np</c>, <c>np.linalg</c>, <c>NDArray</c>, <c>DefaultEngine</c>,
        ///     <c>TensorEngine</c>, <c>NDIter</c> or <c>PowerSeriesModule</c>.</param>
        /// <param name="name">The method name.</param>
        /// <param name="signature">The parameter type names, comma-separated (<c>Nullable`1</c> for an <c>int?</c>).</param>
        /// <exception cref="AssertFailedException">The entry point is missing, ambiguous, or does not root its operands.</exception>
        [TestMethod]
        [DataRow("LT01", "np", "take", "NDArray,NDArray,Nullable`1,NDArray,String")]
        [DataRow("LT02", "np", "put", "NDArray,NDArray,NDArray,String")]
        [DataRow("LT03", "np", "PlaceCore", "NDArray,NDArray,NDArray,Boolean")]
        [DataRow("LT04", "np", "PutMaskCore", "NDArray,NDArray,NDArray,Boolean")]
        [DataRow("LT05", "np", "take_along_axis", "NDArray,NDArray,Nullable`1")]
        [DataRow("LT06", "np", "put_along_axis", "NDArray,NDArray,NDArray,Nullable`1")]
        [DataRow("LT07", "np", "where_internal", "NDArray,NDArray,NDArray")]
        [DataRow("LT08", "np", "copyto", "NDArray,NDArray,String,NDArray")]
        [DataRow("LT09", "NDIter", "Copy", "NDArray,NDArray")]
        [DataRow("LT09", "NDIter", "CopyAs", "DType,NDArray,Char,TensorEngine")]
        [DataRow("LT10", "DefaultEngine", "MaxMinNumPy", "NDArray,Nullable`1,NDArray,Boolean,Object,NDArray,MinMaxOp")]
        [DataRow("LT11", "DefaultEngine", "ReduceAMax", "NDArray,Nullable`1,Boolean,DType")]
        [DataRow("LT12", "np", "cumsum", "NDArray,Nullable`1,DType,NDArray")]
        [DataRow("LT12", "np", "cumprod", "NDArray,Nullable`1,DType,NDArray")]
        [DataRow("LT13", "np", "nancumsum", "NDArray,Nullable`1,DType,NDArray")]
        [DataRow("LT14", "np", "searchsorted", "NDArray,NDArray,String,NDArray")]
        [DataRow("LT15", "np", "digitize", "NDArray,NDArray,Boolean")]
        [DataRow("LT16", "np", "bincount", "NDArray,NDArray,Int32")]
        [DataRow("LT17", "np", "reciprocal", "NDArray,NDArray,NDArray,DType")]
        [DataRow("LT18", "np", "dot", "NDArray,NDArray,NDArray")]
        [DataRow("LT18", "TensorEngine", "Inner", "NDArray,NDArray")]
        [DataRow("LT19", "np", "matvec", "NDArray,NDArray,NDArray,Int32[][],Nullable`1,Boolean,DType")]
        [DataRow("LT19", "np", "vecmat", "NDArray,NDArray,NDArray,Int32[][],Nullable`1,Boolean,DType")]
        [DataRow("LT21", "np.linalg", "det", "NDArray")]
        [DataRow("LT21", "np.linalg", "inv", "NDArray")]
        [DataRow("LT21", "np.linalg", "solve", "NDArray,NDArray")]
        [DataRow("LT22", "NDArray", "convolve", "NDArray,String")]
        [DataRow("LT22", "NDArray", "correlate", "NDArray,String")]
        [DataRow("LT23", "PowerSeriesModule", "polyder", "Object,Int32,Object,Int32")]
        [DataRow("LT24", "PowerSeriesModule", "polyvander", "Object,Int32")]
        [DataRow("LT25", "NDArray", "op_LogicalNot", "NDArray")]
        [DataRow("LT26", "DefaultEngine", "All", "NDArray")]
        [DataRow("LT27", "NDArray", "ToMuliDimArray", "")]
        [DataRow("LT28", "np", "unique_values", "NDArray")]
        [DataRow("LT28", "np", "unique_counts", "NDArray")]
        public void FixedSite_RootsItsOperands(string gap, string owner, string name, string signature)
        {
            var type = owner switch
            {
                "np" => typeof(np),
                "np.linalg" => typeof(np.linalg),
                "NDArray" => typeof(NDArray),
                "DefaultEngine" => typeof(DefaultEngine),
                "TensorEngine" => typeof(TensorEngine),
                "NDIter" => typeof(NDIter),
                "PowerSeriesModule" => typeof(NDArray).Assembly.GetTypes().Single(t => t.Name == "PowerSeriesModule"),
                _ => throw new AssertFailedException($"unknown owner {owner}")
            };
            var wanted = signature.Length == 0 ? Array.Empty<string>() : signature.Split(',');
            var hits = type.GetMethods(Declared)
                .Where(m => m.Name == name && m.GetParameters().Select(p => p.ParameterType.Name).SequenceEqual(wanted))
                .ToList();
            Assert.AreEqual(1, hits.Count, $"{gap}: expected exactly one {owner}.{name}({signature}), found {hits.Count}");
            Assert.IsTrue(CalledMethods(hits[0]).Any(IsAnchor),
                $"{gap}: {owner}.{name}({signature}) does not root its operands (no NDScope.Anchor call in its woven IL)");
        }

        // ------------------------------------------------------------------ helpers

        /// <summary>Every method and constructor with a body declared in NumSharp.Core, the NDScope type itself excluded.</summary>
        /// <returns>The methods (types that fail to load are skipped).</returns>
        private static IEnumerable<MethodBase> AllMethods()
        {
            Type[] types;
            try
            {
                types = typeof(NDArray).Assembly.GetTypes();
            }
            catch (ReflectionTypeLoadException e)
            {
                types = e.Types.Where(t => t != null).ToArray();
            }

            foreach (var t in types)
            {
                if (t == typeof(NDScope))
                    continue;
                foreach (var m in t.GetMethods(Declared).Cast<MethodBase>().Concat(t.GetConstructors(Declared)))
                    if (!m.IsAbstract)
                        yield return m;
            }
        }

        /// <summary>Whether <paramref name="m"/> is <see cref="NDScope.Anchor"/>.</summary>
        /// <param name="m">A called method.</param>
        /// <returns>True for the anchor.</returns>
        private static bool IsAnchor(MethodBase m) => m.DeclaringType == typeof(NDScope) && m.Name == nameof(NDScope.Anchor);

        /// <summary>The method's name with its declaring type and parameter types, for messages.</summary>
        /// <param name="m">The method.</param>
        /// <returns>The description.</returns>
        private static string Describe(MethodBase m)
            => $"{m.DeclaringType?.FullName}.{m.Name}({string.Join(", ", m.GetParameters().Select(p => p.ParameterType.Name))})";

        /// <summary>One-byte opcodes by value.</summary>
        private static readonly OpCode[] OneByte = new OpCode[0x100];

        /// <summary>Two-byte (<c>0xFE</c>-prefixed) opcodes by their second byte.</summary>
        private static readonly OpCode[] TwoByte = new OpCode[0x100];

        /// <summary>Fills the opcode tables from <see cref="OpCodes"/>.</summary>
        static OperandRootsTests()
        {
            foreach (var f in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
            {
                if (f.GetValue(null) is not OpCode op)
                    continue;
                ushort v = unchecked((ushort)op.Value);
                if (v < 0x100)
                    OneByte[v] = op;
                else if ((v & 0xFF00) == 0xFE00)
                    TwoByte[v & 0xFF] = op;
            }
        }

        /// <summary>
        ///     The methods <paramref name="m"/>'s IL calls (<c>call</c>, <c>callvirt</c>, <c>newobj</c>, <c>ldftn</c>: every
        ///     <c>InlineMethod</c> operand), resolved in its generic context. Operands that do not resolve are skipped.
        /// </summary>
        /// <param name="m">A method with a body.</param>
        /// <returns>The called methods, in IL order (empty without a body).</returns>
        private static List<MethodBase> CalledMethods(MethodBase m)
        {
            var result = new List<MethodBase>();
            byte[] il;
            try
            {
                il = m.GetMethodBody()?.GetILAsByteArray();
            }
            catch (Exception)
            {
                return result; // a body the runtime cannot hand out (a dynamic or extern method) calls nothing we can read
            }

            if (il == null)
                return result;
            var typeArgs = m.DeclaringType is { IsGenericType: true } dt ? dt.GetGenericArguments() : null;
            var methodArgs = m.IsGenericMethod ? m.GetGenericArguments() : null;

            int pos = 0;
            while (pos < il.Length)
            {
                byte b = il[pos++];
                OpCode op = b == 0xFE ? TwoByte[il[pos++]] : OneByte[b];
                switch (op.OperandType)
                {
                    case OperandType.InlineNone:
                        break;
                    case OperandType.ShortInlineBrTarget:
                    case OperandType.ShortInlineI:
                    case OperandType.ShortInlineVar:
                        pos += 1;
                        break;
                    case OperandType.InlineVar:
                        pos += 2;
                        break;
                    case OperandType.InlineMethod:
                        int token = BitConverter.ToInt32(il, pos);
                        try
                        {
                            if (m.Module.ResolveMethod(token, typeArgs, methodArgs) is { } target)
                                result.Add(target);
                        }
                        catch (Exception)
                        {
                            // an operand this context cannot resolve (a vararg signature, a missing reference): not a call
                            // to NDScope, which always resolves
                        }

                        pos += 4;
                        break;
                    case OperandType.InlineI8:
                    case OperandType.InlineR:
                        pos += 8;
                        break;
                    case OperandType.InlineSwitch:
                        int n = BitConverter.ToInt32(il, pos);
                        pos += 4 + 4 * n;
                        break;
                    default:
                        // InlineBrTarget, InlineField, InlineI, InlineSig, InlineString, InlineTok, InlineType, ShortInlineR
                        pos += 4;
                        break;
                }
            }

            return result;
        }
    }
}
