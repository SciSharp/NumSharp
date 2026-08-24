using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Proves the build-time <c>[NDScoped]</c> IL weaver (<c>tools/NumSharp.Weaver</c>) actually
    ///     ran over the SHIPPED assembly. Every method — or property accessor — marked
    ///     <see cref="NDScopedAttribute"/> must, after weaving, carry a local of type
    ///     <see cref="NDScope"/>: the scope the weaver opens. A <c>[NDScoped]</c> method that still
    ///     lacks it was NOT woven — a broken <c>NDScopeWeave</c> target, or a build with
    ///     <c>-p:SkipNDScopeWeave=true</c> — which silently drops eager reclamation on that method
    ///     back to the finalizer backstop (the pre-migration status quo). The <see cref="NDScopeTests"/>
    ///     zero-strand counters would eventually notice for the ops they exercise; this gate is the
    ///     direct, total proof that the transform covered the whole attributed surface.
    /// </summary>
    /// <remarks>
    ///     The signal — an <see cref="NDScope"/> local — is a NECESSARY consequence of scoping,
    ///     whether the scope was woven in or hand-written, so the check also (correctly) passes a
    ///     method that carries the attribute AND opens the scope by hand: both honour the attribute's
    ///     promise. It reads only metadata (<see cref="MethodBody.LocalVariables"/>), no IL decoding,
    ///     so it is robust to instruction-layout changes in the weaver.
    /// </remarks>
    [TestClass]
    public class NDScopeWeaveTests
    {
        // DISPOSAL-GUIDELINES / the weaver commit record ~50 attributed methods. Assert a floor so
        // the test can never pass vacuously (attribute stripped, wrong assembly, reflection finding
        // none). Kept well below the true count to tolerate ordinary churn.
        private const int MinScopedMethods = 40;

        private const BindingFlags AllDeclared =
            BindingFlags.Public | BindingFlags.NonPublic |
            BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        private static IEnumerable<System.Type> AllTypes()
        {
            var asm = typeof(NDArray).Assembly;
            try
            {
                return asm.GetTypes();
            }
            catch (ReflectionTypeLoadException ex)
            {
                // A type that fails to load cannot host a [NDScoped] method we care about; use what loaded.
                return ex.Types.Where(t => t != null);
            }
        }

        private static IEnumerable<MethodInfo> AllScopedMethods()
        {
            var seen = new HashSet<MethodInfo>();
            foreach (var type in AllTypes())
            {
                foreach (var m in type.GetMethods(AllDeclared))
                    if (m.GetCustomAttribute<NDScopedAttribute>(inherit: false) != null && seen.Add(m))
                        yield return m;

                // A property-LEVEL attribute resolves to the getter (mirrors the weaver's
                // CollectTargets); accessor-level attributes are already covered by the method loop.
                foreach (var p in type.GetProperties(AllDeclared))
                {
                    if (p.GetCustomAttribute<NDScopedAttribute>(inherit: false) is null)
                        continue;
                    var getter = p.GetGetMethod(nonPublic: true);
                    if (getter != null && seen.Add(getter))
                        yield return getter;
                }
            }
        }

        private static bool HasScopeLocal(MethodInfo m)
        {
            var body = m.GetMethodBody();
            if (body is null)
                return false;
            foreach (var local in body.LocalVariables)
                if (local.LocalType == typeof(NDScope))
                    return true;
            return false;
        }

        [TestMethod]
        public void EveryScopedMethod_WasWoven()
        {
            var scoped = AllScopedMethods().ToList();

            Assert.IsTrue(scoped.Count >= MinScopedMethods,
                $"found only {scoped.Count} [NDScoped] methods (expected >= {MinScopedMethods}); " +
                "the attribute may have been stripped, or reflection did not reach the attributed types");

            var unwoven = scoped
                .Where(m => !HasScopeLocal(m))
                .Select(m => m.DeclaringType!.FullName + "." + m.Name)
                .OrderBy(n => n)
                .ToList();

            Assert.AreEqual(0, unwoven.Count,
                "these [NDScoped] methods carry no NDScope local, so the weaver did not scope them " +
                "(a broken NDScopeWeave target, or a build with -p:SkipNDScopeWeave=true):\n  " +
                string.Join("\n  ", unwoven));
        }
    }
}
