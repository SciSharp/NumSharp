using System;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Assembly
{
    /// <summary>
    ///     Pins NumSharp's strong-name identity.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         This gate exists because the previous arrangement failed SILENTLY for years. Signing was
    ///         configured only under a <c>Publish|AnyCPU</c> condition while CI builds and packs
    ///         <c>Release</c>, so every published NumSharp assembly shipped with
    ///         <c>PublicKeyToken=null</c> while the repository looked, to a reader, like it signed its
    ///         output. Nothing asserted otherwise. Worse, the <c>Publish</c> configuration did not even
    ///         compile — strong-naming turns a keyless <c>InternalsVisibleTo</c> into <c>CS1726</c>, and
    ///         all seven of ours were keyless (five in NumSharp.Core, two in NumSharp.Interop.OpenBLAS) — so
    ///         the one configuration that would have revealed the problem was never built by anything.
    ///     </para>
    ///     <para>
    ///         A build failure is therefore NOT sufficient coverage: the failure mode was output that
    ///         built fine and merely lacked an identity. These tests read the identity off the compiled
    ///         assemblies at runtime, which is the only thing that can catch a regression of that shape.
    ///     </para>
    ///     <para>
    ///         Note what is deliberately NOT claimed here. The key is Microsoft's PUBLISHED open-source
    ///         key (see the repo-root <c>Directory.Build.props</c>), whose private half is public by
    ///         design. Strong-naming is assembly IDENTITY — it lets the loader bind versions and tell
    ///         this NumSharp from another — not authenticity, and these assertions are not a security
    ///         property. Real authenticity would be Authenticode plus NuGet author signing, which needs
    ///         a code-signing certificate and is entirely separate from anything asserted here.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class StrongNameTests
    {
        /// <summary>
        ///     The public key token every NumSharp assembly must carry: Microsoft's open OSS key, the
        ///     same identity borne by <c>netstandard</c>, <c>System.Memory</c> and <c>System.Buffers</c>,
        ///     and the same key TensorFlow.NET signs with (which is what makes the cross-repo
        ///     <c>InternalsVisibleTo("TensorFlowNET.UnitTest")</c> resolve).
        /// </summary>
        private const string ExpectedToken = "cc7b13ffcd2ddd51";

        /// <summary>
        ///     The full public key, as written in <c>Directory.Build.props</c>
        ///     (<c>NumSharpPublicKey</c>), <c>NumSharp.Core/Assembly/Properties.cs</c> and
        ///     <c>NumSharp.Interop.OpenBLAS/AssemblyInfo.cs</c>. Held here a fourth time ON PURPOSE: this
        ///     copy is what proves the other three still describe the key the compiler actually used.
        /// </summary>
        private const string ExpectedPublicKey =
            "00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34" +
            "bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbb" +
            "ad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8" +
            "fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb";

        private static string TokenOf(System.Reflection.Assembly asm)
        {
            byte[] token = asm.GetName().GetPublicKeyToken();
            return token is null || token.Length == 0
                ? "<unsigned>"
                : Convert.ToHexString(token).ToLowerInvariant();
        }

        [TestMethod]
        public void NumSharpCore_IsStrongNamed()
        {
            var asm = typeof(NDArray).Assembly;

            Assert.AreEqual(ExpectedToken, TokenOf(asm),
                $"NumSharp.Core lost its strong name. Shipping an unsigned NumSharp is exactly the " +
                $"regression this gate exists for — check SignAssembly in the repo-root " +
                $"Directory.Build.props, and that no csproj overrides it back to false.");
        }

        [TestMethod]
        public void NumSharpCore_CarriesTheExpectedPublicKey()
        {
            byte[] key = typeof(NDArray).Assembly.GetName().GetPublicKey();

            Assert.IsNotNull(key, "NumSharp.Core carries no public key at all.");
            Assert.AreEqual(ExpectedPublicKey, Convert.ToHexString(key).ToLowerInvariant(),
                "The key NumSharp.Core was signed with is not the one written into " +
                "Directory.Build.props / Properties.cs. Those InternalsVisibleTo literals would no " +
                "longer match, so friend access breaks at runtime rather than at build time.");
        }

        [TestMethod]
        public void TestAssembly_IsStrongNamed_SoFriendAccessResolves()
        {
            // NumSharp.Core grants this assembly InternalsVisibleTo *with* a PublicKey. A friend
            // reference that names a key only matches an assembly signed with that key, so an
            // unsigned test assembly would silently lose access to every internal it tests.
            var asm = typeof(StrongNameTests).Assembly;

            Assert.AreEqual(ExpectedToken, TokenOf(asm),
                "NumSharp.UnitTest is not signed, so NumSharp.Core's keyed InternalsVisibleTo cannot " +
                "match it.");
        }

        [TestMethod]
        public void FriendAccess_ActuallyWorks()
        {
            // Reaching an `internal` member of NumSharp.Core from here only compiles AND runs if the
            // keyed InternalsVisibleTo genuinely matches this assembly's identity. This is the
            // end-to-end proof that the PublicKey= literals are correct, not merely well-formed.
            var shape = new Shape(2, 3);

            Assert.AreEqual(6L, shape.size);                                  // internal field
            CollectionAssert.AreEqual(new long[] { 2, 3 }, shape.dimensions); // internal field (long[])
        }

        [TestMethod]
        public void EveryFriendDeclaration_NamesAPublicKey()
        {
            // A keyless InternalsVisibleTo on a signed assembly is CS1726 and cannot compile — but a
            // friend naming the WRONG key compiles fine and simply never matches. Assert the shape of
            // every declaration so a future entry cannot be added keyless-by-habit and, if someone
            // relaxes signing, quietly ship a grant that matches any assembly of that name.
            foreach (var asm in new[] { typeof(NDArray).Assembly, typeof(StrongNameTests).Assembly })
            {
                var friends = asm.GetCustomAttributes<InternalsVisibleToAttribute>().ToArray();

                foreach (var f in friends)
                {
                    StringAssert.Contains(f.AssemblyName, "PublicKey=",
                        $"InternalsVisibleTo(\"{f.AssemblyName}\") in {asm.GetName().Name} names no " +
                        $"PublicKey.");
                    StringAssert.Contains(f.AssemblyName.ToLowerInvariant(), ExpectedPublicKey,
                        $"InternalsVisibleTo(\"{f.AssemblyName}\") in {asm.GetName().Name} names a key " +
                        $"that is not NumSharp's, so it will never match.");
                }
            }
        }

        [TestMethod]
        public void KeyIsFullyRealised_NotDelaySigned()
        {
            // A delay-signed assembly carries the public key and an EMPTY signature: it looks signed to
            // GetPublicKeyToken() but fails verification and cannot load without skip-verification
            // registration. Loading and executing this assembly's own code is the practical proof the
            // signature is real, so assert we got here through a fully-signed image.
            var name = typeof(NDArray).Assembly.GetName();

            Assert.AreEqual(ExpectedToken, TokenOf(typeof(NDArray).Assembly));
            Assert.IsTrue((name.Flags & AssemblyNameFlags.PublicKey) != 0,
                "NumSharp.Core's identity does not carry the full public key.");
        }
    }
}
