using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Interop;

[TestClass]
public class KarpathyOriginalSourceTests : InteropTestBase
{
    [DataTestMethod, TestCategory("KarpathySourceIntegrity")]
    [DataRow("rnn", 2)] [DataRow("pong", 5)] [DataRow("lstm", 3)] [DataRow("microgpt", 5)]
    [DataRow("nes", 1)] [DataRow("walk", 1)]
    public void OriginalSource_IsHashVerifiedAndOnlyReviewedDefinitionsAreLoaded(string key, int definitions)
    {
        KarpathyOriginalSource.Load(Scope, key);
        Assert.AreEqual(definitions, PyLong("len(original['__loaded_names__'])"));
        Assert.IsTrue(PyBool("all(name in original for name in original['__loaded_names__'])"));
        Assert.IsTrue(PyBool("not any(name in original for name in ('docs','data','env','urllib','os','gym','input','num_steps'))"));
        Assert.AreEqual(64, PyLong("len(original['__source_sha256__'])"));
        Assert.ThrowsException<ArgumentException>(() => KarpathyOriginalSource.Read("unapproved"));
    }
}
