using System;
using System.Collections.Generic;
using System.Threading;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     <see cref="DTypeRegistry"/> without locks on the read side: its three keyed tables are
    ///     <c>ConcurrentDictionary</c>s, the code table is read with <see cref="Volatile"/>, and <see cref="DTypeRegistry.All"/>
    ///     copies an array each registration replaces; only <see cref="DTypeRegistry.Register"/> takes the registry's lock.
    ///     These tests pin that a refused registration leaves nothing behind (every check runs before the first write), that
    ///     lookups stay correct while registrations grow every table, and that <see cref="DTypeRegistry.All"/> hands out a
    ///     copy.
    /// </summary>
    /// <remarks>
    ///     A registration cannot be undone, so every class registered here stays in the registry for the rest of the test
    ///     run. Each test takes a fresh range of type numbers (<see cref="Reserve"/>, far above NumPy's) and names built
    ///     from them, so the tests can run again in one process, and the other tests that read
    ///     <see cref="DTypeRegistry.All"/> only require it to contain the builtins and to hold unique type numbers.
    /// </remarks>
    [TestClass]
    public class DTypeRegistryConcurrencyTests
    {
        /// <summary>The next unused type number for a probe class; <see cref="Reserve"/> hands out ranges of it.</summary>
        private static int s_nextTypeNum = 40_000;

        /// <summary>
        ///     A class refused for its alias, for a storage code another class holds, or for a storage code outside the
        ///     table leaves no trace: its type number, name and alias stay unregistered, it is not in
        ///     <see cref="DTypeRegistry.All"/>, and the corrected class registers under the same number and names.
        /// </summary>
        [TestMethod]
        public void Register_RefusedClass_LeavesNoTrace_AndTheCorrectedClassRegisters()
        {
            int num = Reserve(1);
            string name = $"RegistryProbe{num}DType";
            string alias = $"RegistryProbe{num}Alias";

            var takenAlias = new RegistryProbeMeta(name, num, aliases: new[] { alias, "Int8DType" });
            var takenCode = new RegistryProbeMeta(name, num, NPTypeCode.Int32, aliases: new[] { alias });
            var codeOutOfRange = new RegistryProbeMeta(name, num, (NPTypeCode)500, aliases: new[] { alias });

            Action registerTakenAlias = () => DTypeRegistry.Register(takenAlias);
            registerTakenAlias.Should().Throw<InvalidOperationException>().WithMessage("A DType named 'Int8DType' is already registered.");
            AssertNotRegistered(num, name, alias, takenAlias);

            Action registerTakenCode = () => DTypeRegistry.Register(takenCode);
            registerTakenCode.Should().Throw<InvalidOperationException>()
                .WithMessage("NPTypeCode Int32 is already registered (<class 'numpy.dtypes.Int32DType'>).");
            AssertNotRegistered(num, name, alias, takenCode);
            DTypeRegistry.FromTypeCode(NPTypeCode.Int32).Should().BeSameAs(DTypeRegistry.Int32, "the refused class never took the code");

            Action registerOutOfRange = () => DTypeRegistry.Register(codeOutOfRange);
            registerOutOfRange.Should().Throw<InvalidOperationException>().WithMessage("NPTypeCode 500 is outside the registry's code range.");
            AssertNotRegistered(num, name, alias, codeOutOfRange);

            var corrected = new RegistryProbeMeta(name, num, aliases: new[] { alias });
            DTypeRegistry.Register(corrected);
            DTypeRegistry.FromTypeNum(num).Should().BeSameAs(corrected);
            DTypeRegistry.FromName(name).Should().BeSameAs(corrected);
            DTypeRegistry.FromName(alias).Should().BeSameAs(corrected);
            DTypeRegistry.All.Should().Contain(corrected);

            Action again = () => DTypeRegistry.Register(new RegistryProbeMeta(name, num));
            again.Should().Throw<InvalidOperationException>().WithMessage($"A DType with type number {num} is already registered (<class 'numsharp.tests.{name}'>).");
        }

        /// <summary>
        ///     <see cref="DTypeRegistry.All"/> returns a new array on every read, in registration order, so a caller that
        ///     changes it changes only its own copy.
        /// </summary>
        [TestMethod]
        public void All_ReturnsACopy_InRegistrationOrder()
        {
            var first = DTypeRegistry.All;
            var second = DTypeRegistry.All;
            second.Should().NotBeSameAs(first);
            second.Should().Equal(first);
            first[0].Should().BeSameAs(DTypeRegistry.Bool, "Bool is the first class the static constructor registers");

            ((DTypeMeta[])first)[0] = null;
            DTypeRegistry.All[0].Should().BeSameAs(DTypeRegistry.Bool, "changing the returned array cannot change the registry");

            int num = Reserve(1);
            var probe = new RegistryProbeMeta($"RegistryProbe{num}DType", num);
            DTypeRegistry.Register(probe);
            var after = DTypeRegistry.All;
            after[after.Count - 1].Should().BeSameAs(probe, "a new class is appended");
            after.Count.Should().BeGreaterThan(second.Count);
        }

        /// <summary>
        ///     Writer threads register classes while reader threads look every one of them up by type number, name, alias
        ///     and C# scalar type, and look builtins up by name, scalar type and storage code. Every lookup returns null or
        ///     exactly the class registered under that key, a class once found is never lost, the builtins always resolve,
        ///     <see cref="DTypeRegistry.All"/> never shrinks, and nothing throws. Afterwards every class resolves by every key.
        /// </summary>
        /// <remarks>
        ///     The count is what lets this test see the race it guards against. A plain <c>Dictionary</c> read during a
        ///     resize misses keys whose buckets the rehash has not reached yet, and a resize lasts as long as rehashing every
        ///     entry: about a hundred nanoseconds at the builtin table sizes, which no reader hit in 10 runs with 96 classes.
        ///     1,024 classes grow each table through five or six resizes, the late ones over thousands of entries, so the
        ///     readers land inside one.
        /// </remarks>
        [TestMethod]
        public void Lookups_DuringRegistrations_SeeNullOrTheRegisteredClass()
        {
            const int count = 1_024;
            int writers = System.Math.Clamp(Environment.ProcessorCount / 2, 2, 4);
            int readers = System.Math.Clamp(Environment.ProcessorCount / 2, 1, 4);
            int baseNum = Reserve(count);

            var probes = new RegistryProbeMeta[count];
            var names = new string[count];
            var aliases = new string[count];
            var scalarTypes = new Type[count];
            for (int i = 0; i < count; i++)
            {
                names[i] = $"RegistryProbe{baseNum + i}DType";
                aliases[i] = $"RegistryProbe{baseNum + i}Alias";
                scalarTypes[i] = ScalarProbeType(baseNum - 40_000 + i);
                DTypeRegistry.FromScalarType(scalarTypes[i]).Should().BeNull("each probe scalar type must still be unclaimed");
                probes[i] = new RegistryProbeMeta(names[i], baseNum + i, scalarType: scalarTypes[i], aliases: new[] { aliases[i] });
            }

            int finished = 0;
            int writersLeft = writers;
            int readersLooping = 0;
            var problems = new List<string>();
            using var go = new ManualResetEventSlim(false);

            var threads = new List<Thread>();
            for (int w = 0; w < writers; w++)
            {
                int writer = w;
                threads.Add(new Thread(() =>
                {
                    go.Wait();
                    try
                    {
                        // The registrations take microseconds, less than waking a thread does: start them only once every
                        // reader is looping, or they finish before any lookup runs beside them.
                        var deadline = DateTime.UtcNow.AddSeconds(30);
                        while (Volatile.Read(ref readersLooping) < readers && DateTime.UtcNow < deadline)
                            Thread.Yield();
                        for (int i = writer; i < count; i += writers)
                            DTypeRegistry.Register(probes[i]);
                    }
                    catch (Exception e)
                    {
                        Report($"writer {writer}: {e}");
                    }
                    finally
                    {
                        if (Interlocked.Decrement(ref writersLeft) == 0)
                            Volatile.Write(ref finished, 1);
                    }
                }) { IsBackground = true, Name = $"registry writer {w}" });
            }

            for (int r = 0; r < readers; r++)
            {
                threads.Add(new Thread(() =>
                {
                    var seen = new bool[count, 4];
                    int lastAllCount = 0;
                    go.Wait();
                    Interlocked.Increment(ref readersLooping);
                    try
                    {
                        // One more sweep after the writers finish, so every class is also read from quiet tables.
                        bool lastSweep = false;
                        while (!lastSweep)
                        {
                            lastSweep = Volatile.Read(ref finished) == 1;
                            for (int i = 0; i < count; i++)
                            {
                                Check(seen, i, 0, DTypeRegistry.FromTypeNum(baseNum + i), "FromTypeNum");
                                Check(seen, i, 1, DTypeRegistry.FromName(names[i]), "FromName");
                                Check(seen, i, 2, DTypeRegistry.FromName(aliases[i]), "FromName(alias)");
                                Check(seen, i, 3, DTypeRegistry.FromScalarType(scalarTypes[i]), "FromScalarType");

                                // The builtins are present throughout, so a lookup that misses one ran into a table in the
                                // middle of a resize. Asked between every probe lookup, to be inside one when it happens.
                                if (!ReferenceEquals(DTypeRegistry.FromName("Float64DType"), DTypeRegistry.Double))
                                    Report("FromName(\"Float64DType\") lost the builtin class");
                                if (!ReferenceEquals(DTypeRegistry.FromScalarType(typeof(double)), DTypeRegistry.Double))
                                    Report("FromScalarType(typeof(double)) lost the builtin class");
                                if (!ReferenceEquals(DTypeRegistry.FromTypeNum(DTypeRegistry.Int16.type_num), DTypeRegistry.Int16))
                                    Report("FromTypeNum lost Int16");
                                if (!ReferenceEquals(DTypeRegistry.FromTypeCode(NPTypeCode.Int32), DTypeRegistry.Int32))
                                    Report("FromTypeCode(Int32) lost the builtin class");
                            }

                            int allCount = DTypeRegistry.All.Count;
                            if (allCount < lastAllCount)
                                Report($"All shrank from {lastAllCount} to {allCount}");
                            lastAllCount = allCount;
                        }
                    }
                    catch (Exception e)
                    {
                        Report($"reader: {e}");
                    }

                    void Check(bool[,] seenBy, int i, int lookup, DTypeMeta got, string what)
                    {
                        if (got is null)
                        {
                            if (seenBy[i, lookup])
                                Report($"{what} lost probe {i} after it was found");
                        }
                        else if (!ReferenceEquals(got, probes[i]))
                            Report($"{what} of probe {i} returned {got}");
                        else
                            seenBy[i, lookup] = true;
                    }
                }) { IsBackground = true, Name = $"registry reader {r}" });
            }

            foreach (var t in threads)
                t.Start();
            go.Set();
            foreach (var t in threads)
                t.Join(TimeSpan.FromMinutes(1)).Should().BeTrue("a registry thread must finish instead of hanging the run");

            problems.Should().BeEmpty();
            var all = new HashSet<DTypeMeta>(DTypeRegistry.All);
            for (int i = 0; i < count; i++)
            {
                DTypeRegistry.FromTypeNum(baseNum + i).Should().BeSameAs(probes[i]);
                DTypeRegistry.FromName(names[i]).Should().BeSameAs(probes[i]);
                DTypeRegistry.FromName(aliases[i]).Should().BeSameAs(probes[i]);
                DTypeRegistry.FromScalarType(scalarTypes[i]).Should().BeSameAs(probes[i]);
                all.Should().Contain(probes[i]);
            }

            void Report(string problem)
            {
                lock (problems)
                    if (problems.Count < 20)
                        problems.Add(problem);
            }
        }

        /// <summary>Checks that none of a refused class's keys resolve and that it is not listed.</summary>
        /// <param name="num">The refused class's type number.</param>
        /// <param name="name">Its name.</param>
        /// <param name="alias">Its alias.</param>
        /// <param name="refused">The refused class.</param>
        private static void AssertNotRegistered(int num, string name, string alias, DTypeMeta refused)
        {
            DTypeRegistry.FromTypeNum(num).Should().BeNull("a refused class must leave no type number behind");
            DTypeRegistry.FromName(name).Should().BeNull("a refused class must leave no name behind");
            DTypeRegistry.FromName(alias).Should().BeNull("a refused class must leave no alias behind");
            DTypeRegistry.All.Should().NotContain(refused);
        }

        /// <summary>Reserves <paramref name="count"/> consecutive unused probe type numbers.</summary>
        /// <param name="count">How many numbers to reserve.</param>
        /// <returns>The first number of the range.</returns>
        private static int Reserve(int count) => Interlocked.Add(ref s_nextTypeNum, count) - count;

        /// <summary>
        ///     A C# type no code outside these tests uses, distinct for each <paramref name="index"/> below 10,000:
        ///     <c>ScalarProbe&lt;T1, T2, T3, T4&gt;</c> over ten primitive types, one per decimal digit of the index. Probes
        ///     claim these as their scalar types, so the scalar-type table grows like the others.
        /// </summary>
        /// <param name="index">The type's index.</param>
        /// <returns>The closed generic type.</returns>
        private static Type ScalarProbeType(int index)
        {
            Type[] digits = { typeof(bool), typeof(byte), typeof(sbyte), typeof(short), typeof(ushort),
                              typeof(int), typeof(uint), typeof(long), typeof(ulong), typeof(float) };
            return typeof(ScalarProbe<,,,>).MakeGenericType(
                digits[index / 1000 % 10], digits[index / 100 % 10], digits[index / 10 % 10], digits[index % 10]);
        }

        /// <summary>A type that exists only to be a probe class's scalar type.</summary>
        /// <typeparam name="T1">Thousands digit of the probe index.</typeparam>
        /// <typeparam name="T2">Hundreds digit.</typeparam>
        /// <typeparam name="T3">Tens digit.</typeparam>
        /// <typeparam name="T4">Units digit.</typeparam>
        private struct ScalarProbe<T1, T2, T3, T4>
        {
        }

        /// <summary>
        ///     A registrable DType class that exists only to fill the registry's tables: abstract unless it is given a storage
        ///     code, in the <c>numsharp.tests</c> module.
        /// </summary>
        private sealed class RegistryProbeMeta : DTypeMeta
        {
            /// <summary>Builds a probe class.</summary>
            /// <param name="name">The class name, which the registry indexes and its messages quote.</param>
            /// <param name="typeNum">The type number, from <see cref="Reserve"/>.</param>
            /// <param name="typeCode">A storage code, to exercise the code table's checks; <see cref="NPTypeCode.Empty"/> for none.</param>
            /// <param name="scalarType">The C# scalar type the class claims, or null.</param>
            /// <param name="aliases">Alias names the registry also indexes, or null.</param>
            public RegistryProbeMeta(string name, int typeNum, NPTypeCode typeCode = NPTypeCode.Empty, Type scalarType = null, string[] aliases = null)
                : base(name, "numsharp.tests", typeNum, scalarType, scalarName: null, typeCode,
                    kind: 'V', typeChar: 'V', itemSize: -1, alignment: 1, DTypeFlags.NPY_DT_ABSTRACT, aliases)
            {
            }

            /// <summary>Not used by these tests: a probe class has no descriptors.</summary>
            /// <returns>Always null.</returns>
            public override DType default_descr() => null;

            /// <summary>Not used by these tests: a probe class takes part in no promotion.</summary>
            /// <param name="other">Ignored.</param>
            /// <returns>Always null (NumPy's <c>NotImplemented</c>).</returns>
            public override DTypeMeta common_dtype(DTypeMeta other) => null;
        }
    }
}
