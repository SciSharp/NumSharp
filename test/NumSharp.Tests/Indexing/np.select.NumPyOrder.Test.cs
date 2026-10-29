using System;

namespace NumSharp.Tests.Indexing;

/// <summary>
///     np.select as NumPy's statements run it, found when the np-layer dtype plan's P3 batch 7 moved its body to
///     <c>NDArray.Select</c> and replayed the snapshot against NumPy 2.4.2: the three broadcasts are
///     <c>np.broadcast_arrays</c> calls, whose mismatch text names the two operands that disagree, and a C# number literal
///     reaches np.copyto as the array <c>np.broadcast_arrays</c> makes of the Python scalar (int64 / float64 / complex128,
///     of the choice group's shape). Every expectation is NumPy 2.4.2's output.
/// </summary>
[TestClass]
public class SelectNumPyOrderTests
{
    private static readonly NDArray C1 = np.array(new[] { true, false, true, false });
    private static readonly NDArray C2 = np.array(new[] { false, true, false, true });

    /// <summary>The conditions are one broadcast group: two that disagree are its arguments 0 and 1.</summary>
    [TestMethod]
    public void ConditionGroup_Mismatch_NamesTheTwoConditions()
    {
        // np.select([np.zeros(3, bool), np.zeros(4, bool)], [1, 2])
        Action act = () => np.select(new[] { np.zeros(new Shape(3), np.@bool), np.zeros(new Shape(4), np.@bool) },
            new object[] { 1, 2 });
        act.Should().ThrowExactly<IncorrectShapeException>()
            .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  " +
                         "Mismatch is between arg 0 with shape (3,) and arg 1 with shape (4,).");
    }

    /// <summary>The choices and the default are the second group: the default is its last argument.</summary>
    [TestMethod]
    public void ChoiceGroup_Mismatch_NamesTheChoiceAndTheDefault()
    {
        // np.select([np.zeros((2, 3), bool)], [np.zeros(3)], default=np.zeros(4))
        Action act = () => np.select(new[] { np.zeros(new Shape(2, 3), np.@bool) }, new object[] { np.zeros(new Shape(3)) },
            np.zeros(new Shape(4)));
        act.Should().ThrowExactly<IncorrectShapeException>()
            .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  " +
                         "Mismatch is between arg 0 with shape (3,) and arg 1 with shape (4,).");
    }

    /// <summary>The two groups' shapes are the two arguments of the last broadcast.</summary>
    [TestMethod]
    public void GroupsAgainstEachOther_NameTheTwoGroupShapes()
    {
        // np.select([np.zeros(3, bool)], [np.zeros(4)])
        Action act = () => np.select(new[] { np.zeros(new Shape(3), np.@bool) }, new object[] { np.zeros(new Shape(4)) });
        act.Should().ThrowExactly<IncorrectShapeException>()
            .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  " +
                         "Mismatch is between arg 0 with shape (3,) and arg 1 with shape (4,).");

        // np.select([np.zeros(7, bool), np.zeros(7, bool)], [np.zeros((3, 4)), np.zeros((3, 4))])
        Action act2 = () => np.select(new[] { np.zeros(new Shape(7), np.@bool), np.zeros(new Shape(7), np.@bool) },
            new object[] { np.zeros(new Shape(3, 4)), np.zeros(new Shape(3, 4)) });
        act2.Should().ThrowExactly<IncorrectShapeException>()
            .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  " +
                         "Mismatch is between arg 0 with shape (7,) and arg 1 with shape (3, 4).");
    }

    /// <summary>
    ///     A C# int beside a uint8 array is NumPy's Python int: the result is uint8, and np.copyto receives the int64 array
    ///     np.broadcast_arrays made of it, which same_kind refuses (the C# literal's int32 was refused with another text).
    /// </summary>
    [TestMethod]
    public void IntLiteral_BesideUInt8Array_IsAnInt64ArrayToCopyto()
    {
        // np.select([c1, c2], [np.arange(4, dtype=np.uint8), 5])
        Action act = () => np.select(new[] { C1, C2 }, new object[] { np.arange(4).astype(np.uint8), 5 });
        act.Should().ThrowExactly<InvalidCastException>()
            .WithMessage("Cannot cast array data from dtype('int64') to dtype('uint8') according to the rule 'same_kind'");
    }

    /// <summary>
    ///     A C# byte is a Python int too, whatever its C# width: NumPy refuses it beside a uint8 array, where NumSharp
    ///     copied the literal's own uint8 array in.
    /// </summary>
    [TestMethod]
    public void ByteLiteral_IsAPythonInt_NotAUInt8()
    {
        // np.select([c1, c2], [np.arange(4, dtype=np.uint8), 5])  (the C# literal (byte)5)
        Action act = () => np.select(new[] { C1, C2 }, new object[] { np.arange(4).astype(np.uint8), (byte)5 });
        act.Should().ThrowExactly<InvalidCastException>()
            .WithMessage("Cannot cast array data from dtype('int64') to dtype('uint8') according to the rule 'same_kind'");
    }

    /// <summary>Every choice and the default 0-d: the choice reaches np.copyto as a 0-d array, NumPy's "scalar" text.</summary>
    [TestMethod]
    public void AllChoicesZeroD_RefusedCastReadsScalar()
    {
        // np.select([c1, c2], [np.array(3, dtype=np.uint8), 5])
        Action act = () => np.select(new[] { C1, C2 }, new object[] { np.array((byte)3), 5 });
        act.Should().ThrowExactly<InvalidCastException>()
            .WithMessage("Cannot cast scalar from dtype('int64') to dtype('uint8') according to the rule 'same_kind'");
    }

    /// <summary>The same_kind cast that NumPy accepts still runs: an int beside an int8 array, a float beside float32.</summary>
    [TestMethod]
    public void SameKindLiterals_AreCopiedIn()
    {
        // np.select([c1, c2], [np.arange(4, dtype=np.int8), 5], default=1000) -> int8 [0, 5, 2, 5]
        var r = np.select(new[] { C1, C2 }, new object[] { np.arange(4).astype(np.int8), 5 }, 1000);
        r.dtype.Should().Be(np.int8);
        r.ToArray<sbyte>().Should().Equal((sbyte)0, (sbyte)5, (sbyte)2, (sbyte)5);

        // np.select([c1, c2], [np.arange(4, dtype=np.float32), 2.5]) -> float32 [0, 2.5, 2, 2.5]
        var f = np.select(new[] { C1, C2 }, new object[] { np.arange(4).astype(np.float32), 2.5 });
        f.dtype.Should().Be(np.float32);
        f.ToArray<float>().Should().Equal(0f, 2.5f, 2f, 2.5f);
    }
}
