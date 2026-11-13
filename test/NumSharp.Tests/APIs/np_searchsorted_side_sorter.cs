using System;

namespace NumSharp.Tests.APIs;

/// <summary>
/// Tests for np.searchsorted side / sorter / multidim parameters added per audit T1.27 follow-up.
/// All expected values produced from NumPy 2.x and pasted in the comment above each assertion.
/// </summary>
[TestClass]
public class NpSearchsortedSideSorterTests
{
    #region side='left' vs side='right' on duplicates

    [TestMethod]
    public void Side_Left_DefaultIsLeft()
    {
        // np.searchsorted([1,2,2,2,3], 2) -> 1
        var a = np.array(new[] { 1, 2, 2, 2, 3 });
        Assert.AreEqual(1L, np.searchsorted(a, 2));
        Assert.AreEqual(1L, np.searchsorted(a, 2, side: "left"));
    }

    [TestMethod]
    public void Side_Right_Duplicates()
    {
        // np.searchsorted([1,2,2,2,3], 2, side='right') -> 4
        var a = np.array(new[] { 1, 2, 2, 2, 3 });
        Assert.AreEqual(4L, np.searchsorted(a, 2, side: "right"));
    }

    [TestMethod]
    public void Side_Left_AllDuplicatesMatch()
    {
        // np.searchsorted([5,5,5], 5, side='left') -> 0
        var a = np.array(new[] { 5, 5, 5 });
        Assert.AreEqual(0L, np.searchsorted(a, 5, side: "left"));
    }

    [TestMethod]
    public void Side_Right_AllDuplicatesMatch()
    {
        // np.searchsorted([5,5,5], 5, side='right') -> 3
        var a = np.array(new[] { 5, 5, 5 });
        Assert.AreEqual(3L, np.searchsorted(a, 5, side: "right"));
    }

    [TestMethod]
    public void Side_Right_NoMatch()
    {
        // np.searchsorted([1,3,5], 3, side='right') -> 2
        var a = np.array(new[] { 1, 3, 5 });
        Assert.AreEqual(2L, np.searchsorted(a, 3, side: "right"));
    }

    [TestMethod]
    public void Side_Right_FloatDuplicates()
    {
        // np.searchsorted([1.0, 2.0, 2.0, 3.0], 2.0, side='right') -> 3
        var a = np.array(new[] { 1.0, 2.0, 2.0, 3.0 });
        Assert.AreEqual(3L, np.searchsorted(a, 2.0, side: "right"));
        Assert.AreEqual(1L, np.searchsorted(a, 2.0, side: "left"));
    }

    [TestMethod]
    public void Side_NDArrayValues_Right()
    {
        // np.searchsorted([1,2,2,2,3], [2,3], side='right') -> [4, 5]
        var a = np.array(new[] { 1, 2, 2, 2, 3 });
        var v = np.array(new[] { 2, 3 });
        var r = np.searchsorted(a, v, side: "right");
        Assert.AreEqual(2, r.size);
        Assert.AreEqual(4L, r.GetInt64(0));
        Assert.AreEqual(5L, r.GetInt64(1));
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Side_InvalidValue_Throws()
    {
        // NumPy raises ValueError for invalid side
        var a = np.array(new[] { 1, 2, 3 });
        np.searchsorted(a, 2, side: "middle");
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Side_EmptyString_Throws()
    {
        var a = np.array(new[] { 1, 2, 3 });
        np.searchsorted(a, 2, side: "");
    }

    #endregion

    #region sorter parameter

    [TestMethod]
    public void Sorter_BasicScalar()
    {
        // a = [40,10,20,30], sorter=argsort(a)=[1,2,3,0]
        // np.searchsorted(a, 25, sorter=sorter) -> 2
        var a = np.array(new[] { 40, 10, 20, 30 });
        var sorter = np.array(new[] { 1L, 2L, 3L, 0L });
        Assert.AreEqual(2L, np.searchsorted(a, 25, sorter: sorter));
    }

    [TestMethod]
    public void Sorter_ArrayValues()
    {
        // np.searchsorted([40,10,20,30], [25,40,15], sorter=[1,2,3,0]) -> [2,3,1]
        var a = np.array(new[] { 40, 10, 20, 30 });
        var sorter = np.array(new[] { 1L, 2L, 3L, 0L });
        var v = np.array(new[] { 25, 40, 15 });
        var r = np.searchsorted(a, v, sorter: sorter);
        Assert.AreEqual(3, r.size);
        Assert.AreEqual(2L, r.GetInt64(0));
        Assert.AreEqual(3L, r.GetInt64(1));
        Assert.AreEqual(1L, r.GetInt64(2));
    }

    [TestMethod]
    public void Sorter_WithSideLeft()
    {
        // a=[40,10,20,30] (sorted=[10,20,30,40]), sorter=[1,2,3,0]
        // np.searchsorted(a, 30, sorter=sorter, side='left') -> 2
        var a = np.array(new[] { 40, 10, 20, 30 });
        var sorter = np.array(new[] { 1L, 2L, 3L, 0L });
        Assert.AreEqual(2L, np.searchsorted(a, 30, side: "left", sorter: sorter));
    }

    [TestMethod]
    public void Sorter_WithSideRight()
    {
        // a=[40,10,20,30] (sorted=[10,20,30,40]), sorter=[1,2,3,0]
        // np.searchsorted(a, 30, sorter=sorter, side='right') -> 3
        var a = np.array(new[] { 40, 10, 20, 30 });
        var sorter = np.array(new[] { 1L, 2L, 3L, 0L });
        Assert.AreEqual(3L, np.searchsorted(a, 30, side: "right", sorter: sorter));
    }

    [TestMethod]
    public void Sorter_DoubleValue()
    {
        // a=[40,10,20,30], sorter=[1,2,3,0]
        // np.searchsorted(a, 25.5, sorter=sorter) -> 2
        var a = np.array(new[] { 40, 10, 20, 30 });
        var sorter = np.array(new[] { 1L, 2L, 3L, 0L });
        Assert.AreEqual(2L, np.searchsorted(a, 25.5, sorter: sorter));
    }

    [TestMethod]
    public void Sorter_NDArrayValues_DuplicatesRight()
    {
        // a = [3,1,2,1,3] (sorter = [1,3,2,0,4] -> sorted [1,1,2,3,3])
        // np.searchsorted(a, [1,3], sorter=sorter, side='right') -> [2, 5]
        var a = np.array(new[] { 3, 1, 2, 1, 3 });
        var sorter = np.array(new[] { 1L, 3L, 2L, 0L, 4L });
        var v = np.array(new[] { 1, 3 });
        var r = np.searchsorted(a, v, side: "right", sorter: sorter);
        Assert.AreEqual(2, r.size);
        Assert.AreEqual(2L, r.GetInt64(0));
        Assert.AreEqual(5L, r.GetInt64(1));
    }

    [TestMethod]
    [ExpectedException(typeof(ValueError))]
    public void Sorter_WrongSize_Throws()
    {
        // NumPy: ValueError "sorter.size must equal a.size"
        var a = np.array(new[] { 1, 2, 3 });
        var sorter = np.array(new[] { 0L, 1L });
        np.searchsorted(a, 2, sorter: sorter);
    }

    [TestMethod]
    [ExpectedException(typeof(ValueError))]
    public void Sorter_EmptyForNonEmptyA_Throws()
    {
        // NumPy: ValueError "sorter.size must equal a.size"
        var a = np.array(new[] { 1, 2, 3 });
        var sorter = np.array(Array.Empty<long>());
        np.searchsorted(a, 2, sorter: sorter);
    }

    [TestMethod]
    [ExpectedException(typeof(TypeError))]
    public void Sorter_MultidimSorter_Throws()
    {
        // NumPy: TypeError "could not parse sorter argument" (the sorter's depth check, before its size)
        var a = np.array(new[] { 1, 2, 3, 4 });
        var sorter = np.array(new long[,] { { 0, 1 }, { 2, 3 } });
        np.searchsorted(a, 2, sorter: sorter);
    }

    /// <summary>
    ///     A sorter entry outside the haystack is NumPy's <c>ValueError: Sorter index out of range.</c> when the bisection
    ///     reaches it (argbinsearch checks each entry it reads before the entry addresses the haystack), and never read, so
    ///     never an error, when it does not. NumSharp's kernel read the haystack at the entry: memory outside it, an answer
    ///     that changed from run to run. NumPy 2.5.4's answers for every overload, both sides, an int32 sorter, a strided
    ///     haystack and a huge index. NumPy 2.5's batched bisection reads the middle entry <c>sorter[a.size &gt;&gt; 1]</c>
    ///     before any key, so a bad middle entry fails even with no key (2.4.2 returned the empty result).
    /// </summary>
    [TestMethod]
    public void Sorter_IndexOutsideTheHaystack_IsNumPysValueErrorWhenReached()
    {
        var a = np.array(new long[] { 10, 20, 30, 40 });
        var oob = np.array(new long[] { 0, 1, 2, 4 });
        const string text = "Sorter index out of range.";
        void Refused(Action call) => new Action(call).Should().Throw<ValueError>().WithMessage(text);

        Refused(() => np.searchsorted(a, 35, sorter: oob));
        Refused(() => np.searchsorted(a, 35.0, sorter: oob));
        Refused(() => np.searchsorted(a, np.array(new long[] { 15, 35 }), sorter: oob));
        Refused(() => np.searchsorted(a, 30, side: "right", sorter: oob));
        Refused(() => np.searchsorted(a, 35, sorter: np.array(new long[] { 0, 1, 2, -1 })));
        Refused(() => np.searchsorted(a, 35, sorter: np.array(new long[] { 0, 1, 2, 1L << 40 })));
        Refused(() => np.searchsorted(a, 35, sorter: np.array(new[] { 0, 1, 2, 4 })));
        Refused(() => np.searchsorted(np.array(new long[] { 10, 0, 20, 0, 30, 0, 40, 0 })["::2"], 35, sorter: oob));

        // Not reached: the bisection for 15 reads sorter[2], sorter[1] and sorter[0] only.
        np.searchsorted(a, 15, sorter: oob).Should().Be(1);
        np.searchsorted(a, np.array(new long[] { 15, 12 }), sorter: oob).ToArray<long>().Should().Equal(1L, 1L);

        // The middle entry (sorter[2]) is read before the first key: no key, still refused...
        Refused(() => np.searchsorted(a, np.array(Array.Empty<long>()), sorter: np.array(new long[] { 0, 1, 4, 2 })));
        // ...while a bad FIRST entry is read only by a key that bisects down to it (12 does, 45 does not).
        np.searchsorted(a, np.array(Array.Empty<long>()), sorter: np.array(new long[] { 4, 1, 2, 3 })).size.Should().Be(0);
        Refused(() => np.searchsorted(a, 12, sorter: np.array(new long[] { 5, 1, 2, 3 })));
        np.searchsorted(a, 45, sorter: np.array(new long[] { 5, 1, 2, 3 })).Should().Be(4);
        // A bad LAST entry: 45 reaches it, 12 does not, and an array of keys fails as soon as one key reaches it.
        np.searchsorted(a, 12, sorter: np.array(new long[] { 0, 1, 2, 9 })).Should().Be(1);
        Refused(() => np.searchsorted(a, 45, sorter: np.array(new long[] { 0, 1, 2, 9 })));
        Refused(() => np.searchsorted(a, np.array(new long[] { 12, 45 }), sorter: np.array(new long[] { 0, 1, 2, 9 })));
        Refused(() => np.searchsorted(a, np.array(new long[] { 45, 12 }), sorter: np.array(new long[] { 0, 1, 2, 9 })));
    }

    /// <summary>
    ///     Keys and a sorter that are C-contiguous views at a non-zero offset (np.split children) are read from the view, for
    ///     the scalar keys as for the array ones (gap VA01: the kernel calls passed the storage's start). NumPy 2.4.2: with
    ///     a = [5, 1, 3, 2, 8] and the sorter [1, 3, 2, 0, 4] (offset 2 of its owner), 7.5 and 2 on the right are 4 and 2.
    /// </summary>
    [TestMethod]
    public void Sorter_ViewAtAnOffset_IsReadFromTheView_ForEveryKeyForm()
    {
        var a = np.array(new double[] { 5, 1, 3, 2, 8 });
        var sorter = np.split(np.array(new long[] { 7, 7, 1, 3, 2, 0, 4 }), new[] { 2 })[1];
        np.searchsorted(a, 7.5, side: "right", sorter: sorter).Should().Be(4);
        np.searchsorted(a, 2, side: "right", sorter: sorter).Should().Be(2);
        var keys = np.split(np.array(new double[] { 9, 2, 7.5 }), new[] { 1 })[1];
        np.searchsorted(a, keys, "right", sorter).ToArray<long>().Should().Equal(2L, 4L);
    }

    #endregion

    #region Multidim 'a' validation

    [TestMethod]
    [ExpectedException(typeof(ValueError))]
    public void Multidim_A_2D_Throws()
    {
        // NumPy: ValueError "object too deep for desired array"
        var a = np.arange(20).reshape(4, 5);
        np.searchsorted(a, 5);
    }

    [TestMethod]
    [ExpectedException(typeof(ValueError))]
    public void Multidim_A_3D_Throws()
    {
        // NumPy: ValueError "object too deep for desired array"
        var a = np.arange(24).reshape(2, 3, 4);
        np.searchsorted(a, 5);
    }

    [TestMethod]
    [ExpectedException(typeof(ValueError))]
    public void Multidim_A_2D_WithNDArrayV_Throws()
    {
        // NumPy: ValueError "object too deep for desired array"
        var a = np.arange(20).reshape(4, 5);
        var v = np.array(new[] { 5, 10 });
        np.searchsorted(a, v);
    }

    #endregion

    #region Multidim 'v' preserves shape

    [TestMethod]
    public void Multidim_V_2D_PreservesShape()
    {
        // np.searchsorted([1,2,3,4,5], [[1,2],[3,4]]) -> [[0,1],[2,3]] shape (2,2)
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var v = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var r = np.searchsorted(a, v);
        CollectionAssert.AreEqual(new long[] { 2, 2 }, r.shape);
        Assert.AreEqual(0L, r.GetInt64(0, 0));
        Assert.AreEqual(1L, r.GetInt64(0, 1));
        Assert.AreEqual(2L, r.GetInt64(1, 0));
        Assert.AreEqual(3L, r.GetInt64(1, 1));
    }

    [TestMethod]
    public void Multidim_V_2D_RightSide()
    {
        // np.searchsorted([1,2,3,4,5], [[1,2],[3,4]], side='right') -> [[1,2],[3,4]]
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        var v = np.array(new[,] { { 1, 2 }, { 3, 4 } });
        var r = np.searchsorted(a, v, side: "right");
        CollectionAssert.AreEqual(new long[] { 2, 2 }, r.shape);
        Assert.AreEqual(1L, r.GetInt64(0, 0));
        Assert.AreEqual(2L, r.GetInt64(0, 1));
        Assert.AreEqual(3L, r.GetInt64(1, 0));
        Assert.AreEqual(4L, r.GetInt64(1, 1));
    }

    [TestMethod]
    public void Multidim_V_3D_PreservesShape()
    {
        // a = [10,20,30,40], v = arange(24).reshape(2,3,4)
        // np.searchsorted(a, v) reshape (2,3,4):
        //   [[[0,0,0,0],[0,0,0,0],[0,0,0,1]],
        //    [[1,1,1,1],[1,1,1,1],[1,2,2,2]]]
        var a = np.array(new[] { 10, 20, 30, 40 });
        var v = np.arange(24).reshape(2, 3, 4);
        var r = np.searchsorted(a, v);
        CollectionAssert.AreEqual(new long[] { 2, 3, 4 }, r.shape);

        // First plane: values 0..11
        Assert.AreEqual(0L, r.GetInt64(0, 0, 0));   // v=0  -> insert at 0
        Assert.AreEqual(0L, r.GetInt64(0, 2, 0));   // v=8  -> insert at 0 (8 < 10)
        Assert.AreEqual(0L, r.GetInt64(0, 2, 1));   // v=9  -> 0
        Assert.AreEqual(1L, r.GetInt64(0, 2, 3));   // v=11 -> 1 (between 10 and 20)
        // Second plane: values 12..23
        Assert.AreEqual(1L, r.GetInt64(1, 0, 0));   // v=12 -> 1
        Assert.AreEqual(1L, r.GetInt64(1, 2, 0));   // v=20 -> 1 (left side, 20 found at idx 1)
        Assert.AreEqual(2L, r.GetInt64(1, 2, 1));   // v=21 -> 2
        Assert.AreEqual(2L, r.GetInt64(1, 2, 2));   // v=22 -> 2
        Assert.AreEqual(2L, r.GetInt64(1, 2, 3));   // v=23 -> 2
    }

    [TestMethod]
    public void Multidim_V_Empty_PreservesShape()
    {
        // np.searchsorted([1,2,3], np.array([], dtype=int).reshape(0,3)) -> shape (0,3)
        var a = np.array(new[] { 1, 2, 3 });
        var v = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
        var r = np.searchsorted(a, v);
        CollectionAssert.AreEqual(new long[] { 0, 3 }, r.shape);
        Assert.AreEqual(0, r.size);
    }

    #endregion

    #region Empty 'a'

    [TestMethod]
    public void EmptyA_ScalarV_ReturnsZero_BothSides()
    {
        // np.searchsorted([], 5) -> 0 for both sides
        var a = np.array(Array.Empty<int>());
        Assert.AreEqual(0L, np.searchsorted(a, 5));
        Assert.AreEqual(0L, np.searchsorted(a, 5, side: "right"));
    }

    [TestMethod]
    public void EmptyA_ArrayV_AllZeros()
    {
        // np.searchsorted([], [1, 2, 3]) -> [0, 0, 0]
        var a = np.array(Array.Empty<int>());
        var v = np.array(new[] { 1, 2, 3 });
        var r = np.searchsorted(a, v);
        Assert.AreEqual(3, r.size);
        Assert.AreEqual(0L, r.GetInt64(0));
        Assert.AreEqual(0L, r.GetInt64(1));
        Assert.AreEqual(0L, r.GetInt64(2));
    }

    #endregion

    #region Dtype coverage

    [TestMethod]
    public void Dtype_Float_Right()
    {
        // np.searchsorted([1.0f, 2.0f, 2.0f, 3.0f], 2.0, side='right') -> 3
        var a = np.array(new float[] { 1.0f, 2.0f, 2.0f, 3.0f });
        Assert.AreEqual(3L, np.searchsorted(a, 2.0, side: "right"));
    }

    [TestMethod]
    public void Dtype_Int64_Right()
    {
        // np.searchsorted(int64[1,2,2,3], 2, 'right') -> 3
        var a = np.array(new[] { 1L, 2L, 2L, 3L });
        Assert.AreEqual(3L, np.searchsorted(a, 2, side: "right"));
    }

    [TestMethod]
    public void Dtype_UInt32_Right()
    {
        var a = np.array(new uint[] { 1, 2, 2, 3 });
        Assert.AreEqual(3L, np.searchsorted(a, 2, side: "right"));
    }

    [TestMethod]
    public void Dtype_Int16_Right()
    {
        var a = np.array(new short[] { 1, 2, 2, 3 });
        Assert.AreEqual(3L, np.searchsorted(a, 2, side: "right"));
    }

    [TestMethod]
    public void Dtype_Byte_Right()
    {
        var a = np.array(new byte[] { 1, 2, 2, 3 });
        Assert.AreEqual(3L, np.searchsorted(a, 2, side: "right"));
    }

    [TestMethod]
    public void Dtype_Half_Right()
    {
        var a = np.array(new[] { (Half)1.0f, (Half)2.0f, (Half)2.0f, (Half)3.0f });
        Assert.AreEqual(3L, np.searchsorted(a, 2.0, side: "right"));
        Assert.AreEqual(1L, np.searchsorted(a, 2.0, side: "left"));
    }

    #endregion

    #region View / strided 'a'

    [TestMethod]
    public void Sliced_A_Works()
    {
        // a = [0,1,2,3,4,5,6,7,8,9][2:8] = [2,3,4,5,6,7]
        // searchsorted slice for 5 -> 3 (insert at index 3 of the slice)
        var a = np.arange(10);
        var slice = a["2:8"];
        Assert.AreEqual(3L, np.searchsorted(slice, 5));
        Assert.AreEqual(4L, np.searchsorted(slice, 5, side: "right"));
    }

    [TestMethod]
    public void Strided_A_StepWorks()
    {
        // a = arange(10)[::2] = [0,2,4,6,8]
        // searchsorted for 5 -> 3 (between 4 and 6)
        var a = np.arange(10)["::2"];
        Assert.AreEqual(3L, np.searchsorted(a, 5));
        Assert.AreEqual(3L, np.searchsorted(a, 5, side: "right"));
    }

    [TestMethod]
    public void Sliced_V_PreservesValues()
    {
        // a=[10,20,30], v=arange(10)[1:6:2]=[1,3,5]
        // np.searchsorted([10,20,30], [1,3,5]) -> [0,0,0]
        var a = np.array(new[] { 10, 20, 30 });
        var v = np.arange(10)["1:6:2"];
        var r = np.searchsorted(a, v);
        Assert.AreEqual(3, r.size);
        Assert.AreEqual(0L, r.GetInt64(0));
        Assert.AreEqual(0L, r.GetInt64(1));
        Assert.AreEqual(0L, r.GetInt64(2));
    }

    #endregion

    #region Unsorted haystack (NumPy 2.5's batched bisection)

    /// <summary>
    ///     An unsorted haystack has no single insertion point, so the answer is whichever pivots the bisection visits:
    ///     NumPy 2.5 (gh-30517) runs every key through one shared interval length, halved per pass (median first, then
    ///     the quartiles, ...), where 2.4 bisected each key on its own and carried the previous key's bound. np.unique's
    ///     NaN probe over a masked array's raw data searches such a haystack. NumPy 2.5.4's answers for both sides, with
    ///     and without a sorter (2.4.2's differ in every row but the first keys), and a 1-element haystack.
    /// </summary>
    [TestMethod]
    public void Unsorted_Int64_IsNumPy25sBatchedBisection()
    {
        var u = np.array(new long[] { 3, 1, 2, 5, 4, 0, 7 });
        var keys = np.array(new long[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 1, 0 });
        np.searchsorted(u, keys).ToArray<long>().Should().Equal(0L, 0, 2, 3, 3, 3, 6, 6, 7, 2, 0, 0);
        np.searchsorted(u, keys, side: "right").ToArray<long>().Should().Equal(0L, 2, 3, 3, 3, 6, 6, 7, 7, 3, 2, 0);

        // The sorter is not a sorting permutation of u either: positions are bisected, each read through it.
        var sorter = np.array(new long[] { 6, 1, 4, 3, 0, 2, 5 });
        np.searchsorted(u, keys, sorter: sorter).ToArray<long>().Should().Equal(0L, 0, 2, 2, 2, 3, 7, 7, 7, 2, 0, 0);
        np.searchsorted(u, keys, "right", sorter).ToArray<long>().Should().Equal(0L, 2, 2, 2, 3, 7, 7, 7, 7, 2, 2, 0);

        // One element: half is 0, so the first pass writes 0 and the last pass alone decides.
        np.searchsorted(np.array(new long[] { 5 }), np.array(new long[] { 1, 5, 9 })).ToArray<long>().Should().Equal(0L, 0, 1);
        np.searchsorted(np.array(new long[] { 5 }), np.array(new long[] { 1, 5, 9 }), side: "right").ToArray<long>()
            .Should().Equal(0L, 1, 1);
    }

    /// <summary>
    ///     The NaN-aware comparators over an unsorted haystack (NaN sorts last: a NaN pivot is "greater" than every finite
    ///     key, a NaN key greater than every finite pivot): float64, float16 and complex128 (CDOUBLE_LT's lexicographic
    ///     order), and bool. NumPy 2.5.4's answers; a scalar key runs the same schedule with one key.
    /// </summary>
    [TestMethod]
    public void Unsorted_NaNAwareDtypes_AreNumPy25sBatchedBisection()
    {
        var f = np.array(new[] { double.NaN, 1.0, double.NaN, 3.0, 2.0, double.NaN, 0.5 });
        var fk = np.array(new[] { double.NaN, 0.0, 1.0, 2.5, 3.5, double.NaN, 0.75 });
        np.searchsorted(f, fk).ToArray<long>().Should().Equal(5L, 0, 0, 2, 5, 5, 0);
        np.searchsorted(f, fk, side: "right").ToArray<long>().Should().Equal(7L, 0, 2, 2, 5, 7, 0);
        np.searchsorted(f, 0.5).Should().Be(0);

        var h = np.array(new[] { (Half)3, (Half)1, (Half)2, Half.NaN, (Half)0 });
        var hk = np.array(new[] { (Half)0, (Half)1, (Half)2, (Half)3, Half.NaN });
        np.searchsorted(h, hk).ToArray<long>().Should().Equal(0L, 0, 2, 3, 3);

        var c = np.array(new[]
        {
            new System.Numerics.Complex(1, 1), new System.Numerics.Complex(0, 2), new System.Numerics.Complex(3, 0),
            new System.Numerics.Complex(double.NaN, 1), new System.Numerics.Complex(2, 2),
        });
        var ck = np.array(new[]
        {
            new System.Numerics.Complex(0, 0), new System.Numerics.Complex(1, 1), new System.Numerics.Complex(2, 2),
            new System.Numerics.Complex(1, double.NaN), new System.Numerics.Complex(double.NaN, 0),
        });
        np.searchsorted(c, ck).ToArray<long>().Should().Equal(0L, 2, 2, 3, 3);
        np.searchsorted(c, ck, side: "right").ToArray<long>().Should().Equal(0L, 2, 2, 3, 3);

        var b = np.array(new[] { true, false, true, false, false });
        var bk = np.array(new[] { false, true });
        np.searchsorted(b, bk).ToArray<long>().Should().Equal(0L, 2);
        np.searchsorted(b, bk, side: "right").ToArray<long>().Should().Equal(2L, 5);
    }

    #endregion
}
