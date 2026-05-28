#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;
using NumSharp.Backends.Iteration;

Console.WriteLine("=== NpyIter Behavioral Parity Audit ===\n");

int passed = 0, failed = 0;
var failures = new List<string>();

void Test(string name, bool condition, string details = "")
{
    if (condition) { passed++; Console.WriteLine("OK: " + name); }
    else { failed++; failures.Add(name + ": " + details); Console.WriteLine("FAIL: " + name + " - " + details); }
}

// Test Case 1: Basic_3x4_CIndex
Console.WriteLine("\n--- Test 1: Basic_3x4_CIndex ---");
var arr1 = np.arange(12).reshape(3, 4);
using (var it1 = NpyIterRef.New(arr1, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX))
{
    Test("ndim=2", it1.NDim == 2, "got " + it1.NDim);
    Test("itersize=12", it1.IterSize == 12, "got " + it1.IterSize);

    it1.GotoMultiIndex(new long[] { 0, 0 });
    Test("(0,0) c_index=0", it1.GetIndex() == 0, "got " + it1.GetIndex());

    it1.GotoMultiIndex(new long[] { 1, 0 });
    Test("(1,0) c_index=4", it1.GetIndex() == 4, "got " + it1.GetIndex());

    it1.GotoMultiIndex(new long[] { 2, 3 });
    Test("(2,3) c_index=11", it1.GetIndex() == 11, "got " + it1.GetIndex());
}

// Test Case 2: Basic_3x4_FIndex
Console.WriteLine("\n--- Test 2: Basic_3x4_FIndex ---");
var arr2 = np.arange(12).reshape(3, 4);
using (var it2 = NpyIterRef.New(arr2, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.F_INDEX))
{
    it2.GotoMultiIndex(new long[] { 0, 1 });
    Test("(0,1) f_index=3", it2.GetIndex() == 3, "got " + it2.GetIndex());

    it2.GotoMultiIndex(new long[] { 1, 0 });
    Test("(1,0) f_index=1", it2.GetIndex() == 1, "got " + it2.GetIndex());

    it2.GotoMultiIndex(new long[] { 2, 3 });
    Test("(2,3) f_index=11", it2.GetIndex() == 11, "got " + it2.GetIndex());
}

// Test Case 3: Sliced Array
Console.WriteLine("\n--- Test 3: Sliced ---");
var arr3 = np.arange(20).reshape(4, 5);
var sliced = arr3["::2, 1:4"];
Test("shape=(2,3)", sliced.Shape.Equals(new Shape(2, 3)), "got " + sliced.Shape);
using (var it3 = NpyIterRef.New(sliced, NpyIterGlobalFlags.MULTI_INDEX))
{
    Test("ndim=2", it3.NDim == 2, "got " + it3.NDim);
    Test("itersize=6", it3.IterSize == 6, "got " + it3.IterSize);

    var expected = new[] { 1, 2, 3, 11, 12, 13 };
    var values = new List<int>();
    do
    {
        unsafe { values.Add(*(int*)it3.GetDataPtrArray()[0]); }
    } while (it3.Iternext());
    Test("values match", values.SequenceEqual(expected), "got [" + string.Join(",", values) + "]");
}

// Test Case 4: Transposed
Console.WriteLine("\n--- Test 4: Transposed ---");
var arr4 = np.arange(24).reshape(2, 3, 4);
var trans = np.transpose(arr4, new[] { 2, 0, 1 });
Test("shape=(4,2,3)", trans.Shape.Equals(new Shape(4, 2, 3)), "got " + trans.Shape);
using (var it4 = NpyIterRef.New(trans, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX))
{
    Test("ndim=3", it4.NDim == 3, "got " + it4.NDim);

    it4.GotoMultiIndex(new long[] { 0, 0, 0 });
    Test("(0,0,0) c_index=0", it4.GetIndex() == 0, "got " + it4.GetIndex());

    it4.GotoMultiIndex(new long[] { 1, 0, 0 });
    Test("(1,0,0) c_index=6", it4.GetIndex() == 6, "got " + it4.GetIndex());

    it4.GotoMultiIndex(new long[] { 3, 1, 2 });
    Test("(3,1,2) c_index=23", it4.GetIndex() == 23, "got " + it4.GetIndex());
}

// Test Case 5: Reversed
Console.WriteLine("\n--- Test 5: Reversed ---");
var arr5 = np.arange(10);
var rev = arr5["::-1"];
using (var it5 = NpyIterRef.New(rev, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX))
{
    Test("ndim=1", it5.NDim == 1, "got " + it5.NDim);
    Test("itersize=10", it5.IterSize == 10, "got " + it5.IterSize);

    var coords = new long[1];
    it5.GetMultiIndex(coords);
    Test("first multi_index=9", coords[0] == 9, "got " + coords[0]);
    Test("first c_index=9", it5.GetIndex() == 9, "got " + it5.GetIndex());

    it5.Reset();
    var values = new List<int>();
    do { unsafe { values.Add(*(int*)it5.GetDataPtrArray()[0]); } } while (it5.Iternext());
    var expected = new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    Test("values match memory order", values.SequenceEqual(expected), "got [" + string.Join(",", values) + "]");
}

// Test Case 6: Broadcast
Console.WriteLine("\n--- Test 6: Broadcast ---");
var a6 = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
var b6 = np.array(new int[,] { { 10, 20, 30 } });
using (var it6 = NpyIterRef.MultiNew(2, new[] { a6, b6 }, NpyIterGlobalFlags.MULTI_INDEX,
    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
    new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY }))
{
    Test("ndim=2", it6.NDim == 2, "got " + it6.NDim);
    Test("itersize=9", it6.IterSize == 9, "got " + it6.IterSize);

    var pairs = new List<(int, int)>();
    do
    {
        unsafe
        {
            var p = it6.GetDataPtrArray();
            pairs.Add((*(int*)p[0], *(int*)p[1]));
        }
    } while (it6.Iternext());
    var expected = new[] { (1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30), (3, 10), (3, 20), (3, 30) };
    Test("pairs match", pairs.SequenceEqual(expected), "got " + string.Join(", ", pairs));
}

// Test Case 7: Coalescing
Console.WriteLine("\n--- Test 7: Coalesced ---");
var arr7 = np.arange(24).reshape(2, 3, 4);
using (var it7 = NpyIterRef.New(arr7))
{
    Test("ndim=1 (coalesced)", it7.NDim == 1, "got " + it7.NDim);
    Test("itersize=24", it7.IterSize == 24, "got " + it7.IterSize);
}

// Test Case 8: K-Order Strided
Console.WriteLine("\n--- Test 8: K-Order Strided ---");
var arr8 = np.arange(24).reshape(2, 3, 4);
var strided8 = arr8[":, ::2, :"];
using (var it8 = NpyIterRef.AdvancedNew(1, new[] { strided8 }, NpyIterGlobalFlags.MULTI_INDEX,
    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING, new[] { NpyIterPerOpFlags.READONLY }))
{
    Test("ndim=3", it8.NDim == 3, "got " + it8.NDim);
    Test("itersize=16", it8.IterSize == 16, "got " + it8.IterSize);

    var values = new List<int>();
    do { unsafe { values.Add(*(int*)it8.GetDataPtrArray()[0]); } } while (it8.Iternext());
    var expected = new[] { 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23 };
    Test("values match K-order", values.SequenceEqual(expected), "got [" + string.Join(",", values) + "]");
}

// Test Case 9: High-Dim 5D
Console.WriteLine("\n--- Test 9: HighDim 5D ---");
var arr9 = np.arange(32).reshape(2, 2, 2, 2, 2);
using (var it9 = NpyIterRef.New(arr9, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX))
{
    Test("ndim=5", it9.NDim == 5, "got " + it9.NDim);
    Test("itersize=32", it9.IterSize == 32, "got " + it9.IterSize);

    it9.GotoMultiIndex(new long[] { 0, 0, 0, 0, 0 });
    Test("(0,0,0,0,0) c_index=0", it9.GetIndex() == 0, "got " + it9.GetIndex());

    it9.GotoMultiIndex(new long[] { 0, 1, 0, 0, 0 });
    Test("(0,1,0,0,0) c_index=8", it9.GetIndex() == 8, "got " + it9.GetIndex());

    it9.GotoMultiIndex(new long[] { 1, 1, 1, 1, 1 });
    Test("(1,1,1,1,1) c_index=31", it9.GetIndex() == 31, "got " + it9.GetIndex());
}

// Test Case 10: Reduction (sum along axis 1)
Console.WriteLine("\n--- Test 10: Reduction ---");
var arr10 = np.arange(12).reshape(3, 4);
var out10 = np.zeros(new Shape(3), NPTypeCode.Int64);
using (var it10 = NpyIterRef.AdvancedNew(2, new[] { arr10, out10 },
    NpyIterGlobalFlags.REDUCE_OK,
    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
    new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
    null,
    2,  // opAxesNDim = 2
    new[] { new[] { 0, 1 }, new[] { 0, -1 } },  // Explicit axes for both operands
    new long[] { 3, 4 }))  // Explicit iterShape required when operands don't broadcast
{
    Test("ndim=2", it10.NDim == 2, "got " + it10.NDim);
    Test("itersize=12", it10.IterSize == 12, "got " + it10.IterSize);
    Test("IsReduction=true", it10.IsReduction);
    Test("IsOperandReduction(1)=true", it10.IsOperandReduction(1));

    // Actually perform reduction
    do
    {
        var x = it10.GetValue<long>(0);
        var y = it10.GetValue<long>(1);
        it10.SetValue(y + x, 1);
    } while (it10.Iternext());
}
// Verify: sum along axis 1: [0+1+2+3, 4+5+6+7, 8+9+10+11] = [6, 22, 38]
Test("Reduction result[0]=6", (long)out10[0] == 6, "got " + (long)out10[0]);
Test("Reduction result[1]=22", (long)out10[1] == 22, "got " + (long)out10[1]);
Test("Reduction result[2]=38", (long)out10[2] == 38, "got " + (long)out10[2]);

// Test Case 11: GotoIterIndex and GetIterIndex
Console.WriteLine("\n--- Test 11: GotoIterIndex ---");
var arr11 = np.arange(24).reshape(2, 3, 4);
using (var it11 = NpyIterRef.New(arr11, NpyIterGlobalFlags.MULTI_INDEX))
{
    it11.GotoIterIndex(10);
    Test("GotoIterIndex(10): IterIndex=10", it11.IterIndex == 10, "got " + it11.IterIndex);

    var coords = new long[3];
    it11.GetMultiIndex(coords);
    // Index 10 in shape (2,3,4) = (0, 2, 2) in row-major
    Test("GotoIterIndex(10): coords=(0,2,2)", coords[0] == 0 && coords[1] == 2 && coords[2] == 2,
         "got (" + coords[0] + "," + coords[1] + "," + coords[2] + ")");

    it11.GotoIterIndex(23);
    it11.GetMultiIndex(coords);
    Test("GotoIterIndex(23): coords=(1,2,3)", coords[0] == 1 && coords[1] == 2 && coords[2] == 3,
         "got (" + coords[0] + "," + coords[1] + "," + coords[2] + ")");
}

// Test Case 12: Empty array iteration
Console.WriteLine("\n--- Test 12: Empty Array ---");
var emptyArr = np.array(new int[0]);
using (var it12 = NpyIterRef.New(emptyArr, NpyIterGlobalFlags.ZEROSIZE_OK))
{
    Test("Empty: itersize=0", it12.IterSize == 0, "got " + it12.IterSize);
    Test("Empty: Finished=true", it12.Finished);
}

// Test Case 13: Scalar array
Console.WriteLine("\n--- Test 13: Scalar Array ---");
var scalar = np.array(42);
using (var it13 = NpyIterRef.New(scalar))
{
    Test("Scalar: ndim=0", it13.NDim == 0, "got " + it13.NDim);
    Test("Scalar: itersize=1", it13.IterSize == 1, "got " + it13.IterSize);
    unsafe
    {
        int value = *(int*)it13.GetDataPtrArray()[0];
        Test("Scalar: value=42", value == 42, "got " + value);
    }
}

// Test Case 14: Type casting with BUFFERED
Console.WriteLine("\n--- Test 14: Type Casting ---");
var intArr = np.arange(5);  // int32
using (var it14 = NpyIterRef.AdvancedNew(1, new[] { intArr },
    NpyIterGlobalFlags.BUFFERED,
    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
    new[] { NpyIterPerOpFlags.READONLY },
    new[] { NPTypeCode.Double }))  // Cast to double
{
    Test("Cast: RequiresBuffering=true", it14.RequiresBuffering);

    var values = new List<double>();
    do
    {
        values.Add(it14.GetValue<double>(0));
    } while (it14.Iternext());

    var expected = new[] { 0.0, 1.0, 2.0, 3.0, 4.0 };
    Test("Cast: values match", values.SequenceEqual(expected),
         "got [" + string.Join(",", values) + "]");
}

// Test Case 15: Three operand broadcast
Console.WriteLine("\n--- Test 15: Three Operand Broadcast ---");
var a15 = np.array(new int[] { 1, 2, 3 });        // (3,)
var b15 = np.array(new int[,] { { 10 }, { 20 } }); // (2, 1)
var c15 = np.array(100);                            // scalar
using (var it15 = NpyIterRef.MultiNew(3, new[] { a15, b15, c15 },
    NpyIterGlobalFlags.MULTI_INDEX,
    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
    new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY }))
{
    Test("3-way broadcast: ndim=2", it15.NDim == 2, "got " + it15.NDim);
    Test("3-way broadcast: itersize=6", it15.IterSize == 6, "got " + it15.IterSize);

    var triples = new List<(int, int, int)>();
    do
    {
        unsafe
        {
            var p = it15.GetDataPtrArray();
            triples.Add((*(int*)p[0], *(int*)p[1], *(int*)p[2]));
        }
    } while (it15.Iternext());

    // Expected: (1,10,100), (2,10,100), (3,10,100), (1,20,100), (2,20,100), (3,20,100)
    var expected = new[] { (1, 10, 100), (2, 10, 100), (3, 10, 100), (1, 20, 100), (2, 20, 100), (3, 20, 100) };
    Test("3-way broadcast: triples match", triples.SequenceEqual(expected),
         "got " + string.Join(", ", triples));
}

// ==========================================================
// TECHNIQUE 2: Systematic Edge Case Matrix
// ==========================================================
Console.WriteLine("\n\n=== EDGE CASE MATRIX ===\n");

// Edge Case 16: 2D reversed both axes
// NumPy with NEGPERM iterates in memory order, so multi_index starts at (2,3) with value 0
Console.WriteLine("--- Edge 16: 2D Reversed Both Axes ---");
var arr16 = np.arange(12).reshape(3, 4);
var rev16 = arr16["::-1, ::-1"];  // Reverse both dimensions
using (var it16 = NpyIterRef.New(rev16, NpyIterGlobalFlags.MULTI_INDEX))
{
    var coords = new long[2];
    it16.GetMultiIndex(coords);
    // NumPy: First position is (2,3) in original coordinates (NEGPERM flips iteration order)
    Test("2D reversed: first coords=(2,3)", coords[0] == 2 && coords[1] == 3,
         "got (" + coords[0] + "," + coords[1] + ")");

    // First value is 0 (iterating from bottom-right in memory order)
    unsafe { Test("2D reversed: first value=0", *(int*)it16.GetDataPtrArray()[0] == 0, "got " + *(int*)it16.GetDataPtrArray()[0]); }
}

// Edge Case 17: Single row iteration
Console.WriteLine("\n--- Edge 17: Single Row ---");
var arr17 = np.arange(5).reshape(1, 5);
using (var it17 = NpyIterRef.New(arr17, NpyIterGlobalFlags.MULTI_INDEX))
{
    Test("Single row: ndim=2", it17.NDim == 2, "got " + it17.NDim);
    Test("Single row: itersize=5", it17.IterSize == 5, "got " + it17.IterSize);
}

// Edge Case 18: Single column iteration
Console.WriteLine("\n--- Edge 18: Single Column ---");
var arr18 = np.arange(5).reshape(5, 1);
using (var it18 = NpyIterRef.New(arr18, NpyIterGlobalFlags.MULTI_INDEX))
{
    Test("Single col: ndim=2", it18.NDim == 2, "got " + it18.NDim);
    Test("Single col: itersize=5", it18.IterSize == 5, "got " + it18.IterSize);
}

// Edge Case 19: Very thin slice (step > size)
Console.WriteLine("\n--- Edge 19: Wide Step Slice ---");
var arr19 = np.arange(100);
var wide19 = arr19["::50"];  // Should get 2 elements: [0, 50]
using (var it19 = NpyIterRef.New(wide19))
{
    Test("Wide step: itersize=2", it19.IterSize == 2, "got " + it19.IterSize);
    var vals = new List<int>();
    do { unsafe { vals.Add(*(int*)it19.GetDataPtrArray()[0]); } } while (it19.Iternext());
    Test("Wide step: values=[0,50]", vals.SequenceEqual(new[] { 0, 50 }), "got [" + string.Join(",", vals) + "]");
}

// Edge Case 20: Middle slice
Console.WriteLine("\n--- Edge 20: Middle Slice ---");
var arr20 = np.arange(10);
var mid20 = arr20["3:7"];  // [3, 4, 5, 6]
using (var it20 = NpyIterRef.New(mid20))
{
    Test("Middle slice: itersize=4", it20.IterSize == 4, "got " + it20.IterSize);
    var vals = new List<int>();
    do { unsafe { vals.Add(*(int*)it20.GetDataPtrArray()[0]); } } while (it20.Iternext());
    Test("Middle slice: values=[3,4,5,6]", vals.SequenceEqual(new[] { 3, 4, 5, 6 }), "got [" + string.Join(",", vals) + "]");
}

// Edge Case 21: Negative indexing slice
Console.WriteLine("\n--- Edge 21: Negative Indexing ---");
var arr21 = np.arange(10);
var neg21 = arr21["-3:"];  // Last 3 elements: [7, 8, 9]
using (var it21 = NpyIterRef.New(neg21))
{
    Test("Negative idx: itersize=3", it21.IterSize == 3, "got " + it21.IterSize);
    var vals = new List<int>();
    do { unsafe { vals.Add(*(int*)it21.GetDataPtrArray()[0]); } } while (it21.Iternext());
    Test("Negative idx: values=[7,8,9]", vals.SequenceEqual(new[] { 7, 8, 9 }), "got [" + string.Join(",", vals) + "]");
}

// ==========================================================
// TECHNIQUE 4: Property-Based Invariant Testing
// ==========================================================
Console.WriteLine("\n\n=== PROPERTY INVARIANTS ===\n");

// Invariant 1: Sum of iterated values == sum of array
Console.WriteLine("--- Invariant 1: Sum Preservation ---");
var invArr1 = np.arange(100).reshape(10, 10);
long iterSum = 0;
using (var itInv1 = NpyIterRef.New(invArr1))
{
    do { unsafe { iterSum += *(int*)itInv1.GetDataPtrArray()[0]; } } while (itInv1.Iternext());
}
long arraySum = 0;
for (int i = 0; i < 100; i++) arraySum += i;
Test("Invariant: iter_sum == array_sum (4950)", iterSum == arraySum, "iter_sum=" + iterSum + " array_sum=" + arraySum);

// Invariant 2: IterSize == np.prod(shape)
Console.WriteLine("\n--- Invariant 2: IterSize == prod(shape) ---");
var shapes = new[] { new[] { 2, 3 }, new[] { 5 }, new[] { 2, 3, 4 }, new[] { 1, 1, 1, 1, 1 } };
foreach (var shape in shapes)
{
    var arr = np.ones(new Shape(shape));
    using (var it = NpyIterRef.New(arr))
    {
        int prod = shape.Aggregate(1, (a, b) => a * b);
        Test("IterSize(" + string.Join("x", shape) + ")=" + prod, it.IterSize == prod, "got " + it.IterSize);
    }
}

// Invariant 3: All indices visited exactly once
Console.WriteLine("\n--- Invariant 3: All Indices Visited Once ---");
var invArr3 = np.arange(24).reshape(2, 3, 4);
var visited = new HashSet<int>();
using (var itInv3 = NpyIterRef.New(invArr3, NpyIterGlobalFlags.MULTI_INDEX | NpyIterGlobalFlags.C_INDEX))
{
    do
    {
        int idx = (int)itInv3.GetIndex();
        visited.Add(idx);
    } while (itInv3.Iternext());
}
Test("All indices visited", visited.Count == 24, "visited " + visited.Count + " indices");
Test("Indices 0-23 complete", visited.Min() == 0 && visited.Max() == 23, "range [" + visited.Min() + "," + visited.Max() + "]");

// Invariant 4: Reset returns to start
Console.WriteLine("\n--- Invariant 4: Reset Returns to Start ---");
var invArr4 = np.arange(10);
using (var itInv4 = NpyIterRef.New(invArr4, NpyIterGlobalFlags.MULTI_INDEX))
{
    // Advance some steps
    itInv4.Iternext();
    itInv4.Iternext();
    itInv4.Iternext();

    // Reset
    itInv4.Reset();

    var coords = new long[1];
    itInv4.GetMultiIndex(coords);
    Test("Reset: back to index 0", coords[0] == 0, "got " + coords[0]);
    Test("Reset: IterIndex=0", itInv4.IterIndex == 0, "got " + itInv4.IterIndex);
}

// Invariant 5: GotoIterIndex is reversible
Console.WriteLine("\n--- Invariant 5: GotoIterIndex Reversible ---");
using (var itInv5 = NpyIterRef.New(invArr4, NpyIterGlobalFlags.MULTI_INDEX))
{
    itInv5.GotoIterIndex(7);
    Test("Goto(7): IterIndex=7", itInv5.IterIndex == 7, "got " + itInv5.IterIndex);

    itInv5.GotoIterIndex(2);
    Test("Goto(2): IterIndex=2", itInv5.IterIndex == 2, "got " + itInv5.IterIndex);

    itInv5.GotoIterIndex(9);
    Test("Goto(9): IterIndex=9", itInv5.IterIndex == 9, "got " + itInv5.IterIndex);
}

// Invariant 6: Iternext increments IterIndex by 1
Console.WriteLine("\n--- Invariant 6: Iternext Increments by 1 ---");
using (var itInv6 = NpyIterRef.New(np.arange(5)))
{
    var indices = new List<long>();
    do { indices.Add(itInv6.IterIndex); } while (itInv6.Iternext());
    Test("IterIndex increments: [0,1,2,3,4]", indices.SequenceEqual(new long[] { 0, 1, 2, 3, 4 }),
         "got [" + string.Join(",", indices) + "]");
}

Console.WriteLine("\n" + new string('=', 50));
Console.WriteLine("TOTAL: " + passed + " passed, " + failed + " failed");
Console.WriteLine(new string('=', 50));
if (failures.Count > 0)
{
    Console.WriteLine("\nFAILURES:");
    foreach (var f in failures) Console.WriteLine("  - " + f);
}
