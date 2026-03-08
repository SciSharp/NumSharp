#!/usr/bin/env dotnet run
#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("Testing SIMD Prod Reduction Kernel");
Console.WriteLine("===================================\n");

bool allPassed = true;

// Test 1: Basic prod
var arr1 = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
var prod1 = (double)arr1.prod();
var expected1 = 120.0;
var pass1 = Math.Abs(prod1 - expected1) < 0.001;
Console.WriteLine($"Test 1: [1,2,3,4,5].prod() = {prod1} (expected: {expected1}) - {(pass1 ? "PASS" : "FAIL")}");
allPassed &= pass1;

// Test 2: Ones array (identity check)
var arr2 = np.ones(100);
var prod2 = (double)arr2.prod();
var expected2 = 1.0;
var pass2 = Math.Abs(prod2 - expected2) < 0.001;
Console.WriteLine($"Test 2: ones(100).prod() = {prod2} (expected: {expected2}) - {(pass2 ? "PASS" : "FAIL")}");
allPassed &= pass2;

// Test 3: Array of 2s
var arr3 = np.ones(new int[] { 10 }) * 2.0;
var prod3 = (double)arr3.prod();
var expected3 = 1024.0; // 2^10
var pass3 = Math.Abs(prod3 - expected3) < 0.001;
Console.WriteLine($"Test 3: (ones(10)*2).prod() = {prod3} (expected: {expected3}) - {(pass3 ? "PASS" : "FAIL")}");
allPassed &= pass3;

// Test 4: Integer factorial-like
var arr4 = np.arange(1, 6); // [1,2,3,4,5]
var prod4nd = (NDArray)arr4.prod();
// For 0-D scalar arrays, use GetAtIndex(0) - returns int32 for arange
var prod4 = Convert.ToInt64(prod4nd.GetAtIndex(0));
var expected4 = 120L;
var pass4 = prod4 == expected4;
Console.WriteLine($"Test 4: arange(1,6).prod() = {prod4} (expected: {expected4}) - {(pass4 ? "PASS" : "FAIL")}");
allPassed &= pass4;

// Test 5: Larger array to ensure SIMD path (32 elements for AVX-256 doubles)
var arr5 = np.ones(32) * 1.5;
var prod5 = (double)arr5.prod();
var expected5 = Math.Pow(1.5, 32);
var pass5 = Math.Abs(prod5 - expected5) / expected5 < 0.0001; // relative error
Console.WriteLine($"Test 5: (ones(32)*1.5).prod() = {prod5:E6} (expected: {expected5:E6}) - {(pass5 ? "PASS" : "FAIL")}");
allPassed &= pass5;

// Test 6: Float type
var arr6 = np.array(new float[] { 1f, 2f, 3f, 4f });
var prod6 = (float)arr6.prod();
var expected6 = 24f;
var pass6 = Math.Abs(prod6 - expected6) < 0.001f;
Console.WriteLine($"Test 6: float[1,2,3,4].prod() = {prod6} (expected: {expected6}) - {(pass6 ? "PASS" : "FAIL")}");
allPassed &= pass6;

// Test 7: Empty/scalar
var arr7 = np.array(new[] { 42.0 });
var prod7 = (double)arr7.prod();
var expected7 = 42.0;
var pass7 = Math.Abs(prod7 - expected7) < 0.001;
Console.WriteLine($"Test 7: [42].prod() = {prod7} (expected: {expected7}) - {(pass7 ? "PASS" : "FAIL")}");
allPassed &= pass7;

// Test 8: Zero in array
var arr8 = np.array(new[] { 1.0, 2.0, 0.0, 4.0 });
var prod8 = (double)arr8.prod();
var expected8 = 0.0;
var pass8 = prod8 == expected8;
Console.WriteLine($"Test 8: [1,2,0,4].prod() = {prod8} (expected: {expected8}) - {(pass8 ? "PASS" : "FAIL")}");
allPassed &= pass8;

Console.WriteLine($"\n===================================");
Console.WriteLine($"Result: {(allPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED")}");

return allPassed ? 0 : 1;
