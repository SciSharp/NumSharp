---
name: np-tests
description: Write and migrate numpy tests for NumSharp functions. Use when adding tests for np.* methods, migrating numpy test suites, or battletesting NumSharp implementations against numpy 2.4.2.
---

We are looking to test numpy's np.* implementations to the fullest. we are aligning with numpy 2.4.2 as source of truth and are to validate exact same behavior as numpy does.
This session we focusing on: """$ARGUMENTS"""
Your job is around writing tests for np.* functions (no more than a few. more than one ONLY if they are closely related).

To interact/develop/create tests for np.* functions, high-level development cycle is as follows:
1. Find and read numpy's tests for the function/s you are about to test. Tests are in K:\source\NumSharp\src\numpy under numpy/_core/tests/, numpy/lib/tests/, etc. Remember, numpy is the source of truth.
Definition of Done:
- At the end of this step you understand 100% what numpy tests: inputs, outputs, edge cases, error conditions, dtype behaviors.
- You have identified all test files and test methods related to your function.
2. Migrate numpy's tests to C# following MSTest v3 framework patterns in test/NumSharp.UnitTest. Match numpy's test structure and assertions exactly.
Definition of Done:
    - Every numpy test case has a corresponding C# test.
    - We cover all dtypes NumSharp supports (Boolean, Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Single, Double, Decimal).
    - Test names reflect what they test (e.g., Test_Sort_Axis0_Int32).
    - Assertions match numpy's expected values exactly.
3. Battletest to find gaps. Run python and dotnet side-by-side to discover edge cases numpy handles that we might miss.
Definition of Done:
    - All numpy tests pass in NumSharp.
    - Additional edge cases discovered via battletesting are covered.
    - Any bugs found are reported to implementation teammate or fixed on the spot.
4. Review test coverage: empty arrays, scalar inputs, negative axis, broadcasting, NaN/Inf handling, dtype promotion, error conditions.
5. Commit and report completion.

## Tools:
### Battletesting
Use battletesting to validate behavior matches numpy: 'dotnet run << 'EOF'' and 'python << 'EOF'' side-by-side comparison.

### Test Patterns
```csharp
[TestMethod]
public void FunctionName_Scenario_Dtype()
{
    // Arrange
    var input = np.array(new[] { 3, 1, 2 });

    // Act
    var result = np.sort(input);

    // Assert - values from running actual numpy
    Assert.IsTrue(result.IsContiguous);
    Assert.AreEqual(1, result.GetAtIndex<int>(0));
}
```

## Instructions to Team Leader
- Create at-least 4 users if the task can be parallelised to that level and if not then use less
    - Do not wait for other teammates to complete, always have N teammates developing until all the work is completed by definition of done.
    - If user asked for 1 of something there there is only a reason to launch one teammate and not five.
- When the teammate have completed development all the way to last step and all definition of done: finish and shutdown the teammate.
