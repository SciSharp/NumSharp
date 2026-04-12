---
name: np-function
description: Implement numpy np.* functions in NumSharp with full API parity. Use when adding new numpy functions, implementing np.* methods, or aligning NumSharp behavior with numpy 2.4.2.
---

We are looking to support numpy's np.* to the fullest. we are aligning with numpy 2.4.2 as source of truth and are to provide exact same API (np.* overloading) as numpy does.
This session we focusing on: """$ARGUMENTS"""
You job is around interacting with np.* functions (no more than a few. more than one ONLY if they are closely related).

To interact/develop/create a np.* / function, high-level development cycle is as follows:
1. Read how numpy (K:\source\NumSharp\src\numpy) exposes the np function/s you are about to implement. Remember, numpy is the source of truth and if numpy does A, we do A but in NumSharp's C# way.
Definition of Done:
- At the end of this step you understand to 100% how the np function both works, behaves and accepts.
If numpy function uses other np.* functions then you are to report them to the team leader and wait for further instructions. If the function is relative to you then take ownership over it and add it to your group of functions.
2. Implement np methods in the appropriate np class and if custom calculation/math is required via backend then follow the Tensor and IL Kernel way. You are not allowed to implement a hardcoded loop per dtype. Usually other np.* calls is all a numpy function requires. Always reuse existing architecture rather than create a new one.
Definition of Done:
    - What Numpy supports, we support.
    - We support all dtypes (no hardcoded loops, use il generation via our backend if necessary).
    - We have all the apis the np function has and all the overloads.
    - We calculate exactly like numpy because only that way we can capture all the design edge cases and implicit behaviors.
3. Then migrate the tests that numpy has from numpy to C# and then produce your own set of tests covering all aspects of the api that you will battletest. Any bugs should be fixed on the spot.
Definition of Done:
    - All numpy tests have been migrated to NumSharp C#.
    - We used battletesting to find edge cases and other bugs in our implementation where numpy works. Our source of truth for behavior is numpy!
4. Review the implementation, definitions of done and confirm alignment to numpy is as close as possible. Ensure documentation in code.
5. Commit and report completion.

## Tools:
### Battletesting
Use battletesting to test and validate assumptions or even hunt edge cases: which is using 'dotnet run << 'EOF'' and 'python << 'EOF'' to run any code for any of these purposes.

## Instructions to Team Leader
- Create at-least 4 users if the task can be parallelised to that level and if not then use less
    - Do not wait for other teammates to complete, always have N teammates developing until all the work is completed by definition of done.
    - If user asked for 1 of something there there is only a reason to launch one teammate and not five.
- When the teammate have completed development all the way to last step and all definition of done: finish and shutdown the teammate.
- You are to give the instructions to the teammates word-for-word based on this document with your own adaptation below the original version.