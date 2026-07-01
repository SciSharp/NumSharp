---
uid: NumSharp
summary: Primary NumSharp namespace containing the NDArray type, NumPy-style API facade, dtype metadata, slicing and shape helpers, random state, and NumPy-compatible exception types.
remarks: |
  Use this namespace when writing NumSharp code:

  - <xref href="NumSharp.np" data-throw-if-not-resolved="false"></xref> is the NumPy-style static API, equivalent to Python's `import numpy as np`.
  - <xref href="NumSharp.NDArray" data-throw-if-not-resolved="false"></xref> is the main n-dimensional array container.
  - <xref href="NumSharp.Shape" data-throw-if-not-resolved="false"></xref>, <xref href="NumSharp.Slice" data-throw-if-not-resolved="false"></xref>, and <xref href="NumSharp.NPTypeCode" data-throw-if-not-resolved="false"></xref> define shape, view, indexing, and dtype behavior.
  - <xref href="NumSharp.NumPyRandom" data-throw-if-not-resolved="false"></xref> exposes NumPy-compatible MT19937 random generation.

  NumSharp targets NumPy 2.x API and behavioral compatibility. Arrays use unmanaged storage,
  slicing returns views that share storage, broadcasting follows NumPy shape rules, and the public
  NDArray dtype set covers Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char,
  Half, Single, Double, Decimal, and Complex. `NPTypeCode.Empty`, `NPTypeCode.String`, and the
  `NPTypeCode.Float` alias are enum/compatibility values, not additional public array dtypes;
  `Float` resolves to `Single`.

  ```csharp
  using NumSharp;

  var a = np.arange(6).reshape(2, 3);
  var b = np.array(new[] { 10, 20, 30 });
  NDArray c = a + b;        // NumPy-style broadcasting
  NDArray col = c[":, 1"];  // Slicing returns a view
  ```
---
