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
  Half, Single, Double, Decimal, Complex, and the text families `S`, `U` and `StringDType` (`T`).
  Text uses `NPTypeCode.String` with a full <xref href="NumSharp.DType" data-throw-if-not-resolved="false"></xref>
  descriptor for the width, encoding and missing-value policy. Preserve `array.dtype`, not just its type code.
  `NPTypeCode.Empty` is a sentinel; `NPTypeCode.Float` is an alias of `Single`.

  `np.strings.add` and `np.strings.multiply` concatenate and repeat text. `np.@char` is NumPy's
  `numpy.char` module (C# escapes the keyword); <xref href="NumSharp.NDCharArray" data-throw-if-not-resolved="false"></xref>
  supplies chararray behavior and <xref href="NumSharp.NDListArray" data-throw-if-not-resolved="false"></xref>
  holds split results. None of these is the numeric UTF-16 `Char` dtype.

  ```csharp
  using NumSharp;

  var a = np.arange(6).reshape(2, 3);
  var b = np.array([10, 20, 30]);
  NDArray c = a + b;        // NumPy-style broadcasting
  NDArray col = c[":, 1"];  // Slicing returns a view
  ```
---
