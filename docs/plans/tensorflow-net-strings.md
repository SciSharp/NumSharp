# TensorFlow.NET and NumSharp's text arrays: can TensorFlow.NET use them? (evaluation, 2026-11-06)

> **Scope.** This is an evaluation: nothing in `src/` changed. It answers the owner's request "In strings branch we now
> work with strings and chararray, can you check tensorflow.net and see if it can now use NumSharp's strings
> capabilities". Measured on branch `strings` at `106b67fa` (NumSharp.Core built Release, net10.0) against the published
> TensorFlow.NET 0.150.0 with its TensorFlow 2.16.0 CPU runtime (`SciSharp.TensorFlow.Redist` 2.16.0), on Windows x64.
> Python TensorFlow 2.21.0 with NumPy 2.4.2 is the reference for every expected answer. TensorFlow.NET's source was read
> at `SciSharp/TensorFlow.NET` master `6ce60665`. The probe in [tensorflow-net-strings/](tensorflow-net-strings/)
> reproduces every result (§7).

## 0. The question and the answer

The question has two readings, and they get different answers.

**A TensorFlow.NET program can use them today, with no change to TensorFlow.NET.** NumSharp's `S`, `U` and
`StringDType` arrays cross into TensorFlow string tensors and back byte for byte. That holds for 1-D, 2-D, transposed,
sliced, reversed and offset views, for `StringDType` texts stored in the cell, the arena or the heap, and for 0-d and
empty arrays. TensorFlow's string ops then give Python TensorFlow's answers on those tensors. The probe passes 145 checks
and fails none. There are two ways across:
- The **public route** uses only public API on both sides: `NDArray.AsStringArray` into `new EagerTensor(string[],
  shape)`, and `StringData()` into `np.array(...)`. It works for valid text. It runs 0.66–1.58× Python TensorFlow's
  speed for the same crossing. It has three bad edges: a `StringDType` NA throws `ArgumentNullException` inside
  TensorFlow.NET, a lone surrogate becomes U+FFFD silently, and invalid UTF-8 coming out of TensorFlow becomes U+FFFD
  silently.
- The **direct route** is a NumSharp-side bridge, prototyped here in `TfTextBridge.cs`. It writes TensorFlow's
  `TF_TString` cells straight from NumSharp's buffers and reads them back the same way. It is 2.0–14× faster than Python
  TensorFlow on every crossing and refuses what Python refuses, with Python's error text. Its tensors also survive the
  lifetime defect described in §5.2.

**TensorFlow.NET itself cannot use them without being rewritten.** TensorFlow.NET dropped NumSharp in July 2021
(`f3eb31fb` "Built-in numpy api.", version 0.60.0, which removed `PackageReference NumSharp 0.30.0`). Its replacement,
`Tensorflow.NumPy`, has an `NDArray` that is itself a `Tensor`. It targets `netstandard2.0` and `net6.0`, while NumSharp
targets `net8.0` and `net10.0`. Its last commit is from 2025-01-22, and its last release is 0.150.0. Rebuilding its numpy
layer on NumSharp would be a re-architecture of TensorFlow.NET, not a question about strings.

**What NumSharp's text support gives a TensorFlow.NET user that TensorFlow.NET lacks:**
- **46 `np.strings` functions with NumPy's answers.** TensorFlow.NET binds 7 of Python's 26 `tf.strings` ops. It has no
  `upper`, `strip`, `join` or `to_number`. Its `lower(encoding: "utf-8")` lowers ASCII only (§5.1). On the probe's 15
  texts, NumSharp's `np.strings.lower` and `upper` match NumPy exactly. A tensor round trip through `StringDType` costs
  69 + 47 ms per million texts.
- **NumPy's text files.** NumSharp loads NumPy-written `U`, `S` and `StringDType` `.npy` files, `StringDType` through
  NumPy's pickle. TensorFlow.NET's `np.load` fails on all four kinds the probe wrote, and its `np.save` refuses strings.
- **`StringDType` as the carrier of TensorFlow strings.** It is variable-width UTF-8 like `tf.string`, keeps NUL bytes and
  holds any length. Reading a string tensor into `StringDType` is what Python users write as
  `tensor.numpy().astype(StringDType())`. Python TensorFlow itself refuses `StringDType` arrays as input (§2).

**Two TensorFlow.NET defects turned up on the way** (§5). Both are small fixes in TensorFlow.NET:
- `tf.strings.lower` drops its `encoding`.
- Its string tensors free their long texts while another tensor still shares the buffer. After `tf.identity` and a
  collection, 64 of 64 long texts read `''`.

**Recommendation (§6).** For text, nothing is missing on NumSharp's side. Whether to ship a `NumSharp.Interop.TensorFlow`
package is a product decision. The text half is proven here, and the numeric half would follow the ONNX Runtime bridge's
pattern. The risk is TensorFlow.NET's dormancy: the package would pin an unmaintained, unsigned dependency and TensorFlow
2.16 natives, which exist for Windows x64 and Linux x64 only. If the package is wanted, four decisions come first
(D1–D4). The two TensorFlow.NET defects deserve upstream fixes either way.

## 1. TensorFlow.NET today

| Fact | Value | Source |
|---|---|---|
| Last release | 0.150.0 (TensorFlow 2.15/2.16 natives) | nuget.org `TensorFlow.NET` versions |
| Last commit on master | 2025-01-22 (`6ce60665`) | `git log` |
| Target frameworks | `netstandard2.0;net6.0`, LangVersion 10 | `src/TensorFlowNET.Core/Tensorflow.Binding.csproj` |
| NumPy layer | its own `Tensorflow.NumPy` since 0.60.0; `NDArray : Tensor` | `f3eb31fb` (2021-07-07) |
| NumSharp dependency | none since `f3eb31fb` (it was `NumSharp 0.30.0`) | the same commit |
| Strong name | the published `Tensorflow.Binding.dll` has none (CS8002 when a signed assembly references it) | the probe's build |
| Native runtime | `SciSharp.TensorFlow.Redist` 2.16.0: win-x64 and linux-x64 only (`osx-x64` holds `_._`) | the package |

How TensorFlow.NET holds text:
- **Storage.** `tf.string` is TensorFlow's `TF_TString`, 24 bytes per element. A text of at most 22 bytes lives in the
  cell; a longer one is a `malloc`ed block the cell points at.
- **Building a tensor.** `Tensor.StringTensor(string[] / byte[][], Shape)` calls `TF_AllocateTensor`, then
  `TF_StringInit` + `TF_StringCopy` per element. A `string` is encoded with `Encoding.UTF8.GetBytes`, which replaces a
  lone surrogate with U+FFFD.
- **Reading a tensor.** `StringBytes()` gives a `byte[]` per element. `StringData()` gives `Encoding.UTF8.GetString` of
  each, which replaces invalid UTF-8 with U+FFFD.
- **`tf.strings`.** TensorFlow.NET binds 7 of Python's 26 ops: `lower`, `regex_replace`, `substr`, `string_length`,
  `format`, `split` and `unicode_decode_with_offsets`.
- **`np.save` / `np.load`.** `np.save` throws `NotSupportedException` for a string array (`NpyFormat.GetDtypeName`).
  `np.load` has no `U` mapping and fails on `S` (§4.2).

## 2. What Python TensorFlow does with NumPy text (the parity target)

Probed with TensorFlow 2.21.0 and NumPy 2.4.2 (`py_reference.py`, `conversions.json`):

| Input | `tf.constant(...)` |
|---|---|
| `U` array (any layout, 0-d, empty) | a string tensor of each element's UTF-8, in logical order |
| `U` with a lone surrogate | `ValueError: ... (Unable to convert element to UTF-8)` |
| `S` array | each element's bytes **without trailing NULs** (`b'tail\x00'` → `tail`); an embedded NUL stays |
| `StringDType` array (with or without NA) | `ValueError: Failed to convert a NumPy array to a Tensor (Unsupported numpy type: not a numpy type)` |
| object array of `str` / `bytes` | UTF-8 of each `str`, each `bytes` as is |

| A string tensor `t` | Result |
|---|---|
| `t.numpy()` | an **object** array of `bytes` |
| `t.numpy().astype(StringDType())` | the UTF-8 decode; invalid bytes raise `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data` |
| `t.numpy().astype('S')` | `S` of the longest element, trailing NULs read back stripped (`[b'ab\x00']` → `array([b'ab'], dtype='\|S3')`) |
| `t.numpy().astype('U')` | `UnicodeDecodeError: 'ascii' codec ...` for any non-ASCII byte |

NumSharp has no object dtype, so `S` (bytes) and `StringDType` (decoded text) are the two faithful import targets. `U`
is not: Python itself reaches `U` only through `np.strings.decode(t.numpy().astype('S'))`.

TensorFlow's string ops and NumPy's `np.strings` agree on valid text, measured on 15 texts covering ASCII, Latin,
dotted capital I, final sigma, sharp s, a titlecase digraph, a ligature, CJK, an astral emoji, an embedded NUL and a
40-character text:
- `tf.strings.length(unit='UTF8_CHAR')` equals `np.strings.str_len` 15/15.
- `lower` and `upper` with `encoding='utf-8'` equal NumPy's 14/15. The exception is TensorFlow's own: its ICU case map
  stops at an embedded NUL (`'a\x00b'` lowers to `'a'`).
- `substr(0, 2, 'UTF8_CHAR')` equals `np.strings.slice(0, 2)` 14/15. The exception is `U`'s trailing-NUL rule: TF
  returns `'a\x00'` where NumPy's `U` element reads `'a'`.

## 3. The two routes

### 3.1 The public route (no new code)

```csharp
using Tensorflow.Eager;
using NSArray = NumSharp.NDArray;                 // TF.NET also has np, NDArray, Shape and Slice: alias one side

// NumSharp U / StringDType -> tf.string
var tensor = new EagerTensor(NSArray.AsStringArray(text), new Tensorflow.Shape(text.shape));

// tf.string -> NumSharp StringDType
NSArray back = np.array(tensor.StringData().Cast<object>().ToArray(), np.dtypes.StringDType())
                 .reshape(new NumSharp.Shape(tensor.shape.dims));
```

- **`S`.** `NDArray.AsStringArray` refuses `S`. Its bytes come out through `nd.Unsafe.Storage.GetAtIndex(i)` as `byte[]`.
  TensorFlow.NET's only multi-element bytes constructor is the instance method `Tensor.StringTensor(byte[][], Shape)`,
  wrapped as `new Tensorflow.NumPy.NDArray(new Tensor(handle))`.
- **Two transcodes per crossing.** NumSharp's UCS-4 or UTF-8 becomes UTF-16 strings, then UTF-8 again in TensorFlow.NET.
- **The edges.** A `StringDType` NA is a `null` string, and TensorFlow.NET throws `ArgumentNullException`. A lone
  surrogate and invalid UTF-8 out of TensorFlow both become U+FFFD silently. Python raises in both cases.

### 3.2 The direct route (`TfTextBridge.cs`, the prototype)

**Export** (`ToTensor(nd, Payload.Copy | Payload.View)`):
1. The bridge allocates a 64-byte-aligned native block of `TF_TString` cells and initializes every cell.
2. It writes each element from NumSharp's buffer in C order:
   - `S`: the element's bytes minus trailing NULs, which is Python TF's `S` rule.
   - `U`: UCS-4 encoded to UTF-8 in one reused buffer. A surrogate or a unit past U+10FFFF is refused, as Python TF
     refuses it.
   - `StringDType`: the cell is unpacked by its own allocator (`NpyString_load`) under one lock, and its UTF-8 copied. An
     NA is refused.
   - A layout other than C-contiguous is first put in C order by `np.ascontiguousarray`.
3. The block goes to `TF_NewTensor` with an `[UnmanagedCallersOnly]` deallocator. The deallocator runs `TF_StringDealloc`
   per cell, frees the block, and releases a view's pin. TensorFlow calls it when the last tensor sharing the buffer is
   gone.
4. `TFE_NewTensorHandle` makes the eager tensor.

`Payload.View` (`S` and `StringDType` only) assigns `TF_TSTR_VIEW` cells that point into the NumSharp buffer, so no text
byte is copied. The tensor holds an ARC reference (`TryAddRef`) on the NumSharp buffer, and the deallocator releases it.
The NumSharp array must not be written while a tensor views it:
- an `S` write changes what TensorFlow reads;
- a `StringDType` write can move the arena or free a heap text that a view points at.

**Import** (`ToNDArray(tensor, TextKind.StringDType | TextKind.Bytes)`) reads each cell with `TF_StringGetDataPointer`
and `TF_StringGetSize`:
- `Bytes` copies every text into an `S` of the longest width.
- `StringDType` first checks every element is UTF-8. On invalid bytes it raises NumSharp's own `UnicodeDecodeError`,
  which has Python's exact text because `np.strings.decode` is `bytes.decode`. It then sums the arena bytes, reserves the
  arena once and packs each text (`NpyString_pack`) under the allocator's lock.

**Why the bridge belongs in NumSharp's repo.** A `StringDType` arena or heap text can only be found through the
cell's allocator. `VStringAllocator`, `VStringArraySlice.Allocator` and `UnmanagedStorage.VStringSlice` are internal. An
assembly NumSharp grants `InternalsVisibleTo` (the ONNX Runtime bridge's arrangement) can therefore move `StringDType`
text as UTF-8. Code inside TensorFlow.NET could only decode it to `string`s first, which is the public route. `S` and
`U` bytes are reachable publicly (`nd.Unsafe.Storage.InternalArray as StringArraySlice`).

## 4. Results

### 4.1 Correctness: 145 checks, 0 failures

| Section | Checks | What |
|---|---|---|
| routes | 123 | 3 dtypes × 6 layouts (1-D, 3×4, transposed, `[1:, ::2]`, reversed, `[3:]`): public export (bytes + shape), direct export (bytes + shape), view export (`S`, `StringDType`), direct import, public import; plus `StringDType` short / arena / heap texts through copy, view and import |
| ops | 12 | TensorFlow.NET's `string_length` (BYTE, UTF8_CHAR), `lower` (ASCII), raw `StringLower` with the `encoding` attribute, `substr` (UTF8_CHAR, BYTE) and `regex_replace` on a NumSharp-made tensor equal Python TF 15/15; NumSharp's `str_len`, `lower`, `upper` and `slice` equal NumPy 15/15; `np.strings.upper` over a tensor's text equals NumPy 15/15 |
| edges | 6 | 0-d `U` → TF scalar → 0-d `StringDType`; empty `(0,)` and `(2, 0)` both ways |
| lifetime | 4 | §4.3 |

### 4.2 Edges

| Case | Public route | Direct route | Python TF |
|---|---|---|---|
| `StringDType` with an NA | `ArgumentNullException` (TF.NET) | `NotSupportedException` naming the element | refuses every `StringDType` array |
| `U` with a lone surrogate | silently U+FFFD (`ef bf bd`) | `NotSupportedException` naming the unit | `ValueError` |
| invalid UTF-8 from TF (BYTE `substr` of `"été"`) | `StringData()` silently U+FFFD | to `StringDType`: `UnicodeDecodeError` with Python's exact text; to `S`: the byte `c3` kept | `.astype(StringDType())`: the same `UnicodeDecodeError` |
| `[b'ab\0', b'a\0b']` from TF | — | to `S`: `\|S3`, `ab` and `a\0b`, like Python's `.astype('S')`; to `StringDType`: both NULs kept | the same |
| NumPy-written `.npy` (`U`, `S`, `StringDType`, `StringDType` with NA) | TF.NET's `np.load`: `NotSupportedException` (`U`), `ArgumentException: Object must be an array of primitives` (the other three) | NumSharp's `np.load` reads all four; three cross to TF, the NA file is refused | — |

### 4.3 Lifetime

- **View exports.** A view export reads correctly after the NumSharp array is disposed and collected. The pin goes back
  when the tensor goes.
- **Direct exports with a shared buffer.** The output of `tf.identity` still reads every text after the source tensor
  has been collected. The block is freed with the last tensor.
- **TensorFlow.NET's own `tf.constant(string[])`, the same sequence:**
  - 64 of 64 texts of 30–93 characters read `''` after the source is collected.
  - With the source kept reachable, 0 of 64 change.
  - Texts of 22 bytes or fewer, which live inside the cell, are unaffected (0 of 64).

  §5.2 has the cause.

### 4.4 Speed

1,000,000 texts of 1–40 code points, one character in ten from `é ß 日 😀`. Each process ran pinned to the same two
P-cores (`start /affinity 14`), back to back. Times are the best of 5 after 2 warm-ups. The host was at about 51% CPU
load from other sessions during the run. The ratios use the house convention (Python ms / NumSharp ms; above 1 means
NumSharp is faster).

| Crossing | Public route | Direct route | Python TF 2.21 + NumPy 2.4.2 | Direct ratio |
|---|---|---|---|---|
| `U` → TF | 296 ms | 86 ms | `tf.constant(U)` 196 ms | 2.29 ✅ |
| `StringDType` → TF | 290 ms | copy 47 ms · view 17 ms | refused; `tf.constant(t.astype(object))` 240 ms | 5.13 / 14.1 ✅ |
| `S` → TF | 143 ms | copy 64 ms · view 45 ms | `tf.constant(S)` 129 ms | 2.02 / 2.86 ✅ |
| TF → `StringDType` | 370 ms | 69 ms | `t.numpy().astype(StringDType())` 275 ms | 3.98 ✅ |
| TF → `S` | 131 ms | 40 ms | `t.numpy().astype('S')` 195 ms | 4.82 ✅ |
| TF → `U` | 418 ms | — | `np.strings.decode(t.numpy().astype('S'))` 661 ms | public 1.58 ✅ |

The public route runs at 0.66–0.90× Python TF for exports and 0.74–1.58× for imports. Python's `t.numpy()` alone, which
gives the object array of `bytes`, takes 85 ms.

## 5. TensorFlow.NET defects found on the way

### 5.1 `tf.strings.lower(input, encoding)` ignores `encoding`

`src/TensorFlowNET.Core/Operations/string_ops.cs:25-26` passes the encoding as an op **input**:
`tf.Context.ExecuteOp("StringLower", name, new ExecuteOpArgs(input, encoding))`. `StringLower` takes it as an
**attribute**.

On the probe's 15 texts, `lower(encoding: "utf-8")` returns exactly Python's ASCII-only lowering (0/15 differ). It
differs from Python's UTF-8 lowering on 6/15. In five, À É Î Õ Ü, İ, the Greek capitals, ǅ and Ä Ö Ü stay upper case.
In the sixth, `'a\0b'` keeps its tail, which the UTF-8 path cuts at the NUL. The same op called with the attribute
(`new ExecuteOpArgs(input).SetAttributes(new { encoding = "utf-8" })`) equals Python TF 15/15. So the TensorFlow 2.16
native supports UTF-8 lowering, and the binding drops the attribute.

### 5.2 String tensors free their long texts while another tensor shares the buffer

`Tensor.StringTensor` builds the tensor with `TF_AllocateTensor`. That buffer's deallocator frees the memory but runs no
`tstring` destructor. So TensorFlow.NET frees the long texts itself: `SafeStringTensorHandle.ReleaseHandle`
(`src/TensorFlowNET.Core/Tensors/SafeStringTensorHandle.cs:27-43`) runs `TF_StringDealloc` on every cell when its own
handle is released.

TensorFlow shares buffers between tensors (`tf.identity` here, and any op that forwards its input buffer), so the texts
are freed while another tensor still reads them. That tensor then reads `''` (the cell is re-initialized) or a freed block (a concurrent read).
Measured in §4.3: 64/64 long texts lost, 0/64 with the source kept alive.

The fix is the prototype's ownership: build the cells in a block given to `TF_NewTensor` with a deallocator that runs
`TF_StringDealloc` per cell and then frees the block. TensorFlow calls that deallocator when the last reference is gone.

### 5.3 Text `.npy` files

- `np.save` refuses a string array (`NpyFormat.GetDtypeName` throws `NotSupportedException("")`).
- `np.load` has no `U` mapping (`NumPyImpl.load.cs`, `GetType`). It fails on `S` and on `StringDType` (§4.2).

### 5.4 Silent replacement characters

`Tensor.StringTensor(string[])` encodes with `Encoding.UTF8.GetBytes`, and `StringData()` decodes with
`Encoding.UTF8.GetString` (`Tensor.String.cs:18, 51, 59`). Both replace with U+FFFD where Python TensorFlow and NumPy
raise: a lone surrogate on the way in, invalid UTF-8 on the way out.

## 6. Options

| Option | What it takes | For | Against |
|---|---|---|---|
| A. Nothing in either repo | users write the §3.1 snippet | works today for valid text | the three edges of §3.1, about Python's speed, TensorFlow.NET's §5 defects stay |
| B. `NumSharp.Interop.TensorFlow` (in this repo, like the ONNX Runtime bridge) | the text verbs of `TfTextBridge.cs` hardened into a package; the numeric verbs (zero-copy `TF_NewTensor` over the NumSharp buffer with an ARC pin, as `AsOrtValue` does); a test project over the supported TensorFlow natives | Python TF's strictness, 2–14× Python TF's speed, correct lifetime, `StringDType` without decoding | depends on a dormant, unsigned package and on TensorFlow 2.16 natives (win-x64, linux-x64); TensorFlow.NET's `np`, `NDArray`, `Shape` and `Slice` collide with NumSharp's names in user code (aliases) |
| C. Fix TensorFlow.NET upstream | §5.1 is one line; §5.2 is the `TF_NewTensor` ownership; §5.3/§5.4 are larger | TensorFlow.NET users benefit with or without NumSharp | TensorFlow.NET has had no commit since 2025-01-22 and needs a release to ship anything |
| D. TensorFlow.NET re-adopts NumSharp as its numpy layer | a TensorFlow.NET re-architecture plus a retarget past `netstandard2.0` | one NumPy for .NET | `Tensorflow.NumPy.NDArray` is a `Tensor` (GPU, graph mode) by design; not a strings question |

Decisions for B:
- **D1. Accept `StringDType` at export?** Python TF refuses it ("Unsupported numpy type"), as Python onnxruntime does,
  and NumSharp's ONNX Runtime bridge mirrors that refusal. Accepting it makes the natural carrier of `tf.string` cross
  with no decode.
- **D2. The import target.** The choice is `StringDType` (strict UTF-8, `t.numpy().astype(StringDType())`), `S` (any
  bytes, `t.numpy().astype('S')`, trailing NULs stripped, every element as wide as the longest), or both behind an
  explicit parameter, as the prototype has it.
- **D3. Offer the view payload?** It costs 17 ms instead of 47 ms per million `StringDType` texts, with a no-write
  contract on the NumSharp array.
- **D4. Text only, or text plus numerics?** A TensorFlow bridge without numeric tensors would be unusual.

The recommendation is B if TensorFlow.NET interop is a product goal, with D1–D4 decided first. C is worth doing either
way for §5.1 and §5.2. D is not recommended.

## 7. Reproducing

The folder [tensorflow-net-strings/](tensorflow-net-strings/) holds:
- `py_reference.py`: writes `texts.json`, `reference.json` (TF's and NumPy's answers), `conversions.json` (§2) and the
  four `.npy` files;
- `tfprobe.csproj` + `Program.cs` + `TfTextBridge.cs`: the probe and the prototype bridge;
- `py_timing.py`: the Python twin of the timing section;
- `run_timing.cmd`: runs both timings back to back for a pinned start.

1. Make a Python venv with TensorFlow and NumPy 2.4.2: `python -m venv tfvenv`, then
   `tfvenv/Scripts/python -m pip install numpy==2.4.2 tensorflow==2.21.0`. Do not install TensorFlow into the oracle's
   interpreter: pip may change its NumPy.
2. Run `tfvenv/Scripts/python -I py_reference.py <data dir>`.
3. Run `dotnet build -c Release` in the folder. It restores TensorFlow.NET 0.150.0 and the 191 MB CPU runtime, and
   builds `../../../src/NumSharp.Core`. Build it from a worktree no other session builds in.
4. Run `TF_CPP_MIN_LOG_LEVEL=2 bin/Release/net10.0/NumSharp.DotNetRunScript.exe <data dir> all`. It prints PASS / FAIL /
   INFO, then `FAILURES 0`.
5. For the timing, run ``cmd //c "start /b /wait /affinity 14 cmd /c <absolute path of run_timing.cmd> <data dir> <tfvenv python.exe>"``.
   It writes `timing_cs.txt` and `timing_py.txt` into the data dir.
