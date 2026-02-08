# NumSharp Open Issues - Categorized

**Total open issues: 142**

| Category | Count |
|----------|-------|
| [np.* Bugs](#np-bugs) | 29 |
| [Other Bugs](#other-bugs) | 28 |
| [Missing np.* Functions](#missing-np-functions) | 34 |
| [Feature Requests](#feature-requests) | 21 |
| [Performance Issues](#performance-issues) | 4 |
| [Questions / How-To / Off-Topic](#questions--how-to--off-topic) | 27 |

---

## np.* Bugs

Bugs in existing `np.*` functions: wrong results, crashes, or API typos.

| Issue | Title | Author | Labels |
|-------|-------|--------|--------|
| [#315](https://github.com/SciSharp/NumSharp/issues/315) | ToString should truncate its output | @thomasd3 | bug, enhancement |
| [#362](https://github.com/SciSharp/NumSharp/issues/362) | Implicit operators for >, >=, <, <= | @deepakkumar1984 | help wanted, missing feature/s |
| [#398](https://github.com/SciSharp/NumSharp/issues/398) | Typo in library np.random.stardard | @QadiymStewart |  |
| [#405](https://github.com/SciSharp/NumSharp/issues/405) | np.argsort not sorting properly | @tk4218 |  |
| [#407](https://github.com/SciSharp/NumSharp/issues/407) | np.negative is not working ? | @LordTrololo |  |
| [#408](https://github.com/SciSharp/NumSharp/issues/408) | np.meshgrid() has a hidden error returning wrong results | @LordTrololo |  |
| [#418](https://github.com/SciSharp/NumSharp/issues/418) | help me | @mak27arr |  |
| [#419](https://github.com/SciSharp/NumSharp/issues/419) | np.meshgrid error | @mak27arr |  |
| [#426](https://github.com/SciSharp/NumSharp/issues/426) | arctan2() returning incorrect value | @RoseberryPi |  |
| [#428](https://github.com/SciSharp/NumSharp/issues/428) | Typo in NDArray.ToMuliArray method name | @jpmn |  |
| [#436](https://github.com/SciSharp/NumSharp/issues/436) | np.searchsorted error! | @wangfeixing |  |
| [#437](https://github.com/SciSharp/NumSharp/issues/437) | argmin is not the same with numpy | @tomachristian |  |
| [#443](https://github.com/SciSharp/NumSharp/issues/443) | 0.3.0 from NuGet throwing NotSupportedException on negate function call | @gandalfh |  |
| [#447](https://github.com/SciSharp/NumSharp/issues/447) | np.sum() Is supported on numsharp0.20.5, but not on NumSharp0.30.0 | @lijianxin520 |  |
| [#452](https://github.com/SciSharp/NumSharp/issues/452) | [missing feature/s] NumSharp's np.around() method is missing decimals parameter which is available in NumPy | @shashi4u |  |
| [#456](https://github.com/SciSharp/NumSharp/issues/456) | silent catastrophe in implicit casting singleton array to value type | @dmacd |  |
| [#461](https://github.com/SciSharp/NumSharp/issues/461) | np.save incorrectly saves System.Byte arrays as signed | @rikkitook |  |
| [#466](https://github.com/SciSharp/NumSharp/issues/466) | [Bug] np.random.choice raise Exception | @QingtaoLi1 |  |
| [#468](https://github.com/SciSharp/NumSharp/issues/468) | np_array.convolve returning Null | @dklein9500 |  |
| [#470](https://github.com/SciSharp/NumSharp/issues/470) | Numsharp0.30.0  np.random.choice() method missing cause Exception | @UCtreespring |  |
| [#477](https://github.com/SciSharp/NumSharp/issues/477) | Different Result between NumPy and NumSharp with np.matmul Function | @Koyamin |  |
| [#487](https://github.com/SciSharp/NumSharp/issues/487) | linspace to Array as type float, while other functions as type double | @changjian-github |  |
| [#488](https://github.com/SciSharp/NumSharp/issues/488) | np.random.choice raised  System.NotSupportedException | @alvinfebriando |  |
| [#490](https://github.com/SciSharp/NumSharp/issues/490) | np.random.choice with replace: false produces duplicates | @GThibeault |  |
| [#499](https://github.com/SciSharp/NumSharp/issues/499) | Possible typo "ToMuliDimArray()" | @sappho192 |  |
| [#505](https://github.com/SciSharp/NumSharp/issues/505) | `np.convolve` return null exception  | @behroozbc |  |
| [#507](https://github.com/SciSharp/NumSharp/issues/507) | np.maximum error | @Thanatos0173 |  |
| [#508](https://github.com/SciSharp/NumSharp/issues/508) | np.hstack has diffrent effect from python | @xdqa01 |  |
| [#517](https://github.com/SciSharp/NumSharp/issues/517) | Error when loading a `.npy` file containing a scalar value | @thalesfm |  |

### Breakdown

**Wrong/Unexpected Results:**
- [#398](https://github.com/SciSharp/NumSharp/issues/398) - Typo: np.random.stardard_normal (should be standard)
- [#405](https://github.com/SciSharp/NumSharp/issues/405) - np.argsort returns wrong order
- [#408](https://github.com/SciSharp/NumSharp/issues/408) - np.meshgrid returns wrong results (hidden memory bug)
- [#426](https://github.com/SciSharp/NumSharp/issues/426) - np.arctan2 returns incorrect value
- [#437](https://github.com/SciSharp/NumSharp/issues/437) - np.argmin behavior differs from NumPy
- [#456](https://github.com/SciSharp/NumSharp/issues/456) - Implicit cast of singleton NDArray silently produces wrong value (memory reinterpretation)
- [#466](https://github.com/SciSharp/NumSharp/issues/466) - np.random.choice raises NotSupportedException
- [#470](https://github.com/SciSharp/NumSharp/issues/470) - np.random.choice() throws NotSupportedException (0.30.0)
- [#477](https://github.com/SciSharp/NumSharp/issues/477) - np.matmul returns different results from NumPy for 3D arrays
- [#488](https://github.com/SciSharp/NumSharp/issues/488) - np.random.choice raises NotSupportedException
- [#490](https://github.com/SciSharp/NumSharp/issues/490) - np.random.choice with replace=false produces duplicates
- [#507](https://github.com/SciSharp/NumSharp/issues/507) - np.maximum throws random errors during repeated calls
- [#508](https://github.com/SciSharp/NumSharp/issues/508) - np.hstack produces different results from Python

**Crashes / Exceptions:**
- [#362](https://github.com/SciSharp/NumSharp/issues/362) - Comparison operators >, >=, <, <= return null
- [#418](https://github.com/SciSharp/NumSharp/issues/418) - np.meshgrid second return value is always null
- [#419](https://github.com/SciSharp/NumSharp/issues/419) - np.meshgrid second return value crashes the program
- [#436](https://github.com/SciSharp/NumSharp/issues/436) - np.searchsorted throws error on double arrays
- [#443](https://github.com/SciSharp/NumSharp/issues/443) - np.negate throws NotSupportedException on 0.30.0
- [#447](https://github.com/SciSharp/NumSharp/issues/447) - np.sum throws NotSupportedException on 0.30.0
- [#466](https://github.com/SciSharp/NumSharp/issues/466) - np.random.choice raises NotSupportedException
- [#468](https://github.com/SciSharp/NumSharp/issues/468) - NDArray.convolve() returns null
- [#470](https://github.com/SciSharp/NumSharp/issues/470) - np.random.choice() throws NotSupportedException (0.30.0)
- [#488](https://github.com/SciSharp/NumSharp/issues/488) - np.random.choice raises NotSupportedException
- [#505](https://github.com/SciSharp/NumSharp/issues/505) - np.convolve returns null / NullReferenceException
- [#507](https://github.com/SciSharp/NumSharp/issues/507) - np.maximum throws random errors during repeated calls
- [#517](https://github.com/SciSharp/NumSharp/issues/517) - np.load fails on scalar .npy files (off-by-one in header parsing)

**API Typos / Naming:**
- [#398](https://github.com/SciSharp/NumSharp/issues/398) - Typo: np.random.stardard_normal (should be standard)
- [#428](https://github.com/SciSharp/NumSharp/issues/428) - Typo: NDArray.ToMuliArray (should be ToMultiArray)
- [#452](https://github.com/SciSharp/NumSharp/issues/452) - np.around() missing decimals parameter
- [#499](https://github.com/SciSharp/NumSharp/issues/499) - Typo: ToMuliDimArray() should be ToMultiDimArray()

**Version 0.30.0 Regressions:**
- [#443](https://github.com/SciSharp/NumSharp/issues/443) - np.negate throws NotSupportedException on 0.30.0
- [#447](https://github.com/SciSharp/NumSharp/issues/447) - np.sum throws NotSupportedException on 0.30.0
- [#466](https://github.com/SciSharp/NumSharp/issues/466) - np.random.choice raises NotSupportedException
- [#470](https://github.com/SciSharp/NumSharp/issues/470) - np.random.choice() throws NotSupportedException (0.30.0)
- [#488](https://github.com/SciSharp/NumSharp/issues/488) - np.random.choice raises NotSupportedException

---

## Other Bugs

Bugs in core infrastructure, indexing, memory management, platform compatibility.

| Issue | Title | Author | Labels |
|-------|-------|--------|--------|
| [#366](https://github.com/SciSharp/NumSharp/issues/366) | Masking (ndarray[nd]) | @henon |  |
| [#368](https://github.com/SciSharp/NumSharp/issues/368) | Masking a slice ("...") returns null | @ohjerm |  |
| [#369](https://github.com/SciSharp/NumSharp/issues/369) | Slicing NotSupportedException | @Oceania2018 | missing feature/s |
| [#396](https://github.com/SciSharp/NumSharp/issues/396) | Bitmap.ToNDArray problem with odd bitmap width | @herrvonregen |  |
| [#410](https://github.com/SciSharp/NumSharp/issues/410) | np.save fails with IndexOutOfRangeException for jagged arrays | @Jmerk523 |  |
| [#412](https://github.com/SciSharp/NumSharp/issues/412) | The type 'NDArray' exists in both 'NumSharp.Core, Version=0.20.5.0, ' and 'NumSharp.Lite, Version=0.1.7.0,  | @sportbilly21 |  |
| [#422](https://github.com/SciSharp/NumSharp/issues/422) | Index of element with a condiction | @EnricoBeltramo |  |
| [#423](https://github.com/SciSharp/NumSharp/issues/423) | "System.NotImplementedException: '' --> someArray = np.frombuffer(byteBuffer.ToArray<byte>(), np.uint32); | @mehmetcanbalci-Notrino |  |
| [#430](https://github.com/SciSharp/NumSharp/issues/430) |  NumSharp.Backends.Unmanaged.UnmanagedMemoryBlock`1 fails on Mono on Linux | @kgoderis |  |
| [#433](https://github.com/SciSharp/NumSharp/issues/433) | NDArray exists in both NumSharp.Core, Version=0.20.5.0 and NumSharp.Lite, Version=0.1.9.0 | @gscheck |  |
| [#434](https://github.com/SciSharp/NumSharp/issues/434) | AccessViolationException when selecting indexes using ndarray[ndarray] and setting a scalar value | @lijianxin520 | bug |
| [#440](https://github.com/SciSharp/NumSharp/issues/440) | NDArray.ToBitmap() has critical issue with 24bpp VERTICAL images | @MiroslavKabat | bug |
| [#448](https://github.com/SciSharp/NumSharp/issues/448) | Debug.Assert(...) causes tests to stop the entire process | @Nucs | bug |
| [#455](https://github.com/SciSharp/NumSharp/issues/455) | NumSharp does not allow building with IL2CPP via Unity  | @julia-koziel |  |
| [#467](https://github.com/SciSharp/NumSharp/issues/467) | NumSharp and Tensorflow.NET works on Desktop but fails on Cloud Web Service (.NET 5) | @marsousi |  |
| [#471](https://github.com/SciSharp/NumSharp/issues/471) | Unhandled Exception: System.NotSupportedException: Specified method is not supported. | @KonardAdams |  |
| [#475](https://github.com/SciSharp/NumSharp/issues/475) | ToBitmap fails if not contiguous because of Broadcast mismatch | @ponzis |  |
| [#476](https://github.com/SciSharp/NumSharp/issues/476) | Numsharp.Core contains many Debug.Assert() lines | @rtwalterson |  |
| [#484](https://github.com/SciSharp/NumSharp/issues/484) | np.load System.Exception | @Kiord |  |
| [#491](https://github.com/SciSharp/NumSharp/issues/491) | ToBitmap() - datatype mistmatch | @davidvct |  |
| [#492](https://github.com/SciSharp/NumSharp/issues/492) | critical vulnerability in version 5.0.2 of system.drawing.common | @jkl-ds |  |
| [#493](https://github.com/SciSharp/NumSharp/issues/493) | Numsharp array output in .net interactive notebooks is misleading | @oxygen-dioxide |  |
| [#501](https://github.com/SciSharp/NumSharp/issues/501) | Memory leak? | @TakuNishiumi |  |
| [#506](https://github.com/SciSharp/NumSharp/issues/506) | Cannot create an NDArray of shorts | @NickBotelho |  |
| [#509](https://github.com/SciSharp/NumSharp/issues/509) | Extremely poor performance on sum reduce | @lucdem |  |
| [#514](https://github.com/SciSharp/NumSharp/issues/514) | SetItem for multiple Ids not working | @MaxOmlor |  |
| [#519](https://github.com/SciSharp/NumSharp/issues/519) | BUG:  NDArray filted_array = ori_array[max_prob > conf_threshold]; | @1Zengy |  |
| [#520](https://github.com/SciSharp/NumSharp/issues/520) | Can't convert to Vector3 | @xiaoshux |  |

### Breakdown

**Indexing / Slicing / Masking:**
- [#366](https://github.com/SciSharp/NumSharp/issues/366) - Boolean masking ndarray[nd] throws NotSupportedException on setter
- [#368](https://github.com/SciSharp/NumSharp/issues/368) - Masking a slice ('...') returns null
- [#410](https://github.com/SciSharp/NumSharp/issues/410) - np.save fails with IndexOutOfRangeException for jagged arrays
- [#422](https://github.com/SciSharp/NumSharp/issues/422) - Boolean indexing / conditional filtering not working
- [#434](https://github.com/SciSharp/NumSharp/issues/434) - AccessViolationException on ndarray[ndarray] index + scalar set
- [#519](https://github.com/SciSharp/NumSharp/issues/519) - Boolean filtering randomly returns correct results or empty array

**Memory / Crashes:**
- [#430](https://github.com/SciSharp/NumSharp/issues/430) - UnmanagedMemoryBlock fails on Mono/Linux
- [#434](https://github.com/SciSharp/NumSharp/issues/434) - AccessViolationException on ndarray[ndarray] index + scalar set
- [#501](https://github.com/SciSharp/NumSharp/issues/501) - Memory leak: repeated NDArray operations consume excessive memory

**Platform / Compatibility:**
- [#430](https://github.com/SciSharp/NumSharp/issues/430) - UnmanagedMemoryBlock fails on Mono/Linux
- [#455](https://github.com/SciSharp/NumSharp/issues/455) - IL2CPP build fails in Unity (LAPACKProviderType)
- [#467](https://github.com/SciSharp/NumSharp/issues/467) - Fails on Azure Web Service / cloud (DllNotFoundException tensorflow)
- [#492](https://github.com/SciSharp/NumSharp/issues/492) - Critical vulnerability in System.Drawing.Common 5.0.2
- [#520](https://github.com/SciSharp/NumSharp/issues/520) - Cannot cast NDArray values to Unity Vector3 (implicit cast bug)

**Bitmap / Image:**
- [#396](https://github.com/SciSharp/NumSharp/issues/396) - Bitmap.ToNDArray fails with odd bitmap widths (stride alignment)
- [#440](https://github.com/SciSharp/NumSharp/issues/440) - NDArray.ToBitmap() incorrect for 24bpp vertical images
- [#475](https://github.com/SciSharp/NumSharp/issues/475) - ToBitmap fails on non-contiguous arrays (broadcast mismatch)
- [#491](https://github.com/SciSharp/NumSharp/issues/491) - ToBitmap() type mismatch (int arrays not supported)
- [#492](https://github.com/SciSharp/NumSharp/issues/492) - Critical vulnerability in System.Drawing.Common 5.0.2

---

## Missing np.* Functions

Feature requests for specific NumPy API functions not yet implemented.

| Issue | Function | Author | Labels |
|-------|----------|--------|--------|
| [#75](https://github.com/SciSharp/NumSharp/issues/75) | np.asarray | @Oceania2018 | enhancement |
| [#78](https://github.com/SciSharp/NumSharp/issues/78) | np.where | @Oceania2018 | enhancement |
| [#105](https://github.com/SciSharp/NumSharp/issues/105) | np.vdot | @Oceania2018 | enhancement |
| [#106](https://github.com/SciSharp/NumSharp/issues/106) | np.inner | @Oceania2018 | help wanted |
| [#108](https://github.com/SciSharp/NumSharp/issues/108) | np.tensordot | @Oceania2018 | help wanted |
| [#114](https://github.com/SciSharp/NumSharp/issues/114) | np.fft.fft | @Oceania2018 | enhancement |
| [#202](https://github.com/SciSharp/NumSharp/issues/202) | np.pad | @skywalkerisnull | enhancement |
| [#210](https://github.com/SciSharp/NumSharp/issues/210) | np.all (with axis support) | @Esther2013 |  |
| [#220](https://github.com/SciSharp/NumSharp/issues/220) | np.flip | @pkingwsd | enhancement |
| [#221](https://github.com/SciSharp/NumSharp/issues/221) | np.rot90 | @pkingwsd |  |
| [#239](https://github.com/SciSharp/NumSharp/issues/239) | np.linalg.norm (full implementation) | @henon | enhancement |
| [#298](https://github.com/SciSharp/NumSharp/issues/298) | np.random.choice (weighted sampling) | @Plankton555 | enhancement |
| [#360](https://github.com/SciSharp/NumSharp/issues/360) | np.any (with axis support) | @Oceania2018 | enhancement |
| [#365](https://github.com/SciSharp/NumSharp/issues/365) | np.nonzero | @Oceania2018 | enhancement |
| [#373](https://github.com/SciSharp/NumSharp/issues/373) | np.median | @turowicz | help wanted, missing feature/s |
| [#374](https://github.com/SciSharp/NumSharp/issues/374) | np.append | @solarflarefx | help wanted, missing feature/s |
| [#378](https://github.com/SciSharp/NumSharp/issues/378) | np.frombuffer | @Nucs | enhancement, missing feature/s |
| [#397](https://github.com/SciSharp/NumSharp/issues/397) | np.tile | @QadiymStewart |  |
| [#413](https://github.com/SciSharp/NumSharp/issues/413) | np.split | @lqdev |  |
| [#414](https://github.com/SciSharp/NumSharp/issues/414) | np.delete (currently returns null) | @simonbuehler |  |
| [#415](https://github.com/SciSharp/NumSharp/issues/415) | Boolean indexing and np.where | @joshmyersdean |  |
| [#439](https://github.com/SciSharp/NumSharp/issues/439) | np.where (re-requested) | @minhduc66532 | missing feature/s |
| [#441](https://github.com/SciSharp/NumSharp/issues/441) | np.linalg.norm | @minhduc66532 | missing feature/s |
| [#445](https://github.com/SciSharp/NumSharp/issues/445) | np.dot with preallocated output array | @bigdimboom | missing feature/s |
| [#449](https://github.com/SciSharp/NumSharp/issues/449) | np.isclose / np.allclose (currently dead code) | @koliyo | missing feature/s |
| [#450](https://github.com/SciSharp/NumSharp/issues/450) | np.diag | @syemhusa | missing feature/s |
| [#454](https://github.com/SciSharp/NumSharp/issues/454) | np.linalg.lstsq (currently returns null) | @yangjiandendi |  |
| [#464](https://github.com/SciSharp/NumSharp/issues/464) | np.random.triangular | @ppsdatta |  |
| [#473](https://github.com/SciSharp/NumSharp/issues/473) | Bitwise shift and OR operators on NDArray | @MichielMans |  |
| [#480](https://github.com/SciSharp/NumSharp/issues/480) | np.unravel_index | @iainross |  |
| [#485](https://github.com/SciSharp/NumSharp/issues/485) | np.linalg.norm (re-requested) | @williamlzw |  |
| [#486](https://github.com/SciSharp/NumSharp/issues/486) | Slice assignment (e.g. preds[:,:,0] = ...) | @burungiu |  |
| [#497](https://github.com/SciSharp/NumSharp/issues/497) | np.linalg.pinv | @gsgou |  |
| [#515](https://github.com/SciSharp/NumSharp/issues/515) | np.tile (re-requested) | @MaxOmlor |  |

### By Area

**Linear Algebra:**
- [#105](https://github.com/SciSharp/NumSharp/issues/105) - np.vdot
- [#106](https://github.com/SciSharp/NumSharp/issues/106) - np.inner
- [#108](https://github.com/SciSharp/NumSharp/issues/108) - np.tensordot
- [#239](https://github.com/SciSharp/NumSharp/issues/239) - np.linalg.norm (full implementation)
- [#441](https://github.com/SciSharp/NumSharp/issues/441) - np.linalg.norm
- [#445](https://github.com/SciSharp/NumSharp/issues/445) - np.dot with preallocated output array
- [#454](https://github.com/SciSharp/NumSharp/issues/454) - np.linalg.lstsq (currently returns null)
- [#485](https://github.com/SciSharp/NumSharp/issues/485) - np.linalg.norm (re-requested)
- [#497](https://github.com/SciSharp/NumSharp/issues/497) - np.linalg.pinv

**Array Creation / Manipulation:**
- [#75](https://github.com/SciSharp/NumSharp/issues/75) - np.asarray
- [#202](https://github.com/SciSharp/NumSharp/issues/202) - np.pad
- [#220](https://github.com/SciSharp/NumSharp/issues/220) - np.flip
- [#221](https://github.com/SciSharp/NumSharp/issues/221) - np.rot90
- [#374](https://github.com/SciSharp/NumSharp/issues/374) - np.append
- [#378](https://github.com/SciSharp/NumSharp/issues/378) - np.frombuffer
- [#397](https://github.com/SciSharp/NumSharp/issues/397) - np.tile
- [#413](https://github.com/SciSharp/NumSharp/issues/413) - np.split
- [#414](https://github.com/SciSharp/NumSharp/issues/414) - np.delete (currently returns null)
- [#450](https://github.com/SciSharp/NumSharp/issues/450) - np.diag
- [#480](https://github.com/SciSharp/NumSharp/issues/480) - np.unravel_index
- [#486](https://github.com/SciSharp/NumSharp/issues/486) - Slice assignment (e.g. preds[:,:,0] = ...)
- [#515](https://github.com/SciSharp/NumSharp/issues/515) - np.tile (re-requested)

**Logic / Selection / Indexing:**
- [#78](https://github.com/SciSharp/NumSharp/issues/78) - np.where
- [#210](https://github.com/SciSharp/NumSharp/issues/210) - np.all (with axis support)
- [#360](https://github.com/SciSharp/NumSharp/issues/360) - np.any (with axis support)
- [#365](https://github.com/SciSharp/NumSharp/issues/365) - np.nonzero
- [#415](https://github.com/SciSharp/NumSharp/issues/415) - Boolean indexing and np.where
- [#439](https://github.com/SciSharp/NumSharp/issues/439) - np.where (re-requested)
- [#445](https://github.com/SciSharp/NumSharp/issues/445) - np.dot with preallocated output array
- [#449](https://github.com/SciSharp/NumSharp/issues/449) - np.isclose / np.allclose (currently dead code)

**Statistics:**
- [#373](https://github.com/SciSharp/NumSharp/issues/373) - np.median

**Random:**
- [#298](https://github.com/SciSharp/NumSharp/issues/298) - np.random.choice (weighted sampling)
- [#464](https://github.com/SciSharp/NumSharp/issues/464) - np.random.triangular

**Other:**
- [#114](https://github.com/SciSharp/NumSharp/issues/114) - np.fft.fft
- [#473](https://github.com/SciSharp/NumSharp/issues/473) - Bitwise shift and OR operators on NDArray

---

## Feature Requests

Non-np.* enhancements: architecture, tooling, new capabilities, ecosystem.

| Issue | Title | Author | Labels |
|-------|-------|--------|--------|
| [#95](https://github.com/SciSharp/NumSharp/issues/95) | Extend the guidelines | @dotChris90 |  |
| [#111](https://github.com/SciSharp/NumSharp/issues/111) | NumSharp GPU acceleration | @Oceania2018 | enhancement, help wanted, further discuss |
| [#116](https://github.com/SciSharp/NumSharp/issues/116) | Intel Math Kernel Library (MKL) | @Oceania2018 | enhancement, help wanted |
| [#129](https://github.com/SciSharp/NumSharp/issues/129) | Doc : Better specification for all classes (What class has what task) | @dotChris90 | further discuss |
| [#190](https://github.com/SciSharp/NumSharp/issues/190) | Compressed Sparse Format | @Oceania2018 | enhancement |
| [#211](https://github.com/SciSharp/NumSharp/issues/211) | implement scipy interpolate? | @xinqipony | enhancement |
| [#284](https://github.com/SciSharp/NumSharp/issues/284) | [Discussion] Ground Rules and Library Structure/Architecture | @Nucs | further discuss |
| [#326](https://github.com/SciSharp/NumSharp/issues/326) | Lazy loading | @aidevnn | further discuss |
| [#340](https://github.com/SciSharp/NumSharp/issues/340) | Memory Limitations | @Nucs | enhancement |
| [#341](https://github.com/SciSharp/NumSharp/issues/341) | NDArray string problem | @lokinfey | missing feature/s |
| [#343](https://github.com/SciSharp/NumSharp/issues/343) | Built-in System.Drawing.Image and Bitmap methods | @Nucs | enhancement |
| [#349](https://github.com/SciSharp/NumSharp/issues/349) | Scipy.Signal | @natank1 | missing feature/s |
| [#351](https://github.com/SciSharp/NumSharp/issues/351) | Proper way to iterate using IEnumerable<T> | @Nucs | enhancement |
| [#361](https://github.com/SciSharp/NumSharp/issues/361) | Mixing indices and slices in NDArray[...] | @Oceania2018 | enhancement |
| [#363](https://github.com/SciSharp/NumSharp/issues/363) | Add `NDIterator<T>` overload with support for specific axis. | @Nucs | missing feature/s |
| [#372](https://github.com/SciSharp/NumSharp/issues/372) | Clustering Example | @turowicz | help wanted, missing feature/s |
| [#375](https://github.com/SciSharp/NumSharp/issues/375) | Slice assignment? | @solarflarefx | missing feature/s |
| [#435](https://github.com/SciSharp/NumSharp/issues/435) | Complex number support? | @cgranade |  |
| [#479](https://github.com/SciSharp/NumSharp/issues/479) | Lacking/Outdated Documentation | @Tianmaru |  |
| [#500](https://github.com/SciSharp/NumSharp/issues/500) | The fact that such an excellent project has many unimplemented APIs and is no longer being maintained is regrettable. | @HCareLou |  |
| [#523](https://github.com/SciSharp/NumSharp/issues/523) | Multi-threading supported? | @sawyermade |  |

---

## Performance Issues

Reports of NumSharp being significantly slower than expected.

| Issue | Title | Author | Summary |
|-------|-------|--------|---------|
| [#421](https://github.com/SciSharp/NumSharp/issues/421) | Performance | @mishun | Overall performance far slower than expected vs naive C# |
| [#427](https://github.com/SciSharp/NumSharp/issues/427) | Performance on np.matmul | @Banyc | np.matmul 100x slower than NumPy (3-4s vs 0.03s) |
| [#451](https://github.com/SciSharp/NumSharp/issues/451) | np.argmax is slow | @feiyuhuahuo | np.argmax very slow on large arrays (~500ms) |
| [#509](https://github.com/SciSharp/NumSharp/issues/509) | Extremely poor performance on sum reduce | @lucdem | sum(axis:0) 150x slower than NumPy |

---

## Questions / How-To / Off-Topic

Usage questions, conversion help, off-topic requests.

| Issue | Title | Author |
|-------|-------|--------|
| [#70](https://github.com/SciSharp/NumSharp/issues/70) | Let more people know about NumSharp | @Oceania2018 |
| [#238](https://github.com/SciSharp/NumSharp/issues/238) | How to mimic Python's nice column and row access (i.e matrix[:, 2])? | @henon |
| [#383](https://github.com/SciSharp/NumSharp/issues/383) | is there any way to convert NumSharp.NDArray to Numpy.NDarray? | @lelelemonade |
| [#384](https://github.com/SciSharp/NumSharp/issues/384) | Save NDArray as png image | @solarflarefx |
| [#386](https://github.com/SciSharp/NumSharp/issues/386) | how to read .csv file with Numsharp? | @guang7400613 |
| [#390](https://github.com/SciSharp/NumSharp/issues/390) | How to create an NDArray from pointer and NPTypeCode? | @LarryThermo |
| [#401](https://github.com/SciSharp/NumSharp/issues/401) | How to convert NDArray to list | @Sullivanecidi |
| [#406](https://github.com/SciSharp/NumSharp/issues/406) | C# --> convert image to NDarray | @R06921096Yen |
| [#411](https://github.com/SciSharp/NumSharp/issues/411) | PyObject to NDArray | @aaronavi |
| [#416](https://github.com/SciSharp/NumSharp/issues/416) | how to make NumSharp.NDArray from Numpy.NDarray? | @djagatiya |
| [#424](https://github.com/SciSharp/NumSharp/issues/424) | The type or namespace name 'NumSharp' could not be found (are you missing a using directive or an assembly reference?) [Assembly-CSharp]csharp(CS0246) | @rcffc |
| [#438](https://github.com/SciSharp/NumSharp/issues/438) | How to get the inverse of a 2D matrix？ | @Mingrui-Yu |
| [#446](https://github.com/SciSharp/NumSharp/issues/446) | Unable to use np.dot due to "Specified method unsupported" error | @moonlitlyra |
| [#462](https://github.com/SciSharp/NumSharp/issues/462) | How to use the repo to convert some Python code? | @zydjohnHotmail |
| [#465](https://github.com/SciSharp/NumSharp/issues/465) | how could I transform between NumSharp.NDArray with Tensorflow.Numpy.NDArray? | @cross-hello |
| [#472](https://github.com/SciSharp/NumSharp/issues/472) | How to calculate the rank of a matrix with NumSharp? | @drtujugkhjk |
| [#481](https://github.com/SciSharp/NumSharp/issues/481) | Normal disttribution in NumSharp | @rthota90 |
| [#482](https://github.com/SciSharp/NumSharp/issues/482) | 大佬，能否用C#还原 java 的一个求解器 optaplanner | @JavaScript-zt |
| [#483](https://github.com/SciSharp/NumSharp/issues/483) | How to convert List<NDArray> to NDArray | @williamlzw |
| [#494](https://github.com/SciSharp/NumSharp/issues/494) | Hello, has SciSharp/NumSharp stopped development and maintenance? | @sdyby2006 |
| [#495](https://github.com/SciSharp/NumSharp/issues/495) | What is the exposed method for percentiles & median using np? | @vikassingh281 |
| [#496](https://github.com/SciSharp/NumSharp/issues/496) | Can NumSharp fit polynomial surface equations? | @liyu3519 |
| [#498](https://github.com/SciSharp/NumSharp/issues/498) | Is there an example on how to use it with IronPython? | @william19941994 |
| [#510](https://github.com/SciSharp/NumSharp/issues/510) | How to save a nested dictionary with `save_npz` | @rqx110 |
| [#511](https://github.com/SciSharp/NumSharp/issues/511) | Does Numsharp work in Unity with the IL2CPP backend? | @jacob-jacob-jacob |
| [#512](https://github.com/SciSharp/NumSharp/issues/512) | how to change to Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>? | @BingGitCn |
| [#516](https://github.com/SciSharp/NumSharp/issues/516) | Very helpful work. Keep at it | @duxuan11 |
