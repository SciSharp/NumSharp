# PR #573 File Changes Summary

**Legend:** `A` = Added | `M` = Modified | `D` = Deleted | `R` = Renamed

---


## 📁 .claude/

### [M] `CLAUDE.md`
Refresh .claude/CLAUDE.md documentation docs: add ILKernelGenerator documentation and refactor plan docs: update CLAUDE.md - mark medium severity bugs as fixed docs: update CLAUDE.md bug list - mark fixed bugs feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): add SIMD axis reduction kernels and fix NaN edge cases feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD fix(unique): sort unique values to match NumPy behavior 


## 📁 Root

### [A] `.gitattributes`
chore: add .gitattributes for consistent line endings 

### [M] `.gitignore`
chore(.gitignore): exclude local design documents chore: remove worktree tracking from git, add to .gitignore 

### [A] `CHANGES.md`
chore: remove internal planning docs and changelog from PR docs: fix broken documentation URLs (scipy → numpy.org) refactor(random): align parameter names and docs with NumPy API refactor: move broadcast utilities from DefaultEngine to Shape struct 

### [M] `README.md`
docs: fix broken documentation URLs (scipy → numpy.org) docs: fix versioned numpy URLs in README to use /stable/ 

### [A] `RELEASE_0.41.0-prerelease.md`
docs: fix broken documentation URLs (scipy → numpy.org) 


## 📁 benchmark/NumSharp.Benchmark.GraphEngine/Benchmarks/Allocation/

### [A] `AllocationMicroBenchmarks.cs`
feat(benchmark): add NativeMemory allocation benchmarks for issue #528 feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling 

### [A] `AllocationSizeBenchmarks.cs`
feat(benchmark): add NativeMemory allocation benchmarks for issue #528 feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling 

### [A] `NumSharpAllocationBenchmarks.cs`
feat(benchmark): add NativeMemory allocation benchmarks for issue #528 feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling 

### [A] `ZeroInitBenchmarks.cs`
feat(benchmark): add NativeMemory allocation benchmarks for issue #528 feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling 


## 📁 benchmark/NumSharp.Benchmark.GraphEngine/Benchmarks/Arithmetic/

### [M] `ModuloBenchmarks.cs`
feat(benchmark): comprehensive benchmark infrastructure with full NumPy comparison 


## 📁 benchmark/NumSharp.Benchmark.GraphEngine/Benchmarks/

### [A] `SimdVsScalarBenchmarks.cs`
bench: add SIMD vs scalar benchmark suite 


## 📁 benchmark/NumSharp.Benchmark.GraphEngine/Infrastructure/

### [M] `BenchmarkBase.cs`
feat(benchmark): comprehensive benchmark infrastructure with full NumPy comparison 


## 📁 benchmark/NumSharp.Benchmark.GraphEngine/

### [M] `Program.cs`
bench: add SIMD vs scalar benchmark suite 


## 📁 benchmark/NumSharp.Benchmark.Python/

### [M] `numpy_benchmark.py`
feat(benchmark): comprehensive benchmark infrastructure with full NumPy comparison 


## 📁 benchmark/

### [M] `README.md`
feat(benchmark): comprehensive benchmark infrastructure with full NumPy comparison 

### [M] `benchmark-report.md`
feat(benchmark): comprehensive benchmark infrastructure with full NumPy comparison 

### [M] `run-benchmarks.ps1`
feat(benchmark): comprehensive benchmark infrastructure with full NumPy comparison 


## 📁 benchmark/scripts/

### [M] `merge-results.py`
feat(benchmark): comprehensive benchmark infrastructure with full NumPy comparison 


## 📁 docs/

### [A] `exceptions.md`
Added plans to unfinished work 


## 📁 docs/issues/

### [M] `issue-0070-let-more-people-know-about-numsharp.md`
chore: normalize line endings in issue docs 

### [M] `issue-0075-implement-numpy.asarray.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0078-implement-numpy.where.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0095-extend-the-guidelines.md`
chore: normalize line endings in issue docs 

### [M] `issue-0105-implement-numpy.vdot.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0106-implement-numpy.inner.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0108-implement-numpy.tensordot.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0114-implement-numpy.fft.fft.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0116-intel-math-kernel-library-mkl.md`
chore: normalize line endings in issue docs 

### [M] `issue-0129-doc-better-specification-for-all-classes-what-class-has-what-task.md`
chore: normalize line endings in issue docs 

### [M] `issue-0190-compressed-sparse-format.md`
chore: normalize line endings in issue docs 

### [M] `issue-0202-implement-np.pad.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0210-add-numpy.all.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0211-implement-scipy-interpolate.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0220-numpy.flip.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0221-numpy.rot90.md`
docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0238-how-to-mimic-pythons-nice-column-and-row-access-i.e-matrix-2.md`
chore: normalize line endings in issue docs 

### [M] `issue-0239-np.linalg.norm.md`
chore: normalize line endings in issue docs 

### [M] `issue-0284-discussion-ground-rules-and-library-structure-architecture.md`
chore: normalize line endings in issue docs 

### [M] `issue-0298-implement-numpy.random.choice.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0315-tostring-should-truncate-its-output.md`
chore: normalize line endings in issue docs 

### [M] `issue-0326-lazy-loading.md`
chore: normalize line endings in issue docs 

### [M] `issue-0340-memory-limitations.md`
chore: normalize line endings in issue docs 

### [M] `issue-0341-ndarray-string-problem.md`
chore: normalize line endings in issue docs 

### [M] `issue-0343-built-in-system.drawing.image-and-bitmap-methods.md`
chore: normalize line endings in issue docs 

### [M] `issue-0349-scipy.signal.md`
chore: normalize line endings in issue docs 

### [M] `issue-0351-proper-way-to-iterate-using-ienumerable-t.md`
chore: normalize line endings in issue docs 

### [M] `issue-0360-np.any.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0361-mixing-indices-and-slices-in-ndarray.md`
chore: normalize line endings in issue docs 

### [M] `issue-0362-implicit-operators-for.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0363-add-nditerator-t-overload-with-support-for-specific-axis.md`
chore: normalize line endings in issue docs 

### [M] `issue-0365-np.nonzero.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0366-masking-ndarray-nd.md`
chore: normalize line endings in issue docs 

### [M] `issue-0368-masking-a-slice-...-returns-null.md`
chore: normalize line endings in issue docs 

### [M] `issue-0369-slicing-notsupportedexception.md`
chore: normalize line endings in issue docs 

### [M] `issue-0372-clustering-example.md`
chore: normalize line endings in issue docs 

### [M] `issue-0374-np.append.md`
chore: normalize line endings in issue docs 

### [M] `issue-0375-slice-assignment.md`
chore: normalize line endings in issue docs 

### [M] `issue-0378-add-np.frombuffer.md`
chore: normalize line endings in issue docs docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `issue-0383-is-there-any-way-to-convert-numsharp.ndarray-to-numpy.ndarray.md`
chore: normalize line endings in issue docs 

### [M] `issue-0384-save-ndarray-as-png-image.md`
chore: normalize line endings in issue docs 

### [M] `issue-0386-how-to-read-.csv-file-with-numsharp.md`
chore: normalize line endings in issue docs 

### [M] `issue-0390-how-to-create-an-ndarray-from-pointer-and-nptypecode.md`
chore: normalize line endings in issue docs 

### [M] `issue-0396-bitmap.tondarray-problem-with-odd-bitmap-width.md`
chore: normalize line endings in issue docs 

### [M] `issue-0397-missing-np.tile.md`
chore: normalize line endings in issue docs 

### [M] `issue-0398-typo-in-library-np.random.stardard.md`
chore: normalize line endings in issue docs 

### [M] `issue-0401-how-to-convert-ndarray-to-list.md`
chore: normalize line endings in issue docs 

### [M] `issue-0405-np.argsort-not-sorting-properly.md`
chore: normalize line endings in issue docs 

### [M] `issue-0406-c-convert-image-to-ndarray.md`
chore: normalize line endings in issue docs 

### [M] `issue-0407-np.negative-is-not-working.md`
chore: normalize line endings in issue docs 

### [M] `issue-0408-np.meshgrid-has-a-hidden-error-returning-wrong-results.md`
chore: normalize line endings in issue docs 

### [M] `issue-0410-np.save-fails-with-indexoutofrangeexception-for-jagged-arrays.md`
chore: normalize line endings in issue docs 

### [M] `issue-0411-pyobject-to-ndarray.md`
chore: normalize line endings in issue docs 

### [M] `issue-0412-the-type-ndarray-exists-in-both-numsharp.core-version-0.20.5.0-and-numsh.md`
chore: normalize line endings in issue docs 

### [M] `issue-0413-ndarray-split.md`
chore: normalize line endings in issue docs 

### [M] `issue-0414-implementation-of-np.delete-working.md`
chore: normalize line endings in issue docs 

### [M] `issue-0416-how-to-make-numsharp.ndarray-from-numpy.ndarray.md`
chore: normalize line endings in issue docs 

### [M] `issue-0418-help-me.md`
chore: normalize line endings in issue docs 

### [M] `issue-0421-performance.md`
chore: normalize line endings in issue docs 

### [M] `issue-0422-index-of-element-with-a-condiction.md`
chore: normalize line endings in issue docs 

### [M] `issue-0423-system.notimplementedexception-somearray-np.frombuffer-bytebuffer.toa.md`
chore: normalize line endings in issue docs 

### [M] `issue-0424-the-type-or-namespace-name-numsharp-could-not-be-found-are-you-missing-a-usin.md`
chore: normalize line endings in issue docs 

### [M] `issue-0426-arctan2-returning-incorrect-value.md`
chore: normalize line endings in issue docs 

### [M] `issue-0427-performance-on-np.matmul.md`
chore: normalize line endings in issue docs 

### [M] `issue-0428-typo-in-ndarray.tomuliarray-method-name.md`
chore: normalize line endings in issue docs 

### [M] `issue-0430-numsharp.backends.unmanaged.unmanagedmemoryblock1-fails-on-mono-on-linux.md`
chore: normalize line endings in issue docs 

### [M] `issue-0433-ndarray-exists-in-both-numsharp.core-version-0.20.5.0-and-numsharp.lite-versio.md`
chore: normalize line endings in issue docs 

### [M] `issue-0434-accessviolationexception-when-selecting-indexes-using-ndarray-ndarray-and-setti.md`
chore: normalize line endings in issue docs 

### [M] `issue-0435-complex-number-support.md`
chore: normalize line endings in issue docs 

### [M] `issue-0436-np.searchsorted-error.md`
chore: normalize line endings in issue docs 

### [M] `issue-0437-argmin-is-not-the-same-with-numpy.md`
chore: normalize line endings in issue docs 

### [M] `issue-0438-how-to-get-the-inverse-of-a-2d-matrix.md`
chore: normalize line endings in issue docs 

### [M] `issue-0439-where-is-np.where-function.md`
chore: normalize line endings in issue docs 

### [M] `issue-0440-ndarray.tobitmap-has-critical-issue-with-24bpp-vertical-images.md`
chore: normalize line endings in issue docs 

### [M] `issue-0443-0.3.0-from-nuget-throwing-notsupportedexception-on-negate-function-call.md`
chore: normalize line endings in issue docs 

### [M] `issue-0445-how-can-provide-output-for-np.dot.md`
chore: normalize line endings in issue docs 

### [M] `issue-0446-unable-to-use-np.dot-due-to-specified-method-unsupported-error.md`
chore: normalize line endings in issue docs 

### [M] `issue-0447-np.sum-is-supported-on-numsharp0.20.5-but-not-on-numsharp0.30.0.md`
chore: normalize line endings in issue docs 

### [M] `issue-0448-debug.assert-...-causes-tests-to-stop-the-entire-process.md`
chore: normalize line endings in issue docs 

### [M] `issue-0449-isclose-is-not-implemented-and-allclose-test-is-ignored.md`
chore: normalize line endings in issue docs 

### [M] `issue-0451-np.argmax-is-slow.md`
chore: normalize line endings in issue docs 

### [M] `issue-0452-missing-feature-s-numsharps-np.around-method-is-missing-decimals-parameter.md`
chore: normalize line endings in issue docs 

### [M] `issue-0454-ndarray.lstqr-doesnt-work.md`
chore: normalize line endings in issue docs 

### [M] `issue-0455-numsharp-does-not-allow-building-with-il2cpp-via-unity.md`
chore: normalize line endings in issue docs 

### [M] `issue-0456-silent-catastrophe-in-implicit-casting-singleton-array-to-value-type.md`
chore: normalize line endings in issue docs 

### [M] `issue-0461-np.save-incorrectly-saves-system.byte-arrays-as-signed.md`
chore: normalize line endings in issue docs 

### [M] `issue-0462-how-to-use-the-repo-to-convert-some-python-code.md`
chore: normalize line endings in issue docs 

### [M] `issue-0464-new-api-request-to-port-np.random.triangular.md`
chore: normalize line endings in issue docs 

### [M] `issue-0465-how-could-i-transform-between-numsharp.ndarray-with-tensorflow.numpy.ndarray.md`
chore: normalize line endings in issue docs 

### [M] `issue-0466-bug-np.random.choice-raise-exception.md`
chore: normalize line endings in issue docs 

### [M] `issue-0467-numsharp-and-tensorflow.net-works-on-desktop-but-fails-on-cloud-web-service-.ne.md`
chore: normalize line endings in issue docs 

### [M] `issue-0468-np-array.convolve-returning-null.md`
chore: normalize line endings in issue docs 

### [M] `issue-0470-numsharp0.30.0-np.random.choice-method-missing-cause-exception.md`
chore: normalize line endings in issue docs 

### [M] `issue-0471-unhandled-exception-system.notsupportedexception-specified-method-is-not-suppo.md`
chore: normalize line endings in issue docs 

### [M] `issue-0472-how-to-calculate-the-rank-of-a-matrix-with-numsharp.md`
chore: normalize line endings in issue docs 

### [M] `issue-0473-bit-shift-and-bit-or.md`
chore: normalize line endings in issue docs 

### [M] `issue-0475-tobitmap-fails-if-not-contiguous-because-of-broadcast-mismatch.md`
chore: normalize line endings in issue docs 

### [M] `issue-0476-numsharp.core-contains-many-debug.assert-lines.md`
chore: normalize line endings in issue docs 

### [M] `issue-0477-different-result-between-numpy-and-numsharp-with-np.matmul-function.md`
chore: normalize line endings in issue docs 

### [M] `issue-0479-lacking-outdated-documentation.md`
chore: normalize line endings in issue docs 

### [M] `issue-0480-numsharp-equivalent-for-unravel-index.md`
chore: normalize line endings in issue docs 

### [M] `issue-0481-normal-disttribution-in-numsharp.md`
chore: normalize line endings in issue docs 

### [M] `issue-0483-how-to-convert-list-ndarray-to-ndarray.md`
chore: normalize line endings in issue docs 

### [M] `issue-0484-np.load-system.exception.md`
chore: normalize line endings in issue docs 

### [M] `issue-0486-slice-assign.md`
chore: normalize line endings in issue docs 

### [M] `issue-0487-linspace-to-array-as-type-float-while-other-functions-as-type-double.md`
chore: normalize line endings in issue docs 

### [M] `issue-0488-np.random.choice-raised-system.notsupportedexception.md`
chore: normalize line endings in issue docs 

### [M] `issue-0490-np.random.choice-with-replace-false-produces-duplicates.md`
chore: normalize line endings in issue docs 

### [M] `issue-0491-tobitmap-datatype-mistmatch.md`
chore: normalize line endings in issue docs 

### [M] `issue-0492-critical-vulnerability-in-version-5.0.2-of-system.drawing.common.md`
chore: normalize line endings in issue docs 

### [M] `issue-0493-numsharp-array-output-in-.net-interactive-notebooks-is-misleading.md`
chore: normalize line endings in issue docs 

### [M] `issue-0494-hello-has-scisharp-numsharp-stopped-development-and-maintenance.md`
chore: normalize line endings in issue docs 

### [M] `issue-0497-np.linalg.pinv-not-supported.md`
chore: normalize line endings in issue docs 

### [M] `issue-0498-is-there-an-example-on-how-to-use-it-with-ironpython.md`
chore: normalize line endings in issue docs 

### [M] `issue-0499-possible-typo-tomulidimarray.md`
chore: normalize line endings in issue docs 

### [M] `issue-0501-memory-leak.md`
chore: normalize line endings in issue docs 

### [M] `issue-0505-np.convolve-return-null-exception.md`
chore: normalize line endings in issue docs 

### [M] `issue-0506-cannot-create-an-ndarray-of-shorts.md`
chore: normalize line endings in issue docs 

### [M] `issue-0507-np.maximum-error.md`
chore: normalize line endings in issue docs 

### [M] `issue-0508-np.hstack-has-diffrent-effect-from-python.md`
chore: normalize line endings in issue docs 

### [M] `issue-0509-extremely-poor-performance-on-sum-reduce.md`
chore: normalize line endings in issue docs 

### [M] `issue-0510-how-to-save-a-nested-dictionary-with-save-npz.md`
chore: normalize line endings in issue docs 

### [M] `issue-0511-does-numsharp-work-in-unity-with-the-il2cpp-backend.md`
chore: normalize line endings in issue docs 

### [M] `issue-0512-how-to-change-to-microsoft.ml.onnxruntime.tensors.densetensor-float.md`
chore: normalize line endings in issue docs 

### [M] `issue-0514-setitem-for-multiple-ids-not-working.md`
chore: normalize line endings in issue docs 

### [M] `issue-0517-error-when-loading-a-.npy-file-containing-a-scalar-value.md`
chore: normalize line endings in issue docs 

### [M] `issue-0519-bug-ndarray-filted-array-ori-array-max-prob-conf-threshold.md`
chore: normalize line endings in issue docs 


## 📁 docs/neps/

### [M] `NEP42.md`
docs: fix broken documentation URLs (scipy → numpy.org) 

### [M] `NEP55.md`
docs: fix broken documentation URLs (scipy → numpy.org) 


## 📁 docs/numpy-alignment/

### [A] `kernel-ops-test-matrix.md`
fix: comprehensive bug fixes from parallel agent battle-testing 


## 📁 docs/numpy/

### [A] `NUMPY_STRING_TYPES.md`
feat: added our own numpy docs 

### [A] `PERFORMANCE_EXECUTION_PATHS.md`
feat: added our own numpy docs 

### [A] `PERFORMANCE_NUMSHARP_RECOMMENDATIONS.md`
feat: added our own numpy docs 

### [A] `PERFORMANCE_OPTIMIZATION_STRATEGIES.md`
feat: added our own numpy docs 

### [A] `PERFORMANCE_SHARED_INFRASTRUCTURE.md`
feat: added our own numpy docs 


## 📁 docs/plans/

### [A] `EXCEPTION_REDESIGN.md`
Added plans to unfinished work 

### [A] `UNIFIED_ITERATOR_DESIGN.md`
Added plans to unfinished work 

### [M] `numpy-1x-deprecation-findings.md`
docs: fix broken documentation URLs (scipy → numpy.org) 


## 📁 src/NumSharp.Core/APIs/

### [M] `np.array_manipulation.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static 

### [A] `np.count_nonzero.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [M] `np.cs`
fix: np.intp now uses nint (native-sized integer) instead of int 

### [A] `np.cumprod.cs`
docs: fix broken documentation URLs (scipy → numpy.org) perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 


## 📁 src/NumSharp.Core/Backends/Default/ArrayManipulation/

### [M] `Default.Broadcasting.cs`
refactor: move broadcast utilities from DefaultEngine to Shape struct 

### [M] `Default.Transpose.cs`
fix: multiple NumPy alignment bug fixes 


## 📁 src/NumSharp.Core/Backends/Default/

### [M] `DefaultEngine.cs`
refactor(kernel): integrate IKernelProvider into DefaultEngine refactor(kernel): use DefaultKernelProvider for Enabled/VectorBits checks refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove all Parallel.For usage and switch to single-threaded execution refactor: route np.* and NDArray operations through TensorEngine 


## 📁 src/NumSharp.Core/Backends/Default/Indexing/

### [A] `Default.BooleanMask.cs`
refactor: route np.* and NDArray operations through TensorEngine 

### [M] `Default.NonZero.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling feat: IL kernel migration for reductions, scans, and math ops fix: kernel day bug fixes and SIMD enhancements fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests refactor(kernel): use DefaultKernelProvider for Enabled/VectorBits checks refactor: remove IKernelProvider interface, make ILKernelGenerator static 


## 📁 src/NumSharp.Core/Backends/Default/Logic/

### [M] `Default.All.cs`
refactor: route np.* and NDArray operations through TensorEngine 

### [A] `Default.Any.cs`
refactor: route np.* and NDArray operations through TensorEngine 

### [M] `Default.IsClose.cs`
chore: cleanup dead code and fix IsClose/All/Any fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) 

### [M] `Default.IsFinite.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) refactor: proper NumPy-aligned implementations replacing hacks 

### [A] `Default.IsInf.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [M] `Default.IsNan.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) refactor: proper NumPy-aligned implementations replacing hacks 


## 📁 src/NumSharp.Core/Backends/Default/Math/Add/

### [D] `Default.Add.Boolean.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.Byte.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.Char.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.Decimal.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.Double.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.Int16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.Int32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.Int64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.Single.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.UInt16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.UInt32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Add.UInt64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 src/NumSharp.Core/Backends/Default/Math/BLAS/

### [M] `Default.Dot.NDMD.cs`
feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD fix: IL kernel battle-tested fixes for shift overflow and Dot non-contiguous arrays fix: multiple NumPy alignment bug fixes 

### [M] `Default.Dot.cs`
fix: implement np.dot(1D, 2D) - treats 1D as row vector 

### [M] `Default.MatMul.2D2D.cs`
feat: SIMD-optimized MatMul with 35-100x speedup over scalar path feat: SIMD-optimized MatMul with cache blocking (no parallelization) feat: cache-blocked SIMD MatMul achieving 14-17 GFLOPS fix: IL kernel battle-tested fixes for shift overflow and Dot non-contiguous arrays 

### [M] `Default.MatMul.cs`
fix: np.matmul broadcasting crash with >2D arrays 


## 📁 src/NumSharp.Core/Backends/Default/Math/

### [M] `Default.ACos.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.ASin.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.ATan.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.ATan2.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions refactor: replace DecimalMath dependency with internal implementation refactor: route np.* and NDArray operations through TensorEngine 

### [M] `Default.Abs.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: multiple NumPy alignment bug fixes 

### [M] `Default.Add.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `Default.Cbrt.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [M] `Default.Ceil.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Clip.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: IL kernel migration for reductions, scans, and math ops fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [M] `Default.ClipNDArray.cs`
feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [M] `Default.Cos.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Cosh.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `Default.Deg2Rad.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [M] `Default.Divide.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Exp.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Exp2.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Expm1.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Floor.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `Default.FloorDivide.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [A] `Default.Invert.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [M] `Default.Log.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Log10.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Log1p.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Log2.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Mod.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Modf.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: IL kernel migration for reductions, scans, and math ops 

### [M] `Default.Multiply.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Negate.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Power.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat: IL kernel migration for reductions, scans, and math ops fix: multiple NumPy alignment bug fixes 

### [A] `Default.Rad2Deg.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [A] `Default.Reciprocal.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [M] `Default.Round.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `Default.Shift.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD fix: correct shift operations and ATan2 tests 

### [M] `Default.Sign.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: multiple NumPy alignment bug fixes 

### [M] `Default.Sin.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Sinh.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Sqrt.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `Default.Square.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [M] `Default.Subtract.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Tan.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `Default.Tanh.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `Default.Truncate.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [A] `DefaultEngine.BinaryOp.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code feat: IL kernel migration for reductions, scans, and math ops refactor(kernel): complete scalar delegate integration via IKernelProvider refactor(kernel): integrate IKernelProvider into DefaultEngine refactor: route np.* and NDArray operations through TensorEngine 

### [A] `DefaultEngine.BitwiseOp.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `DefaultEngine.CompareOp.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code refactor(kernel): complete scalar delegate integration via IKernelProvider refactor(kernel): integrate IKernelProvider into DefaultEngine refactor: route np.* and NDArray operations through TensorEngine 

### [A] `DefaultEngine.ReductionOp.cs`
Unify ArgMin/ArgMax to IL kernels (NaN/bool) feat(kernel): wire axis reduction SIMD to production + port NumPy tests feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling feat: IL Kernel Generator replaces 500K+ lines of generated code feat: NumPy 2.4.2 alignment investigation with bug fixes refactor(kernel): integrate IKernelProvider into DefaultEngine refactor: route np.* and NDArray operations through TensorEngine 

### [A] `DefaultEngine.UnaryOp.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix(unary): preserve Boolean type for LogicalNot operation refactor(kernel): complete scalar delegate integration via IKernelProvider refactor(kernel): integrate IKernelProvider into DefaultEngine refactor: proper NumPy-aligned implementations replacing hacks refactor: route np.* and NDArray operations through TensorEngine 


## 📁 src/NumSharp.Core/Backends/Default/Math/Divide/

### [D] `Default.Divide.Boolean.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.Byte.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.Char.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.Decimal.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.Double.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.Int16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.Int32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.Int64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.Single.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.UInt16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.UInt32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Divide.UInt64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 src/NumSharp.Core/Backends/Default/Math/Mod/

### [D] `Default.Mod.Boolean.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.Byte.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.Char.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.Decimal.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.Double.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.Int16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.Int32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.Int64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.Single.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.UInt16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.UInt32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Mod.UInt64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 src/NumSharp.Core/Backends/Default/Math/Multiply/

### [D] `Default.Multiply.Boolean.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.Byte.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.Char.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.Decimal.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.Double.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.Int16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.Int32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.Int64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.Single.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.UInt16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.UInt32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Multiply.UInt64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 src/NumSharp.Core/Backends/Default/Math/Reduction/

### [M] `Default.Reduction.AMax.cs`
feat(kernel): wire axis reduction SIMD to production + port NumPy tests feat: IL Kernel Generator replaces 500K+ lines of generated code feat: NumPy 2.4.2 alignment investigation with bug fixes fix: comprehensive bug fixes from parallel agent battle-testing fix: extend keepdims fix to all reduction operations fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests refactor: replace Regen axis reduction templates with IL kernel dispatch 

### [M] `Default.Reduction.AMin.cs`
feat(kernel): wire axis reduction SIMD to production + port NumPy tests feat: IL Kernel Generator replaces 500K+ lines of generated code feat: NumPy 2.4.2 alignment investigation with bug fixes fix: comprehensive bug fixes from parallel agent battle-testing fix: extend keepdims fix to all reduction operations fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests refactor: replace Regen axis reduction templates with IL kernel dispatch 

### [M] `Default.Reduction.Add.cs`
feat(kernel): wire axis reduction SIMD to production + port NumPy tests feat: IL Kernel Generator replaces 500K+ lines of generated code fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) fix: comprehensive bug fixes from parallel agent battle-testing fix: keepdims returns correct shape for element-wise reductions fix: sum axis reduction for broadcast arrays + NEP50 test fixes (6 more OpenBugs) fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests refactor: proper NumPy-aligned implementations replacing hacks refactor: replace Regen axis reduction templates with IL kernel dispatch 

### [M] `Default.Reduction.ArgMax.cs`
Unify ArgMin/ArgMax to IL kernels (NaN/bool) chore: cleanup dead code and fix IsClose/All/Any feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): add SIMD axis reduction kernels and fix NaN edge cases feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling feat: IL Kernel Generator replaces 500K+ lines of generated code feat: IL kernel migration for reductions, scans, and math ops feat: NumPy 2.4.2 alignment investigation with bug fixes fix: comprehensive bug fixes from parallel agent battle-testing 

### [M] `Default.Reduction.ArgMin.cs`
Unify ArgMin/ArgMax to IL kernels (NaN/bool) chore: cleanup dead code and fix IsClose/All/Any feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): add SIMD axis reduction kernels and fix NaN edge cases feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling feat: IL Kernel Generator replaces 500K+ lines of generated code feat: IL kernel migration for reductions, scans, and math ops feat: NumPy 2.4.2 alignment investigation with bug fixes fix: comprehensive bug fixes from parallel agent battle-testing 

### [M] `Default.Reduction.CumAdd.cs`
feat: IL kernel migration for reductions, scans, and math ops feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) fix: comprehensive bug fixes from parallel agent battle-testing fix: empty array handling for std/var, cumsum refactor removing 4K lines Regen fix: resolve 6 OpenBugs (3 fixed, 3 verified already working) refactor: proper NumPy-aligned implementations replacing hacks 

### [A] `Default.Reduction.CumMul.cs`
perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 

### [M] `Default.Reduction.Mean.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: comprehensive bug fixes from parallel agent battle-testing fix: keepdims returns correct shape for element-wise reductions refactor: replace Regen axis reduction templates with IL kernel dispatch 

### [A] `Default.Reduction.Nan.cs`
refactor: route np.* and NDArray operations through TensorEngine 

### [M] `Default.Reduction.Product.cs`
feat(kernel): wire axis reduction SIMD to production + port NumPy tests feat: IL Kernel Generator replaces 500K+ lines of generated code fix: comprehensive bug fixes from parallel agent battle-testing fix: extend keepdims fix to all reduction operations refactor: replace Regen axis reduction templates with IL kernel dispatch 

### [M] `Default.Reduction.Std.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: IL kernel migration for reductions, scans, and math ops feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD fix: comprehensive bug fixes from parallel agent battle-testing fix: empty array handling for std/var, cumsum refactor removing 4K lines Regen fix: extend keepdims fix to all reduction operations refactor: replace DecimalMath dependency with internal implementation 

### [M] `Default.Reduction.Var.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: IL kernel migration for reductions, scans, and math ops feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD fix: comprehensive bug fixes from parallel agent battle-testing fix: empty array handling for std/var, cumsum refactor removing 4K lines Regen fix: extend keepdims fix to all reduction operations 


## 📁 src/NumSharp.Core/Backends/Default/Math/Subtract/

### [D] `Default.Subtract.Boolean.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.Byte.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.Char.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.Decimal.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.Double.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.Int16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.Int32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.Int64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.Single.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.UInt16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.UInt32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Subtract.UInt64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 src/NumSharp.Core/Backends/Default/Math/Templates/

### [D] `Default.Op.Dot.Boolean.template.cs`
chore: cleanup dead code and fix IsClose/All/Any 

### [D] `Default.Op.Dot.template.cs`
chore: cleanup dead code and fix IsClose/All/Any 

### [D] `Default.Op.General.template.cs`
chore: cleanup dead code and fix IsClose/All/Any 


## 📁 src/NumSharp.Core/Backends/Default/Statistics/

### [M] `Default.ArgMax.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `Default.ArgMin.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 


## 📁 src/NumSharp.Core/Backends/Iterators/

### [M] `MultiIterator.cs`
refactor: move broadcast utilities from DefaultEngine to Shape struct 

### [D] `NDIterator.template.cs`
Delete NDIterator.template.cs 


## 📁 src/NumSharp.Core/Backends/Kernels/

### [A] `BinaryKernel.cs`
feat(kernel): add IKernelProvider abstraction layer feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `ILKernelGenerator.Binary.cs`
feat(kernel): add IKernelProvider abstraction layer feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling perf: 4x loop unrolling for SIMD kernels refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods refactor: split ILKernelGenerator.cs into partial classes 

### [A] `ILKernelGenerator.Clip.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: IL kernel migration for reductions, scans, and math ops feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 refactor: remove IKernelProvider interface, make ILKernelGenerator static 

### [A] `ILKernelGenerator.Comparison.cs`
feat(kernel): add IKernelProvider abstraction layer feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling perf: 4x loop unrolling for SIMD kernels refactor(kernel): complete scalar delegate integration via IKernelProvider refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods refactor: split ILKernelGenerator.cs into partial classes 

### [A] `ILKernelGenerator.Masking.Boolean.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: split ILKernelGenerator.Reduction.Axis.cs and Masking.cs into focused partial classes 

### [A] `ILKernelGenerator.Masking.NaN.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: split ILKernelGenerator.Reduction.Axis.cs and Masking.cs into focused partial classes 

### [A] `ILKernelGenerator.Masking.VarStd.cs`
perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: split ILKernelGenerator.Reduction.Axis.cs and Masking.cs into focused partial classes 

### [A] `ILKernelGenerator.Masking.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: split ILKernelGenerator.Reduction.Axis.cs and Masking.cs into focused partial classes refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes 

### [A] `ILKernelGenerator.MatMul.cs`
feat: SIMD-optimized MatMul with 35-100x speedup over scalar path feat: SIMD-optimized MatMul with cache blocking (no parallelization) fix: IL MatMul - declare locals before executable code fix: SIMD MatMul IL generation - method lookup and Store argument order refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods 

### [A] `ILKernelGenerator.MixedType.cs`
feat(kernel): add IKernelProvider abstraction layer feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling feat: IL kernel migration for reductions, scans, and math ops feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD perf: 4x loop unrolling for SIMD kernels refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods refactor: split ILKernelGenerator.cs into partial classes 

### [A] `ILKernelGenerator.Modf.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: IL kernel migration for reductions, scans, and math ops perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 refactor: remove IKernelProvider interface, make ILKernelGenerator static 

### [A] `ILKernelGenerator.Reduction.Arg.cs`
Unify ArgMin/ArgMax to IL kernels (NaN/bool) refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes 

### [A] `ILKernelGenerator.Reduction.Axis.Arg.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: split ILKernelGenerator.Reduction.Axis.cs and Masking.cs into focused partial classes 

### [A] `ILKernelGenerator.Reduction.Axis.NaN.cs`
feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure 

### [A] `ILKernelGenerator.Reduction.Axis.Simd.cs`
perf: SIMD axis reductions with AVX2 gather and parallel outer loop (#576) refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove all Parallel.For usage and switch to single-threaded execution refactor: split ILKernelGenerator.Reduction.Axis.cs and Masking.cs into focused partial classes 

### [A] `ILKernelGenerator.Reduction.Axis.VarStd.cs`
perf: SIMD axis reductions with AVX2 gather and parallel outer loop (#576) refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove all Parallel.For usage and switch to single-threaded execution refactor: split ILKernelGenerator.Reduction.Axis.cs and Masking.cs into focused partial classes 

### [A] `ILKernelGenerator.Reduction.Axis.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure refactor: split ILKernelGenerator.Reduction.Axis.cs and Masking.cs into focused partial classes refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes 

### [A] `ILKernelGenerator.Reduction.Boolean.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes 

### [A] `ILKernelGenerator.Reduction.cs`
Unify ArgMin/ArgMax to IL kernels (NaN/bool) feat(kernel): add IKernelProvider abstraction layer feat(kernel): add SIMD axis reduction kernels and fix NaN edge cases feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling feat: IL kernel migration for reductions, scans, and math ops feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD fix: kernel day bug fixes and SIMD enhancements perf: 4x loop unrolling for SIMD kernels refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure refactor: replace Regen axis reduction templates with IL kernel dispatch refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes refactor: split ILKernelGenerator.cs into partial classes 

### [A] `ILKernelGenerator.Scalar.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes 

### [A] `ILKernelGenerator.Scan.cs`
feat: IL kernel migration for reductions, scans, and math ops feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods 

### [A] `ILKernelGenerator.Shift.cs`
feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD fix: IL kernel battle-tested fixes for shift overflow and Dot non-contiguous arrays refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove redundant BUG 81 references from Shift kernel comments refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods 

### [A] `ILKernelGenerator.Unary.Decimal.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes 

### [A] `ILKernelGenerator.Unary.Math.cs`
perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes 

### [A] `ILKernelGenerator.Unary.Predicate.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes 

### [A] `ILKernelGenerator.Unary.Vector.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes 

### [A] `ILKernelGenerator.Unary.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): add IKernelProvider abstraction layer feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling fix: kernel day bug fixes and SIMD enhancements perf: 4x loop unrolling for SIMD kernels refactor: proper NumPy-aligned implementations replacing hacks refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods refactor: split ILKernelGenerator.Reduction.cs and Unary.cs into focused partial classes refactor: split ILKernelGenerator.cs into partial classes 

### [A] `ILKernelGenerator.cs`
feat(SIMD): add SIMD scalar paths to IL kernel generator feat(SIMD): dynamic vector width detection for IL kernels feat(kernel): add IKernelProvider abstraction layer feat(kernel): add SIMD axis reduction kernels and fix NaN edge cases feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling feat: IL Kernel Generator replaces 500K+ lines of generated code feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 refactor(kernel): complete scalar delegate integration via IKernelProvider refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: remove dead code and cleanup IL kernel infrastructure refactor: replace DecimalMath dependency with internal implementation refactor: replace null-forgiving operators with fail-fast exceptions in CachedMethods refactor: split ILKernelGenerator.cs into partial classes 

### [A] `KernelKey.cs`
feat(kernel): add IKernelProvider abstraction layer 

### [A] `KernelOp.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): add IKernelProvider abstraction layer feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 refactor: proper NumPy-aligned implementations replacing hacks 

### [A] `KernelSignatures.cs`
feat(kernel): add IKernelProvider abstraction layer refactor: remove IKernelProvider interface, make ILKernelGenerator static 

### [A] `ReductionKernel.cs`
feat(kernel): add IKernelProvider abstraction layer feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling feat: IL Kernel Generator replaces 500K+ lines of generated code feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD 

### [A] `ScalarKernel.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code refactor(kernel): complete scalar delegate integration via IKernelProvider 

### [A] `SimdMatMul.cs`
feat: cache-blocked SIMD MatMul achieving 14-17 GFLOPS perf: full panel packing for MatMul achieving 20+ GFLOPS 

### [A] `SimdThresholds.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `StrideDetector.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `TypeRules.cs`
feat(kernel): add IKernelProvider abstraction layer fix: resolve 6 OpenBugs (3 fixed, 3 verified already working) 


## 📁 src/NumSharp.Core/Backends/

### [M] `NDArray.cs`
refactor: remove IKernelProvider interface, make ILKernelGenerator static 

### [M] `NPTypeCode.cs`
fix: multiple NumPy alignment bug fixes 

### [M] `TensorEngine.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: IL Kernel Generator replaces 500K+ lines of generated code fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 refactor: route np.* and NDArray operations through TensorEngine 


## 📁 src/NumSharp.Core/Backends/Unmanaged/Pooling/

### [M] `StackedMemoryPool.cs`
refactor: modernize allocation with NativeMemory API (#528) 


## 📁 src/NumSharp.Core/Backends/Unmanaged/

### [M] `UnmanagedMemoryBlock`1.cs`
refactor: modernize allocation with NativeMemory API (#528) refactor: rename AllocationType.AllocHGlobal to Native 

### [M] `UnmanagedStorage.Getters.cs`
fix: np.matmul broadcasting crash with >2D arrays 

### [M] `UnmanagedStorage.Slicing.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: NumPy 2.x alignment for array creation and indexing refactor: proper NumPy-aligned implementations replacing hacks 


## 📁 src/NumSharp.Core/Casting/Implicit/

### [M] `NdArray.Implicit.ValueTypes.cs`
fix: resolve 6 OpenBugs (3 fixed, 3 verified already working) 


## 📁 src/NumSharp.Core/Casting/

### [M] `NdArray.ToString.cs`
fix: NumPy 2.x alignment for array creation and indexing 


## 📁 src/NumSharp.Core/Creation/

### [M] `np.arange.cs`
fix: NumPy 2.x alignment for array creation and indexing 

### [M] `np.are_broadcastable.cs`
refactor: move broadcast utilities from DefaultEngine to Shape struct 

### [M] `np.array.cs`
refactor: remove all Parallel.For usage and switch to single-threaded execution 

### [M] `np.broadcast.cs`
refactor: move broadcast utilities from DefaultEngine to Shape struct 

### [M] `np.broadcast_arrays.cs`
refactor: move broadcast utilities from DefaultEngine to Shape struct 

### [M] `np.broadcast_to.cs`
refactor: move broadcast utilities from DefaultEngine to Shape struct 

### [M] `np.full.cs`
fix: NumPy 2.x alignment for array creation and indexing 

### [M] `np.linspace.cs`
fix: NumPy 2.x alignment for array creation and indexing fix: multiple NumPy alignment bug fixes 


## 📁 src/NumSharp.Core/Generics/

### [A] `NDArray`1.Operators.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 src/NumSharp.Core/Logic/

### [M] `np.all.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling fix: kernel day bug fixes and SIMD enhancements refactor(kernel): use DefaultKernelProvider for Enabled/VectorBits checks refactor: route np.* and NDArray operations through TensorEngine 

### [M] `np.any.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling fix: kernel day bug fixes and SIMD enhancements refactor(kernel): use DefaultKernelProvider for Enabled/VectorBits checks refactor: route np.* and NDArray operations through TensorEngine 

### [A] `np.comparison.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `np.is.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [A] `np.logical.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 


## 📁 src/NumSharp.Core/Manipulation/

### [M] `NDArray.unique.cs`
fix(unique): sort unique values to match NumPy behavior fix: comprehensive bug fixes from parallel agent battle-testing 

### [M] `np.repeat.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) refactor: proper NumPy-aligned implementations replacing hacks 


## 📁 src/NumSharp.Core/Math/

### [M] `NDArray.negative.cs`
fix: multiple NumPy alignment bug fixes 

### [M] `NDArray.positive.cs`
fix: multiple NumPy alignment bug fixes 

### [M] `NdArray.Convolve.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) fix: multiple NumPy alignment bug fixes 

### [M] `np.absolute.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [A] `np.cbrt.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [M] `np.ceil.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `np.clip.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `np.cos.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [A] `np.deg2rad.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [M] `np.floor.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [A] `np.floor_divide.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [A] `np.invert.cs`
docs: fix broken documentation URLs (scipy → numpy.org) feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [A] `np.left_shift.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [M] `np.log.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `np.maximum.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `np.minimum.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `np.modf.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [A] `np.nanprod.cs`
feat(kernel): add SIMD axis reduction kernels and fix NaN edge cases feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions refactor: route np.* and NDArray operations through TensorEngine 

### [A] `np.nansum.cs`
feat(kernel): add SIMD axis reduction kernels and fix NaN edge cases feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions refactor: route np.* and NDArray operations through TensorEngine 

### [M] `np.power.cs`
docs: fix broken documentation URLs (scipy → numpy.org) feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [A] `np.rad2deg.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [A] `np.reciprocal.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [A] `np.right_shift.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [M] `np.round.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `np.sign.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `np.sin.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `np.sqrt.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [M] `np.tan.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [A] `np.trunc.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 


## 📁 src/NumSharp.Core/

### [M] `NumSharp.Core.csproj`
chore: remove obsolete template file references from csproj feat(SIMD): dynamic vector width detection for IL kernels feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling 


## 📁 src/NumSharp.Core/Operations/Elementwise/Equals/

### [D] `Default.Equals.Boolean.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.Byte.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.Char.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.Decimal.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.Double.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.Int16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.Int32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.Int64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.Single.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.UInt16.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.UInt32.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.UInt64.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [D] `Default.Equals.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 src/NumSharp.Core/Operations/Elementwise/

### [M] `NDArray.AND.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [M] `NDArray.Equals.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: comprehensive bug fixes from parallel agent battle-testing 

### [M] `NDArray.Greater.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: comprehensive bug fixes from parallel agent battle-testing 

### [M] `NDArray.Lower.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: comprehensive bug fixes from parallel agent battle-testing 

### [M] `NDArray.NotEquals.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: comprehensive bug fixes from parallel agent battle-testing 

### [M] `NDArray.OR.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 src/NumSharp.Core/Operations/Elementwise/Templates/

### [D] `Default.Op.Boolean.template.cs`
Unify ArgMin/ArgMax to IL kernels (NaN/bool) refactor: move broadcast utilities from DefaultEngine to Shape struct 

### [D] `Default.Op.Equals.template.cs`
Unify ArgMin/ArgMax to IL kernels (NaN/bool) refactor: move broadcast utilities from DefaultEngine to Shape struct 


## 📁 src/NumSharp.Core/RandomSampling/

### [M] `np.random.bernoulli.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.beta.cs`
feat(random): add Shape overloads for randn, normal, standard_normal refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.binomial.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.chisquare.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.choice.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.exponential.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.gamma.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.geometric.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.lognormal.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.permutation.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.poisson.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.rand.cs`
feat(random): add Shape overloads for randn, normal, standard_normal fix(random): fix standard_normal typo and add random() alias refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.randint.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.randn.cs`
feat(random): add Shape overloads for randn, normal, standard_normal fix(random): fix standard_normal typo and add random() alias refactor(random): align parameter names and docs with NumPy API refactor(random): remove stardard_normal backwards compat alias 

### [M] `np.random.shuffle.cs`
feat(shuffle): add axis parameter and fix NumPy alignment (closes #582) fix(shuffle): align with NumPy legacy API (no axis parameter) 

### [M] `np.random.uniform.cs`
refactor(random): align parameter names and docs with NumPy API 


## 📁 src/NumSharp.Core/Selection/

### [M] `NDArray.Indexing.Masking.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling fix: NumPy 2.x alignment for array creation and indexing fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests refactor(kernel): use DefaultKernelProvider for Enabled/VectorBits checks refactor: remove IKernelProvider interface, make ILKernelGenerator static refactor: route np.* and NDArray operations through TensorEngine 

### [M] `NDArray.Indexing.Selection.Getter.cs`
fix: kernel day bug fixes and SIMD enhancements 

### [M] `NDArray.Indexing.Selection.Setter.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) 


## 📁 src/NumSharp.Core/Sorting_Searching_Counting/

### [M] `ndarray.argsort.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [M] `np.amax.cs`
fix: extend keepdims fix to all reduction operations 

### [M] `np.argmax.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat: NumPy 2.4.2 alignment investigation with bug fixes 

### [M] `np.argsort.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) 

### [M] `np.min.cs`
fix: extend keepdims fix to all reduction operations 

### [A] `np.nanmax.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions refactor: route np.* and NDArray operations through TensorEngine 

### [A] `np.nanmin.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions refactor: route np.* and NDArray operations through TensorEngine 

### [M] `np.searchsorted.cs`
fix(searchsorted): use type-agnostic value extraction for all dtypes fix: medium severity bug fixes (BUG-12, BUG-16, BUG-17) 


## 📁 src/NumSharp.Core/Statistics/

### [M] `NDArray.argmax.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling feat: NumPy 2.4.2 alignment investigation with bug fixes 

### [M] `NDArray.argmin.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling feat: NumPy 2.4.2 alignment investigation with bug fixes 

### [A] `np.nanmean.cs`
feat: NumPy 2.4.2 alignment investigation with bug fixes 

### [A] `np.nanstd.cs`
feat: NumPy 2.4.2 alignment investigation with bug fixes 

### [A] `np.nanvar.cs`
feat: NumPy 2.4.2 alignment investigation with bug fixes 

### [M] `np.std.cs`
feat: IL kernel migration for reductions, scans, and math ops 

### [M] `np.var.cs`
feat: IL kernel migration for reductions, scans, and math ops 


## 📁 src/NumSharp.Core/Utilities/

### [M] `ArrayConvert.cs`
refactor: remove all Parallel.For usage and switch to single-threaded execution 

### [M] `Converts.Native.cs`
fix: medium severity bug fixes (BUG-12, BUG-16, BUG-17) test: comprehensive edge case battle-testing for recent fixes 

### [D] `DecimalEx.cs`
refactor: replace DecimalMath dependency with internal implementation 

### [A] `DecimalMath.cs`
refactor: replace DecimalMath dependency with internal implementation 


## 📁 src/NumSharp.Core/View/

### [A] `Shape.Broadcasting.cs`
docs: fix broken documentation URLs (scipy → numpy.org) refactor: move broadcast utilities from DefaultEngine to Shape struct 

### [M] `Shape.cs`
docs: fix misleading ALIGNED flag comment 


## 📁 src/

### [A] `numpy`
feat(simd): add SIMD helpers for reductions, fix np.any bug, NumPy NaN handling 


## 📁 test/NumSharp.Benchmark/

### [A] `GlobalUsings.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 


## 📁 test/NumSharp.UnitTest/APIs/

### [A] `CountNonzeroTests.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [A] `np_searchsorted_edge_cases.cs`
test: comprehensive edge case battle-testing for recent fixes 


## 📁 test/NumSharp.UnitTest/Backends/Kernels/

### [A] `ArgMaxArgMinComprehensiveTests.cs`
feat: IL kernel migration for reductions, scans, and math ops 

### [A] `ArgMaxNaNTests.cs`
feat: NumPy 2.4.2 alignment investigation with bug fixes fix: kernel day bug fixes and SIMD enhancements test: add comprehensive tests for SIMD optimizations and NumPy compatibility 

### [A] `AxisReductionBenchmarkTests.cs`
perf: SIMD axis reductions with AVX2 gather and parallel outer loop (#576) 

### [A] `AxisReductionEdgeCaseTests.cs`
test: comprehensive edge case battle-testing for recent fixes 

### [A] `AxisReductionMemoryTests.cs`
fix: comprehensive bug fixes from parallel agent battle-testing 

### [A] `AxisReductionSimdTests.cs`
feat(kernel): add SIMD axis reduction kernels and fix NaN edge cases 

### [A] `BattleProofTests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `BinaryOpTests.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations feat: IL Kernel Generator replaces 500K+ lines of generated code feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions fix: correct shift operations and ATan2 tests 

### [A] `BitwiseOpTests.cs`
Unify ArgMin/ArgMax to IL kernels (NaN/bool) feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `ComparisonOpTests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `CumSumComprehensiveTests.cs`
feat: IL kernel migration for reductions, scans, and math ops 

### [A] `DtypeCoverageTests.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment fix: sum axis reduction for broadcast arrays + NEP50 test fixes (6 more OpenBugs) 

### [A] `DtypePromotionTests.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [A] `EdgeCaseTests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) fix: empty array handling for std/var, cumsum refactor removing 4K lines Regen 

### [A] `EmptyArrayReductionTests.cs`
feat: NumPy 2.4.2 alignment investigation with bug fixes 

### [A] `EmptyAxisReductionTests.cs`
fix: comprehensive bug fixes from parallel agent battle-testing 

### [A] `IndexingEdgeCaseTests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `KernelMisalignmentTests.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment test: add [Misaligned] tests documenting NumPy behavioral differences 

### [A] `LinearAlgebraTests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: implement np.dot(1D, 2D) - treats 1D as row vector test: add comprehensive np.dot(1D, 2D) battle tests test: update tests for bug fixes and NEP50 alignment 

### [A] `ManipulationEdgeCaseTests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `NanReductionTests.cs`
feat(kernel): add SIMD axis reduction kernels and fix NaN edge cases feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions 

### [A] `NonContiguousTests.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment 

### [A] `NumpyAlignmentBugTests.cs`
feat: NumPy 2.4.2 alignment investigation with bug fixes fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) fix: implement np.dot(1D, 2D) - treats 1D as row vector fix: kernel day bug fixes and SIMD enhancements fix: sum axis reduction for broadcast arrays + NEP50 test fixes (6 more OpenBugs) refactor: proper NumPy-aligned implementations replacing hacks test: update tests for bug fixes and NEP50 alignment 

### [A] `ReductionOpTests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [A] `ShiftOpTests.cs`
feat(kernel): complete ILKernelGenerator coverage with SIMD optimizations 

### [A] `SimdOptimizationTests.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) test: add comprehensive tests for SIMD optimizations and NumPy compatibility 

### [A] `SlicedArrayOpTests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `TypePromotionTests.cs`
feat: NumPy 2.4.2 alignment investigation with bug fixes 

### [A] `UnaryOpTests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `UnarySpecialValuesTests.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [A] `VarStdComprehensiveTests.cs`
feat: IL kernel migration for reductions, scans, and math ops 


## 📁 test/NumSharp.UnitTest/Backends/

### [M] `NDArray.Base.MemoryLeakTest.cs`
refactor: remove Parallel.For from MemoryLeakTest 

### [M] `NDArray.Base.Test.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 


## 📁 test/NumSharp.UnitTest/Backends/Unmanaged/Math/Reduction/

### [M] `ReduceAddTests.cs`
fix: extend keepdims fix to all reduction operations fix: keepdims returns correct shape for element-wise reductions test: update tests for bug fixes and NEP50 alignment 


## 📁 test/NumSharp.UnitTest/Backends/Unmanaged/Math/

### [M] `np.prod.tests.cs`
test: move Bug 75/76 fix tests to proper test files 

### [M] `np_divide_tests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 test/NumSharp.UnitTest/Backends/Unmanaged/

### [M] `UmanagedArrayTests.cs`
feat(SIMD): dynamic vector width detection for IL kernels 


## 📁 test/NumSharp.UnitTest/Casting/

### [A] `ScalarConversionTests.cs`
fix: resolve 6 OpenBugs (3 fixed, 3 verified already working) 


## 📁 test/NumSharp.UnitTest/Creation/

### [M] `NpBroadcastFromNumPyTests.cs`
test: update tests for bug fixes and NEP50 alignment 

### [M] `np.concatenate.Test.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) 


## 📁 test/NumSharp.UnitTest/Extensions/

### [M] `ndarray.argsort.Test.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) 


## 📁 test/NumSharp.UnitTest/

### [A] `GlobalUsings.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 test/NumSharp.UnitTest/Indexing/

### [A] `NonzeroTests.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [A] `np_nonzero_edge_cases.cs`
test: add comprehensive tests for SIMD optimizations and NumPy compatibility 

### [A] `np_nonzero_strided_tests.cs`
feat: IL kernel migration for reductions, scans, and math ops 


## 📁 test/NumSharp.UnitTest/LinearAlgebra/

### [A] `np.dot.BattleTest.cs`
test(dot): add comprehensive battle tests for np.dot NumPy alignment 

### [M] `np.dot.Test.cs`
feat: complete IL kernel migration batch 2 - Dot.NDMD, CumSum axis, Shift, Var/Std SIMD 

### [A] `np.matmul.BattleTest.cs`
test(linalg): add battle tests for np.matmul and np.outer 

### [M] `np.matmul.Test.cs`
fix: np.matmul broadcasting crash with >2D arrays 

### [A] `np.outer.BattleTest.cs`
test(linalg): add battle tests for np.matmul and np.outer 


## 📁 test/NumSharp.UnitTest/Logic/

### [M] `NEP50.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

### [A] `TypePromotionTests.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [M] `np.allclose.Test.cs`
chore: cleanup dead code and fix IsClose/All/Any 

### [M] `np.any.Test.cs`
chore: cleanup dead code and fix IsClose/All/Any 

### [A] `np.comparison.Test.cs`
feat(api): complete kernel API audit with NumPy 2.x alignment fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [M] `np.isclose.Test.cs`
chore: cleanup dead code and fix IsClose/All/Any 

### [M] `np.isfinite.Test.cs`
refactor: proper NumPy-aligned implementations replacing hacks 

### [A] `np.isinf.Test.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [M] `np.isnan.Test.cs`
refactor: proper NumPy-aligned implementations replacing hacks 

### [M] `np_all_axis_Test.cs`
chore: cleanup dead code and fix IsClose/All/Any 


## 📁 test/NumSharp.UnitTest/Manipulation/

### [A] `NDArray.astype.Truncation.Test.cs`
test: comprehensive edge case battle-testing for recent fixes 

### [M] `NDArray.flat.Test.cs`
fix(tests): correct IsBroadcasted expectations in broadcast_arrays tests 

### [M] `NDArray.unique.Test.cs`
fix(unique): sort unique values to match NumPy behavior 

### [A] `ReshapeScalarTests.cs`
fix: resolve 6 OpenBugs (3 fixed, 3 verified already working) 

### [M] `np.repeat.Test.cs`
refactor: proper NumPy-aligned implementations replacing hacks 

### [A] `np.unique.EdgeCases.Test.cs`
fix: comprehensive bug fixes from parallel agent battle-testing test: comprehensive edge case battle-testing for recent fixes 


## 📁 test/NumSharp.UnitTest/Math/

### [M] `NDArray.Absolute.Test.cs`
test: update tests for bug fixes and NEP50 alignment 

### [A] `NDArray.cumprod.Test.cs`
perf: CumProd, MethodInfo cache, Integer Abs/Sign bitwise, Vector512 

### [M] `NDArray.cumsum.Test.cs`
test: move Bug 75/76 fix tests to proper test files test: update tests for bug fixes and NEP50 alignment 

### [M] `NDArray.negative.Test.cs`
test: update tests for bug fixes and NEP50 alignment 

### [M] `NDArray.positive.Test.cs`
test: update tests for bug fixes and NEP50 alignment 

### [M] `NdArray.Convolve.Test.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) 

### [A] `SignDtypeTests.cs`
fix: resolve 6 OpenBugs (3 fixed, 3 verified already working) 

### [A] `np.minimum.Test.cs`
fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) 


## 📁 test/NumSharp.UnitTest/NumPyPortedTests/

### [A] `ArgMaxArgMinEdgeCaseTests.cs`
feat: IL kernel migration for reductions, scans, and math ops fix: correct assertion syntax and API usage in edge case tests 

### [A] `ClipEdgeCaseTests.cs`
feat: IL kernel migration for reductions, scans, and math ops fix: correct assertion syntax and API usage in edge case tests fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [A] `ClipNDArrayTests.cs`
feat: complete IL kernel migration - ATan2, ClipNDArray, NaN reductions 

### [A] `CumSumEdgeCaseTests.cs`
feat: IL kernel migration for reductions, scans, and math ops fix: correct assertion syntax and API usage in edge case tests 

### [A] `ModfEdgeCaseTests.cs`
feat: IL kernel migration for reductions, scans, and math ops fix: correct assertion syntax and API usage in edge case tests 

### [A] `NonzeroEdgeCaseTests.cs`
feat: IL kernel migration for reductions, scans, and math ops fix: correct assertion syntax and API usage in edge case tests 

### [A] `PowerEdgeCaseTests.cs`
feat: IL kernel migration for reductions, scans, and math ops fix: correct assertion syntax and API usage in edge case tests fix: remove async from non-async test methods in PowerEdgeCaseTests fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 

### [A] `VarStdEdgeCaseTests.cs`
feat: IL kernel migration for reductions, scans, and math ops fix: correct assertion syntax and API usage in edge case tests 


## 📁 test/NumSharp.UnitTest/

### [M] `OpenBugs.ApiAudit.cs`
OpenBugs.ApiAudit.cs: test updates for int64 changes feat: NumPy 2.4.2 alignment investigation with bug fixes fix: empty array handling for std/var, cumsum refactor removing 4K lines Regen fix: resolve 6 OpenBugs (3 fixed, 3 verified already working) test: move Bug 75/76 fix tests to proper test files test: update tests for bug fixes and NEP50 alignment 

### [A] `OpenBugs.ILKernelBattle.cs`
fix: IL kernel battle-tested fixes for shift overflow and Dot non-contiguous arrays fix: implement np.dot(1D, 2D) - treats 1D as row vector 

### [M] `OpenBugs.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code fix(unique): sort unique values to match NumPy behavior fix: comprehensive OpenBugs fixes (45 tests fixed, 108→63 failures) fix: sum axis reduction for broadcast arrays + NEP50 test fixes (6 more OpenBugs) 


## 📁 test/NumSharp.UnitTest/Operations/

### [A] `EmptyArrayComparisonTests.cs`
fix: comprehensive bug fixes from parallel agent battle-testing 


## 📁 test/NumSharp.UnitTest/Random/

### [M] `np.random.choice.Test.cs`
refactor(random): align parameter names and docs with NumPy API 

### [M] `np.random.seed.Test.cs`
refactor(random): align parameter names and docs with NumPy API 


## 📁 test/NumSharp.UnitTest/RandomSampling/

### [A] `np.random.shuffle.NumPyAligned.Test.cs`
fix(shuffle): align with NumPy legacy API (no axis parameter) 

### [M] `np.random.shuffle.Test.cs`
feat(shuffle): add axis parameter and fix NumPy alignment (closes #582) 


## 📁 test/NumSharp.UnitTest/Selection/

### [A] `BooleanIndexing.BattleTests.cs`
test: add comprehensive boolean indexing battle tests (76 tests) 

### [A] `BooleanMaskingTests.cs`
test: add comprehensive tests for SIMD optimizations and NumPy compatibility 

### [M] `NDArray.AMax.Test.cs`
feat: NumPy 2.4.2 alignment investigation with bug fixes 

### [M] `NDArray.Indexing.Test.cs`
fix(tests): correct IsBroadcasted expectations in broadcast_arrays tests 


## 📁 test/NumSharp.UnitTest/Sorting/

### [A] `ArgsortNaNTests.cs`
fix: test assertion bugs and API mismatches in PowerEdgeCaseTests and ClipEdgeCaseTests 


## 📁 test/NumSharp.UnitTest/Statistics/

### [M] `NdArray.Mean.Test.cs`
fix: keepdims returns correct shape for element-wise reductions 


## 📁 test/NumSharp.UnitTest/Utilities/

### [M] `FluentExtension.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 


## 📁 test/NumSharp.UnitTest/View/

### [M] `Shape.OffsetParity.Tests.cs`
feat: IL Kernel Generator replaces 500K+ lines of generated code 

