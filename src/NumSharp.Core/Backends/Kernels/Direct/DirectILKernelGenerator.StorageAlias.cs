using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using NumSharp.Backends;

// =============================================================================
// DirectILKernelGenerator.StorageAlias.cs — IL-emitted typed-field copier for
// UnmanagedStorage.Alias view construction
// =============================================================================
//
// RESPONSIBILITY:
//   Replace UnmanagedStorage.Alias's 15-case switch-on-NPTypeCode that copies
//   the one live ArraySlice<T> field from parent to alias with a per-dtype
//   IL-emitted delegate. Each delegate emits a single `ldfld` + `stfld` pair
//   targeting the typed `_array{T}` field that matches the parent's typecode.
//
// WHY:
//   The /np-function rule forbids `switch (typecode)` patterns even when each
//   case does the same operation. Splitting the dispatch into a cached
//   per-dtype delegate keeps the same "copy the one live typed field"
//   semantics but moves the branch out of the hot path: the lookup is a
//   `ConcurrentDictionary` get (or a pre-warmed array, since the dtype set is
//   closed at startup) and the call is a direct delegate invoke.
//
// PERF:
//   Single-field-copy IL body is ~2-3 cycles; delegate invocation adds ~3-5
//   cycles for the indirect call. The original switch compiled to a small
//   jump table at ~3-5 cycles, so per-call cost is comparable. The win is
//   compliance with the IL-generation rule, not raw speed — but the rule
//   exists for cases where it DOES dominate (per-element kernels), so we
//   apply it uniformly even where the gain is small.
//
// CACHE:
//   ConcurrentDictionary<NPTypeCode, StorageTypedFieldCopier>. One entry per
//   supported dtype. NPTypeCode.Empty / .String have no `_array{T}` field;
//   their entries return a no-op delegate.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Copies one strongly-typed <c>_array{T}</c> field from a parent
    ///     <see cref="UnmanagedStorage"/> into an alias <see cref="UnmanagedStorage"/>.
    ///     The two arguments are <c>(dst, src)</c>; the kernel emits
    ///     <c>dst._array{T} = src._array{T}</c>.
    /// </summary>
    /// <param name="dst">Destination storage (the alias being initialised).</param>
    /// <param name="src">Source storage (the parent whose buffer is being aliased).</param>
    internal delegate void StorageTypedFieldCopier(UnmanagedStorage dst, UnmanagedStorage src);

    public static partial class DirectILKernelGenerator
    {
        // Per-typecode delegate cache. ConcurrentDictionary so the first call
        // for each dtype emits IL once; subsequent calls hit a cached lookup.
        private static readonly ConcurrentDictionary<NPTypeCode, StorageTypedFieldCopier> _storageAliasFieldCopiers = new();

        // No-op fallback for typecodes that have no backing typed field (Empty,
        // String). UnmanagedStorage.Alias still copies InternalArray + Address
        // by hand; only the typed-field mirror is skipped here.
        private static readonly StorageTypedFieldCopier _storageAliasNoop = static (_, _) => { };

        /// <summary>
        ///     Returns a cached <see cref="StorageTypedFieldCopier"/> for the
        ///     given <paramref name="typeCode"/>. First call per dtype emits
        ///     IL via <see cref="DynamicMethod"/>; subsequent calls are a
        ///     ConcurrentDictionary lookup.
        /// </summary>
        /// <remarks>
        ///     When <see cref="Enabled"/> is false this falls back to the
        ///     no-op kernel. Callers must then handle the typed-field copy
        ///     themselves (currently nobody opts out, but the contract
        ///     mirrors the rest of DirectILKernelGenerator).
        /// </remarks>
        internal static StorageTypedFieldCopier GetStorageAliasFieldCopier(NPTypeCode typeCode)
        {
            if (!Enabled) return _storageAliasNoop;

            return _storageAliasFieldCopiers.GetOrAdd(typeCode, static tc =>
            {
                try { return GenerateStorageAliasFieldCopierIL(tc); }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine(
                        $"[ILKernel] GetStorageAliasFieldCopier({tc}): {ex.GetType().Name}: {ex.Message}");
                    return _storageAliasNoop;
                }
            });
        }

        /// <summary>
        ///     Emit IL for one (typecode → typed field copy) kernel. Body is:
        ///     <code>
        ///         ldarg.0           // dst
        ///         ldarg.1           // src
        ///         ldfld src._arrayT // pull the typed slice
        ///         stfld dst._arrayT // store into dst
        ///         ret
        ///     </code>
        ///     The field name pattern is <c>_array</c> + the enum name (so
        ///     <c>NPTypeCode.Int32</c> → <c>_arrayInt32</c>). <c>Empty</c> /
        ///     <c>String</c> have no matching field and return the no-op.
        /// </summary>
        private static StorageTypedFieldCopier GenerateStorageAliasFieldCopierIL(NPTypeCode typeCode)
        {
            // Empty / String have no backing typed slice field. Return the
            // shared no-op so the alias path can still call the delegate
            // unconditionally.
            if (typeCode == NPTypeCode.Empty || typeCode == NPTypeCode.String)
                return _storageAliasNoop;

            string fieldName = "_array" + typeCode.ToString();
            FieldInfo field = typeof(UnmanagedStorage).GetField(
                fieldName, BindingFlags.NonPublic | BindingFlags.Instance);
            if (field is null)
                return _storageAliasNoop;

            // skipVisibility:true lets us touch UnmanagedStorage's protected
            // _array{T} fields from the generated DynamicMethod's anonymous
            // module — same trick used by every other partial kernel that
            // pokes at internal/protected backing fields.
            var dm = new DynamicMethod(
                $"StorageAliasFieldCopier_{typeCode}",
                typeof(void),
                new[] { typeof(UnmanagedStorage), typeof(UnmanagedStorage) },
                typeof(UnmanagedStorage),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            il.Emit(OpCodes.Ldarg_0);          // dst
            il.Emit(OpCodes.Ldarg_1);          // src
            il.Emit(OpCodes.Ldfld, field);     // src._array{T}
            il.Emit(OpCodes.Stfld, field);     // dst._array{T} = src._array{T}
            il.Emit(OpCodes.Ret);

            return (StorageTypedFieldCopier)dm.CreateDelegate(typeof(StorageTypedFieldCopier));
        }
    }
}
