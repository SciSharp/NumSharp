#if !NET9_0_OR_GREATER
namespace System.Runtime.CompilerServices
{
    /// <summary>
    ///     net8.0 polyfill of the BCL's <c>OverloadResolutionPriorityAttribute</c> (.NET 9+): among the applicable
    ///     overloads a C# 13+ compiler keeps only those with the highest priority (default 0) before picking the best one.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumSharp uses it where NumPy's dynamically-typed parameters meet C#'s static overloads and a literal fits several
    ///     of them equally well: <c>null</c> (Python's <c>None</c>) against a family of array-typed seed overloads
    ///     (<c>int[]</c>/<c>long[]</c>/<c>uint[]</c> — no array converts to another, so plain C# rules call
    ///     <c>new PCG64(null)</c> ambiguous), and <c>default</c> against the <c>int[]</c>/<c>long[]</c>/<c>long</c> size
    ///     shims of the legacy samplers (every one of them is a "better" target than <c>Shape</c>, and the <c>long</c> one
    ///     silently turned <c>size: default</c> into a zero-length size). Each use says which overload wins and why.
    ///     </para>
    ///     <para>
    ///     The compiler recognizes the attribute by its full name on a member's metadata, so this internal copy works for
    ///     callers of the net8.0 build too, provided they compile with C# 13 or later; an older language version ignores
    ///     it and sees the plain C# rules (the literal stays ambiguous there — never a different binding than before).
    ///     net9.0+ builds use the BCL type, so this file compiles to nothing there.
    ///     </para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Constructor | AttributeTargets.Property, AllowMultiple = false,
        Inherited = false)]
    internal sealed class OverloadResolutionPriorityAttribute : Attribute
    {
        /// <summary>Marks a member with an overload-resolution priority.</summary>
        /// <param name="priority">Higher wins; the default of an unmarked member is 0, and negative values rank a member
        /// below every unmarked overload (it is then chosen only when nothing of higher priority applies).</param>
        public OverloadResolutionPriorityAttribute(int priority) => Priority = priority;

        /// <summary>The priority given to the constructor; read by the compiler, never at run time.</summary>
        public int Priority { get; }
    }
}
#endif
