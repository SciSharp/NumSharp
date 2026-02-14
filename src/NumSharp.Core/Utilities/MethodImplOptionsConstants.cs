using System.Runtime.CompilerServices;

namespace NumSharp;

/// <summary>
/// Method implementation option constants for use with <see cref="MethodImplAttribute"/>.
/// </summary>
/// <remarks>
/// Exposed globally via <c>global using static</c> — use directly without class prefix.
/// </remarks>
/// <example>
/// <code>
/// [MethodImpl(OptimizeAndInline)]
/// public void HotPath() { }
/// </code>
/// </example>
public static class MethodImplOptionsConstants
{
    /// <summary>
    /// Aggressive inlining + aggressive optimization (768).
    /// Use for hot paths where both inlining and JIT optimization are critical.
    /// </summary>
    public const MethodImplOptions OptimizeAndInline = MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization;

    /// <summary>
    /// Aggressive optimization only (512).
    /// Use for methods too large to inline but still performance-critical.
    /// </summary>
    public const MethodImplOptions Optimize = MethodImplOptions.AggressiveOptimization;

    /// <summary>
    /// Aggressive inlining only (256).
    /// Use for small methods where inlining is beneficial.
    /// </summary>
    public const MethodImplOptions Inline = MethodImplOptions.AggressiveInlining;
}
