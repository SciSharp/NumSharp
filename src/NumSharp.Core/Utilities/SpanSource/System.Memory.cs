// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// ------------------------------------------------------------------------------
// Changes to this file must follow the https://aka.ms/api-review process.
// ------------------------------------------------------------------------------

#if !BUILDING_CORELIB_REFERENCE
namespace System
{
    public readonly partial struct SequencePosition : System.IEquatable<System.SequencePosition>
    {
        private readonly object _dummy;
        private readonly int _dummyPrimitive;
        public SequencePosition(object? @object, int integer) { throw null; }
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Never)]
        public override bool Equals([System.Diagnostics.CodeAnalysis.NotNullWhenAttribute(true)] object? obj) { throw null; }
        public bool Equals(System.SequencePosition other) { throw null; }
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Never)]
        public override int GetHashCode() { throw null; }
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Never)]
        public int GetInteger() { throw null; }
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Never)]
        public object? GetObject() { throw null; }
    }
}
namespace System.Buffers
{
    public sealed partial class ArrayBufferWriter<T> : System.Buffers.IBufferWriter<T>
    {
        public ArrayBufferWriter() { }
        public ArrayBufferWriter(int initialCapacity) { }
        public int Capacity { get { throw null; } }
        public int FreeCapacity { get { throw null; } }
        public int WrittenCount { get { throw null; } }
        public System.ReadOnlyMemory<T> WrittenMemory { get { throw null; } }
        public System.ReadOnlyUnmanagedSpan<T> WrittenSpan { get { throw null; } }
        public void Advance(int count) { }
        public void Clear() { }
        public void ResetWrittenCount() { }
        public System.Memory<T> GetMemory(int sizeHint = 0) { throw null; }
        public System.UnmanagedSpan<T> GetSpan(int sizeHint = 0) { throw null; }
    }
    public static partial class BuffersExtensions
    {
        public static void CopyTo<T>(this in System.Buffers.ReadOnlySequence<T> source, System.UnmanagedSpan<T> destination) { }
        public static System.SequencePosition? PositionOf<T>(this in System.Buffers.ReadOnlySequence<T> source, T value) where T : System.IEquatable<T>? { throw null; }
        public static T[] ToArray<T>(this in System.Buffers.ReadOnlySequence<T> sequence) { throw null; }
        public static void Write<T>(this System.Buffers.IBufferWriter<T> writer, System.ReadOnlyUnmanagedSpan<T> value) { }
    }
    public partial interface IBufferWriter<T>
    {
        void Advance(int count);
        System.Memory<T> GetMemory(int sizeHint = 0);
        System.UnmanagedSpan<T> GetSpan(int sizeHint = 0);
    }
    public abstract partial class MemoryPool<T> : System.IDisposable
    {
        protected MemoryPool() { }
        public abstract int MaxBufferSize { get; }
        public static System.Buffers.MemoryPool<T> Shared { get { throw null; } }
        public void Dispose() { }
        protected abstract void Dispose(bool disposing);
        public abstract System.Buffers.IMemoryOwner<T> Rent(int minBufferSize = -1);
    }
    public abstract partial class ReadOnlySequenceSegment<T>
    {
        protected ReadOnlySequenceSegment() { }
        public System.ReadOnlyMemory<T> Memory { get { throw null; } protected set { } }
        public System.Buffers.ReadOnlySequenceSegment<T>? Next { get { throw null; } protected set { } }
        public long RunningIndex { get { throw null; } protected set { } }
    }
    public readonly partial struct ReadOnlySequence<T>
    {
        private readonly object _dummy;
        private readonly int _dummyPrimitive;
        public static readonly System.Buffers.ReadOnlySequence<T> Empty;
        public ReadOnlySequence(System.Buffers.ReadOnlySequenceSegment<T> startSegment, int startIndex, System.Buffers.ReadOnlySequenceSegment<T> endSegment, int endIndex) { throw null; }
        public ReadOnlySequence(System.ReadOnlyMemory<T> memory) { throw null; }
        public ReadOnlySequence(T[] array) { throw null; }
        public ReadOnlySequence(T[] array, int start, int length) { throw null; }
        public System.SequencePosition End { get { throw null; } }
        public System.ReadOnlyMemory<T> First { get { throw null; } }
        public System.ReadOnlyUnmanagedSpan<T> FirstSpan { get { throw null; } }
        public bool IsEmpty { get { throw null; } }
        public bool IsSingleSegment { get { throw null; } }
        public long Length { get { throw null; } }
        public System.SequencePosition Start { get { throw null; } }
        public System.Buffers.ReadOnlySequence<T>.Enumerator GetEnumerator() { throw null; }
        public long GetOffset(System.SequencePosition position) { throw null; }
        public System.SequencePosition GetPosition(long offset) { throw null; }
        public System.SequencePosition GetPosition(long offset, System.SequencePosition origin) { throw null; }
        public System.Buffers.ReadOnlySequence<T> Slice(int start, int length) { throw null; }
        public System.Buffers.ReadOnlySequence<T> Slice(int start, System.SequencePosition end) { throw null; }
        public System.Buffers.ReadOnlySequence<T> Slice(long start) { throw null; }
        public System.Buffers.ReadOnlySequence<T> Slice(long start, long length) { throw null; }
        public System.Buffers.ReadOnlySequence<T> Slice(long start, System.SequencePosition end) { throw null; }
        public System.Buffers.ReadOnlySequence<T> Slice(System.SequencePosition start) { throw null; }
        public System.Buffers.ReadOnlySequence<T> Slice(System.SequencePosition start, int length) { throw null; }
        public System.Buffers.ReadOnlySequence<T> Slice(System.SequencePosition start, long length) { throw null; }
        public System.Buffers.ReadOnlySequence<T> Slice(System.SequencePosition start, System.SequencePosition end) { throw null; }
        public override string ToString() { throw null; }
        public bool TryGet(ref System.SequencePosition position, out System.ReadOnlyMemory<T> memory, bool advance = true) { throw null; }
        public partial struct Enumerator
        {
            private object _dummy;
            private int _dummyPrimitive;
            public Enumerator(in System.Buffers.ReadOnlySequence<T> sequence) { throw null; }
            public System.ReadOnlyMemory<T> Current { get { throw null; } }
            public bool MoveNext() { throw null; }
        }
    }
    public static partial class SequenceReaderExtensions
    {
        public static bool TryReadBigEndian(this ref System.Buffers.SequenceReader<byte> reader, out short value) { throw null; }
        public static bool TryReadBigEndian(this ref System.Buffers.SequenceReader<byte> reader, out int value) { throw null; }
        public static bool TryReadBigEndian(this ref System.Buffers.SequenceReader<byte> reader, out long value) { throw null; }
        public static bool TryReadLittleEndian(this ref System.Buffers.SequenceReader<byte> reader, out short value) { throw null; }
        public static bool TryReadLittleEndian(this ref System.Buffers.SequenceReader<byte> reader, out int value) { throw null; }
        public static bool TryReadLittleEndian(this ref System.Buffers.SequenceReader<byte> reader, out long value) { throw null; }
    }
    public ref partial struct SequenceReader<T> where T : unmanaged, System.IEquatable<T>
    {
        private object _dummy;
        private int _dummyPrimitive;
        public SequenceReader(System.Buffers.ReadOnlySequence<T> sequence) { throw null; }
        public readonly long Consumed { get { throw null; } }
        public readonly System.ReadOnlyUnmanagedSpan<T> CurrentSpan { get { throw null; } }
        public readonly int CurrentSpanIndex { get { throw null; } }
        public readonly bool End { get { throw null; } }
        public readonly long Length { get { throw null; } }
        public readonly System.SequencePosition Position { get { throw null; } }
        public readonly long Remaining { get { throw null; } }
        public readonly System.Buffers.ReadOnlySequence<T> Sequence { get { throw null; } }
        public readonly System.Buffers.ReadOnlySequence<T> UnreadSequence { get { throw null; } }
        public readonly System.ReadOnlyUnmanagedSpan<T> UnreadSpan { get { throw null; } }
        public void Advance(long count) { }
        public long AdvancePast(T value) { throw null; }
        public long AdvancePastAny(scoped System.ReadOnlyUnmanagedSpan<T> values) { throw null; }
        public long AdvancePastAny(T value0, T value1) { throw null; }
        public long AdvancePastAny(T value0, T value1, T value2) { throw null; }
        public long AdvancePastAny(T value0, T value1, T value2, T value3) { throw null; }
        public void AdvanceToEnd() { throw null; }
        public bool IsNext(scoped System.ReadOnlyUnmanagedSpan<T> next, bool advancePast = false) { throw null; }
        public bool IsNext(T next, bool advancePast = false) { throw null; }
        public void Rewind(long count) { }
        public bool TryAdvanceTo(T delimiter, bool advancePastDelimiter = true) { throw null; }
        public bool TryAdvanceToAny(scoped System.ReadOnlyUnmanagedSpan<T> delimiters, bool advancePastDelimiter = true) { throw null; }
        public readonly bool TryCopyTo(System.UnmanagedSpan<T> destination) { throw null; }
        public readonly bool TryPeek(out T value) { throw null; }
        public readonly bool TryPeek(long offset, out T value) { throw null; }
        public bool TryRead(out T value) { throw null; }
        public bool TryReadTo(out System.Buffers.ReadOnlySequence<T> sequence, scoped System.ReadOnlyUnmanagedSpan<T> delimiter, bool advancePastDelimiter = true) { throw null; }
        public bool TryReadTo(out System.Buffers.ReadOnlySequence<T> sequence, T delimiter, bool advancePastDelimiter = true) { throw null; }
        public bool TryReadTo(out System.Buffers.ReadOnlySequence<T> sequence, T delimiter, T delimiterEscape, bool advancePastDelimiter = true) { throw null; }
        public bool TryReadTo(out System.ReadOnlyUnmanagedSpan<T> span, scoped System.ReadOnlyUnmanagedSpan<T> delimiter, bool advancePastDelimiter = true) { throw null; }
        public bool TryReadTo(out System.ReadOnlyUnmanagedSpan<T> span, T delimiter, bool advancePastDelimiter = true) { throw null; }
        public bool TryReadTo(out System.ReadOnlyUnmanagedSpan<T> span, T delimiter, T delimiterEscape, bool advancePastDelimiter = true) { throw null; }
        public bool TryReadToAny(out System.Buffers.ReadOnlySequence<T> sequence, scoped System.ReadOnlyUnmanagedSpan<T> delimiters, bool advancePastDelimiter = true) { throw null; }
        public bool TryReadToAny(out System.ReadOnlyUnmanagedSpan<T> span, scoped System.ReadOnlyUnmanagedSpan<T> delimiters, bool advancePastDelimiter = true) { throw null; }
        public bool TryReadExact(int count, out System.Buffers.ReadOnlySequence<T> sequence) { throw null; }
    }
}
namespace System.Runtime.InteropServices
{
    public static partial class SequenceMarshal
    {
        public static bool TryGetArray<T>(System.Buffers.ReadOnlySequence<T> sequence, out System.ArraySegment<T> segment) { throw null; }
        public static bool TryGetReadOnlyMemory<T>(System.Buffers.ReadOnlySequence<T> sequence, out System.ReadOnlyMemory<T> memory) { throw null; }
        public static bool TryGetReadOnlySequenceSegment<T>(System.Buffers.ReadOnlySequence<T> sequence, [System.Diagnostics.CodeAnalysis.NotNullWhenAttribute(true)] out System.Buffers.ReadOnlySequenceSegment<T>? startSegment, out int startIndex, [System.Diagnostics.CodeAnalysis.NotNullWhenAttribute(true)] out System.Buffers.ReadOnlySequenceSegment<T>? endSegment, out int endIndex) { throw null; }
        public static bool TryRead<T>(ref System.Buffers.SequenceReader<byte> reader, out T value) where T : unmanaged { throw null; }
    }
}
namespace System.Text
{
    public static partial class EncodingExtensions
    {
        public static void Convert(this System.Text.Decoder decoder, in System.Buffers.ReadOnlySequence<byte> bytes, System.Buffers.IBufferWriter<char> writer, bool flush, out long charsUsed, out bool completed) { throw null; }
        public static void Convert(this System.Text.Decoder decoder, System.ReadOnlyUnmanagedSpan<byte> bytes, System.Buffers.IBufferWriter<char> writer, bool flush, out long charsUsed, out bool completed) { throw null; }
        public static void Convert(this System.Text.Encoder encoder, in System.Buffers.ReadOnlySequence<char> chars, System.Buffers.IBufferWriter<byte> writer, bool flush, out long bytesUsed, out bool completed) { throw null; }
        public static void Convert(this System.Text.Encoder encoder, System.ReadOnlyUnmanagedSpan<char> chars, System.Buffers.IBufferWriter<byte> writer, bool flush, out long bytesUsed, out bool completed) { throw null; }
        public static byte[] GetBytes(this System.Text.Encoding encoding, in System.Buffers.ReadOnlySequence<char> chars) { throw null; }
        public static long GetBytes(this System.Text.Encoding encoding, in System.Buffers.ReadOnlySequence<char> chars, System.Buffers.IBufferWriter<byte> writer) { throw null; }
        public static int GetBytes(this System.Text.Encoding encoding, in System.Buffers.ReadOnlySequence<char> chars, System.UnmanagedSpan<byte> bytes) { throw null; }
        public static long GetBytes(this System.Text.Encoding encoding, System.ReadOnlyUnmanagedSpan<char> chars, System.Buffers.IBufferWriter<byte> writer) { throw null; }
        public static long GetChars(this System.Text.Encoding encoding, in System.Buffers.ReadOnlySequence<byte> bytes, System.Buffers.IBufferWriter<char> writer) { throw null; }
        public static int GetChars(this System.Text.Encoding encoding, in System.Buffers.ReadOnlySequence<byte> bytes, System.UnmanagedSpan<char> chars) { throw null; }
        public static long GetChars(this System.Text.Encoding encoding, System.ReadOnlyUnmanagedSpan<byte> bytes, System.Buffers.IBufferWriter<char> writer) { throw null; }
        public static string GetString(this System.Text.Encoding encoding, in System.Buffers.ReadOnlySequence<byte> bytes) { throw null; }
    }
}
#endif // !BUILDING_CORELIB_REFERENCE
namespace System
{
    public static partial class MemoryExtensions
    {
        public static System.ReadOnlyMemory<char> AsMemory(this string? text) { throw null; }
        public static System.ReadOnlyMemory<char> AsMemory(this string? text, System.Index startIndex) { throw null; }
        public static System.ReadOnlyMemory<char> AsMemory(this string? text, int start) { throw null; }
        public static System.ReadOnlyMemory<char> AsMemory(this string? text, int start, int length) { throw null; }
        public static System.ReadOnlyMemory<char> AsMemory(this string? text, System.Range range) { throw null; }
        public static System.Memory<T> AsMemory<T>(this System.ArraySegment<T> segment) { throw null; }
        public static System.Memory<T> AsMemory<T>(this System.ArraySegment<T> segment, int start) { throw null; }
        public static System.Memory<T> AsMemory<T>(this System.ArraySegment<T> segment, int start, int length) { throw null; }
        public static System.Memory<T> AsMemory<T>(this T[]? array) { throw null; }
        public static System.Memory<T> AsMemory<T>(this T[]? array, System.Index startIndex) { throw null; }
        public static System.Memory<T> AsMemory<T>(this T[]? array, int start) { throw null; }
        public static System.Memory<T> AsMemory<T>(this T[]? array, int start, int length) { throw null; }
        public static System.Memory<T> AsMemory<T>(this T[]? array, System.Range range) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> AsUnmanagedSpan(this string? text) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> AsUnmanagedSpan(this string? text, int start) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> AsUnmanagedSpan(this string? text, int start, int length) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> AsUnmanagedSpan(this string? text, System.Index startIndex) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> AsUnmanagedSpan(this string? text, System.Range range) { throw null; }
        public static System.UnmanagedSpan<T> AsUnmanagedSpan<T>(this System.ArraySegment<T> segment) { throw null; }
        public static System.UnmanagedSpan<T> AsUnmanagedSpan<T>(this System.ArraySegment<T> segment, System.Index startIndex) { throw null; }
        public static System.UnmanagedSpan<T> AsUnmanagedSpan<T>(this System.ArraySegment<T> segment, int start) { throw null; }
        public static System.UnmanagedSpan<T> AsUnmanagedSpan<T>(this System.ArraySegment<T> segment, int start, int length) { throw null; }
        public static System.UnmanagedSpan<T> AsUnmanagedSpan<T>(this System.ArraySegment<T> segment, System.Range range) { throw null; }
        public static System.UnmanagedSpan<T> AsUnmanagedSpan<T>(this T[]? array) { throw null; }
        public static System.UnmanagedSpan<T> AsUnmanagedSpan<T>(this T[]? array, System.Index startIndex) { throw null; }
        public static System.UnmanagedSpan<T> AsUnmanagedSpan<T>(this T[]? array, int start) { throw null; }
        public static System.UnmanagedSpan<T> AsUnmanagedSpan<T>(this T[]? array, int start, int length) { throw null; }
        public static System.UnmanagedSpan<T> AsUnmanagedSpan<T>(this T[]? array, System.Range range) { throw null; }
        public static int BinarySearch<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.IComparable<T> comparable) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int BinarySearch<T>(this System.UnmanagedSpan<T> span, System.IComparable<T> comparable) { throw null; }
        public static int BinarySearch<T, TComparer>(this System.ReadOnlyUnmanagedSpan<T> span, T value, TComparer comparer) where TComparer : System.Collections.Generic.IComparer<T>, allows ref struct { throw null; }
        public static int BinarySearch<T, TComparable>(this System.ReadOnlyUnmanagedSpan<T> span, TComparable comparable) where TComparable : System.IComparable<T>, allows ref struct { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int BinarySearch<T, TComparer>(this System.UnmanagedSpan<T> span, T value, TComparer comparer) where TComparer : System.Collections.Generic.IComparer<T>, allows ref struct { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int BinarySearch<T, TComparable>(this System.UnmanagedSpan<T> span, TComparable comparable) where TComparable : System.IComparable<T>, allows ref struct { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int CommonPrefixLength<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int CommonPrefixLength<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other, System.Collections.Generic.IEqualityComparer<T>? comparer) { throw null; }
        public static int CommonPrefixLength<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other) { throw null; }
        public static int CommonPrefixLength<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other, System.Collections.Generic.IEqualityComparer<T>? comparer) { throw null; }
        public static int CompareTo(this System.ReadOnlyUnmanagedSpan<char> span, System.ReadOnlyUnmanagedSpan<char> other, System.StringComparison comparisonType) { throw null; }
        public static bool Contains(this System.ReadOnlyUnmanagedSpan<char> span, System.ReadOnlyUnmanagedSpan<char> value, System.StringComparison comparisonType) { throw null; }
        public static bool Contains<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static bool Contains<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool Contains<T>(this System.UnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAny(this System.ReadOnlyUnmanagedSpan<char> span, System.Buffers.SearchValues<string> values) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAny(this System.UnmanagedSpan<char> span, System.Buffers.SearchValues<string> values) { throw null; }
        public static bool ContainsAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static bool ContainsAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static bool ContainsAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAny<T>(this System.UnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAny<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAny<T>(this System.UnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAny<T>(this System.UnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static bool ContainsAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static bool ContainsAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static bool ContainsAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAnyExcept<T>(this System.UnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAnyExcept<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAnyExcept<T>(this System.UnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAnyExcept<T>(this System.UnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAnyExcept<T>(this System.UnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        public static bool ContainsAnyExceptInRange<T>(this System.ReadOnlyUnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAnyExceptInRange<T>(this System.UnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        public static bool ContainsAnyInRange<T>(this System.ReadOnlyUnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool ContainsAnyInRange<T>(this System.UnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        public static void CopyTo<T>(this T[]? source, System.Memory<T> destination) { }
        public static void CopyTo<T>(this T[]? source, System.UnmanagedSpan<T> destination) { }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int Count<T>(this System.UnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static int Count<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static int Count<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int Count<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value) where T : System.IEquatable<T>? { throw null; }
        public static int Count<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value) where T : System.IEquatable<T>? { throw null; }
        public static int Count<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int CountAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : IEquatable<T>? { throw null; }
        public static int CountAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, params System.ReadOnlyUnmanagedSpan<T> values) where T : IEquatable<T>? { throw null; }
        public static int CountAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static bool EndsWith(this System.ReadOnlyUnmanagedSpan<char> span, System.ReadOnlyUnmanagedSpan<char> value, System.StringComparison comparisonType) { throw null; }
        public static bool EndsWith<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value) where T : System.IEquatable<T>? { throw null; }
        public static bool EndsWith<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool EndsWith<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value) where T : System.IEquatable<T>? { throw null; }
        public static bool EndsWith<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static bool EndsWith<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static System.Text.UnmanagedSpanLineEnumerator EnumerateLines(this System.ReadOnlyUnmanagedSpan<char> span) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static System.Text.UnmanagedSpanLineEnumerator EnumerateLines(this System.UnmanagedSpan<char> span) { throw null; }
        public static System.Text.UnmanagedSpanRuneEnumerator EnumerateRunes(this System.ReadOnlyUnmanagedSpan<char> span) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static System.Text.UnmanagedSpanRuneEnumerator EnumerateRunes(this System.UnmanagedSpan<char> span) { throw null; }
        public static bool Equals(this System.ReadOnlyUnmanagedSpan<char> span, System.ReadOnlyUnmanagedSpan<char> other, System.StringComparison comparisonType) { throw null; }
        public static int IndexOf(this System.ReadOnlyUnmanagedSpan<char> span, System.ReadOnlyUnmanagedSpan<char> value, System.StringComparison comparisonType) { throw null; }
        public static int IndexOfAny(this System.ReadOnlyUnmanagedSpan<char> span, System.Buffers.SearchValues<string> values) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAny(this System.UnmanagedSpan<char> span, System.Buffers.SearchValues<string> values) { throw null; }
        public static int IndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int IndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int IndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAny<T>(this System.UnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAny<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAny<T>(this System.UnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAny<T>(this System.UnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAnyExcept<T>(this System.UnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAnyExcept<T>(this System.UnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAnyExcept<T>(this System.UnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAnyExcept<T>(this System.UnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAnyExcept<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int IndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int IndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int IndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int IndexOfAnyExceptInRange<T>(this System.ReadOnlyUnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAnyExceptInRange<T>(this System.UnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        public static int IndexOf<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOf<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int IndexOf<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOf<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOf<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOf<T>(this System.UnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static int IndexOfAnyInRange<T>(this System.ReadOnlyUnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int IndexOfAnyInRange<T>(this System.UnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        public static bool IsWhiteSpace(this System.ReadOnlyUnmanagedSpan<char> span) { throw null; }
        public static int LastIndexOf(this System.ReadOnlyUnmanagedSpan<char> span, System.ReadOnlyUnmanagedSpan<char> value, System.StringComparison comparisonType) { throw null; }
        public static int LastIndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int LastIndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int LastIndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAny<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAny<T>(this System.UnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAny<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAny<T>(this System.UnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAny<T>(this System.UnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAnyExcept<T>(this System.UnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAnyExcept<T>(this System.UnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAnyExcept<T>(this System.UnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAnyExcept<T>(this System.UnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAnyExcept<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int LastIndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int LastIndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int LastIndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.Buffers.SearchValues<T> values) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> values, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int LastIndexOfAnyExceptInRange<T>(this System.ReadOnlyUnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAnyExceptInRange<T>(this System.UnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        public static int LastIndexOf<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOf<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int LastIndexOf<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOf<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOf<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOf<T>(this System.UnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static int LastIndexOfAnyInRange<T>(this System.ReadOnlyUnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int LastIndexOfAnyInRange<T>(this System.UnmanagedSpan<T> span, T lowInclusive, T highInclusive) where T : System.IComparable<T> { throw null; }
        public static bool Overlaps<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other) { throw null; }
        public static bool Overlaps<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other, out int elementOffset) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool Overlaps<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool Overlaps<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other, out int elementOffset) { throw null; }
        public static void Replace<T>(this System.UnmanagedSpan<T> span, T oldValue, T newValue) where T : System.IEquatable<T>? { }
        public static void Replace<T>(this System.UnmanagedSpan<T> span, T oldValue, T newValue, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { }
        public static void Replace<T>(this System.ReadOnlyUnmanagedSpan<T> source, System.UnmanagedSpan<T> destination, T oldValue, T newValue) where T : System.IEquatable<T>? { }
        public static void Replace<T>(this System.ReadOnlyUnmanagedSpan<T> source, System.UnmanagedSpan<T> destination, T oldValue, T newValue, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { }
        public static void ReplaceAny<T>(this System.ReadOnlyUnmanagedSpan<T> source, System.UnmanagedSpan<T> destination, System.Buffers.SearchValues<T> values, T newValue) where T : IEquatable<T>? { }
        public static void ReplaceAny<T>(this System.UnmanagedSpan<T> span, System.Buffers.SearchValues<T> values, T newValue) where T : IEquatable<T>? { throw null; }
        public static void ReplaceAnyExcept<T>(this System.ReadOnlyUnmanagedSpan<T> source, System.UnmanagedSpan<T> destination, System.Buffers.SearchValues<T> values, T newValue) where T : IEquatable<T>? { }
        public static void ReplaceAnyExcept<T>(this System.UnmanagedSpan<T> span, System.Buffers.SearchValues<T> values, T newValue) where T : IEquatable<T>? { throw null; }
        public static void Reverse<T>(this System.UnmanagedSpan<T> span) { }
        public static int SequenceCompareTo<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other) where T : System.IComparable<T>? { throw null; }
        public static int SequenceCompareTo<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other, System.Collections.Generic.IComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static int SequenceCompareTo<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other) where T : System.IComparable<T>? { throw null; }
        public static bool SequenceEqual<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other) where T : System.IEquatable<T>? { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool SequenceEqual<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other) where T : System.IEquatable<T>? { throw null; }
        public static bool SequenceEqual<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool SequenceEqual<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> other, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static void Sort<T>(this System.UnmanagedSpan<T> span) { }
        public static void Sort<T>(this System.UnmanagedSpan<T> span, System.Comparison<T> comparison) { }
        public static void Sort<TKey, TValue>(this System.UnmanagedSpan<TKey> keys, System.UnmanagedSpan<TValue> items) { }
        public static void Sort<TKey, TValue>(this System.UnmanagedSpan<TKey> keys, System.UnmanagedSpan<TValue> items, System.Comparison<TKey> comparison) { }
        public static void Sort<T, TComparer>(this System.UnmanagedSpan<T> span, TComparer comparer) where TComparer : System.Collections.Generic.IComparer<T>? { }
        public static void Sort<TKey, TValue, TComparer>(this System.UnmanagedSpan<TKey> keys, System.UnmanagedSpan<TValue> items, TComparer comparer) where TComparer : System.Collections.Generic.IComparer<TKey>? { }
        public static System.MemoryExtensions.SpanSplitEnumerator<T> Split<T>(this System.ReadOnlyUnmanagedSpan<T> source, T separator) where T : IEquatable<T> { throw null; }
        public static System.MemoryExtensions.SpanSplitEnumerator<T> Split<T>(this System.ReadOnlyUnmanagedSpan<T> source, System.ReadOnlyUnmanagedSpan<T> separator) where T : IEquatable<T> { throw null; }
        public static System.MemoryExtensions.SpanSplitEnumerator<T> SplitAny<T>(this System.ReadOnlyUnmanagedSpan<T> source, [System.Diagnostics.CodeAnalysis.UnscopedRef] params System.ReadOnlyUnmanagedSpan<T> separators) where T : IEquatable<T> { throw null; }
        public static System.MemoryExtensions.SpanSplitEnumerator<T> SplitAny<T>(this System.ReadOnlyUnmanagedSpan<T> source, System.Buffers.SearchValues<T> separators) where T : IEquatable<T> { throw null; }
        public static int Split(this System.ReadOnlyUnmanagedSpan<char> source, System.UnmanagedSpan<System.Range> destination, char separator, System.StringSplitOptions options = System.StringSplitOptions.None) { throw null; }
        public static int Split(this System.ReadOnlyUnmanagedSpan<char> source, System.UnmanagedSpan<System.Range> destination, System.ReadOnlyUnmanagedSpan<char> separator, System.StringSplitOptions options = System.StringSplitOptions.None) { throw null; }
        public static int SplitAny(this System.ReadOnlyUnmanagedSpan<char> source, System.UnmanagedSpan<System.Range> destination, System.ReadOnlyUnmanagedSpan<char> separators, System.StringSplitOptions options = System.StringSplitOptions.None) { throw null; }
        public static int SplitAny(this System.ReadOnlyUnmanagedSpan<char> source, System.UnmanagedSpan<System.Range> destination, System.ReadOnlyUnmanagedSpan<string> separators, System.StringSplitOptions options = System.StringSplitOptions.None) { throw null; }
        public static bool StartsWith(this System.ReadOnlyUnmanagedSpan<char> span, System.ReadOnlyUnmanagedSpan<char> value, System.StringComparison comparisonType) { throw null; }
        public static bool StartsWith<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value) where T : System.IEquatable<T>? { throw null; }
        public static bool StartsWith<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        [System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute(-1)]
        public static bool StartsWith<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> value) where T : System.IEquatable<T>? { throw null; }
        public static bool StartsWith<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value) where T : System.IEquatable<T>? { throw null; }
        public static bool StartsWith<T>(this System.ReadOnlyUnmanagedSpan<T> span, T value, System.Collections.Generic.IEqualityComparer<T>? comparer = null) { throw null; }
        public static int ToLower(this System.ReadOnlyUnmanagedSpan<char> source, System.UnmanagedSpan<char> destination, System.Globalization.CultureInfo? culture) { throw null; }
        public static int ToLowerInvariant(this System.ReadOnlyUnmanagedSpan<char> source, System.UnmanagedSpan<char> destination) { throw null; }
        public static int ToUpper(this System.ReadOnlyUnmanagedSpan<char> source, System.UnmanagedSpan<char> destination, System.Globalization.CultureInfo? culture) { throw null; }
        public static int ToUpperInvariant(this System.ReadOnlyUnmanagedSpan<char> source, System.UnmanagedSpan<char> destination) { throw null; }
        public static System.Memory<char> Trim(this System.Memory<char> memory) { throw null; }
        public static System.ReadOnlyMemory<char> Trim(this System.ReadOnlyMemory<char> memory) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> Trim(this System.ReadOnlyUnmanagedSpan<char> span) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> Trim(this System.ReadOnlyUnmanagedSpan<char> span, char trimChar) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> Trim(this System.ReadOnlyUnmanagedSpan<char> span, System.ReadOnlyUnmanagedSpan<char> trimChars) { throw null; }
        public static System.UnmanagedSpan<char> Trim(this System.UnmanagedSpan<char> span) { throw null; }
        public static System.Memory<char> TrimEnd(this System.Memory<char> memory) { throw null; }
        public static System.ReadOnlyMemory<char> TrimEnd(this System.ReadOnlyMemory<char> memory) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> TrimEnd(this System.ReadOnlyUnmanagedSpan<char> span) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> TrimEnd(this System.ReadOnlyUnmanagedSpan<char> span, char trimChar) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> TrimEnd(this System.ReadOnlyUnmanagedSpan<char> span, System.ReadOnlyUnmanagedSpan<char> trimChars) { throw null; }
        public static System.UnmanagedSpan<char> TrimEnd(this System.UnmanagedSpan<char> span) { throw null; }
        public static System.Memory<T> TrimEnd<T>(this System.Memory<T> memory, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.Memory<T> TrimEnd<T>(this System.Memory<T> memory, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyMemory<T> TrimEnd<T>(this System.ReadOnlyMemory<T> memory, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyMemory<T> TrimEnd<T>(this System.ReadOnlyMemory<T> memory, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyUnmanagedSpan<T> TrimEnd<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyUnmanagedSpan<T> TrimEnd<T>(this System.ReadOnlyUnmanagedSpan<T> span, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.UnmanagedSpan<T> TrimEnd<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.UnmanagedSpan<T> TrimEnd<T>(this System.UnmanagedSpan<T> span, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.Memory<char> TrimStart(this System.Memory<char> memory) { throw null; }
        public static System.ReadOnlyMemory<char> TrimStart(this System.ReadOnlyMemory<char> memory) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> TrimStart(this System.ReadOnlyUnmanagedSpan<char> span) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> TrimStart(this System.ReadOnlyUnmanagedSpan<char> span, char trimChar) { throw null; }
        public static System.ReadOnlyUnmanagedSpan<char> TrimStart(this System.ReadOnlyUnmanagedSpan<char> span, System.ReadOnlyUnmanagedSpan<char> trimChars) { throw null; }
        public static System.UnmanagedSpan<char> TrimStart(this System.UnmanagedSpan<char> span) { throw null; }
        public static System.Memory<T> TrimStart<T>(this System.Memory<T> memory, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.Memory<T> TrimStart<T>(this System.Memory<T> memory, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyMemory<T> TrimStart<T>(this System.ReadOnlyMemory<T> memory, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyMemory<T> TrimStart<T>(this System.ReadOnlyMemory<T> memory, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyUnmanagedSpan<T> TrimStart<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyUnmanagedSpan<T> TrimStart<T>(this System.ReadOnlyUnmanagedSpan<T> span, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.UnmanagedSpan<T> TrimStart<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.UnmanagedSpan<T> TrimStart<T>(this System.UnmanagedSpan<T> span, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.Memory<T> Trim<T>(this System.Memory<T> memory, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.Memory<T> Trim<T>(this System.Memory<T> memory, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyMemory<T> Trim<T>(this System.ReadOnlyMemory<T> memory, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyMemory<T> Trim<T>(this System.ReadOnlyMemory<T> memory, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyUnmanagedSpan<T> Trim<T>(this System.ReadOnlyUnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.ReadOnlyUnmanagedSpan<T> Trim<T>(this System.ReadOnlyUnmanagedSpan<T> span, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static System.UnmanagedSpan<T> Trim<T>(this System.UnmanagedSpan<T> span, System.ReadOnlyUnmanagedSpan<T> trimElements) where T : System.IEquatable<T>? { throw null; }
        public static System.UnmanagedSpan<T> Trim<T>(this System.UnmanagedSpan<T> span, T trimElement) where T : System.IEquatable<T>? { throw null; }
        public static bool TryWrite(this System.UnmanagedSpan<char> destination, [System.Runtime.CompilerServices.InterpolatedStringHandlerArgumentAttribute("destination")] ref System.MemoryExtensions.TryWriteInterpolatedStringHandler handler, out int charsWritten) { throw null; }
        public static bool TryWrite(this System.UnmanagedSpan<char> destination, IFormatProvider? provider, [System.Runtime.CompilerServices.InterpolatedStringHandlerArgumentAttribute("destination", "provider")] ref System.MemoryExtensions.TryWriteInterpolatedStringHandler handler, out int charsWritten) { throw null; }
        public static bool TryWrite<TArg0>(this System.UnmanagedSpan<char> destination, System.IFormatProvider? provider, System.Text.CompositeFormat format, out int charsWritten, TArg0 arg0) { throw null; }
        public static bool TryWrite<TArg0, TArg1>(this System.UnmanagedSpan<char> destination, System.IFormatProvider? provider, System.Text.CompositeFormat format, out int charsWritten, TArg0 arg0, TArg1 arg1) { throw null; }
        public static bool TryWrite<TArg0, TArg1, TArg2>(this System.UnmanagedSpan<char> destination, System.IFormatProvider? provider, System.Text.CompositeFormat format, out int charsWritten, TArg0 arg0, TArg1 arg1, TArg2 arg2) { throw null; }
        public static bool TryWrite(this UnmanagedSpan<char> destination, System.IFormatProvider? provider, System.Text.CompositeFormat format, out int charsWritten, params object?[] args) { throw null; }
        public static bool TryWrite(this UnmanagedSpan<char> destination, System.IFormatProvider? provider, System.Text.CompositeFormat format, out int charsWritten, params System.ReadOnlyUnmanagedSpan<object?> args) { throw null; }
        public ref struct SpanSplitEnumerator<T> : System.Collections.Generic.IEnumerator<System.Range>, System.Collections.IEnumerator, System.IDisposable where T : System.IEquatable<T>
        {
            private object _dummy;
            private int _dummyPrimitive;
            public readonly System.Range Current { get { throw null; } }
            public readonly System.ReadOnlyUnmanagedSpan<T> Source { get { throw null; } }
            public System.MemoryExtensions.SpanSplitEnumerator<T> GetEnumerator() { throw null; }
            public bool MoveNext() { throw null; }
            object System.Collections.IEnumerator.Current { get { throw null; } }
            void System.Collections.IEnumerator.Reset() { throw null; }
            void System.IDisposable.Dispose() { throw null; }
        }
        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        [System.Runtime.CompilerServices.InterpolatedStringHandlerAttribute]
        public ref struct TryWriteInterpolatedStringHandler
        {
            private readonly object _dummy;
            private readonly int _dummyPrimitive;
            public TryWriteInterpolatedStringHandler(int literalLength, int formattedCount, System.UnmanagedSpan<char> destination, out bool shouldAppend) { throw null; }
            public TryWriteInterpolatedStringHandler(int literalLength, int formattedCount, System.UnmanagedSpan<char> destination, IFormatProvider? provider, out bool shouldAppend) { throw null; }
            public bool AppendLiteral(string value) { throw null; }
            public bool AppendFormatted(scoped System.ReadOnlyUnmanagedSpan<char> value) { throw null; }
            public bool AppendFormatted(scoped System.ReadOnlyUnmanagedSpan<char> value, int alignment = 0, string? format = null) { throw null; }
            public bool AppendFormatted<T>(T value) { throw null; }
            public bool AppendFormatted<T>(T value, string? format) { throw null; }
            public bool AppendFormatted<T>(T value, int alignment) { throw null; }
            public bool AppendFormatted<T>(T value, int alignment, string? format) { throw null; }
            public bool AppendFormatted(object? value, int alignment = 0, string? format = null) { throw null; }
            public bool AppendFormatted(string? value) { throw null; }
            public bool AppendFormatted(string? value, int alignment = 0, string? format = null) { throw null; }
        }
    }
}
namespace System.Buffers
{
    public readonly partial struct StandardFormat : System.IEquatable<System.Buffers.StandardFormat>
    {
        private readonly int _dummyPrimitive;
        public const byte MaxPrecision = (byte)99;
        public const byte NoPrecision = (byte)255;
        public StandardFormat(char symbol, byte precision = (byte)255) { throw null; }
        public bool HasPrecision { get { throw null; } }
        public bool IsDefault { get { throw null; } }
        public byte Precision { get { throw null; } }
        public char Symbol { get { throw null; } }
        public bool Equals(System.Buffers.StandardFormat other) { throw null; }
        public override bool Equals([System.Diagnostics.CodeAnalysis.NotNullWhenAttribute(true)] object? obj) { throw null; }
        public override int GetHashCode() { throw null; }
        public static bool operator ==(System.Buffers.StandardFormat left, System.Buffers.StandardFormat right) { throw null; }
        public static implicit operator System.Buffers.StandardFormat (char symbol) { throw null; }
        public static bool operator !=(System.Buffers.StandardFormat left, System.Buffers.StandardFormat right) { throw null; }
        public static System.Buffers.StandardFormat Parse([System.Diagnostics.CodeAnalysis.StringSyntaxAttribute("NumericFormat")] System.ReadOnlyUnmanagedSpan<char> format) { throw null; }
        public static System.Buffers.StandardFormat Parse([System.Diagnostics.CodeAnalysis.StringSyntaxAttribute("NumericFormat")] string? format) { throw null; }
        public override string ToString() { throw null; }
        public static bool TryParse([System.Diagnostics.CodeAnalysis.StringSyntaxAttribute("NumericFormat")] System.ReadOnlyUnmanagedSpan<char> format, out System.Buffers.StandardFormat result) { throw null; }
    }
}
namespace System.Buffers.Binary
{
    public static partial class BinaryPrimitives
    {
        public static System.Numerics.BFloat16 ReadBFloat16BigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static System.Numerics.BFloat16 ReadBFloat16LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static double ReadDoubleBigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static double ReadDoubleLittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static System.Half ReadHalfBigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static System.Half ReadHalfLittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static short ReadInt16BigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static short ReadInt16LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static int ReadInt32BigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static int ReadInt32LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static long ReadInt64BigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static long ReadInt64LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static System.Int128 ReadInt128BigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static System.Int128 ReadInt128LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static nint ReadIntPtrBigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static nint ReadIntPtrLittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static float ReadSingleBigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static float ReadSingleLittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static ushort ReadUInt16BigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static ushort ReadUInt16LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static uint ReadUInt32BigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static uint ReadUInt32LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static ulong ReadUInt64BigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static ulong ReadUInt64LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static System.UInt128 ReadUInt128BigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static System.UInt128 ReadUInt128LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static nuint ReadUIntPtrBigEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static nuint ReadUIntPtrLittleEndian(System.ReadOnlyUnmanagedSpan<byte> source) { throw null; }
        public static byte ReverseEndianness(byte value) { throw null; }
        public static short ReverseEndianness(short value) { throw null; }
        public static int ReverseEndianness(int value) { throw null; }
        public static long ReverseEndianness(long value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static sbyte ReverseEndianness(sbyte value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static ushort ReverseEndianness(ushort value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static uint ReverseEndianness(uint value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static ulong ReverseEndianness(ulong value) { throw null; }
        public static nint ReverseEndianness(nint value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static nuint ReverseEndianness(nuint value) { throw null; }
        public static System.Int128 ReverseEndianness(System.Int128 value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static System.UInt128 ReverseEndianness(System.UInt128 value) { throw null; }
        public static void ReverseEndianness(System.ReadOnlyUnmanagedSpan<int> source, System.UnmanagedSpan<int> destination) { }
        public static void ReverseEndianness(System.ReadOnlyUnmanagedSpan<Int128> source, System.UnmanagedSpan<Int128> destination) { }
        public static void ReverseEndianness(System.ReadOnlyUnmanagedSpan<long> source, System.UnmanagedSpan<long> destination) { }
        public static void ReverseEndianness(System.ReadOnlyUnmanagedSpan<nint> source, System.UnmanagedSpan<nint> destination) { }
        public static void ReverseEndianness(System.ReadOnlyUnmanagedSpan<short> source, System.UnmanagedSpan<short> destination) { }
        [System.CLSCompliant(false)]
        public static void ReverseEndianness(System.ReadOnlyUnmanagedSpan<nuint> source, System.UnmanagedSpan<nuint> destination) { }
        [System.CLSCompliant(false)]
        public static void ReverseEndianness(System.ReadOnlyUnmanagedSpan<uint> source, System.UnmanagedSpan<uint> destination) { }
        [System.CLSCompliant(false)]
        public static void ReverseEndianness(System.ReadOnlyUnmanagedSpan<UInt128> source, System.UnmanagedSpan<UInt128> destination) { }
        [System.CLSCompliant(false)]
        public static void ReverseEndianness(System.ReadOnlyUnmanagedSpan<ulong> source, System.UnmanagedSpan<ulong> destination) { }
        [System.CLSCompliant(false)]
        public static void ReverseEndianness(System.ReadOnlyUnmanagedSpan<ushort> source, System.UnmanagedSpan<ushort> destination) { }
        public static bool TryReadBFloat16BigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out System.Numerics.BFloat16 value) { throw null; }
        public static bool TryReadBFloat16LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out System.Numerics.BFloat16 value) { throw null; }
        public static bool TryReadDoubleBigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out double value) { throw null; }
        public static bool TryReadDoubleLittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out double value) { throw null; }
        public static bool TryReadHalfBigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out System.Half value) { throw null; }
        public static bool TryReadHalfLittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out System.Half value) { throw null; }
        public static bool TryReadInt16BigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out short value) { throw null; }
        public static bool TryReadInt16LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out short value) { throw null; }
        public static bool TryReadInt32BigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out int value) { throw null; }
        public static bool TryReadInt32LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out int value) { throw null; }
        public static bool TryReadInt64BigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out long value) { throw null; }
        public static bool TryReadInt64LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out long value) { throw null; }
        public static bool TryReadInt128BigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out System.Int128 value) { throw null; }
        public static bool TryReadInt128LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out System.Int128 value) { throw null; }
        public static bool TryReadIntPtrBigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out nint value) { throw null; }
        public static bool TryReadIntPtrLittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out nint value) { throw null; }
        public static bool TryReadSingleBigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out float value) { throw null; }
        public static bool TryReadSingleLittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out float value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryReadUInt16BigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out ushort value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryReadUInt16LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out ushort value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryReadUInt32BigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out uint value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryReadUInt32LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out uint value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryReadUInt64BigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out ulong value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryReadUInt64LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out ulong value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryReadUInt128BigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out System.UInt128 value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryReadUInt128LittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out System.UInt128 value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryReadUIntPtrBigEndian(System.ReadOnlyUnmanagedSpan<byte> source, out nuint value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryReadUIntPtrLittleEndian(System.ReadOnlyUnmanagedSpan<byte> source, out nuint value) { throw null; }
        public static bool TryWriteBFloat16BigEndian(System.UnmanagedSpan<byte> destination, System.Numerics.BFloat16 value) { throw null; }
        public static bool TryWriteBFloat16LittleEndian(System.UnmanagedSpan<byte> destination, System.Numerics.BFloat16 value) { throw null; }
        public static bool TryWriteDoubleBigEndian(System.UnmanagedSpan<byte> destination, double value) { throw null; }
        public static bool TryWriteDoubleLittleEndian(System.UnmanagedSpan<byte> destination, double value) { throw null; }
        public static bool TryWriteHalfBigEndian(System.UnmanagedSpan<byte> destination, System.Half value) { throw null; }
        public static bool TryWriteHalfLittleEndian(System.UnmanagedSpan<byte> destination, System.Half value) { throw null; }
        public static bool TryWriteInt16BigEndian(System.UnmanagedSpan<byte> destination, short value) { throw null; }
        public static bool TryWriteInt16LittleEndian(System.UnmanagedSpan<byte> destination, short value) { throw null; }
        public static bool TryWriteInt32BigEndian(System.UnmanagedSpan<byte> destination, int value) { throw null; }
        public static bool TryWriteInt32LittleEndian(System.UnmanagedSpan<byte> destination, int value) { throw null; }
        public static bool TryWriteInt64BigEndian(System.UnmanagedSpan<byte> destination, long value) { throw null; }
        public static bool TryWriteInt64LittleEndian(System.UnmanagedSpan<byte> destination, long value) { throw null; }
        public static bool TryWriteInt128BigEndian(System.UnmanagedSpan<byte> destination, System.Int128 value) { throw null; }
        public static bool TryWriteInt128LittleEndian(System.UnmanagedSpan<byte> destination, System.Int128 value) { throw null; }
        public static bool TryWriteIntPtrBigEndian(System.UnmanagedSpan<byte> destination, nint value) { throw null; }
        public static bool TryWriteIntPtrLittleEndian(System.UnmanagedSpan<byte> destination, nint value) { throw null; }
        public static bool TryWriteSingleBigEndian(System.UnmanagedSpan<byte> destination, float value) { throw null; }
        public static bool TryWriteSingleLittleEndian(System.UnmanagedSpan<byte> destination, float value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryWriteUInt16BigEndian(System.UnmanagedSpan<byte> destination, ushort value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryWriteUInt16LittleEndian(System.UnmanagedSpan<byte> destination, ushort value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryWriteUInt32BigEndian(System.UnmanagedSpan<byte> destination, uint value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryWriteUInt32LittleEndian(System.UnmanagedSpan<byte> destination, uint value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryWriteUInt64BigEndian(System.UnmanagedSpan<byte> destination, ulong value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryWriteUInt64LittleEndian(System.UnmanagedSpan<byte> destination, ulong value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryWriteUInt128BigEndian(System.UnmanagedSpan<byte> destination, System.UInt128 value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryWriteUInt128LittleEndian(System.UnmanagedSpan<byte> destination, System.UInt128 value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryWriteUIntPtrBigEndian(System.UnmanagedSpan<byte> destination, nuint value) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryWriteUIntPtrLittleEndian(System.UnmanagedSpan<byte> destination, nuint value) { throw null; }
        public static void WriteBFloat16BigEndian(System.UnmanagedSpan<byte> destination, System.Numerics.BFloat16 value) { }
        public static void WriteBFloat16LittleEndian(System.UnmanagedSpan<byte> destination, System.Numerics.BFloat16 value) { }
        public static void WriteDoubleBigEndian(System.UnmanagedSpan<byte> destination, double value) { }
        public static void WriteDoubleLittleEndian(System.UnmanagedSpan<byte> destination, double value) { }
        public static void WriteHalfBigEndian(System.UnmanagedSpan<byte> destination, System.Half value) { }
        public static void WriteHalfLittleEndian(System.UnmanagedSpan<byte> destination, System.Half value) { }
        public static void WriteInt16BigEndian(System.UnmanagedSpan<byte> destination, short value) { }
        public static void WriteInt16LittleEndian(System.UnmanagedSpan<byte> destination, short value) { }
        public static void WriteInt32BigEndian(System.UnmanagedSpan<byte> destination, int value) { }
        public static void WriteInt32LittleEndian(System.UnmanagedSpan<byte> destination, int value) { }
        public static void WriteInt64BigEndian(System.UnmanagedSpan<byte> destination, long value) { }
        public static void WriteInt64LittleEndian(System.UnmanagedSpan<byte> destination, long value) { }
        public static void WriteInt128BigEndian(System.UnmanagedSpan<byte> destination, System.Int128 value) { }
        public static void WriteInt128LittleEndian(System.UnmanagedSpan<byte> destination, System.Int128 value) { }
        public static void WriteIntPtrBigEndian(System.UnmanagedSpan<byte> destination, nint value) { }
        public static void WriteIntPtrLittleEndian(System.UnmanagedSpan<byte> destination, nint value) { }
        public static void WriteSingleBigEndian(System.UnmanagedSpan<byte> destination, float value) { }
        public static void WriteSingleLittleEndian(System.UnmanagedSpan<byte> destination, float value) { }
        [System.CLSCompliantAttribute(false)]
        public static void WriteUInt16BigEndian(System.UnmanagedSpan<byte> destination, ushort value) { }
        [System.CLSCompliantAttribute(false)]
        public static void WriteUInt16LittleEndian(System.UnmanagedSpan<byte> destination, ushort value) { }
        [System.CLSCompliantAttribute(false)]
        public static void WriteUInt32BigEndian(System.UnmanagedSpan<byte> destination, uint value) { }
        [System.CLSCompliantAttribute(false)]
        public static void WriteUInt32LittleEndian(System.UnmanagedSpan<byte> destination, uint value) { }
        [System.CLSCompliantAttribute(false)]
        public static void WriteUInt64BigEndian(System.UnmanagedSpan<byte> destination, ulong value) { }
        [System.CLSCompliantAttribute(false)]
        public static void WriteUInt64LittleEndian(System.UnmanagedSpan<byte> destination, ulong value) { }
        [System.CLSCompliantAttribute(false)]
        public static void WriteUInt128BigEndian(System.UnmanagedSpan<byte> destination, System.UInt128 value) { }
        [System.CLSCompliantAttribute(false)]
        public static void WriteUInt128LittleEndian(System.UnmanagedSpan<byte> destination, System.UInt128 value) { }
        [System.CLSCompliantAttribute(false)]
        public static void WriteUIntPtrBigEndian(System.UnmanagedSpan<byte> destination, nuint value) { }
        [System.CLSCompliantAttribute(false)]
        public static void WriteUIntPtrLittleEndian(System.UnmanagedSpan<byte> destination, nuint value) { }
    }
}
namespace System.Buffers.Text
{
    public static partial class Utf8Formatter
    {
        public static bool TryFormat(bool value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(byte value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(System.DateTime value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(System.DateTimeOffset value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(decimal value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(double value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(System.Guid value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(short value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(int value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(long value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryFormat(sbyte value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(float value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        public static bool TryFormat(System.TimeSpan value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryFormat(ushort value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryFormat(uint value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryFormat(ulong value, System.UnmanagedSpan<byte> destination, out int bytesWritten, System.Buffers.StandardFormat format = default(System.Buffers.StandardFormat)) { throw null; }
    }
    public static partial class Utf8Parser
    {
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out bool value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out byte value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out System.DateTime value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out System.DateTimeOffset value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out decimal value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out double value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out System.Guid value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out short value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out int value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out long value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out sbyte value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out float value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out System.TimeSpan value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out ushort value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out uint value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
        [System.CLSCompliantAttribute(false)]
        public static bool TryParse(System.ReadOnlyUnmanagedSpan<byte> source, out ulong value, out int bytesConsumed, char standardFormat = '\0') { throw null; }
    }
}
namespace System.Text
{
    public ref partial struct UnmanagedSpanLineEnumerator : System.Collections.Generic.IEnumerator<System.ReadOnlyUnmanagedSpan<char>>, System.Collections.IEnumerator, System.IDisposable
    {
        private object _dummy;
        private int _dummyPrimitive;
        public System.ReadOnlyUnmanagedSpan<char> Current { get { throw null; } }
        public System.Text.UnmanagedSpanLineEnumerator GetEnumerator() { throw null; }
        public bool MoveNext() { throw null; }
        object System.Collections.IEnumerator.Current { get { throw null; } }
        void System.Collections.IEnumerator.Reset() { throw null; }
        void IDisposable.Dispose() { throw null; }
    }
    public ref partial struct UnmanagedSpanRuneEnumerator : System.Collections.Generic.IEnumerator<System.Text.Rune>, System.Collections.IEnumerator, System.IDisposable
    {
        private object _dummy;
        private int _dummyPrimitive;
        public System.Text.Rune Current { get { throw null; } }
        public System.Text.UnmanagedSpanRuneEnumerator GetEnumerator() { throw null; }
        public bool MoveNext() { throw null; }
        object System.Collections.IEnumerator.Current { get { throw null; } }
        void System.Collections.IEnumerator.Reset() { throw null; }
        void IDisposable.Dispose() { throw null; }
    }
}
