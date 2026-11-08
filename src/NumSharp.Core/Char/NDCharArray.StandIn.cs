using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

// =============================================================================
// NDCharArray.StandIn.cs — the element stand-in, built for the cost of NumPy's str
// =============================================================================
//
// c[i] in NumPy is a Python str / bytes: one small object, ~330-530 ns with the Python call around it. NumSharp's
// stand-in is a read-only 0-d chararray that owns a copy of the stripped text, and an array is several objects: the
// NDArray (finalizable, reference counted), its storage, the element slice, and for a buffer of its own a pooled block
// with a finalizable Disposer. Built the general way (UnmanagedStorage.AllocateString), a stand-in cost ~360 ns, more than
// NumPy's whole S access. Three things make it cheap, without changing what it is:
//
//   THE ARENA   The text is cut from a per-thread block of BlockBytes (StandInArena). A piece is an ArraySlice of the
//               block, so it shares the block's Disposer: a stand-in's reference counts on the block, and the block goes
//               back to the pool when the arena has moved on AND the last stand-in cut from it is disposed (or collected:
//               the Disposer's finalizer frees it once nothing reaches it). No pool round trip and no Disposer per element.
//               A stand-in still owns its text exclusively — no other array reads or writes its piece — so it is the
//               immutable snapshot NumPy's str is. What a live stand-in keeps is its block (at most BlockBytes), not just
//               its text; a piece wider than MaxPieceBytes gets a block of its own as before.
//   LEAN OBJECTS  The element slice and the storage are built from fields the caller already holds
//               (StringArraySlice's trusted constructor, UnmanagedStorage.ReadOnlyStringScalar): no descriptor checks, no
//               integer divisions, no lane switch.
//   SHARED DESCRIPTORS  The <U5 / |S5 descriptors of the stand-ins are created once per width (StandInDescr).
//
// The ownership a test or a user can see is unchanged: the stand-in owns its data (OWNDATA), is read-only, and its
// buffer is released by Dispose like any array's. Two things differ from a block of its own, neither part of NumPy's str:
// IsUniquelyReferenced (ndarray.resize's refcheck) is false while another stand-in of the same block is alive, and the
// block's memory returns to the pool with the LAST stand-in of the block, not with each one.
//
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        /// <summary>
        ///     The scalar stand-in for the fixed-width element at <paramref name="first"/>: a NEW read-only 0-d chararray of
        ///     the element's class that owns a copy of its text, stripped when <paramref name="strip"/>. The element access
        ///     paths call it with the element's address directly, so no view of the indexed array is created.
        /// </summary>
        /// <param name="first">The element's first byte, inside <paramref name="source"/>'s buffer.</param>
        /// <param name="source">The buffer the element lives in: its class (<c>S</c> / <c>U</c>) and width are read here.</param>
        /// <param name="strip">True for <c>chararray.__getitem__</c>'s <c>rstrip()</c>; false for the unstripped NumPy scalars
        ///     <c>take</c> and the flat iterator return.</param>
        /// <returns>The stand-in (<see cref="IsScalarStandIn"/>); its width is the kept length, at least one character.</returns>
        /// <exception cref="SystemError">A <c>U</c> code unit above <c>U+10FFFF</c> in the kept text: NumPy cannot create the
        ///     <c>str</c> (<c>invalid maximum character passed to PyUnicode_New</c>).</exception>
        /// <exception cref="OutOfMemoryException">No memory for the stand-in's block.</exception>
        /// <remarks>The caller keeps <paramref name="source"/>'s buffer alive for the duration of the call (it holds the
        ///     indexed array); the stand-in copies the text and keeps no reference on that buffer. The validation runs
        ///     before any memory is taken, so a refused element leaves nothing behind.</remarks>
        internal static unsafe NDCharArray StandIn(byte* first, StringArraySlice source, bool strip)
        {
            bool unicode = source.IsUnicode;
            int kept = strip ? KeptUnits(first, source.Chars, unicode) : TrimmedUnits(first, source.Chars, unicode);
            if (unicode)
                RequireValidCodePoints((uint*)first, kept);
            // np.array('') is <U1: a value with no character left still takes one character, a NUL.
            int width = Math.Max(kept, 1);
            int bytes = unicode ? width * 4 : width;
            ArraySlice<byte> piece = bytes <= StandInArena.MaxPieceBytes
                ? StandInArena.Piece(bytes)
                : ArraySlice<byte>.Allocate(bytes);
            byte* to = piece.Address;
            if (kept == 0)
            {
                // Arena and pool memory is not zeroed: the NUL character is written here.
                if (unicode)
                    *(uint*)to = 0;
                else
                    *to = 0;
            }
            else
            {
                Buffer.MemoryCopy(first, to, bytes, bytes);   // kept == width: the whole piece is text
            }

            var element = new StringArraySlice(piece, StandInDescr(unicode, width), bytes, width, unicode, count: 1);
            // The NDArray constructor takes the stand-in's counted reference on the piece's block.
            return new NDCharArray(UnmanagedStorage.ReadOnlyStringScalar(element)) { IsScalarStandIn = true };
        }

        /// <summary>The <c>U</c> stand-in descriptors by width, filled on first use (see <see cref="StandInDescr"/>).</summary>
        private static readonly DType[] UnicodeStandInDescrs = new DType[StandInDescrCacheWidths + 1];

        /// <summary>The <c>S</c> stand-in descriptors by width, filled on first use (see <see cref="StandInDescr"/>).</summary>
        private static readonly DType[] BytesStandInDescrs = new DType[StandInDescrCacheWidths + 1];

        /// <summary>The widest stand-in whose descriptor is cached; a wider element gets a fresh descriptor per access.</summary>
        private const int StandInDescrCacheWidths = 256;

        /// <summary>
        ///     The descriptor of a stand-in holding <paramref name="width"/> characters (<c>&lt;U5</c>, <c>|S5</c>). Element
        ///     access asks for one per call, and a new descriptor object cost a sixth of a stand-in's time, so the descriptors up
        ///     to <see cref="StandInDescrCacheWidths"/> characters are created once and shared: a descriptor is immutable, so
        ///     stand-ins of one width can hold the same object (NumPy's arrays share descriptors the same way).
        /// </summary>
        /// <param name="unicode">True for <c>U</c>, false for <c>S</c>.</param>
        /// <param name="width">The character count, at least 1.</param>
        /// <returns>The sized native descriptor.</returns>
        /// <remarks>Two threads that fill the same slot at once each create a descriptor; the slot keeps one of them. Both are
        ///     equal, so the race costs one extra allocation and nothing else.</remarks>
        private static DType StandInDescr(bool unicode, int width)
        {
            if (width > StandInDescrCacheWidths)
                return unicode ? DTypeRegistry.Str.Descr(width) : DTypeRegistry.Bytes.Descr(width);
            var cache = unicode ? UnicodeStandInDescrs : BytesStandInDescrs;
            return cache[width] ??= unicode ? DTypeRegistry.Str.Descr(width) : DTypeRegistry.Bytes.Descr(width);
        }

        /// <summary>
        ///     The per-thread block the element stand-ins are cut from (see the file header): each piece is an
        ///     <see cref="ArraySlice{T}"/> of the current block and so shares its reference count, which saves the pool round
        ///     trip and the finalizable Disposer a buffer of its own costs.
        /// </summary>
        /// <remarks>
        ///     The arena holds ONE counted reference on its current block, taken when the block is allocated and released when
        ///     the arena moves to the next one; every stand-in cut from the block holds one more. The block is freed by the
        ///     last of those releases, on whichever thread it happens. One instance per thread (<see cref="ThreadStaticAttribute"/>),
        ///     so cutting needs no synchronization; the reference counts are atomic, so a stand-in may be disposed anywhere.
        /// </remarks>
        private sealed class StandInArena
        {
            /// <summary>The size of one block: 64 pieces of a <c>U8</c> stand-in, 256 of an <c>S8</c> one.</summary>
            internal const int BlockBytes = 2048;

            /// <summary>The widest piece cut from a block (a <c>U64</c> stand-in); a wider stand-in gets a block of its own,
            ///     so a block never wastes more than this at its end.</summary>
            internal const int MaxPieceBytes = 256;

            /// <summary>The arena of the current thread, created on its first stand-in.</summary>
            [ThreadStatic] private static StandInArena _current;

            /// <summary>The current block; meaningful only while <see cref="_hasBlock"/>.</summary>
            private ArraySlice<byte> _block;

            /// <summary>Whether <see cref="_block"/> holds a block (and the arena's reference on it).</summary>
            private bool _hasBlock;

            /// <summary>The bytes of <see cref="_block"/> already cut, rounded up to 8 per piece.</summary>
            private int _used;

            /// <summary>Cuts a piece of <paramref name="bytes"/> bytes from the current thread's block.</summary>
            /// <param name="bytes">The piece size, 1 to <see cref="MaxPieceBytes"/>.</param>
            /// <returns>A piece sharing the block's reference count; its bytes are not initialized.</returns>
            /// <exception cref="OutOfMemoryException">No memory for a new block.</exception>
            internal static ArraySlice<byte> Piece(int bytes) => (_current ??= new StandInArena()).Cut(bytes);

            /// <summary>Cuts a piece from this arena's block, moving to a new block when the rest is too small.</summary>
            /// <param name="bytes">The piece size, 1 to <see cref="MaxPieceBytes"/>.</param>
            /// <returns>The piece.</returns>
            /// <exception cref="OutOfMemoryException">No memory for a new block.</exception>
            private ArraySlice<byte> Cut(int bytes)
            {
                // 8-byte steps keep every piece aligned for its UCS-4 units (the pool's blocks start 64-byte aligned).
                int step = (bytes + 7) & ~7;
                if (!_hasBlock || _used + step > BlockBytes)
                    Renew();
                var piece = _block.Slice(_used, bytes);
                _used += step;
                return piece;
            }

            /// <summary>
            ///     Moves to a new block: allocates it and takes the arena's reference first, then releases the arena's
            ///     reference on the old block, which is freed there and then unless a stand-in still holds a piece of it.
            /// </summary>
            /// <exception cref="OutOfMemoryException">No memory for the new block; the arena keeps its old block.</exception>
            private void Renew()
            {
                var fresh = ArraySlice<byte>.Allocate(BlockBytes);
                fresh.TryAddRef();
                var retired = _block;
                bool hadBlock = _hasBlock;
                _block = fresh;
                _hasBlock = true;
                _used = 0;
                if (hadBlock)
                    retired.Release();
            }
        }
    }
}
