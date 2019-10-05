using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends;
#pragma warning disable IDE0052 // Remove unread private members

namespace NumSharp.Unmanaged.Memory
{
    /// <summary>
    ///     Pool of allocated buffers managed by internal garbage collection mechanism.
    /// </summary>
    /// <remarks>Used to speed up scalar allocation. Thread-safe.</remarks>
    public class StackedMemoryPool : IDisposable
    {
        //TODO! Instead of allocating n buffer, allocate one block and split it.
        private static readonly int DefaultSingleSize = Enum.GetValues(typeof(NPTypeCode)).Cast<NPTypeCode>().Where(v => v != 0).Max(n => n.SizeOf());

        private long _originalTotals;
        private long _originalTotalsThreshold;
        private long _totalCount;

        public readonly int SingleSize;

        private readonly Stack<IntPtr> availables;
        private readonly Stack<IntPtr> availables_blocks;
        private readonly List<IntPtr> individualyAllocated;
        private List<UnmanagedMemoryBlock<byte>> _blocks;

        private bool _isGcRunning = false;
        private System.Threading.Timer _gcTimer = null;
        private volatile bool _abortGc = false;
#if DEBUG
        public event Action GCInvoked;
#endif
        /// <summary>
        ///     How many Scalar pointers are allocated.
        /// </summary>
        public long TotalAllocated => _totalCount;

        /// <summary>
        ///     How many pointers are currently preallocated and available for the taking.
        /// </summary>
        public long Available
        {
            get
            {
                lock (this)
                    return availables.Count + availables_blocks.Count;
            }
        }

        /// <summary>
        ///     After how many milliseconds should unused excess memory be deallocated. (only if allocated exceeded above 133% of firstly allocated).<br></br>
        ///     Default: 5000ms.
        /// </summary>
        public int GarbageCollectionDelay = 5000;

        public StackedMemoryPool(int totalSize) : this(DefaultSingleSize, totalSize / (int)Math.Ceiling(totalSize / (double)DefaultSingleSize)) { }

        public StackedMemoryPool(int singleSize, int count)
        {
            SingleSize = singleSize;

            availables = new Stack<IntPtr>(0);
            availables_blocks = new Stack<IntPtr>(count);
            individualyAllocated = new List<IntPtr>(count);
            _blocks = new List<UnmanagedMemoryBlock<byte>>(1);

            AllocateCount(count);

            _originalTotals = count;
            _originalTotalsThreshold = unchecked((int)Math.Min((count * 1.333f), int.MaxValue));
        }


        public IntPtr Take()
        {
            lock (this)
            {
                if (availables.Count == 0)
                {
                    if (availables_blocks.Count != 0)
                        return availables_blocks.Pop();

                    var ret = Marshal.AllocHGlobal(SingleSize);
                    individualyAllocated.Add(ret);
                    _totalCount += 1;
                    return ret;
                }

                return availables.Pop();
            }
        }

        public unsafe void Return(IntPtr x)
        {
            //check if it belongs to a block
            foreach (var block in _blocks)
            {
                var x_addr = x.ToPointer();
                if (block.Address <= x_addr && x_addr <= (block.Address + block.BytesCount - SingleSize))
                {
                    lock (this)
                    {
                        availables_blocks.Push(x);
                        tryStartGC();
                        return;
                    }

                }
            }

            //belongs to an individual
            lock (this)
            {
                Debug.Assert(availables.Count == 0 || availables.Contains(x) == false, "availables.Contains(x)==false");

                availables.Push(x);
                tryStartGC();
            }
        }


        #region Garbage Collection

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void tryStartGC()
        {
            //try start GC
            if (_isGcRunning || availables.Count + availables_blocks.Count < _originalTotalsThreshold)
                return;

            _isGcRunning = true;
            _runGC(GarbageCollectionDelay);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void _runGC(int delay, int restarts = 0)
        {
            //assinging to prevent GC from collecting.
            _gcTimer = new System.Threading.Timer(state => CollectGarbage((GCContext)state), new GCContext(availables.Count, restarts), delay, -1);
        }

        private void CollectGarbage(GCContext ctx)
        {
            lock (this)
            {
                if (_abortGc)
                {
                    _abortGc = false;
                    return;
                }

                //if there were pushes or pops during the 5000 delay, postpone for an other 5000 seconds.
                if (!ctx.Immediate && availables.Count != ctx.LastAvailablePtrs) //postpone
                    goto _restart;

                var removeCount = Math.Min(_totalCount - _originalTotals, availables.Count);
                for (int i = 0; i < removeCount; i++)
                {
                    var addr = availables.Pop();
                    Debug.Assert(individualyAllocated.Contains(addr), "individualyAllocated.Contains(addr)");
                    individualyAllocated.Remove(addr);
                    Marshal.FreeHGlobal(addr);
                }

                availables.TrimExcess();
                individualyAllocated.TrimExcess();

                if (removeCount > 0)
                    _totalCount -= removeCount;
#if DEBUG
                GCInvoked?.Invoke();
#endif
                _isGcRunning = false;
                return;
            }

//first restart is 1000ms, second and above is GarbageCollectionDelay
_restart:
            _runGC(ctx.Restarts >= 1 ? GarbageCollectionDelay : 1000, ctx.Restarts + 1);
        }

        #endregion

        public unsafe void AllocateBytes(long bytes)
        {
            AllocateCount(Math.Max(bytes / SingleSize, 1));
        }

        public unsafe void AllocateCount(long count)
        {
            if (count <= 0)
                throw new ArgumentException(nameof(count));

            lock (this)
            {
                var blocksize = SingleSize * count;
                var block = new UnmanagedMemoryBlock<byte>(blocksize);

                var addr = new IntPtr(block.Address);
                for (int i = 0; i < count; i++, addr += SingleSize)
                    availables_blocks.Push(addr);

                _blocks.Add(block);

                _totalCount += count;
            }
        }

        public void TrimExcess()
        {
            CollectGarbage(new GCContext() { Immediate = true });
        }

        /// <summary>
        ///     Set the point of GC activation to the current <see cref="TotalAllocated"/> multiplied by 1.33.
        /// </summary>
        public void UpdateGarbageCollectionThreshold()
        {
            _originalTotals = _totalCount;
            _originalTotalsThreshold = unchecked((int)Math.Min((_totalCount * 1.333f), int.MaxValue));
        }

        private void ReleaseUnmanagedResources()
        {
            lock (this)
            {
                _blocks.Clear(); //clearing block removes all existing references causing GC to collect and dispose the memory block.
                availables.Clear();
                availables_blocks.Clear();

                individualyAllocated.ForEach(Marshal.FreeHGlobal);
                individualyAllocated.Clear();
            }
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
            _abortGc = true;
        }

        ~StackedMemoryPool() => ReleaseUnmanagedResources();

        private class GCContext
        {
            public int LastAvailablePtrs;
            public int Restarts;
            public bool Immediate;
            public GCContext() { }
            public GCContext(int lastAvailablePtrs, int restarts)
            {
                LastAvailablePtrs = lastAvailablePtrs;
                Restarts = restarts;
            }
        }

    }
}
