using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.Utilities.Linq;

namespace NumSharp.Unmanaged.Memory
{
    /// <summary>
    ///     Pool of allocated buffers.
    /// </summary>
    /// <remarks>Used to speed up scalar allocation.</remarks>
    public class StackedMemoryPool : IDisposable
    {

        //TODO! this should have a mechanism of auto-trimming via task creation.
        //todo      We only start monitoring once we exceed the existing pool size.
        //todo      If we surpassed 1/3 of the original size:
        //todo      after 10 seconds, a trim should occur - a GC clean in other words.

        private static readonly int DefaultSingleSize = Enum.GetValues(typeof(NPTypeCode)).Cast<NPTypeCode>().Where(v => v != 0).Max(n => n.SizeOf());
        public readonly int SingleSize;
        public readonly int TotalSize;

        private readonly Stack<IntPtr> ptrs;
        private readonly List<IntPtr> allptrs;

        public int Available
        {
            get
            {
                lock (this)
                    return ptrs.Count;
            }
        }

        public StackedMemoryPool(int totalSize) : this(DefaultSingleSize, totalSize / (int)Math.Ceiling(totalSize / (double)DefaultSingleSize)) { }

        public StackedMemoryPool(int singleSize, int count)
        {
            TotalSize = singleSize * count;
            SingleSize = singleSize;

            var queue = new Stack<IntPtr>(count);
            allptrs = new List<IntPtr>(count);
            for (int i = 0; i < count; i++)
            {
                var alloc = Marshal.AllocHGlobal(SingleSize);
                queue.Push(alloc);
                allptrs.Add(alloc);
            }

            ptrs = queue;
        }

        public IntPtr Take()
        {
            lock (this)
            {
                if (ptrs.Count == 0)
                {
                    var ret = Marshal.AllocHGlobal(SingleSize);
                    allptrs.Add(ret);
                    return ret;
                }

                return ptrs.Pop();
            }
        }

        public void Return(IntPtr x)
        {

            lock (this)
            {
                Debug.Assert(ptrs.Count == 0 || ptrs.Contains(x) == false, "ptrs.Contains(x)==false");
                Debug.Assert(allptrs.Contains(x) == true, "allptrs.Contains(x)==true");

                ptrs.Push(x);
            }
        }

        public void AddAllocations(int count)
        {
            lock (this)
                for (int i = 0; i < count; i++)
                {
                    var alloc = Marshal.AllocHGlobal(SingleSize);
                    ptrs.Push(alloc);
                    allptrs.Add(alloc);
                }
        }

        private void ReleaseUnmanagedResources()
        {
            lock (this)
            {
                foreach (var ptr in allptrs)
                    Marshal.FreeHGlobal(ptr);
                ptrs.Clear();
                allptrs.Clear();
            }
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }

        ~StackedMemoryPool() => ReleaseUnmanagedResources();
    }
}
