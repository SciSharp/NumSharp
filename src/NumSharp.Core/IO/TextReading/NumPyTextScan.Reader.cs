using System;
using System.Numerics;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.IO
{
    internal static partial class NumPyTextScan
    {
        /// <summary>NumPy's <c>FROM_BUFFER_SIZE</c>: the element count a read of unknown length starts with.</summary>
        private const long FromBufferSize = 4096;

        /// <summary>
        ///     The values one text read has scanned, in its family's staging lane (<see cref="StagingType"/>), in a pooled
        ///     buffer that grows by doubling (NumPy grows its result by <see cref="FromBufferSize"/> elements at a time; only
        ///     the capacity sequence differs, never an element).
        /// </summary>
        /// <remarks>Owns pooled memory: the reader disposes it whether the read succeeds or raises.</remarks>
        private sealed unsafe class Stage : IDisposable
        {
            /// <summary>The staging buffer (null before the first element).</summary>
            internal byte* Buffer;

            /// <summary>The bytes of one staged value.</summary>
            internal readonly int ItemSize;

            private long _capacityBytes;

            /// <summary>Creates an empty stage.</summary>
            /// <param name="itemSize">The bytes of one staged value.</param>
            internal Stage(int itemSize) => ItemSize = itemSize;

            /// <summary>The slot of element <paramref name="index"/>, growing the buffer to hold it.</summary>
            /// <param name="index">The element index (at most one past the last slot handed out).</param>
            /// <returns>The slot's address.</returns>
            /// <exception cref="OutOfMemoryException">The pool cannot provide the larger buffer.</exception>
            internal byte* Slot(long index)
            {
                long needed = (index + 1) * ItemSize;
                if (needed > _capacityBytes)
                {
                    long grown = Math.Max(_capacityBytes * 2, FromBufferSize * ItemSize);
                    while (grown < needed)
                        grown *= 2;
                    var fresh = (byte*)SizeBucketedBufferPool.Take(grown);
                    if (Buffer != null)
                    {
                        System.Buffer.MemoryCopy(Buffer, fresh, grown, _capacityBytes);
                        // returned with the size it was taken with: the pool's rule (the size selects the bucket)
                        SizeBucketedBufferPool.Return((IntPtr)Buffer, _capacityBytes);
                    }
                    Buffer = fresh;
                    _capacityBytes = grown;
                }
                return Buffer + index * ItemSize;
            }

            /// <summary>Returns the buffer to the pool (idempotent).</summary>
            public void Dispose()
            {
                if (Buffer != null)
                {
                    SizeBucketedBufferPool.Return((IntPtr)Buffer, _capacityBytes);
                    Buffer = null;
                    _capacityBytes = 0;
                }
            }
        }

        /// <summary>
        ///     NumPy's <c>array_from_text</c> over a string (<c>np.fromstring</c>'s text mode, <c>fromstr_next_element</c> and
        ///     <c>fromstr_skip_separator</c>): elements and separators alternate until the text ends, <paramref name="num"/>
        ///     elements are read, or a separator does not match.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     <b>The result is NumPy's.</b> With <paramref name="num"/> &gt;= 0 NumPy never shrinks the array, so the result has
        ///     <paramref name="num"/> elements whatever was read. The element after the last one read holds the value the failed
        ///     scan stored (NumPy's <c>fromstr</c> writes its result before the reader looks at the progress: 0 for an integer,
        ///     -1.0 for a float, -1+0j for a complex); NumPy leaves the rest uninitialized, NumSharp zeroes it. With
        ///     <paramref name="num"/> &lt; 0 the result holds exactly the elements read.
        ///     </para>
        ///     <para>
        ///     <b>A read without progress</b> is the end when the text has ended and a failure otherwise; so is a separator that
        ///     does not match, except right after the last requested element, where the separator is optional.
        ///     </para>
        /// </remarks>
        /// <param name="data">The text's bytes (NumPy's <c>s#</c> argument: a <c>str</c> encoded as UTF-8, or the bytes).</param>
        /// <param name="dtype">The numeric dtype (a non-native byte order reads as its native dtype).</param>
        /// <param name="num">The element count to read; negative reads until the text ends.</param>
        /// <param name="sep">The separator's UTF-8 bytes (non-empty, no NUL).</param>
        /// <returns>The 1-D array.</returns>
        /// <exception cref="ValueError">A failed read or an unmatched separator (<c>string or file could not be read to its end
        ///     due to unmatched data</c>), or an element count too large to allocate (<c>array is too big</c>).</exception>
        internal static unsafe NDArray FromString(ReadOnlySpan<byte> data, DType dtype, long num, ReadOnlySpan<byte> sep)
        {
            NPTypeCode tc = dtype.GetTypeCode();
            // array_from_text allocates the result before it reads anything: a count no array can have fails first.
            if (num >= 0)
                AllocationGuard.CheckElementCount(num, tc.SizeOf());
            TextScanFamily family = FamilyOf(tc);
            NPTypeCode staging = StagingType(family);
            byte[] swabbed = SwabSeparator(sep);
            using var stage = new Stage(staging.SizeOf());

            int pos = 0, stop = 0;
            long nread = 0, written = 0;
            for (long i = 0; num < 0 || i < num; i++)
            {
                stop = FromstrNextElement(data, ref pos, family, stage.Slot(nread));
                // the scan wrote its slot whether or not it read anything (the element NumPy leaves behind)
                written = nread + 1;
                if (stop < 0)
                    break;
                nread++;
                stop = FromstrSkipSeparator(data, ref pos, swabbed);
                if (stop < 0)
                {
                    // the separator after the last requested element is optional
                    if (num == i + 1)
                        stop = -1;
                    break;
                }
            }
            if (stop == -2)
                throw Unmatched();
            long length = num >= 0 ? num : nread;
            return Finish(stage, staging, tc, length, Math.Min(written, length));
        }

        /// <summary>
        ///     NumPy's <c>array_from_text</c> over a file (<c>np.fromfile</c>'s text mode, <c>fromfile_next_element</c> and
        ///     <c>fromfile_skip_separator</c>); <c>PyArray_FromFile</c> then shrinks the result to the elements read.
        /// </summary>
        /// <param name="fp">The file, at the position the read starts (after <c>offset</c>).</param>
        /// <param name="dtype">The numeric dtype (a non-native byte order reads as its native dtype).</param>
        /// <param name="num">The element count to read; negative reads until the file ends.</param>
        /// <param name="sep">The separator's UTF-8 bytes (non-empty, no NUL).</param>
        /// <returns>The 1-D array of the elements read.</returns>
        /// <exception cref="ValueError">A failed read or an unmatched separator (<c>string or file could not be read to its end
        ///     due to unmatched data</c>), or an element count too large to allocate (<c>array is too big</c>).</exception>
        internal static unsafe NDArray FromFile(CFileReader fp, DType dtype, long num, ReadOnlySpan<byte> sep)
        {
            NPTypeCode tc = dtype.GetTypeCode();
            if (num >= 0)
                AllocationGuard.CheckElementCount(num, tc.SizeOf());
            TextScanFamily family = FamilyOf(tc);
            NPTypeCode staging = StagingType(family);
            byte[] swabbed = SwabSeparator(sep);
            using var stage = new Stage(staging.SizeOf());

            int stop = 0;
            long nread = 0;
            for (long i = 0; num < 0 || i < num; i++)
            {
                stop = FromfileNextElement(fp, family, stage.Slot(nread));
                if (stop < 0)
                    break;
                nread++;
                stop = FromfileSkipSeparator(fp, swabbed);
                if (stop < 0)
                {
                    if (num == i + 1)
                        stop = -1;
                    break;
                }
            }
            if (stop == -2)
                throw Unmatched();
            return Finish(stage, staging, tc, nread, nread);
        }

        /// <summary>
        ///     NumPy's <c>fromstr_next_element</c>: scans one element with the family's <c>fromstr</c> routine into
        ///     <paramref name="slot"/>, then judges the progress.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="pos">The position; moved past the element when it was read.</param>
        /// <param name="family">The scan family.</param>
        /// <param name="slot">The staging slot (always written).</param>
        /// <returns>0 when read, -1 when nothing was read and the text has ended, -2 when nothing was read before its
        ///     end (or the value cannot be stored).</returns>
        private static unsafe int FromstrNextElement(ReadOnlySpan<byte> s, ref int pos, TextScanFamily family, byte* slot)
        {
            int end, r = 0;
            switch (family)
            {
                case TextScanFamily.Signed:
                    *(long*)slot = PyOSStrtol(s, pos, out end);
                    break;
                case TextScanFamily.Unsigned:
                    *(ulong*)slot = PyOSStrtoul(s, pos, out end);
                    break;
                case TextScanFamily.Complex:
                    *(Complex*)slot = CDoubleFromstr(s, pos, out end);
                    break;
                case TextScanFamily.Decimal:
                    r = DecimalFromstr(s, pos, out end, out decimal d);
                    *(decimal*)slot = d;
                    break;
                default:
                    *(double*)slot = NumPyOSStrtod(s, pos, out end);
                    break;
            }
            if (end == pos || r < 0)
                // string_is_fully_read: the end, or a failure before it
                return pos >= s.Length ? -1 : -2;
            pos = end;
            return 0;
        }

        /// <summary>
        ///     NumPy's <c>fromfile_next_element</c>: one element with the family's <c>scanfunc</c> (<see cref="ScanInteger"/>,
        ///     <see cref="Ftolf"/>, <see cref="ScanComplex"/>), mapping <c>fscanf</c>'s 1 / 0 / EOF.
        /// </summary>
        /// <param name="fp">The file.</param>
        /// <param name="family">The scan family.</param>
        /// <param name="slot">The staging slot (written when the element is read).</param>
        /// <returns>0 when read, -1 at the end of the file, -2 on a matching failure.</returns>
        private static unsafe int FromfileNextElement(CFileReader fp, TextScanFamily family, byte* slot)
        {
            int r;
            switch (family)
            {
                case TextScanFamily.Signed:
                case TextScanFamily.Unsigned:
                    r = ScanInteger(fp, family == TextScanFamily.Unsigned, out ulong bits);
                    *(ulong*)slot = bits;
                    break;
                case TextScanFamily.Complex:
                    r = ScanComplex(fp, out Complex z);
                    *(Complex*)slot = z;
                    break;
                case TextScanFamily.Decimal:
                    r = ScanDecimal(fp, out decimal d);
                    *(decimal*)slot = d;
                    break;
                default:
                    r = Ftolf(fp, out double x);
                    *(double*)slot = x;
                    break;
            }
            return r == 1 ? 0 : r < 0 ? -1 : -2;
        }

        /// <summary>
        ///     Builds the result: <paramref name="length"/> elements of <paramref name="tc"/>, the first
        ///     <paramref name="written"/> converted from the staging lane with the dtype's C assignment (the cast kernels:
        ///     integers truncate, a double rounds to float32 / float16, a double is true when not 0), the rest zero.
        /// </summary>
        /// <param name="stage">The staged values.</param>
        /// <param name="staging">The staging lane.</param>
        /// <param name="tc">The result's dtype.</param>
        /// <param name="length">The result's element count.</param>
        /// <param name="written">The staged values to convert (at most <paramref name="length"/>).</param>
        /// <returns>A fresh 1-D array that owns its data.</returns>
        private static unsafe NDArray Finish(Stage stage, NPTypeCode staging, NPTypeCode tc, long length, long written)
        {
            var result = new NDArray(tc, Shape.Vector(length), written < length);
            if (written == 0)
                return result;
            byte* dst = (byte*)result.Storage.Address;
            if (staging == tc)
            {
                long bytes = written * stage.ItemSize;
                System.Buffer.MemoryCopy(stage.Buffer, dst, bytes, bytes);
                return result;
            }
            var kernel = DirectILKernelGenerator.TryGetCastKernel(staging, tc);
            if (kernel != null)
            {
                kernel(stage.Buffer, dst, written);
                return result;
            }
            // No contiguous kernel for the pair: the general cast of a view over the staged values.
            using var view = np.frombuffer((IntPtr)stage.Buffer, written * stage.ItemSize, staging);
            using var cast = view.astype(tc);
            long castBytes = written * tc.SizeOf();
            System.Buffer.MemoryCopy((byte*)cast.Storage.Address, dst, castBytes, castBytes);
            return result;
        }

        /// <summary>The error NumPy's reader raises for a failed read or an unmatched separator.</summary>
        /// <returns>The exception.</returns>
        internal static ValueError Unmatched()
            => new ValueError("string or file could not be read to its end due to unmatched data");
    }
}
