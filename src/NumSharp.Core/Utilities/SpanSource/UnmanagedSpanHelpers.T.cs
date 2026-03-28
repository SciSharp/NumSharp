using System;
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Utilities
{
    internal static partial class UnmanagedSpanHelpers // .T
    {
        public static unsafe void Fill<T>(ref T refData, nuint numElements, T value) where T : unmanaged
        {
            // Early checks to see if it's even possible to vectorize - JIT will turn these checks into consts.
            // - Vectorization must be hardware-accelerated
            // - T's size must not exceed the vector's size
            // - T's size must be a whole power of 2
            // Note: T is constrained to unmanaged, so no reference check needed

            if (!Vector.IsHardwareAccelerated)
            {
                goto CannotVectorize;
            }

            if (sizeof(T) > Vector<byte>.Count)
            {
                goto CannotVectorize;
            }

            if (!BitOperations.IsPow2(sizeof(T)))
            {
                goto CannotVectorize;
            }

            if (numElements >= (uint)(Vector<byte>.Count / sizeof(T)))
            {
                // We have enough data for at least one vectorized write.
                Vector<byte> vector;

                if (sizeof(T) == 1)
                {
                    vector = new Vector<byte>(Unsafe.BitCast<T, byte>(value));
                }
                else if (sizeof(T) == 2)
                {
                    vector = (Vector<byte>)new Vector<ushort>(Unsafe.BitCast<T, ushort>(value));
                }
                else if (sizeof(T) == 4)
                {
                    // special-case float since it's already passed in a SIMD reg
                    vector = (typeof(T) == typeof(float))
                        ? (Vector<byte>)new Vector<float>(Unsafe.BitCast<T, float>(value))
                        : (Vector<byte>)new Vector<uint>(Unsafe.BitCast<T, uint>(value));
                }
                else if (sizeof(T) == 8)
                {
                    // special-case double since it's already passed in a SIMD reg
                    vector = (typeof(T) == typeof(double))
                        ? (Vector<byte>)new Vector<double>(Unsafe.BitCast<T, double>(value))
                        : (Vector<byte>)new Vector<ulong>(Unsafe.BitCast<T, ulong>(value));
                }
                else if (sizeof(T) == Vector<byte>.Count)
                {
                    vector = Unsafe.BitCast<T, Vector<byte>>(value);
                }
                else
                {
                    // For types larger than 8 bytes (e.g., decimal, Guid), fall back to scalar
                    goto CannotVectorize;
                }

                ref byte refDataAsBytes = ref Unsafe.As<T, byte>(ref refData);
                nuint totalByteLength = numElements * (nuint)sizeof(T); // get this calculation ready ahead of time
                nuint stopLoopAtOffset = totalByteLength & (nuint)(nint)(2 * (int)-Vector<byte>.Count); // intentional sign extension carries the negative bit
                nuint offset = 0;

                // Loop, writing 2 vectors at a time.
                // Compare 'numElements' rather than 'stopLoopAtOffset' because we don't want a dependency
                // on the very recently calculated 'stopLoopAtOffset' value.

                if (numElements >= (uint)(2 * Vector<byte>.Count / sizeof(T)))
                {
                    do
                    {
                        Unsafe.WriteUnaligned(ref Unsafe.AddByteOffset(ref refDataAsBytes, offset), vector);
                        Unsafe.WriteUnaligned(ref Unsafe.AddByteOffset(ref refDataAsBytes, offset + (nuint)Vector<byte>.Count), vector);
                        offset += (uint)(2 * Vector<byte>.Count);
                    } while (offset < stopLoopAtOffset);
                }

                // At this point, if any data remains to be written, it's strictly less than
                // 2 * sizeof(Vector) bytes. The loop above had us write an even number of vectors.
                // If the total byte length instead involves us writing an odd number of vectors, write
                // one additional vector now. The bit check below tells us if we're in an "odd vector
                // count" situation.

                if ((totalByteLength & (nuint)Vector<byte>.Count) != 0)
                {
                    Unsafe.WriteUnaligned(ref Unsafe.AddByteOffset(ref refDataAsBytes, offset), vector);
                }

                // It's possible that some small buffer remains to be populated - something that won't
                // fit an entire vector's worth of data. Instead of falling back to a loop, we'll write
                // a vector at the very end of the buffer. This may involve overwriting previously
                // populated data, which is fine since we're splatting the same value for all entries.
                // There's no need to perform a length check here because we already performed this
                // check before entering the vectorized code path.

                Unsafe.WriteUnaligned(ref Unsafe.AddByteOffset(ref refDataAsBytes, totalByteLength - (nuint)Vector<byte>.Count), vector);

                // And we're done!

                return;
            }

        CannotVectorize:

            // If we reached this point, we cannot vectorize this T, or there are too few
            // elements for us to vectorize. Fall back to an unrolled loop.

            nuint i = 0;

            // Write 8 elements at a time

            if (numElements >= 8)
            {
                nuint stopLoopAtOffset = numElements & ~(nuint)7;
                do
                {
                    Unsafe.Add(ref refData, (nint)i + 0) = value;
                    Unsafe.Add(ref refData, (nint)i + 1) = value;
                    Unsafe.Add(ref refData, (nint)i + 2) = value;
                    Unsafe.Add(ref refData, (nint)i + 3) = value;
                    Unsafe.Add(ref refData, (nint)i + 4) = value;
                    Unsafe.Add(ref refData, (nint)i + 5) = value;
                    Unsafe.Add(ref refData, (nint)i + 6) = value;
                    Unsafe.Add(ref refData, (nint)i + 7) = value;
                } while ((i += 8) < stopLoopAtOffset);
            }

            // Write next 4 elements if needed

            if ((numElements & 4) != 0)
            {
                Unsafe.Add(ref refData, (nint)i + 0) = value;
                Unsafe.Add(ref refData, (nint)i + 1) = value;
                Unsafe.Add(ref refData, (nint)i + 2) = value;
                Unsafe.Add(ref refData, (nint)i + 3) = value;
                i += 4;
            }

            // Write next 2 elements if needed

            if ((numElements & 2) != 0)
            {
                Unsafe.Add(ref refData, (nint)i + 0) = value;
                Unsafe.Add(ref refData, (nint)i + 1) = value;
                i += 2;
            }

            // Write final element if needed

            if ((numElements & 1) != 0)
            {
                Unsafe.Add(ref refData, (nint)i) = value;
            }
        }

        public static long IndexOf<T>(ref T searchSpace, long searchSpaceLength, ref T value, long valueLength) where T : IEquatable<T>?
        {
            Debug.Assert(searchSpaceLength >= 0);
            Debug.Assert(valueLength >= 0);

            if (valueLength == 0)
                return 0;  // A zero-length sequence is always treated as "found" at the start of the search space.

            T valueHead = value;
            ref T valueTail = ref Unsafe.Add(ref value, 1);
            long valueTailLength = valueLength - 1;

            long index = 0;
            while (true)
            {
                Debug.Assert(0 <= index && index <= searchSpaceLength); // Ensures no deceptive underflows in the computation of "remainingSearchSpaceLength".
                long remainingSearchSpaceLength = searchSpaceLength - index - valueTailLength;
                if (remainingSearchSpaceLength <= 0)
                {
                    break;  // The unsearched portion is now shorter than the sequence we're looking for. So it can't be there.
                }

                // Do a quick search for the first element of "value".
                long relativeIndex = IndexOf(ref Unsafe.Add(ref searchSpace, (nint)index), valueHead, remainingSearchSpaceLength);
                if (relativeIndex < 0)
                {
                    break;
                }
                index += relativeIndex;

                // Found the first element of "value". See if the tail matches.
                if (SequenceEqual(ref Unsafe.Add(ref searchSpace, (nint)(index + 1)), ref valueTail, valueTailLength))
                {
                    return index;  // The tail matched. Return a successful find.
                }

                index++;
            }
            return -1;
        }

        // Adapted from IndexOf(...)
        public static bool Contains<T>(ref T searchSpace, T value, long length) where T : IEquatable<T>?
        {
            Debug.Assert(length >= 0);

            nint index = 0; // Use nint for arithmetic to avoid unnecessary 64->32->64 truncations

            if (default(T) != null || (object?)value != null)
            {
                Debug.Assert(value is not null);

                while (length >= 8)
                {
                    length -= 8;

                    if (value.Equals(Unsafe.Add(ref searchSpace, index + 0)) ||
                        value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 1))) ||
                        value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 2))) ||
                        value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 3))) ||
                        value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 4))) ||
                        value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 5))) ||
                        value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 6))) ||
                        value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 7))))
                    {
                        return true;
                    }

                    index += 8;
                }

                if (length >= 4)
                {
                    length -= 4;

                    if (value.Equals(Unsafe.Add(ref searchSpace, index + 0)) ||
                        value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 1))) ||
                        value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 2))) ||
                        value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 3))))
                    {
                        return true;
                    }

                    index += 4;
                }

                while (length > 0)
                {
                    length--;

                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)index)))
                    {
                        return true;
                    }

                    index += 1;
                }
            }
            else
            {
                nint len = (nint)length;
                for (index = 0; index < len; index++)
                {
                    if ((object?)Unsafe.Add(ref searchSpace, (nint)index) is null)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        public static long IndexOf<T>(ref T searchSpace, T value, long length) where T : IEquatable<T>?
        {
            Debug.Assert(length >= 0);

            nint index = 0; // Use nint for arithmetic to avoid unnecessary 64->32->64 truncations
            if (default(T) != null || (object?)value != null)
            {
                Debug.Assert(value is not null);

                while (length >= 8)
                {
                    length -= 8;

                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)index)))
                    {
                        return (long)index;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 1))))
                    {
                        return (long)(index + 1);
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 2))))
                    {
                        return (long)(index + 2);
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 3))))
                    {
                        return (long)(index + 3);
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 4))))
                    {
                        return (long)(index + 4);
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 5))))
                    {
                        return (long)(index + 5);
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 6))))
                    {
                        return (long)(index + 6);
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 7))))
                    {
                        return (long)(index + 7);
                    }

                    index += 8;
                }

                if (length >= 4)
                {
                    length -= 4;

                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)index)))
                    {
                        return (long)index;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 1))))
                    {
                        return (long)(index + 1);
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 2))))
                    {
                        return (long)(index + 2);
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(index + 3))))
                    {
                        return (long)(index + 3);
                    }

                    index += 4;
                }

                while (length > 0)
                {
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)index)))
                    {
                        return (long)index;
                    }

                    index += 1;
                    length--;
                }
            }
            else
            {
                nint len = (nint)length;
                for (index = 0; index < len; index++)
                {
                    if ((object?)Unsafe.Add(ref searchSpace, (nint)index) is null)
                    {
                        return (long)index;
                    }
                }
            }

            return -1;
        }

        public static long IndexOfAny<T>(ref T searchSpace, T value0, T value1, long length) where T : IEquatable<T>?
        {
            Debug.Assert(length >= 0);

            T lookUp;
            long index = 0;
            if (default(T) != null || ((object?)value0 != null && (object?)value1 != null))
            {
                Debug.Assert(value0 is not null && value1 is not null);

                while ((length - index) >= 8)
                {
                    lookUp = Unsafe.Add(ref searchSpace, (nint)index);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 1));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index + 1;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 2));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index + 2;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 3));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index + 3;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 4));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index + 4;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 5));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index + 5;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 6));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index + 6;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 7));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index + 7;
                    }

                    index += 8;
                }

                if ((length - index) >= 4)
                {
                    lookUp = Unsafe.Add(ref searchSpace, (nint)index);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 1));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index + 1;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 2));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index + 2;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 3));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index + 3;
                    }

                    index += 4;
                }

                while (index < length)
                {
                    lookUp = Unsafe.Add(ref searchSpace, (nint)index);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return index;
                    }

                    index++;
                }
            }
            else
            {
                for (index = 0; index < length; index++)
                {
                    lookUp = Unsafe.Add(ref searchSpace, (nint)index);
                    if ((object?)lookUp is null)
                    {
                        if ((object?)value0 is null || (object?)value1 is null)
                        {
                            return index;
                        }
                    }
                    else if (lookUp.Equals(value0) || lookUp.Equals(value1))
                    {
                        return index;
                    }
                }
            }

            return -1;
        }

        public static long IndexOfAny<T>(ref T searchSpace, T value0, T value1, T value2, long length) where T : IEquatable<T>?
        {
            Debug.Assert(length >= 0);

            T lookUp;
            long index = 0;
            if (default(T) != null || ((object?)value0 != null && (object?)value1 != null && (object?)value2 != null))
            {
                Debug.Assert(value0 is not null && value1 is not null && value2 is not null);

                while ((length - index) >= 8)
                {
                    lookUp = Unsafe.Add(ref searchSpace, (nint)index);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 1));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index + 1;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 2));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index + 2;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 3));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index + 3;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 4));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index + 4;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 5));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index + 5;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 6));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index + 6;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 7));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index + 7;
                    }

                    index += 8;
                }

                if ((length - index) >= 4)
                {
                    lookUp = Unsafe.Add(ref searchSpace, (nint)index);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 1));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index + 1;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 2));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index + 2;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(index + 3));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index + 3;
                    }

                    index += 4;
                }

                while (index < length)
                {
                    lookUp = Unsafe.Add(ref searchSpace, (nint)index);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return index;
                    }

                    index++;
                }
            }
            else
            {
                for (index = 0; index < length; index++)
                {
                    lookUp = Unsafe.Add(ref searchSpace, (nint)index);
                    if ((object?)lookUp is null)
                    {
                        if ((object?)value0 is null || (object?)value1 is null || (object?)value2 is null)
                        {
                            return index;
                        }
                    }
                    else if (lookUp.Equals(value0) || lookUp.Equals(value1) || lookUp.Equals(value2))
                    {
                        return index;
                    }
                }
            }

            return -1;
        }

        public static long IndexOfAny<T>(ref T searchSpace, long searchSpaceLength, ref T value, long valueLength) where T : IEquatable<T>?
        {
            Debug.Assert(searchSpaceLength >= 0);
            Debug.Assert(valueLength >= 0);

            if (valueLength == 0)
                return -1;  // A zero-length set of values is always treated as "not found".

            // For the following paragraph, let:
            //   n := length of haystack
            //   i := index of first occurrence of any needle within haystack
            //   l := length of needle array
            //
            // We use a naive non-vectorized search because we want to bound the complexity of IndexOfAny
            // to O(i * l) rather than O(n * l), or just O(n * l) if no needle is found. The reason for
            // this is that it's common for callers to invoke IndexOfAny immediately before slicing,
            // and when this is called in a loop, we want the entire loop to be bounded by O(n * l)
            // rather than O(n^2 * l).

            if (typeof(T).IsValueType)
            {
                // Calling ValueType.Equals (devirtualized), which takes 'this' byref. We'll make
                // a byval copy of the candidate from the search space in the outer loop, then in
                // the inner loop we'll pass a ref (as 'this') to each element in the needle.
                for (long i = 0; i < searchSpaceLength; i++)
                {
                    T candidate = Unsafe.Add(ref searchSpace, (nint)i);
                    for (long j = 0; j < valueLength; j++)
                    {
                        if (Unsafe.Add(ref value, (nint)j)!.Equals(candidate))
                        {
                            return i;
                        }
                    }
                }
            }
            else
            {
                // Calling IEquatable<T>.Equals (virtual dispatch). We'll perform the null check
                // in the outer loop instead of in the inner loop to save some branching.
                for (long i = 0; i < searchSpaceLength; i++)
                {
                    T candidate = Unsafe.Add(ref searchSpace, (nint)i);
                    if (candidate is not null)
                    {
                        for (long j = 0; j < valueLength; j++)
                        {
                            if (candidate.Equals(Unsafe.Add(ref value, (nint)j)))
                            {
                                return i;
                            }
                        }
                    }
                    else
                    {
                        for (long j = 0; j < valueLength; j++)
                        {
                            if (Unsafe.Add(ref value, (nint)j) is null)
                            {
                                return i;
                            }
                        }
                    }
                }
            }

            return -1; // not found
        }

        public static long LastIndexOf<T>(ref T searchSpace, long searchSpaceLength, ref T value, long valueLength) where T : IEquatable<T>?
        {
            Debug.Assert(searchSpaceLength >= 0);
            Debug.Assert(valueLength >= 0);

            if (valueLength == 0)
                return searchSpaceLength;  // A zero-length sequence is always treated as "found" at the end of the search space.

            long valueTailLength = valueLength - 1;
            if (valueTailLength == 0)
            {
                return LastIndexOf(ref searchSpace, value, searchSpaceLength);
            }

            long index = 0;

            T valueHead = value;
            ref T valueTail = ref Unsafe.Add(ref value, 1);

            while (true)
            {
                Debug.Assert(0 <= index && index <= searchSpaceLength); // Ensures no deceptive underflows in the computation of "remainingSearchSpaceLength".
                long remainingSearchSpaceLength = searchSpaceLength - index - valueTailLength;
                if (remainingSearchSpaceLength <= 0)
                {
                    break;  // The unsearched portion is now shorter than the sequence we're looking for. So it can't be there.
                }

                // Do a quick search for the first element of "value".
                long relativeIndex = LastIndexOf(ref searchSpace, valueHead, remainingSearchSpaceLength);
                if (relativeIndex < 0)
                {
                    break;
                }

                // Found the first element of "value". See if the tail matches.
                if (SequenceEqual(ref Unsafe.Add(ref searchSpace, (nint)(relativeIndex + 1)), ref valueTail, valueTailLength))
                {
                    return relativeIndex;  // The tail matched. Return a successful find.
                }

                index += remainingSearchSpaceLength - relativeIndex;
            }
            return -1;
        }

        public static long LastIndexOf<T>(ref T searchSpace, T value, long length) where T : IEquatable<T>?
        {
            Debug.Assert(length >= 0);

            if (default(T) != null || (object?)value != null)
            {
                Debug.Assert(value is not null);

                while (length >= 8)
                {
                    length -= 8;

                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(length + 7))))
                    {
                        return length + 7;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(length + 6))))
                    {
                        return length + 6;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(length + 5))))
                    {
                        return length + 5;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(length + 4))))
                    {
                        return length + 4;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(length + 3))))
                    {
                        return length + 3;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(length + 2))))
                    {
                        return length + 2;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(length + 1))))
                    {
                        return length + 1;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)length)))
                    {
                        return length;
                    }
                }

                if (length >= 4)
                {
                    length -= 4;

                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(length + 3))))
                    {
                        return length + 3;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(length + 2))))
                    {
                        return length + 2;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)(length + 1))))
                    {
                        return length + 1;
                    }
                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)length)))
                    {
                        return length;
                    }
                }

                while (length > 0)
                {
                    length--;

                    if (value.Equals(Unsafe.Add(ref searchSpace, (nint)length)))
                    {
                        return length;
                    }
                }
            }
            else
            {
                for (length--; length >= 0; length--)
                {
                    if ((object?)Unsafe.Add(ref searchSpace, (nint)length) is null)
                    {
                        return length;
                    }
                }
            }

            return -1;
        }

        public static long LastIndexOfAny<T>(ref T searchSpace, T value0, T value1, long length) where T : IEquatable<T>?
        {
            Debug.Assert(length >= 0);

            T lookUp;
            if (default(T) != null || ((object?)value0 != null && (object?)value1 != null))
            {
                Debug.Assert(value0 is not null && value1 is not null);

                while (length >= 8)
                {
                    length -= 8;

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 7));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length + 7;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 6));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length + 6;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 5));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length + 5;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 4));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length + 4;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 3));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length + 3;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 2));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length + 2;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 1));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length + 1;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)length);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length;
                    }
                }

                if (length >= 4)
                {
                    length -= 4;

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 3));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length + 3;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 2));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length + 2;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 1));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length + 1;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)length);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length;
                    }
                }

                while (length > 0)
                {
                    length--;

                    lookUp = Unsafe.Add(ref searchSpace, (nint)length);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp))
                    {
                        return length;
                    }
                }
            }
            else
            {
                for (length--; length >= 0; length--)
                {
                    lookUp = Unsafe.Add(ref searchSpace, (nint)length);
                    if ((object?)lookUp is null)
                    {
                        if ((object?)value0 is null || (object?)value1 is null)
                        {
                            return length;
                        }
                    }
                    else if (lookUp.Equals(value0) || lookUp.Equals(value1))
                    {
                        return length;
                    }
                }
            }

            return -1;
        }

        public static long LastIndexOfAny<T>(ref T searchSpace, T value0, T value1, T value2, long length) where T : IEquatable<T>?
        {
            Debug.Assert(length >= 0);

            T lookUp;
            if (default(T) != null || ((object?)value0 != null && (object?)value1 != null && (object?)value2 != null))
            {
                Debug.Assert(value0 is not null && value1 is not null && value2 is not null);

                while (length >= 8)
                {
                    length -= 8;

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 7));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length + 7;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 6));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length + 6;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 5));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length + 5;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 4));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length + 4;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 3));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length + 3;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 2));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length + 2;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 1));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length + 1;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)length);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length;
                    }
                }

                if (length >= 4)
                {
                    length -= 4;

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 3));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length + 3;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 2));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length + 2;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)(length + 1));
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length + 1;
                    }

                    lookUp = Unsafe.Add(ref searchSpace, (nint)length);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length;
                    }
                }

                while (length > 0)
                {
                    length--;

                    lookUp = Unsafe.Add(ref searchSpace, (nint)length);
                    if (value0.Equals(lookUp) || value1.Equals(lookUp) || value2.Equals(lookUp))
                    {
                        return length;
                    }
                }
            }
            else
            {
                for (length--; length >= 0; length--)
                {
                    lookUp = Unsafe.Add(ref searchSpace, (nint)length);
                    if ((object?)lookUp is null)
                    {
                        if ((object?)value0 is null || (object?)value1 is null || (object?)value2 is null)
                        {
                            return length;
                        }
                    }
                    else if (lookUp.Equals(value0) || lookUp.Equals(value1) || lookUp.Equals(value2))
                    {
                        return length;
                    }
                }
            }

            return -1;
        }

        public static long LastIndexOfAny<T>(ref T searchSpace, long searchSpaceLength, ref T value, long valueLength) where T : IEquatable<T>?
        {
            Debug.Assert(searchSpaceLength >= 0);
            Debug.Assert(valueLength >= 0);

            if (valueLength == 0)
                return -1;  // A zero-length set of values is always treated as "not found".

            // See comments in IndexOfAny(ref T, int, ref T, int) above regarding algorithmic complexity concerns.
            // This logic is similar, but it runs backward.
            if (typeof(T).IsValueType)
            {
                for (long i = searchSpaceLength - 1; i >= 0; i--)
                {
                    T candidate = Unsafe.Add(ref searchSpace, (nint)i);
                    for (long j = 0; j < valueLength; j++)
                    {
                        if (Unsafe.Add(ref value, (nint)j)!.Equals(candidate))
                        {
                            return i;
                        }
                    }
                }
            }
            else
            {
                for (long i = searchSpaceLength - 1; i >= 0; i--)
                {
                    T candidate = Unsafe.Add(ref searchSpace, (nint)i);
                    if (candidate is not null)
                    {
                        for (long j = 0; j < valueLength; j++)
                        {
                            if (candidate.Equals(Unsafe.Add(ref value, (nint)j)))
                            {
                                return i;
                            }
                        }
                    }
                    else
                    {
                        for (long j = 0; j < valueLength; j++)
                        {
                            if (Unsafe.Add(ref value, (nint)j) is null)
                            {
                                return i;
                            }
                        }
                    }
                }
            }

            return -1; // not found
        }

        internal static long IndexOfAnyExcept<T>(ref T searchSpace, T value0, long length)
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            for (long i = 0; i < length; i++)
            {
                if (!EqualityComparer<T>.Default.Equals(Unsafe.Add(ref searchSpace, (nint)i), value0))
                {
                    return i;
                }
            }

            return -1;
        }

        internal static long LastIndexOfAnyExcept<T>(ref T searchSpace, T value0, long length)
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            for (long i = length - 1; i >= 0; i--)
            {
                if (!EqualityComparer<T>.Default.Equals(Unsafe.Add(ref searchSpace, (nint)i), value0))
                {
                    return i;
                }
            }

            return -1;
        }

        internal static long IndexOfAnyExcept<T>(ref T searchSpace, T value0, T value1, long length)
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            for (long i = 0; i < length; i++)
            {
                ref T current = ref Unsafe.Add(ref searchSpace, (nint)i);
                if (!EqualityComparer<T>.Default.Equals(current, value0) && !EqualityComparer<T>.Default.Equals(current, value1))
                {
                    return i;
                }
            }

            return -1;
        }

        internal static long LastIndexOfAnyExcept<T>(ref T searchSpace, T value0, T value1, long length)
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            for (long i = length - 1; i >= 0; i--)
            {
                ref T current = ref Unsafe.Add(ref searchSpace, (nint)i);
                if (!EqualityComparer<T>.Default.Equals(current, value0) && !EqualityComparer<T>.Default.Equals(current, value1))
                {
                    return i;
                }
            }

            return -1;
        }

        internal static long IndexOfAnyExcept<T>(ref T searchSpace, T value0, T value1, T value2, long length)
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            for (long i = 0; i < length; i++)
            {
                ref T current = ref Unsafe.Add(ref searchSpace, (nint)i);
                if (!EqualityComparer<T>.Default.Equals(current, value0)
                    && !EqualityComparer<T>.Default.Equals(current, value1)
                    && !EqualityComparer<T>.Default.Equals(current, value2))
                {
                    return i;
                }
            }

            return -1;
        }

        internal static long LastIndexOfAnyExcept<T>(ref T searchSpace, T value0, T value1, T value2, long length)
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            for (long i = length - 1; i >= 0; i--)
            {
                ref T current = ref Unsafe.Add(ref searchSpace, (nint)i);
                if (!EqualityComparer<T>.Default.Equals(current, value0)
                    && !EqualityComparer<T>.Default.Equals(current, value1)
                    && !EqualityComparer<T>.Default.Equals(current, value2))
                {
                    return i;
                }
            }

            return -1;
        }

        internal static long IndexOfAnyExcept<T>(ref T searchSpace, T value0, T value1, T value2, T value3, long length)
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            for (long i = 0; i < length; i++)
            {
                ref T current = ref Unsafe.Add(ref searchSpace, (nint)i);
                if (!EqualityComparer<T>.Default.Equals(current, value0)
                    && !EqualityComparer<T>.Default.Equals(current, value1)
                    && !EqualityComparer<T>.Default.Equals(current, value2)
                    && !EqualityComparer<T>.Default.Equals(current, value3))
                {
                    return i;
                }
            }

            return -1;
        }

        internal static long LastIndexOfAnyExcept<T>(ref T searchSpace, T value0, T value1, T value2, T value3, long length)
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            for (long i = length - 1; i >= 0; i--)
            {
                ref T current = ref Unsafe.Add(ref searchSpace, (nint)i);
                if (!EqualityComparer<T>.Default.Equals(current, value0)
                    && !EqualityComparer<T>.Default.Equals(current, value1)
                    && !EqualityComparer<T>.Default.Equals(current, value2)
                    && !EqualityComparer<T>.Default.Equals(current, value3))
                {
                    return i;
                }
            }

            return -1;
        }

        public static bool SequenceEqual<T>(ref T first, ref T second, long length) where T : IEquatable<T>?
        {
            Debug.Assert(length >= 0);

            if (Unsafe.AreSame(ref first, ref second))
            {
                return true;
            }

            nint index = 0; // Use nint for arithmetic to avoid unnecessary 64->32->64 truncations
            T lookUp0;
            T lookUp1;
            while (length >= 8)
            {
                length -= 8;

                lookUp0 = Unsafe.Add(ref first, index);
                lookUp1 = Unsafe.Add(ref second, index);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                lookUp0 = Unsafe.Add(ref first, index + 1);
                lookUp1 = Unsafe.Add(ref second, index + 1);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                lookUp0 = Unsafe.Add(ref first, index + 2);
                lookUp1 = Unsafe.Add(ref second, index + 2);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                lookUp0 = Unsafe.Add(ref first, index + 3);
                lookUp1 = Unsafe.Add(ref second, index + 3);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                lookUp0 = Unsafe.Add(ref first, index + 4);
                lookUp1 = Unsafe.Add(ref second, index + 4);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                lookUp0 = Unsafe.Add(ref first, index + 5);
                lookUp1 = Unsafe.Add(ref second, index + 5);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                lookUp0 = Unsafe.Add(ref first, index + 6);
                lookUp1 = Unsafe.Add(ref second, index + 6);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                lookUp0 = Unsafe.Add(ref first, index + 7);
                lookUp1 = Unsafe.Add(ref second, index + 7);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                index += 8;
            }

            if (length >= 4)
            {
                length -= 4;

                lookUp0 = Unsafe.Add(ref first, index);
                lookUp1 = Unsafe.Add(ref second, index);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                lookUp0 = Unsafe.Add(ref first, index + 1);
                lookUp1 = Unsafe.Add(ref second, index + 1);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                lookUp0 = Unsafe.Add(ref first, index + 2);
                lookUp1 = Unsafe.Add(ref second, index + 2);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                lookUp0 = Unsafe.Add(ref first, index + 3);
                lookUp1 = Unsafe.Add(ref second, index + 3);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                index += 4;
            }

            while (length > 0)
            {
                lookUp0 = Unsafe.Add(ref first, index);
                lookUp1 = Unsafe.Add(ref second, index);
                if (!(lookUp0?.Equals(lookUp1) ?? (object?)lookUp1 is null))
                {
                    return false;
                }

                index += 1;
                length--;
            }

            return true;
        }

        public static int SequenceCompareTo<T>(ref T first, long firstLength, ref T second, long secondLength)
            where T : IComparable<T>?
        {
            Debug.Assert(firstLength >= 0);
            Debug.Assert(secondLength >= 0);

            long minLength = firstLength;
            if (minLength > secondLength)
                minLength = secondLength;
            for (long i = 0; i < minLength; i++)
            {
                T lookUp = Unsafe.Add(ref second, (nint)i);
                int result = (Unsafe.Add(ref first, (nint)i)?.CompareTo(lookUp) ?? (((object?)lookUp is null) ? 0 : -1));
                if (result != 0)
                {
                    return result;
                }
            }

            return firstLength.CompareTo(secondLength);
        }

        // ==================================================================================
        // SIMD-optimized value type methods below
        // These use INumber<T> constraint for == operator and SIMD acceleration
        // ==================================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool ContainsValueType<T>(ref T searchSpace, T value, long length) where T : struct, INumber<T>
        {
            return NonPackedContainsValueType(ref searchSpace, value, length);
        }

        internal static bool NonPackedContainsValueType<T>(ref T searchSpace, T value, long length) where T : struct, INumber<T>
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            if (!Vector128.IsHardwareAccelerated || length < Vector128<T>.Count)
            {
                nuint offset = 0;

                while (length >= 8)
                {
                    length -= 8;

                    if (Unsafe.Add(ref searchSpace, offset) == value
                     || Unsafe.Add(ref searchSpace, offset + 1) == value
                     || Unsafe.Add(ref searchSpace, offset + 2) == value
                     || Unsafe.Add(ref searchSpace, offset + 3) == value
                     || Unsafe.Add(ref searchSpace, offset + 4) == value
                     || Unsafe.Add(ref searchSpace, offset + 5) == value
                     || Unsafe.Add(ref searchSpace, offset + 6) == value
                     || Unsafe.Add(ref searchSpace, offset + 7) == value)
                    {
                        return true;
                    }

                    offset += 8;
                }

                if (length >= 4)
                {
                    length -= 4;

                    if (Unsafe.Add(ref searchSpace, offset) == value
                     || Unsafe.Add(ref searchSpace, offset + 1) == value
                     || Unsafe.Add(ref searchSpace, offset + 2) == value
                     || Unsafe.Add(ref searchSpace, offset + 3) == value)
                    {
                        return true;
                    }

                    offset += 4;
                }

                while (length > 0)
                {
                    length -= 1;

                    if (Unsafe.Add(ref searchSpace, offset) == value)
                    {
                        return true;
                    }

                    offset += 1;
                }
            }
            else if (Vector512.IsHardwareAccelerated && length >= Vector512<T>.Count)
            {
                Vector512<T> current, values = Vector512.Create(value);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector512<T>.Count));

                // Loop until either we've finished all elements or there's less than a vector's-worth remaining.
                do
                {
                    current = Vector512.LoadUnsafe(ref currentSearchSpace);

                    if (Vector512.EqualsAny(values, current))
                    {
                        return true;
                    }

                    currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector512<T>.Count);
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                // If any elements remain, process the last vector in the search space.
                if (length % Vector512<T>.Count != 0)
                {
                    current = Vector512.LoadUnsafe(ref oneVectorAwayFromEnd);

                    if (Vector512.EqualsAny(values, current))
                    {
                        return true;
                    }
                }
            }
            else if (Vector256.IsHardwareAccelerated && length >= Vector256<T>.Count)
            {
                Vector256<T> equals, values = Vector256.Create(value);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector256<T>.Count));

                // Loop until either we've finished all elements or there's less than a vector's-worth remaining.
                do
                {
                    equals = Vector256.Equals(values, Vector256.LoadUnsafe(ref currentSearchSpace));
                    if (equals == Vector256<T>.Zero)
                    {
                        currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector256<T>.Count);
                        continue;
                    }

                    return true;
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                // If any elements remain, process the last vector in the search space.
                if (length % Vector256<T>.Count != 0)
                {
                    equals = Vector256.Equals(values, Vector256.LoadUnsafe(ref oneVectorAwayFromEnd));
                    if (equals != Vector256<T>.Zero)
                    {
                        return true;
                    }
                }
            }
            else
            {
                Vector128<T> equals, values = Vector128.Create(value);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector128<T>.Count));

                // Loop until either we've finished all elements or there's less than a vector's-worth remaining.
                do
                {
                    equals = Vector128.Equals(values, Vector128.LoadUnsafe(ref currentSearchSpace));
                    if (equals == Vector128<T>.Zero)
                    {
                        currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector128<T>.Count);
                        continue;
                    }

                    return true;
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                // If any elements remain, process the first vector in the search space.
                if (length % Vector128<T>.Count != 0)
                {
                    equals = Vector128.Equals(values, Vector128.LoadUnsafe(ref oneVectorAwayFromEnd));
                    if (equals != Vector128<T>.Zero)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long IndexOfValueType<T>(ref T searchSpace, T value, long length) where T : struct, INumber<T>
        {
            return NonPackedIndexOfValueType(ref searchSpace, value, length);
        }

        internal static long NonPackedIndexOfValueType<T>(ref T searchSpace, T value, long length) where T : struct, INumber<T>
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            if (!Vector128.IsHardwareAccelerated || length < Vector128<T>.Count)
            {
                nuint offset = 0;

                while (length >= 8)
                {
                    length -= 8;

                    if (Unsafe.Add(ref searchSpace, offset) == value) return (long)offset;
                    if (Unsafe.Add(ref searchSpace, offset + 1) == value) return (long)(offset + 1);
                    if (Unsafe.Add(ref searchSpace, offset + 2) == value) return (long)(offset + 2);
                    if (Unsafe.Add(ref searchSpace, offset + 3) == value) return (long)(offset + 3);
                    if (Unsafe.Add(ref searchSpace, offset + 4) == value) return (long)(offset + 4);
                    if (Unsafe.Add(ref searchSpace, offset + 5) == value) return (long)(offset + 5);
                    if (Unsafe.Add(ref searchSpace, offset + 6) == value) return (long)(offset + 6);
                    if (Unsafe.Add(ref searchSpace, offset + 7) == value) return (long)(offset + 7);

                    offset += 8;
                }

                if (length >= 4)
                {
                    length -= 4;

                    if (Unsafe.Add(ref searchSpace, offset) == value) return (long)offset;
                    if (Unsafe.Add(ref searchSpace, offset + 1) == value) return (long)(offset + 1);
                    if (Unsafe.Add(ref searchSpace, offset + 2) == value) return (long)(offset + 2);
                    if (Unsafe.Add(ref searchSpace, offset + 3) == value) return (long)(offset + 3);

                    offset += 4;
                }

                while (length > 0)
                {
                    length -= 1;

                    if (Unsafe.Add(ref searchSpace, offset) == value)
                    {
                        return (long)offset;
                    }

                    offset += 1;
                }

                return -1;
            }
            else if (Vector512.IsHardwareAccelerated && length >= Vector512<T>.Count)
            {
                Vector512<T> current, values = Vector512.Create(value);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector512<T>.Count));

                // Loop until either we've finished all elements or there's less than a vector's-worth remaining.
                do
                {
                    current = Vector512.LoadUnsafe(ref currentSearchSpace);
                    var equals = Vector512.Equals(values, current);

                    if (equals != Vector512<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref currentSearchSpace, equals);
                    }

                    currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector512<T>.Count);
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                // If any elements remain, process the last vector in the search space.
                if (length % Vector512<T>.Count != 0)
                {
                    current = Vector512.LoadUnsafe(ref oneVectorAwayFromEnd);
                    var equals = Vector512.Equals(values, current);

                    if (equals != Vector512<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref oneVectorAwayFromEnd, equals);
                    }
                }
            }
            else if (Vector256.IsHardwareAccelerated && length >= Vector256<T>.Count)
            {
                Vector256<T> equals, values = Vector256.Create(value);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector256<T>.Count));

                // Loop until either we've finished all elements or there's less than a vector's-worth remaining.
                do
                {
                    equals = Vector256.Equals(values, Vector256.LoadUnsafe(ref currentSearchSpace));
                    if (equals == Vector256<T>.Zero)
                    {
                        currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector256<T>.Count);
                        continue;
                    }

                    return ComputeFirstIndex(ref searchSpace, ref currentSearchSpace, equals);
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                // If any elements remain, process the last vector in the search space.
                if (length % Vector256<T>.Count != 0)
                {
                    equals = Vector256.Equals(values, Vector256.LoadUnsafe(ref oneVectorAwayFromEnd));
                    if (equals != Vector256<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref oneVectorAwayFromEnd, equals);
                    }
                }
            }
            else
            {
                Vector128<T> equals, values = Vector128.Create(value);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector128<T>.Count));

                // Loop until either we've finished all elements or there's less than a vector's-worth remaining.
                do
                {
                    equals = Vector128.Equals(values, Vector128.LoadUnsafe(ref currentSearchSpace));
                    if (equals == Vector128<T>.Zero)
                    {
                        currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector128<T>.Count);
                        continue;
                    }

                    return ComputeFirstIndex(ref searchSpace, ref currentSearchSpace, equals);
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                // If any elements remain, process the first vector in the search space.
                if (length % Vector128<T>.Count != 0)
                {
                    equals = Vector128.Equals(values, Vector128.LoadUnsafe(ref oneVectorAwayFromEnd));
                    if (equals != Vector128<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref oneVectorAwayFromEnd, equals);
                    }
                }
            }

            return -1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long LastIndexOfValueType<T>(ref T searchSpace, T value, long length) where T : struct, INumber<T>
        {
            return NonPackedLastIndexOfValueType(ref searchSpace, value, length);
        }

        internal static long NonPackedLastIndexOfValueType<T>(ref T searchSpace, T value, long length) where T : struct, INumber<T>
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            if (!Vector128.IsHardwareAccelerated || length < Vector128<T>.Count)
            {
                nuint offset = (nuint)length - 1;

                while (length >= 8)
                {
                    length -= 8;

                    if (Unsafe.Add(ref searchSpace, offset) == value) return (long)offset;
                    if (Unsafe.Add(ref searchSpace, offset - 1) == value) return (long)(offset - 1);
                    if (Unsafe.Add(ref searchSpace, offset - 2) == value) return (long)(offset - 2);
                    if (Unsafe.Add(ref searchSpace, offset - 3) == value) return (long)(offset - 3);
                    if (Unsafe.Add(ref searchSpace, offset - 4) == value) return (long)(offset - 4);
                    if (Unsafe.Add(ref searchSpace, offset - 5) == value) return (long)(offset - 5);
                    if (Unsafe.Add(ref searchSpace, offset - 6) == value) return (long)(offset - 6);
                    if (Unsafe.Add(ref searchSpace, offset - 7) == value) return (long)(offset - 7);

                    offset -= 8;
                }

                if (length >= 4)
                {
                    length -= 4;

                    if (Unsafe.Add(ref searchSpace, offset) == value) return (long)offset;
                    if (Unsafe.Add(ref searchSpace, offset - 1) == value) return (long)(offset - 1);
                    if (Unsafe.Add(ref searchSpace, offset - 2) == value) return (long)(offset - 2);
                    if (Unsafe.Add(ref searchSpace, offset - 3) == value) return (long)(offset - 3);

                    offset -= 4;
                }

                while (length > 0)
                {
                    length -= 1;

                    if (Unsafe.Add(ref searchSpace, offset) == value)
                    {
                        return (long)offset;
                    }

                    offset -= 1;
                }

                return -1;
            }
            else if (Vector512.IsHardwareAccelerated && length >= Vector512<T>.Count)
            {
                Vector512<T> current, values = Vector512.Create(value);
                long offset = length - Vector512<T>.Count;

                // Loop until either we've finished all elements or there's less than a vector's-worth remaining.
                do
                {
                    current = Vector512.LoadUnsafe(ref searchSpace, (nuint)offset);
                    var equals = Vector512.Equals(values, current);

                    if (equals != Vector512<T>.Zero)
                    {
                        return ComputeLastIndex(offset, equals);
                    }

                    offset -= Vector512<T>.Count;
                }
                while (offset >= 0);

                // If any elements remain, process the first vector in the search space.
                if (length % Vector512<T>.Count != 0)
                {
                    current = Vector512.LoadUnsafe(ref searchSpace);
                    var equals = Vector512.Equals(values, current);

                    if (equals != Vector512<T>.Zero)
                    {
                        return ComputeLastIndex(0, equals);
                    }
                }
            }
            else if (Vector256.IsHardwareAccelerated && length >= Vector256<T>.Count)
            {
                Vector256<T> equals, values = Vector256.Create(value);
                long offset = length - Vector256<T>.Count;

                // Loop until either we've finished all elements or there's less than a vector's-worth remaining.
                do
                {
                    equals = Vector256.Equals(values, Vector256.LoadUnsafe(ref searchSpace, (nuint)offset));
                    if (equals == Vector256<T>.Zero)
                    {
                        offset -= Vector256<T>.Count;
                        continue;
                    }

                    return ComputeLastIndex(offset, equals);
                }
                while (offset >= 0);

                // If any elements remain, process the first vector in the search space.
                if (length % Vector256<T>.Count != 0)
                {
                    equals = Vector256.Equals(values, Vector256.LoadUnsafe(ref searchSpace));
                    if (equals != Vector256<T>.Zero)
                    {
                        return ComputeLastIndex(0, equals);
                    }
                }
            }
            else
            {
                Vector128<T> equals, values = Vector128.Create(value);
                long offset = length - Vector128<T>.Count;

                // Loop until either we've finished all elements or there's less than a vector's-worth remaining.
                do
                {
                    equals = Vector128.Equals(values, Vector128.LoadUnsafe(ref searchSpace, (nuint)offset));
                    if (equals == Vector128<T>.Zero)
                    {
                        offset -= Vector128<T>.Count;
                        continue;
                    }

                    return ComputeLastIndex(offset, equals);
                }
                while (offset >= 0);

                // If any elements remain, process the first vector in the search space.
                if (length % Vector128<T>.Count != 0)
                {
                    equals = Vector128.Equals(values, Vector128.LoadUnsafe(ref searchSpace));
                    if (equals != Vector128<T>.Zero)
                    {
                        return ComputeLastIndex(0, equals);
                    }
                }
            }

            return -1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long IndexOfAnyValueType<T>(ref T searchSpace, T value0, T value1, long length) where T : struct, INumber<T>
        {
            return NonPackedIndexOfAnyValueType(ref searchSpace, value0, value1, length);
        }

        internal static long NonPackedIndexOfAnyValueType<T>(ref T searchSpace, T value0, T value1, long length) where T : struct, INumber<T>
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            if (!Vector128.IsHardwareAccelerated || length < Vector128<T>.Count)
            {
                nuint offset = 0;
                T lookUp;

                while (length >= 4)
                {
                    length -= 4;

                    ref T current = ref Unsafe.Add(ref searchSpace, offset);
                    lookUp = current;
                    if (lookUp == value0 || lookUp == value1) return (long)offset;

                    lookUp = Unsafe.Add(ref current, 1);
                    if (lookUp == value0 || lookUp == value1) return (long)(offset + 1);

                    lookUp = Unsafe.Add(ref current, 2);
                    if (lookUp == value0 || lookUp == value1) return (long)(offset + 2);

                    lookUp = Unsafe.Add(ref current, 3);
                    if (lookUp == value0 || lookUp == value1) return (long)(offset + 3);

                    offset += 4;
                }

                while (length > 0)
                {
                    length -= 1;

                    lookUp = Unsafe.Add(ref searchSpace, offset);
                    if (lookUp == value0 || lookUp == value1) return (long)offset;

                    offset += 1;
                }

                return -1;
            }
            else if (Vector512.IsHardwareAccelerated && length >= Vector512<T>.Count)
            {
                Vector512<T> current, values0 = Vector512.Create(value0), values1 = Vector512.Create(value1);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector512<T>.Count));

                do
                {
                    current = Vector512.LoadUnsafe(ref currentSearchSpace);
                    var equals = Vector512.Equals(current, values0) | Vector512.Equals(current, values1);

                    if (equals != Vector512<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref currentSearchSpace, equals);
                    }

                    currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector512<T>.Count);
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                if (length % Vector512<T>.Count != 0)
                {
                    current = Vector512.LoadUnsafe(ref oneVectorAwayFromEnd);
                    var equals = Vector512.Equals(current, values0) | Vector512.Equals(current, values1);

                    if (equals != Vector512<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref oneVectorAwayFromEnd, equals);
                    }
                }
            }
            else if (Vector256.IsHardwareAccelerated && length >= Vector256<T>.Count)
            {
                Vector256<T> current, equals, values0 = Vector256.Create(value0), values1 = Vector256.Create(value1);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector256<T>.Count));

                do
                {
                    current = Vector256.LoadUnsafe(ref currentSearchSpace);
                    equals = Vector256.Equals(current, values0) | Vector256.Equals(current, values1);

                    if (equals == Vector256<T>.Zero)
                    {
                        currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector256<T>.Count);
                        continue;
                    }

                    return ComputeFirstIndex(ref searchSpace, ref currentSearchSpace, equals);
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                if (length % Vector256<T>.Count != 0)
                {
                    current = Vector256.LoadUnsafe(ref oneVectorAwayFromEnd);
                    equals = Vector256.Equals(current, values0) | Vector256.Equals(current, values1);

                    if (equals != Vector256<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref oneVectorAwayFromEnd, equals);
                    }
                }
            }
            else
            {
                Vector128<T> current, equals, values0 = Vector128.Create(value0), values1 = Vector128.Create(value1);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector128<T>.Count));

                do
                {
                    current = Vector128.LoadUnsafe(ref currentSearchSpace);
                    equals = Vector128.Equals(current, values0) | Vector128.Equals(current, values1);

                    if (equals == Vector128<T>.Zero)
                    {
                        currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector128<T>.Count);
                        continue;
                    }

                    return ComputeFirstIndex(ref searchSpace, ref currentSearchSpace, equals);
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                if (length % Vector128<T>.Count != 0)
                {
                    current = Vector128.LoadUnsafe(ref oneVectorAwayFromEnd);
                    equals = Vector128.Equals(current, values0) | Vector128.Equals(current, values1);

                    if (equals != Vector128<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref oneVectorAwayFromEnd, equals);
                    }
                }
            }

            return -1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long IndexOfAnyValueType<T>(ref T searchSpace, T value0, T value1, T value2, long length) where T : struct, INumber<T>
        {
            return NonPackedIndexOfAnyValueType(ref searchSpace, value0, value1, value2, length);
        }

        internal static long NonPackedIndexOfAnyValueType<T>(ref T searchSpace, T value0, T value1, T value2, long length) where T : struct, INumber<T>
        {
            Debug.Assert(length >= 0, "Expected non-negative length");

            if (!Vector128.IsHardwareAccelerated || length < Vector128<T>.Count)
            {
                nuint offset = 0;
                T lookUp;

                while (length >= 4)
                {
                    length -= 4;

                    ref T current = ref Unsafe.Add(ref searchSpace, offset);
                    lookUp = current;
                    if (lookUp == value0 || lookUp == value1 || lookUp == value2) return (long)offset;

                    lookUp = Unsafe.Add(ref current, 1);
                    if (lookUp == value0 || lookUp == value1 || lookUp == value2) return (long)(offset + 1);

                    lookUp = Unsafe.Add(ref current, 2);
                    if (lookUp == value0 || lookUp == value1 || lookUp == value2) return (long)(offset + 2);

                    lookUp = Unsafe.Add(ref current, 3);
                    if (lookUp == value0 || lookUp == value1 || lookUp == value2) return (long)(offset + 3);

                    offset += 4;
                }

                while (length > 0)
                {
                    length -= 1;

                    lookUp = Unsafe.Add(ref searchSpace, offset);
                    if (lookUp == value0 || lookUp == value1 || lookUp == value2) return (long)offset;

                    offset += 1;
                }

                return -1;
            }
            else if (Vector512.IsHardwareAccelerated && length >= Vector512<T>.Count)
            {
                Vector512<T> current, values0 = Vector512.Create(value0), values1 = Vector512.Create(value1), values2 = Vector512.Create(value2);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector512<T>.Count));

                do
                {
                    current = Vector512.LoadUnsafe(ref currentSearchSpace);
                    var equals = Vector512.Equals(current, values0) | Vector512.Equals(current, values1) | Vector512.Equals(current, values2);

                    if (equals != Vector512<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref currentSearchSpace, equals);
                    }

                    currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector512<T>.Count);
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                if (length % Vector512<T>.Count != 0)
                {
                    current = Vector512.LoadUnsafe(ref oneVectorAwayFromEnd);
                    var equals = Vector512.Equals(current, values0) | Vector512.Equals(current, values1) | Vector512.Equals(current, values2);

                    if (equals != Vector512<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref oneVectorAwayFromEnd, equals);
                    }
                }
            }
            else if (Vector256.IsHardwareAccelerated && length >= Vector256<T>.Count)
            {
                Vector256<T> current, equals, values0 = Vector256.Create(value0), values1 = Vector256.Create(value1), values2 = Vector256.Create(value2);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector256<T>.Count));

                do
                {
                    current = Vector256.LoadUnsafe(ref currentSearchSpace);
                    equals = Vector256.Equals(current, values0) | Vector256.Equals(current, values1) | Vector256.Equals(current, values2);

                    if (equals == Vector256<T>.Zero)
                    {
                        currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector256<T>.Count);
                        continue;
                    }

                    return ComputeFirstIndex(ref searchSpace, ref currentSearchSpace, equals);
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                if (length % Vector256<T>.Count != 0)
                {
                    current = Vector256.LoadUnsafe(ref oneVectorAwayFromEnd);
                    equals = Vector256.Equals(current, values0) | Vector256.Equals(current, values1) | Vector256.Equals(current, values2);

                    if (equals != Vector256<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref oneVectorAwayFromEnd, equals);
                    }
                }
            }
            else
            {
                Vector128<T> current, equals, values0 = Vector128.Create(value0), values1 = Vector128.Create(value1), values2 = Vector128.Create(value2);
                ref T currentSearchSpace = ref searchSpace;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref searchSpace, (nint)(length - Vector128<T>.Count));

                do
                {
                    current = Vector128.LoadUnsafe(ref currentSearchSpace);
                    equals = Vector128.Equals(current, values0) | Vector128.Equals(current, values1) | Vector128.Equals(current, values2);

                    if (equals == Vector128<T>.Zero)
                    {
                        currentSearchSpace = ref Unsafe.Add(ref currentSearchSpace, Vector128<T>.Count);
                        continue;
                    }

                    return ComputeFirstIndex(ref searchSpace, ref currentSearchSpace, equals);
                }
                while (IsAddressLessThanOrEqualTo(ref currentSearchSpace, ref oneVectorAwayFromEnd));

                if (length % Vector128<T>.Count != 0)
                {
                    current = Vector128.LoadUnsafe(ref oneVectorAwayFromEnd);
                    equals = Vector128.Equals(current, values0) | Vector128.Equals(current, values1) | Vector128.Equals(current, values2);

                    if (equals != Vector128<T>.Zero)
                    {
                        return ComputeFirstIndex(ref searchSpace, ref oneVectorAwayFromEnd, equals);
                    }
                }
            }

            return -1;
        }

        /// <summary>
        /// SIMD-accelerated sequence equality check for value types.
        /// </summary>
        internal static unsafe bool SequenceEqualValueType<T>(ref T first, ref T second, long length) where T : struct, INumber<T>
        {
            Debug.Assert(length >= 0);

            if (Unsafe.AreSame(ref first, ref second))
            {
                return true;
            }

            if (!Vector128.IsHardwareAccelerated || length < Vector128<T>.Count)
            {
                nuint offset = 0;
                nuint lengthToCheck = (nuint)length;

                while (lengthToCheck >= 8)
                {
                    lengthToCheck -= 8;

                    if (Unsafe.Add(ref first, offset) != Unsafe.Add(ref second, offset)
                     || Unsafe.Add(ref first, offset + 1) != Unsafe.Add(ref second, offset + 1)
                     || Unsafe.Add(ref first, offset + 2) != Unsafe.Add(ref second, offset + 2)
                     || Unsafe.Add(ref first, offset + 3) != Unsafe.Add(ref second, offset + 3)
                     || Unsafe.Add(ref first, offset + 4) != Unsafe.Add(ref second, offset + 4)
                     || Unsafe.Add(ref first, offset + 5) != Unsafe.Add(ref second, offset + 5)
                     || Unsafe.Add(ref first, offset + 6) != Unsafe.Add(ref second, offset + 6)
                     || Unsafe.Add(ref first, offset + 7) != Unsafe.Add(ref second, offset + 7))
                    {
                        return false;
                    }

                    offset += 8;
                }

                if (lengthToCheck >= 4)
                {
                    lengthToCheck -= 4;

                    if (Unsafe.Add(ref first, offset) != Unsafe.Add(ref second, offset)
                     || Unsafe.Add(ref first, offset + 1) != Unsafe.Add(ref second, offset + 1)
                     || Unsafe.Add(ref first, offset + 2) != Unsafe.Add(ref second, offset + 2)
                     || Unsafe.Add(ref first, offset + 3) != Unsafe.Add(ref second, offset + 3))
                    {
                        return false;
                    }

                    offset += 4;
                }

                while (lengthToCheck > 0)
                {
                    lengthToCheck -= 1;

                    if (Unsafe.Add(ref first, offset) != Unsafe.Add(ref second, offset))
                    {
                        return false;
                    }

                    offset += 1;
                }

                return true;
            }
            else if (Vector512.IsHardwareAccelerated && length >= Vector512<T>.Count)
            {
                ref T currentFirst = ref first;
                ref T currentSecond = ref second;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref first, (nint)(length - Vector512<T>.Count));

                do
                {
                    if (Vector512.LoadUnsafe(ref currentFirst) != Vector512.LoadUnsafe(ref currentSecond))
                    {
                        return false;
                    }

                    currentFirst = ref Unsafe.Add(ref currentFirst, Vector512<T>.Count);
                    currentSecond = ref Unsafe.Add(ref currentSecond, Vector512<T>.Count);
                }
                while (IsAddressLessThanOrEqualTo(ref currentFirst, ref oneVectorAwayFromEnd));

                if (length % Vector512<T>.Count != 0)
                {
                    ref T lastVectorFirst = ref Unsafe.Add(ref first, (nint)(length - Vector512<T>.Count));
                    ref T lastVectorSecond = ref Unsafe.Add(ref second, (nint)(length - Vector512<T>.Count));
                    if (Vector512.LoadUnsafe(ref lastVectorFirst) != Vector512.LoadUnsafe(ref lastVectorSecond))
                    {
                        return false;
                    }
                }

                return true;
            }
            else if (Vector256.IsHardwareAccelerated && length >= Vector256<T>.Count)
            {
                ref T currentFirst = ref first;
                ref T currentSecond = ref second;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref first, (nint)(length - Vector256<T>.Count));

                do
                {
                    if (Vector256.LoadUnsafe(ref currentFirst) != Vector256.LoadUnsafe(ref currentSecond))
                    {
                        return false;
                    }

                    currentFirst = ref Unsafe.Add(ref currentFirst, Vector256<T>.Count);
                    currentSecond = ref Unsafe.Add(ref currentSecond, Vector256<T>.Count);
                }
                while (IsAddressLessThanOrEqualTo(ref currentFirst, ref oneVectorAwayFromEnd));

                if (length % Vector256<T>.Count != 0)
                {
                    ref T lastVectorFirst = ref Unsafe.Add(ref first, (nint)(length - Vector256<T>.Count));
                    ref T lastVectorSecond = ref Unsafe.Add(ref second, (nint)(length - Vector256<T>.Count));
                    if (Vector256.LoadUnsafe(ref lastVectorFirst) != Vector256.LoadUnsafe(ref lastVectorSecond))
                    {
                        return false;
                    }
                }

                return true;
            }
            else
            {
                ref T currentFirst = ref first;
                ref T currentSecond = ref second;
                ref T oneVectorAwayFromEnd = ref Unsafe.Add(ref first, (nint)(length - Vector128<T>.Count));

                do
                {
                    if (Vector128.LoadUnsafe(ref currentFirst) != Vector128.LoadUnsafe(ref currentSecond))
                    {
                        return false;
                    }

                    currentFirst = ref Unsafe.Add(ref currentFirst, Vector128<T>.Count);
                    currentSecond = ref Unsafe.Add(ref currentSecond, Vector128<T>.Count);
                }
                while (IsAddressLessThanOrEqualTo(ref currentFirst, ref oneVectorAwayFromEnd));

                if (length % Vector128<T>.Count != 0)
                {
                    ref T lastVectorFirst = ref Unsafe.Add(ref first, (nint)(length - Vector128<T>.Count));
                    ref T lastVectorSecond = ref Unsafe.Add(ref second, (nint)(length - Vector128<T>.Count));
                    if (Vector128.LoadUnsafe(ref lastVectorFirst) != Vector128.LoadUnsafe(ref lastVectorSecond))
                    {
                        return false;
                    }
                }

                return true;
            }
        }

        // ==================================================================================
        // Helper methods for computing indices from SIMD match results
        // ==================================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ComputeFirstIndex<T>(ref T searchSpace, ref T current, Vector128<T> equals) where T : struct
        {
            uint notEqualsElements = IsAllNegativeOnes(equals.AsByte());
            int index = BitOperations.TrailingZeroCount(notEqualsElements) / sizeof(T);
            return index + (long)((nuint)Unsafe.ByteOffset(ref searchSpace, ref current) / (nuint)sizeof(T));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ComputeFirstIndex<T>(ref T searchSpace, ref T current, Vector256<T> equals) where T : struct
        {
            uint notEqualsElements = IsAllNegativeOnes(equals.AsByte());
            int index = BitOperations.TrailingZeroCount(notEqualsElements) / sizeof(T);
            return index + (long)((nuint)Unsafe.ByteOffset(ref searchSpace, ref current) / (nuint)sizeof(T));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ComputeFirstIndex<T>(ref T searchSpace, ref T current, Vector512<T> equals) where T : struct
        {
            ulong notEqualsElements = IsAllNegativeOnes(equals.AsByte());
            int index = BitOperations.TrailingZeroCount(notEqualsElements) / sizeof(T);
            return index + (long)((nuint)Unsafe.ByteOffset(ref searchSpace, ref current) / (nuint)sizeof(T));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ComputeLastIndex<T>(long offset, Vector128<T> equals) where T : struct
        {
            uint notEqualsElements = IsAllNegativeOnes(equals.AsByte());
            int index = 31 - BitOperations.LeadingZeroCount(notEqualsElements); // 31 = 32 - 1 for 0-indexed
            return offset + index / sizeof(T);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ComputeLastIndex<T>(long offset, Vector256<T> equals) where T : struct
        {
            uint notEqualsElements = IsAllNegativeOnes(equals.AsByte());
            int index = 31 - BitOperations.LeadingZeroCount(notEqualsElements);
            return offset + index / sizeof(T);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ComputeLastIndex<T>(long offset, Vector512<T> equals) where T : struct
        {
            ulong notEqualsElements = IsAllNegativeOnes(equals.AsByte());
            int index = 63 - BitOperations.LeadingZeroCount(notEqualsElements);
            return offset + index / sizeof(T);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint IsAllNegativeOnes(Vector128<byte> vector)
        {
            if (Sse2.IsSupported)
            {
                return (uint)Sse2.MoveMask(vector);
            }
            else
            {
                // Fallback for non-SSE2 platforms
                uint mask = 0;
                for (int i = 0; i < Vector128<byte>.Count; i++)
                {
                    if ((vector.GetElement(i) & 0x80) != 0)
                        mask |= 1u << i;
                }
                return mask;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint IsAllNegativeOnes(Vector256<byte> vector)
        {
            if (Avx2.IsSupported)
            {
                return (uint)Avx2.MoveMask(vector);
            }
            else
            {
                // Fallback
                uint mask = 0;
                for (int i = 0; i < Vector256<byte>.Count; i++)
                {
                    if ((vector.GetElement(i) & 0x80) != 0)
                        mask |= 1u << i;
                }
                return mask;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong IsAllNegativeOnes(Vector512<byte> vector)
        {
#if NET9_0_OR_GREATER
            if (Avx512BW.IsSupported)
            {
                return (ulong)Avx512BW.MoveMask(vector);
            }
#endif
            // Fallback for .NET 8 or when AVX-512 is not supported
            ulong mask = 0;
            for (int i = 0; i < Vector512<byte>.Count; i++)
            {
                if ((vector.GetElement(i) & 0x80) != 0)
                    mask |= 1ul << i;
            }
            return mask;
        }
    }
}
