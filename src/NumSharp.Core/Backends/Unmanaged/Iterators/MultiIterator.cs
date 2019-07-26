using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends.Unmanaged {
    public class MultiIterator<T> where T : unmanaged
    {
        public readonly NDIterator<T> Lhs;
        public readonly NDIterator<T> Rhs;
        public readonly int size;

        private readonly Func<T> Lhs_MoveNext;
        private readonly Func<T> Rhs_MoveNext;
        private readonly Func<bool> Lhs_HasNext;
        private readonly Func<bool> Rhs_HasNext;
        private readonly Action Lhs_Reset;
        private readonly Action Rhs_Reset;

        public MultiIterator(NDIterator<T> lhs, NDIterator<T> rhs)
        {
            size = Math.Max(lhs.Shape.size, rhs.Shape.size);
            Lhs = lhs;
            Rhs = rhs;
            Lhs_MoveNext = Lhs.MoveNext;
            Rhs_MoveNext = Rhs.MoveNext;

            Lhs_HasNext = Lhs.HasNext;
            Rhs_HasNext = Rhs.HasNext;

            Lhs_Reset = Lhs.Reset;
            Rhs_Reset = Rhs.Reset;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public (T, T) MoveNext()
        {
            return (Lhs_MoveNext(), Rhs_MoveNext());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool HasNext()
        {
            return Lhs_HasNext() && Rhs_HasNext();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool AnyHasNext()
        {
            return Lhs_HasNext() || Rhs_HasNext();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Reset()
        {
            Lhs_Reset();
            Rhs_Reset();
        }
    }
}
