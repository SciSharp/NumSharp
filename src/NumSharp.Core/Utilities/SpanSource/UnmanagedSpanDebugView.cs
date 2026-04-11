using System;
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;

namespace NumSharp.Utilities
{
    internal sealed class UnmanagedSpanDebugView<T> where T : unmanaged
    {
        private readonly T[] _array;

        public UnmanagedSpanDebugView(UnmanagedSpan<T> span)
        {
            _array = span.ToArray();
        }

        public UnmanagedSpanDebugView(ReadOnlyUnmanagedSpan<T> span)
        {
            _array = span.ToArray();
        }

        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public T[] Items => _array;
    }
}
