using System;
using System.Runtime.CompilerServices;

// =============================================================================
// NDExpr.Structure.cs — structural identity of an expression tree (the global program cache's key)
// =============================================================================
//
// np.evaluate's compiled program (NDExpr.Program.cs) depends on the tree's STRUCTURE — node kinds,
// ops, literal values, which array leaf is which operand — and on the operand dtype signature,
// never on which NDArray instances sit at the leaves. The per-root cache serves a tree the caller
// hoisted, but the natural spelling `np.evaluate((NDExpr)a * b + c)` inside a loop builds a NEW root
// every call and could only pay the full bind + typing + string-key path again (~0.5 µs and ~2.5 KB
// over the prebuilt path at n = 8 — the whole reason the 1K fused rows lost to the unfused chain).
//
// This file gives every node two allocation-free walks:
//   • HashStructure — folds the node's identity into a 64-bit hash while collecting the distinct
//     array leaves in first-visit order (the SAME order BindArrays assigns InputNode indices, so an
//     ArrayNode hashes as the input index it will bind to);
//   • StructureEquals — verifies a hash hit against the cached program's BOUND tree node by node.
//     A hash match alone must never select a program: a false match would run the wrong kernel
//     (different op, different literal, different operand dedup) and return silently wrong data.
//
// The walks mirror BindArrays' child order exactly (left → right, cond → a → b, args in order);
// changing one without the other desynchronizes the operand indices and the cache would pair a
// program with the wrong operand list — which is why both live here, side by side.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// A 64-bit order-sensitive structural hash accumulator (a one-word struct carried through the
    /// tree walk by <c>ref</c>, so hashing allocates nothing; a plain struct rather than a ref struct
    /// so a stack-allocated dtype span can be folded in without tripping the ref-safety rules).
    /// Two trees that differ ONLY in child order hash differently (<c>a*b+c</c> vs <c>c+a*b</c> are
    /// different programs — their kernels read the operands in a different order), which a
    /// commutative fold would conflate. The hash is the cache's LOOKUP key only;
    /// <see cref="NDExpr.StructureEquals"/> decides identity.
    /// </summary>
    internal struct NDExprStructureHasher
    {
        private ulong _h;

        /// <summary>Start a hash; the seed keeps an empty structure from hashing to zero.</summary>
        public static NDExprStructureHasher Create() => new() { _h = 0x9E3779B97F4A7C15UL };

        /// <summary>Fold one 64-bit word in (a multiply-xorshift mix — cheap, order-sensitive, avalanching).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(ulong v)
        {
            ulong h = _h ^ v;
            h *= 0xFF51AFD7ED558CCDUL;
            h ^= h >> 33;
            h *= 0xC4CEB9FE1A85EC53UL;
            h ^= h >> 29;
            _h = h;
        }

        /// <summary>Fold a node tag and a small integer field in one step (tag in the high byte, so a tag/field pair never aliases a plain value).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(NDExprNodeTag tag, long field) => Add(((ulong)tag << 56) ^ (ulong)field);

        /// <summary>The accumulated hash.</summary>
        public ulong Value => _h;
    }

    /// <summary>
    /// The node kinds the structural hash distinguishes. One byte per node class; a change of kind at
    /// any position changes the hash even when every other field agrees.
    /// </summary>
    internal enum NDExprNodeTag : byte
    {
        Input = 1,
        Const = 2,
        Binary = 3,
        Unary = 4,
        Comparison = 5,
        MinMax = 6,
        Where = 7,
        Call = 8,
        Reduce = 9,
    }

    /// <summary>
    /// The distinct array leaves of a tree in first-visit order — the operand list a bound program
    /// evaluates against — collected during the structural walk with no per-call allocation: one
    /// instance is kept per thread and reset after every use, so it never retains a caller's array
    /// past the call that collected it. Not safe to share across threads; not re-entrant within a
    /// walk (the walk invokes no user code, so it cannot be).
    /// </summary>
    [NDBorrowed] // transient references to the caller's arrays; cleared by Reset before the call returns
    internal sealed class NDExprOperandScratch
    {
        private NDArray[] _items = new NDArray[8];
        private int _count;

        /// <summary>Number of distinct arrays collected so far.</summary>
        public int Count => _count;

        /// <summary>The i-th distinct array (bound operand <c>Input(i)</c>).</summary>
        public NDArray this[int i] => _items[i];

        /// <summary>
        /// The operand index of <paramref name="array"/>, adding it when unseen — BindArrays'
        /// <c>NDExprBindContext.IndexOf</c> rule (dedup by reference, first visit wins), so the index
        /// an <see cref="ArrayNode"/> hashes as is the <see cref="InputNode"/> index it binds to.
        /// </summary>
        /// <param name="array">The array leaf being visited.</param>
        /// <returns>Its 0-based operand index.</returns>
        public int IndexOrAdd(NDArray array)
        {
            int n = _count;
            var items = _items;
            for (int i = 0; i < n; i++)
                if (ReferenceEquals(items[i], array))
                    return i;

            if (n == items.Length)
                Array.Resize(ref _items, n * 2);
            _items[n] = array;
            _count = n + 1;
            return n;
        }

        /// <summary>
        /// The operand index of an already-collected array, or -1 — the verification walk only looks
        /// up (the collection walk ran first over the same tree, so every leaf is present).
        /// </summary>
        /// <param name="array">The array leaf being compared.</param>
        /// <returns>Its operand index, or -1 when the array was not collected.</returns>
        public int IndexOf(NDArray array)
        {
            var items = _items;
            for (int i = 0, n = _count; i < n; i++)
                if (ReferenceEquals(items[i], array))
                    return i;
            return -1;
        }

        /// <summary>Does the collected operand list carry exactly <paramref name="types"/> as its dtype signature?</summary>
        /// <param name="types">The dtype signature a cached program was compiled for.</param>
        /// <returns>True when the counts and every typecode agree.</returns>
        public bool MatchesTypes(NPTypeCode[] types)
        {
            if (types.Length != _count)
                return false;
            var items = _items;
            for (int i = 0; i < types.Length; i++)
                if (items[i].typecode != types[i])
                    return false;
            return true;
        }

        /// <summary>The collected operands as a fresh exact-size array (the list handed to the engine).</summary>
        /// <returns>A new array; the scratch keeps no reference to it.</returns>
        public NDArray[] ToArray()
        {
            var result = new NDArray[_count];
            Array.Copy(_items, result, _count);
            return result;
        }

        /// <summary>The dtype signature of the collected operands (a fresh array — the miss path only).</summary>
        /// <returns>One <see cref="NPTypeCode"/> per operand, in operand order.</returns>
        public NPTypeCode[] ToTypes()
        {
            var types = new NPTypeCode[_count];
            for (int i = 0; i < types.Length; i++)
                types[i] = _items[i].typecode;
            return types;
        }

        /// <summary>Drop every reference (the caller's arrays must not outlive the call through this scratch).</summary>
        public void Reset()
        {
            Array.Clear(_items, 0, _count);
            _count = 0;
        }
    }

    public abstract partial class NDExpr
    {
        /// <summary>
        /// Fold this node and its subtree into <paramref name="h"/> while collecting the distinct array
        /// leaves into <paramref name="operands"/> in the order <see cref="BindArrays"/> would bind
        /// them. Child order MUST match <see cref="BindArrays"/> (see the file header).
        /// </summary>
        /// <param name="h">The running structural hash.</param>
        /// <param name="operands">The operand collector; an <see cref="ArrayNode"/> registers its array here.</param>
        internal abstract void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands);

        /// <summary>
        /// Is this (possibly unbound) tree structurally identical to <paramref name="bound"/>, a tree whose
        /// array leaves are already <see cref="InputNode"/>s? Same node kinds, ops, literal values (by
        /// bits), reduction settings, call targets, and the same operand for every array leaf — the
        /// condition under which a compiled program of <paramref name="bound"/> is correct for this tree.
        /// </summary>
        /// <param name="bound">The cached program's bound tree.</param>
        /// <param name="operands">The operand list collected by <see cref="HashStructure"/> over THIS tree.</param>
        /// <returns>True when a program compiled for <paramref name="bound"/> computes exactly this tree.</returns>
        internal abstract bool StructureEquals(NDExpr bound, NDExprOperandScratch operands);
    }

    public sealed partial class InputNode
    {
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
            => h.Add(NDExprNodeTag.Input, _index);

        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
            => bound is InputNode o && o._index == _index;
    }

    public sealed partial class ArrayNode
    {
        // An array leaf IS the input its binding would produce: hash the dedup index, not the array.
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
            => h.Add(NDExprNodeTag.Input, operands.IndexOrAdd(_array));

        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
            => bound is InputNode o && operands.IndexOf(_array) == o.Index;
    }

    public sealed partial class ConstNode
    {
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
        {
            h.Add(NDExprNodeTag.Const, (long)_kind);
            // The value by BITS: a literal is baked into the kernel's IL, so 2 and 2.0 (different kinds),
            // 0.0 and -0.0, or two NaN payloads are different programs. Bit identity is stricter than
            // the kernel key's "R" formatting (which prints every NaN alike) — a stricter key can only
            // split entries, never pair a tree with a foreign kernel.
            switch (_kind)
            {
                case NDExprLiteralKind.UInt64: h.Add(_u); break;
                case NDExprLiteralKind.Float:
                case NDExprLiteralKind.Half: h.Add((ulong)BitConverter.DoubleToInt64Bits(_f)); break;
                case NDExprLiteralKind.Complex:
                    h.Add((ulong)BitConverter.DoubleToInt64Bits(_c.Real));
                    h.Add((ulong)BitConverter.DoubleToInt64Bits(_c.Imaginary));
                    break;
                case NDExprLiteralKind.Decimal:
                {
                    Span<int> bits = stackalloc int[4];
                    decimal.GetBits(_m, bits);
                    h.Add(((ulong)(uint)bits[0] << 32) | (uint)bits[1]);
                    h.Add(((ulong)(uint)bits[2] << 32) | (uint)bits[3]);
                    break;
                }
                default: h.Add((ulong)_i); break;              // Bool / Int / Char
            }
        }

        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
        {
            if (bound is not ConstNode o || o._kind != _kind)
                return false;
            return _kind switch
            {
                NDExprLiteralKind.UInt64 => o._u == _u,
                NDExprLiteralKind.Float or NDExprLiteralKind.Half
                    => BitConverter.DoubleToInt64Bits(o._f) == BitConverter.DoubleToInt64Bits(_f),
                NDExprLiteralKind.Complex
                    => BitConverter.DoubleToInt64Bits(o._c.Real) == BitConverter.DoubleToInt64Bits(_c.Real)
                       && BitConverter.DoubleToInt64Bits(o._c.Imaginary) == BitConverter.DoubleToInt64Bits(_c.Imaginary),
                // decimal.Equals is VALUE equality (1.0m == 1.00m) but the IL bakes the scale bits, so two
                // spellings of one value emit different constants — compare the raw bits.
                NDExprLiteralKind.Decimal => DecimalBitsEqual(o._m, _m),
                _ => o._i == _i,
            };
        }

        private static bool DecimalBitsEqual(decimal x, decimal y)
        {
            Span<int> bx = stackalloc int[4];
            Span<int> by = stackalloc int[4];
            decimal.GetBits(x, bx);
            decimal.GetBits(y, by);
            return bx.SequenceEqual(by);
        }
    }

    public sealed partial class BinaryNode
    {
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
        {
            h.Add(NDExprNodeTag.Binary, (long)_op);
            _left.HashStructure(ref h, operands);
            _right.HashStructure(ref h, operands);
        }

        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
            => bound is BinaryNode o && o._op == _op
               && _left.StructureEquals(o._left, operands)
               && _right.StructureEquals(o._right, operands);
    }

    public sealed partial class UnaryNode
    {
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
        {
            h.Add(NDExprNodeTag.Unary, (long)_op);
            _child.HashStructure(ref h, operands);
        }

        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
            => bound is UnaryNode o && o._op == _op && _child.StructureEquals(o._child, operands);
    }

    public sealed partial class ComparisonNode
    {
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
        {
            h.Add(NDExprNodeTag.Comparison, (long)_op);
            _left.HashStructure(ref h, operands);
            _right.HashStructure(ref h, operands);
        }

        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
            => bound is ComparisonNode o && o._op == _op
               && _left.StructureEquals(o._left, operands)
               && _right.StructureEquals(o._right, operands);
    }

    public sealed partial class MinMaxNode
    {
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
        {
            h.Add(NDExprNodeTag.MinMax, _isMin ? 1 : 0);
            _left.HashStructure(ref h, operands);
            _right.HashStructure(ref h, operands);
        }

        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
            => bound is MinMaxNode o && o._isMin == _isMin
               && _left.StructureEquals(o._left, operands)
               && _right.StructureEquals(o._right, operands);
    }

    public sealed partial class WhereNode
    {
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
        {
            h.Add(NDExprNodeTag.Where, 0);
            _cond.HashStructure(ref h, operands);
            _a.HashStructure(ref h, operands);
            _b.HashStructure(ref h, operands);
        }

        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
            => bound is WhereNode o
               && _cond.StructureEquals(o._cond, operands)
               && _a.StructureEquals(o._a, operands)
               && _b.StructureEquals(o._b, operands);
    }

    public sealed partial class CallNode
    {
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
        {
            // The emitted IL is decided by (kind, method identity, slot): a direct `call` to a static
            // method, or a slot lookup of a captured delegate / bound target followed by callvirt. Two
            // closures over the same lambda body share the method but not the slot, so the slot is part
            // of the identity (the kernel key carries it too — see AppendSignature).
            h.Add(NDExprNodeTag.Call, (long)_kind);
            h.Add((ulong)(uint)_signatureId.GetHashCode());
            h.Add((ulong)_slotId);
            h.Add((ulong)_args.Length);
            for (int i = 0; i < _args.Length; i++)
                _args[i].HashStructure(ref h, operands);
        }

        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
        {
            if (bound is not CallNode o || o._kind != _kind || o._slotId != _slotId
                || o._args.Length != _args.Length
                || !ReferenceEquals(o._delegateType, _delegateType)
                || !string.Equals(o._signatureId, _signatureId, StringComparison.Ordinal))
                return false;
            for (int i = 0; i < _args.Length; i++)
                if (!_args[i].StructureEquals(o._args[i], operands))
                    return false;
            return true;
        }
    }

    public sealed partial class ReduceNode
    {
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
        {
            // Axis and keepdims are read by the HOST per evaluation (flat vs axis path, output shape), so
            // two roots that differ only there must be two programs even though the child kernel is shared.
            h.Add(NDExprNodeTag.Reduce, (long)_kind);
            h.Add((ulong)(_axis is int ax ? (long)ax : long.MinValue));
            h.Add(_keepdims ? 1UL : 0UL);
            _child.HashStructure(ref h, operands);
        }

        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
            => bound is ReduceNode o && o._kind == _kind && o._axis == _axis && o._keepdims == _keepdims
               && _child.StructureEquals(o._child, operands);
    }
}
