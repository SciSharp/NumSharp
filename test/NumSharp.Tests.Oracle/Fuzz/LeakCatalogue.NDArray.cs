using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Catalogue entries for the <c>ndarray</c> module's C#-side surface (the PascalCase
    ///     infrastructure the value oracle exempts by convention but which allocates like any other
    ///     member — printing, conversions, the typed getters/setters, data replacement), the
    ///     NumPy-named members no corpus key carries, <see cref="NumSharp.Generic.NDArray{TDType}"/>,
    ///     and every operator (<c>op_*</c> special names — a surface the ApiInventory tool filters out
    ///     entirely, so it is enumerated separately by <see cref="LeakSurfaceCoverageTests"/>).
    /// </summary>
    internal static partial class LeakCatalogue
    {
        /// <summary>Entries for NDArray instance/static members and NDArray&lt;T&gt;.</summary>
        /// <param name="l">The entry list.</param>
        private static void AddNDArray(List<LeakCase> l)
        {
            // ---- generic views / clones / raw data ----
            E(l, "ndarray.AsGeneric", "typed wrapper", f => f.M.AsGeneric<double>());
            E(l, "ndarray.AsOrMakeGeneric", "typed wrapper", f => f.M.AsOrMakeGeneric<double>());
            E(l, "ndarray.MakeGeneric", "typed alias", f => f.M.MakeGeneric<double>());
            E(l, "ndarray.Clone", "deep copy of a view", f => f.MT.Clone());
            // CloneData returns a BARE slice holding no counted reference; a caller frees it with one AddRef/Release
            // pair (the ARC protocol — Release alone on a zero count is a no-op). Consumed that way here.
            E(l, "ndarray.CloneData", "slice copy", f =>
            {
                var s = f.MT.CloneData();
                long n = s.Count;
                s.TryAddRef();
                s.Release();
                return Box(n);
            });
            E(l, "ndarray.CloneData", "typed slice copy", f =>
            {
                var s = f.MT.CloneData<double>();
                long n = s.Count;
                s.TryAddRef();
                s.Release();
                return Box(n);
            });
            E(l, "ndarray.Data", "typed slice", f => Box(f.M.Data<double>().Count));
            E(l, "ndarray.GetData", "untyped slice of a view", f => Box(f.MT.GetData().Count));
            E(l, "ndarray.GetData", "typed slice", f => Box(f.M.GetData<double>().Count));
            E(l, "ndarray.GetData", "coordinate sub-array", f => f.M.GetData(new[] { 1 }));
            E(l, "ndarray.CopyTo", "managed array", f =>
            {
                var dst = new double[12];
                f.MT.CopyTo(dst);
                return dst;
            });
            E(l, "ndarray.Contains", "value", f => Box(f.M.Contains(1.75)));
            E(l, "ndarray.__contains__", "value", f => Box(f.M.__contains__(1.75)));
            E(l, "ndarray.Dispose", "construct + dispose", f =>
            {
                np.zeros(new Shape(64)).Dispose();
                return null;
            });
            E(l, "ndarray.Equals", "reference", f => Box(f.M.Equals(f.M)));
            T(l, "ndarray.GetHashCode", "unhashable by design (NumPy: mutable ndarray)", f => Box(f.M.GetHashCode()));
            T(l, "ndarray.__hash__", "unhashable by design (NumPy: mutable ndarray)", f => Box(f.M.__hash__()));
            E(l, "ndarray.FromMultiDimArray", "copy", f => NDArray.FromMultiDimArray<double>(new double[,] { { 1, 2 }, { 3, 4 } }));
            E(l, "ndarray.FromString", "parse", f => NDArray.FromString("[1, 2, 3]"));
            E(l, "ndarray.Scalar", "object", f => NDArray.Scalar(2.5));
            E(l, "ndarray.Scalar", "object + dtype", f => NDArray.Scalar(3, np.float32));
            E(l, "ndarray.Scalar", "typed", f => NDArray.Scalar<long>(7L));
            E(l, "ndarray.AsString", "char array", f => NDArray.AsString(f.Chars));
            T(l, "ndarray.AsStringArray", "broken: parses a legacy ToString layout the NumPy-parity printer no longer emits", f => NDArray.AsStringArray(f.Chars));
            E(l, "ndarray.GetString", "char row", f => f.Chars.GetString());
            T(l, "ndarray.GetStringAt", "broken: passes ndim coordinates where GetString requires ndim-1", f => f.Chars.GetStringAt(0));
            E(l, "ndarray.SetString", "same text", f =>
            {
                var x = np.array("world".ToCharArray());
                x.SetString("world");
                return x;
            });
            T(l, "ndarray.SetStringAt", "broken: passes ndim coordinates where SetString requires ndim-1", f =>
            {
                using var x = np.array("world".ToCharArray());   // the entry's own scratch — released on the throw
                x.SetStringAt("world", 0);
                return null;
            });

            // ---- element access (typed getters/setters, one per dtype) ----
            E(l, "ndarray.GetAtIndex", "object", f => f.MT.GetAtIndex(5));
            E(l, "ndarray.GetAtIndex", "typed", f => Box(f.MT.GetAtIndex<double>(5)));
            E(l, "ndarray.SetAtIndex", "object", f =>
            {
                var t = f.Typed[NPTypeCode.Double];
                t.SetAtIndex(t.GetAtIndex(1), 1);
                return null;
            });
            E(l, "ndarray.GetValue", "object", f => f.MT.GetValue(1, 2));
            E(l, "ndarray.GetValue", "typed", f => Box(f.MT.GetValue<double>(1, 2)));
            E(l, "ndarray.SetValue", "object", f =>
            {
                var t = f.Typed[NPTypeCode.Int64];
                t.SetValue(t.GetValue(0, 1), 0, 1);
                return null;
            });
            E(l, "ndarray.GetBoolean", "bool", f => Box(f.Typed[NPTypeCode.Boolean].GetBoolean(0, 1)));
            E(l, "ndarray.GetByte", "uint8", f => Box(f.Typed[NPTypeCode.Byte].GetByte(0, 1)));
            E(l, "ndarray.GetSByte", "int8", f => Box(f.Typed[NPTypeCode.SByte].GetSByte(0, 1)));
            E(l, "ndarray.GetInt16", "int16", f => Box(f.Typed[NPTypeCode.Int16].GetInt16(0, 1)));
            E(l, "ndarray.GetUInt16", "uint16", f => Box(f.Typed[NPTypeCode.UInt16].GetUInt16(0, 1)));
            E(l, "ndarray.GetInt32", "int32", f => Box(f.Typed[NPTypeCode.Int32].GetInt32(0, 1)));
            E(l, "ndarray.GetUInt32", "uint32", f => Box(f.Typed[NPTypeCode.UInt32].GetUInt32(0, 1)));
            E(l, "ndarray.GetInt64", "int64", f => Box(f.Typed[NPTypeCode.Int64].GetInt64(0, 1)));
            E(l, "ndarray.GetUInt64", "uint64", f => Box(f.Typed[NPTypeCode.UInt64].GetUInt64(0, 1)));
            E(l, "ndarray.GetChar", "char", f => Box(f.Typed[NPTypeCode.Char].GetChar(0, 1)));
            E(l, "ndarray.GetHalf", "float16", f => Box(f.Typed[NPTypeCode.Half].GetHalf(0, 1)));
            E(l, "ndarray.GetSingle", "float32", f => Box(f.Typed[NPTypeCode.Single].GetSingle(0, 1)));
            E(l, "ndarray.GetDouble", "float64", f => Box(f.Typed[NPTypeCode.Double].GetDouble(0, 1)));
            E(l, "ndarray.GetDecimal", "decimal", f => Box(f.Typed[NPTypeCode.Decimal].GetDecimal(0, 1)));
            E(l, "ndarray.GetComplex", "complex128", f => Box(f.Typed[NPTypeCode.Complex].GetComplex(0, 1)));
            // Setters write back the value already stored — the region is re-executable and state-neutral.
            E(l, "ndarray.SetBoolean", "bool", f => { var t = f.Typed[NPTypeCode.Boolean]; t.SetBoolean(t.GetBoolean(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetByte", "uint8", f => { var t = f.Typed[NPTypeCode.Byte]; t.SetByte(t.GetByte(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetSByte", "int8", f => { var t = f.Typed[NPTypeCode.SByte]; t.SetSByte(t.GetSByte(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetInt16", "int16", f => { var t = f.Typed[NPTypeCode.Int16]; t.SetInt16(t.GetInt16(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetUInt16", "uint16", f => { var t = f.Typed[NPTypeCode.UInt16]; t.SetUInt16(t.GetUInt16(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetInt32", "int32", f => { var t = f.Typed[NPTypeCode.Int32]; t.SetInt32(t.GetInt32(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetUInt32", "uint32", f => { var t = f.Typed[NPTypeCode.UInt32]; t.SetUInt32(t.GetUInt32(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetInt64", "int64", f => { var t = f.Typed[NPTypeCode.Int64]; t.SetInt64(t.GetInt64(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetUInt64", "uint64", f => { var t = f.Typed[NPTypeCode.UInt64]; t.SetUInt64(t.GetUInt64(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetChar", "char", f => { var t = f.Typed[NPTypeCode.Char]; t.SetChar(t.GetChar(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetHalf", "float16", f => { var t = f.Typed[NPTypeCode.Half]; t.SetHalf(t.GetHalf(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetSingle", "float32", f => { var t = f.Typed[NPTypeCode.Single]; t.SetSingle(t.GetSingle(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetDouble", "float64", f => { var t = f.Typed[NPTypeCode.Double]; t.SetDouble(t.GetDouble(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetDecimal", "decimal", f => { var t = f.Typed[NPTypeCode.Decimal]; t.SetDecimal(t.GetDecimal(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetComplex", "complex128", f => { var t = f.Typed[NPTypeCode.Complex]; t.SetComplex(t.GetComplex(0, 1), 0, 1); return null; });
            E(l, "ndarray.SetData", "NDArray row", f =>
            {
                var t = f.Typed[NPTypeCode.Double];
                using var row = t.GetData(new[] { 0 }).copy();
                t.SetData(row, 0);
                return null;
            });
            E(l, "ndarray.SetData", "object scalar", f =>
            {
                var t = f.Typed[NPTypeCode.Double];
                t.SetData(t.GetAtIndex(2), new[] { 0, 2 });
                return null;
            });

            // ---- fancy index helpers ----
            E(l, "ndarray.GetIndices", "fancy rows", f => f.M.GetIndices(null, new[] { f.Idx }));
            E(l, "ndarray.SetIndices", "fancy rows, same values", f =>
            {
                var t = f.Typed[NPTypeCode.Double];
                using var ix = np.array(new long[] { 1, 0 });
                using var vals = t.GetIndices(null, new[] { ix });
                t.SetIndices(vals, new[] { ix });
                return null;
            });
            E(l, "ndarray.GetNDArrays", "axis 0", f => f.M.GetNDArrays(0));
            E(l, "ndarray.GetEnumerator", "walk rows", f =>
            {
                long n = 0;
                foreach (var _ in f.M)
                    n++;
                return Box(n);
            });
            E(l, "ndarray.__iter__", "walk rows", f =>
            {
                var e = f.M.__iter__();
                long n = 0;
                while (e.MoveNext())
                    n++;
                return Box(n);
            });
            E(l, "ndarray.__getitem__", "int", f => f.M.__getitem__(1));
            E(l, "ndarray.__getitem__", "slice string", f => f.M.__getitem__("1:, ::2"));
            E(l, "ndarray.__setitem__", "slice string, same values", f =>
            {
                var t = f.Typed[NPTypeCode.Double];
                using var row = t["0"].copy();
                t.__setitem__("0", row);
                return null;
            });

            // ---- conversions / printing ----
            E(l, "ndarray.ToArray", "typed copy of a view", f => f.MT.ToArray<double>());
            E(l, "ndarray.ToJaggedArray", "jagged", f => f.MT.ToJaggedArray<double>());
            E(l, "ndarray.ToMuliDimArray", "multi-dim", f => f.MT.ToMuliDimArray<double>());
            E(l, "ndarray.ToString", "str", f => f.MT.ToString());
            E(l, "ndarray.ToString", "repr", f => f.C.ToString(true));
            E(l, "ndarray.tolist", "nested lists", f => f.MT.tolist());
            E(l, "ndarray.tofile", "binary path", f =>
            {
                f.MT.tofile(Path.Combine(f.Dir, "tofile.bin"));
                return null;
            });
            E(l, "ndarray.tofile", "text stream", f =>
            {
                using var s = new MemoryStream();
                f.MT.tofile(s, " ", "%.3f");
                return Box(s.Length);
            });

            // ---- data replacement / in-place (on arrays the entry owns) ----
            E(l, "ndarray.ReplaceData", "NDArray", f =>
            {
                var x = np.zeros(new Shape(3, 4));
                x.ReplaceData(f.M);
                return x;
            });
            E(l, "ndarray.ReplaceData", "managed array", f =>
            {
                var x = np.zeros(new Shape(2));
                x.ReplaceData(new double[] { 1, 2 });
                return x;
            });
            T(l, "ndarray.Normalize", "unimplemented (NotImplementedException)", f =>
            {
                using var x = np.array(new double[] { 1, 2, 3, 4 });   // the entry's own scratch — released on the throw
                x.Normalize();
                return null;
            });
            E(l, "ndarray.itemset", "shape + value", f =>
            {
                var x = np.zeros(new Shape(2, 2));
                x.itemset(new[] { 1, 1 }, 3.0);
                return x;
            });
            E(l, "ndarray.negate", "copy", f => f.MT.negate());
            E(l, "ndarray.reshape_unsafe", "shape", f => f.M.reshape_unsafe(new Shape(4, 3)));
            E(l, "ndarray.setfield", "window write", f =>
            {
                var x = np.zeros(new Shape(4), np.int64);
                x.setfield(7, np.int32, 0);
                return x;
            });
            E(l, "ndarray.setflags", "round trip", f =>
            {
                var x = f.M.copy();
                x.setflags(write: false);
                x.setflags(write: true);
                return x;
            });
            E(l, "ndarray.to_device", "cpu", f => f.MT.to_device("cpu"));

            // ---- NDArray<T> (the generic surface) ----
            E(l, "NDArray<T>.Clone", "typed clone", f => f.GM.Clone());
            E(l, "NDArray<T>.GetAtIndex", "typed read", f => Box(f.GM.GetAtIndex(3)));
            E(l, "NDArray<T>.reshape", "typed reshape", f => f.GM.reshape(4, 3));
            E(l, "NDArray<T>.reshape_unsafe", "typed reshape", f => f.GM.reshape_unsafe(new Shape(12)));
        }

        /// <summary>
        ///     One entry per operator name on <see cref="NDArray"/>, <see cref="NumSharp.Generic.NDArray{TDType}"/>
        ///     and <see cref="NDMaskedArray"/>; binary operators additionally cover their <c>object</c>
        ///     overloads (the asanyarray-wrapping path the value corpus — which drives np.* functions —
        ///     never reaches).
        /// </summary>
        /// <param name="l">The entry list.</param>
        private static void AddOperators(List<LeakCase> l)
        {
            // ---- NDArray arithmetic/bitwise/shift, all three overload shapes ----
            E(l, "ndarray.op_Addition", "NDArray+NDArray", f => f.M + f.M);
            E(l, "ndarray.op_Addition", "NDArray+object", f => f.M + (object)f.V3["0:1"]);
            E(l, "ndarray.op_Addition", "object+NDArray", f => (object)f.I + f.I);
            E(l, "ndarray.op_Addition", "array+cast-needing held scalar", f => f.M + f.Zero);
            E(l, "ndarray.op_Subtraction", "NDArray-NDArray", f => f.M - f.MT.T);
            E(l, "ndarray.op_Subtraction", "object-NDArray", f => (object)f.M - f.M);
            E(l, "ndarray.op_Multiply", "NDArray*NDArray", f => f.MT * f.MT);
            E(l, "ndarray.op_Multiply", "NDArray*object", f => f.M * (object)f.M);
            E(l, "ndarray.op_Division", "NDArray/NDArray", f => f.M / f.M);
            E(l, "ndarray.op_Division", "int true-divide", f => f.I / f.I);
            E(l, "ndarray.op_Modulus", "NDArray%NDArray", f => f.I % f.I["::-1"]);
            E(l, "ndarray.op_BitwiseAnd", "int&int", f => f.I & f.I);
            E(l, "ndarray.op_BitwiseAnd", "bool&bool", f => f.B & f.B);
            E(l, "ndarray.op_BitwiseOr", "int|int", f => f.I | f.I);
            E(l, "ndarray.op_ExclusiveOr", "int^int", f => f.I ^ f.I);
            E(l, "ndarray.op_LeftShift", "int<<int", f => f.I << f.I["::-1"]);
            E(l, "ndarray.op_LeftShift", "int<<object", f => f.I << (object)f.Zero);
            E(l, "ndarray.op_RightShift", "int>>int", f => f.I >> f.I["::-1"]);
            // ---- NDArray comparisons ----
            E(l, "ndarray.op_Equality", "array==array", f => f.M == f.M);
            E(l, "ndarray.op_Equality", "array==object", f => f.M == (object)f.M);
            E(l, "ndarray.op_Inequality", "array!=array", f => f.M != f.MT.T);
            E(l, "ndarray.op_LessThan", "array<array", f => f.M < f.M);
            E(l, "ndarray.op_LessThanOrEqual", "array<=array", f => f.M <= f.M);
            E(l, "ndarray.op_GreaterThan", "array>array", f => f.MT > f.MT);
            E(l, "ndarray.op_GreaterThanOrEqual", "array>=array", f => f.M >= f.M);
            // ---- NDArray unary ----
            E(l, "ndarray.op_LogicalNot", "bool", f => !f.B);
            E(l, "ndarray.op_LogicalNot", "int", f => !f.I);
            E(l, "ndarray.op_OnesComplement", "int", f => ~f.I);
            E(l, "ndarray.op_UnaryNegation", "view", f => -f.MT);
            E(l, "ndarray.op_UnaryPlus", "view", f => +f.MT);
            // ---- NDArray conversions ----
            E(l, "ndarray.op_Implicit", "scalar -> 0-d", f => { NDArray x = 5.5; return x; });
            E(l, "ndarray.op_Implicit", "managed array -> NDArray", f => { NDArray x = new[] { 1.0, 2.0, 3.0 }; return x; });
            E(l, "ndarray.op_Implicit", "string -> char array", f => { NDArray x = "abc"; return x; });
            E(l, "ndarray.op_Explicit", "0-d -> scalar", f => Box((double)f.M["0, 0"]));
            E(l, "ndarray.op_Explicit", "NDArray -> managed array", f => (Array)f.MT);

            // ---- NDArray<T> ----
            E(l, "NDArray<T>.op_BitwiseAnd", "bool&bool", f => f.GB & f.GB);
            E(l, "NDArray<T>.op_BitwiseOr", "bool|bool", f => f.GB | f.GB);
            E(l, "NDArray<T>.op_ExclusiveOr", "bool^bool", f => f.GB ^ f.GB);
            E(l, "NDArray<T>.op_Explicit", "T[] -> NDArray<T>", f => (NumSharp.Generic.NDArray<double>)new[] { 1.0, 2.0 });
            E(l, "NDArray<T>.op_Implicit", "NDArray<T> -> ArraySlice<T>", f => { ArraySlice<double> s = f.GM; return Box(s.Count); });
            E(l, "NDArray<T>.Item", "typed element get/set by int[] and long[] coordinates", f =>
            {
                // The coordinate overloads read and write the TYPED element itself — no NDArray is built —
                // so a read plus a write-back of the same value on the fixture's wrapper (leaving it
                // unchanged) must read zero pool traffic.
                double v = f.GM[new[] { 1, 2 }];
                f.GM[new[] { 1, 2 }] = v;
                double w = f.GM[2L, 3L];
                f.GM[2L, 3L] = w;
                return Box(v + w);
            });
            E(l, "NDArray<T>.Item", "typed slice get (string + Slice[]) and set, on an owned copy", f =>
            {
                // The slice getters wrap an untyped view as a typed alias — a fresh view object per call, the
                // caller's to release — and the setters copy a typed view back through the base indexer. The
                // array is the entry's own copy, so the fixture is never written.
                using var c = f.M.copy();
                using var gc = c.MakeGeneric<double>();
                using var row = gc["1"];
                using var col = gc[Slice.All, Slice.Index(2)];
                gc["0"] = row;                          // row 1's values into row 0
                gc[Slice.All, Slice.Index(3)] = col;    // column 2's values into column 3
                return Box(row.size + col.size);
            });

            // ---- NDMaskedArray ----
            E(l, "NDMaskedArray.op_Addition", "masked+masked", f => f.MA + f.MB);
            E(l, "NDMaskedArray.op_Addition", "masked+NDArray", f => f.MA + f.M);
            E(l, "NDMaskedArray.op_Addition", "NDArray+masked", f => f.M + f.MA);
            E(l, "NDMaskedArray.op_Subtraction", "masked-masked", f => f.MA - f.MB);
            E(l, "NDMaskedArray.op_Multiply", "masked*masked", f => f.MA * f.MB);
            E(l, "NDMaskedArray.op_Division", "masked/masked", f => f.MA / f.MB);
            E(l, "NDMaskedArray.op_Modulus", "masked%masked", f => f.MI % f.MI2);
            E(l, "NDMaskedArray.op_BitwiseAnd", "masked&masked", f => f.MI & f.MI2);
            E(l, "NDMaskedArray.op_BitwiseOr", "masked|object", f => f.MI | (object)f.MI2);
            E(l, "NDMaskedArray.op_ExclusiveOr", "object^masked", f => (object)f.MI ^ f.MI2);
            E(l, "NDMaskedArray.op_LessThan", "masked<masked", f => f.MA < f.MB);
            E(l, "NDMaskedArray.op_LessThanOrEqual", "masked<=NDArray", f => f.MA <= f.M);
            E(l, "NDMaskedArray.op_GreaterThan", "masked>masked", f => f.MA > f.MB);
            E(l, "NDMaskedArray.op_GreaterThanOrEqual", "NDArray>=masked", f => f.M >= f.MA);
            E(l, "NDMaskedArray.op_OnesComplement", "masked int", f => ~f.MI);
            E(l, "NDMaskedArray.op_UnaryNegation", "masked", f => -f.MA);
            E(l, "NDMaskedArray.op_Implicit", "NDArray -> masked", f => { NDMaskedArray m = f.M; return m; });
        }
    }
}
