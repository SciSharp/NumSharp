using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The Chebyshev points and Gauss-Chebyshev quadrature of <c>numpy.polynomial.chebyshev</c> (plan
    ///     <c>docs/plans/numpy-polynomial-remaining-compute.md</c> §3.3, unit U8): <c>chebpts1</c>, <c>chebpts2</c>,
    ///     <c>chebgauss</c>. NumPy 2.4.2's own tests ported; NumPy's bytes for small counts; NumPy's outcome for every argument
    ///     kind with a C# spelling (the probe scripts are quoted next to <see cref="Pins"/> and <see cref="KindOutcomes"/>);
    ///     np.arange's length rule; the node formulas against the C runtime's sin / cos on the exact ramp; the fill kernel's
    ///     bounds; and the inputs the oracle (<c>polypts.jsonl</c> / <c>polypts_host.jsonl</c>) does not record — NumPy's empty
    ///     object array, decimal and char.
    /// </summary>
    /// <remarks>
    ///     Every node is the C runtime's <c>sin</c> / <c>cos</c> on both sides, and NumPy's bytes are win-amd64's (ucrtbase).
    ///     On another host the values are held to 2 ULP of NumPy's; the formula test (<see cref="Nodes_AreNumPysStatementChain"/>)
    ///     and every error, dtype and shape are exact on every host.
    /// </remarks>
    [TestClass]
    public class PolynomialChebPointsTests
    {
        /// <summary>The three functions by name, each returning its arrays (chebgauss: x then w).</summary>
        private static readonly Dictionary<string, Func<object, NDArray[]>> Functions = new()
        {
            ["chebpts1"] = v => new[] { np.polynomial.chebyshev.chebpts1(v) },
            ["chebpts2"] = v => new[] { np.polynomial.chebyshev.chebpts2(v) },
            ["chebgauss"] = v =>
            {
                var (x, w) = np.polynomial.chebyshev.chebgauss(v);
                return new[] { x, w };
            },
        };

        /// <summary>2**63, the edge of np.arange's double length rule.</summary>
        private static readonly BigInteger T = BigInteger.One << 63;

        /// <summary>
        ///     The C# spelling of every argument kind of <see cref="KindOutcomes"/> (the house map: a C# integer, bool, float,
        ///     double, <see cref="Complex"/> or <see cref="BigInteger"/> is a Python scalar, <see cref="Half"/> / <c>char</c> are
        ///     np.float16 / np.uint16 scalars, a typed C# array or an <see cref="NDArray"/> is an ndarray, <c>object[]</c> a
        ///     list, a <c>ValueTuple</c> a tuple, null Python's None).
        /// </summary>
        private static readonly Dictionary<string, Func<object>> Inputs = new()
        {
            ["int 3"] = () => 3, ["int 0"] = () => 0, ["int 1"] = () => 1, ["int 2"] = () => 2, ["int -1"] = () => -1,
            ["long 3"] = () => 3L, ["byte 3"] = () => (byte)3, ["ulong max"] = () => ulong.MaxValue, ["double 3.0"] = () => 3.0,
            ["double 3.7"] = () => 3.7, ["double -0.0"] = () => -0.0, ["float 3.5f"] = () => 3.5f, ["double nan"] = () => double.NaN,
            ["double inf"] = () => double.PositiveInfinity, ["double -inf"] = () => double.NegativeInfinity,
            ["double 1e300"] = () => 1e300, ["double 1e19"] = () => 1e19, ["double 2**63"] = () => 9223372036854775808.0,
            ["bool true"] = () => true, ["bool false"] = () => false, ["str 3"] = () => "3", ["str 3.0"] = () => "3.0",
            ["str sp3sp"] = () => " 3 ", ["str fullwidth3"] = () => "３", ["str 3_0"] = () => "3_0", ["str 0x3"] = () => "0x3",
            ["str -0"] = () => "-0", ["str empty"] = () => "", ["null"] = () => null, ["complex 3+0j"] = () => new Complex(3, 0),
            ["half 3"] = () => (Half)3, ["half 3.5"] = () => (Half)3.5, ["half nan"] = () => Half.NaN, ["char 3"] = () => (char)3,
            ["nd0 int64 3"] = () => np.array(3L), ["nd0 double 3.5"] = () => np.array(3.5), ["nd0 double 3.0"] = () => np.array(3.0),
            ["nd0 bool true"] = () => np.array(true), ["nd0 half 3.5"] = () => np.array((Half)3.5),
            ["nd0 complex 3"] = () => np.array(new Complex(3, 0)), ["nd0 uint64 2**63"] = () => np.array(1UL << 63),
            ["nd0 uint16 3"] = () => np.array((ushort)3), ["nd1 [3]"] = () => np.array(new long[] { 3 }),
            ["int[] {3}"] = () => new int[] { 3 }, ["object[] {3}"] = () => new object[] { 3 }, ["tuple (3,)"] = () => ValueTuple.Create(3),
            ["big 2**60"] = () => BigInteger.One << 60, ["big 2**63-513"] = () => T - 513, ["big 2**63-512"] = () => T - 512,
            ["big 2**63-511"] = () => T - 511, ["big 2**63-1"] = () => T - 1, ["big 2**63"] = () => T, ["big 2**63+1"] = () => T + 1,
            ["big 2**63+2"] = () => T + 2, ["big 2**63+1024"] = () => T + 1024, ["big 2**63+1025"] = () => T + 1025,
            ["big 2**64"] = () => BigInteger.One << 64, ["big 2**1023"] = () => BigInteger.One << 1023,
            ["big 2**1024"] = () => BigInteger.One << 1024, ["big -2**70"] = () => -(BigInteger.One << 70),
        };

        // NumPy 2.4.2's bytes (scratchpad pts_pins.py):
        //   for n in list(range(1, 9)) + [13, 33]: chebpts1(n).tobytes().hex(), chebpts2(n) (n >= 2), chebgauss(n)'s x and w.
        /// <summary>NumPy's float64 bytes of each function at small counts (<c>chebgauss.x</c> / <c>chebgauss.w</c>: the
        ///     tuple's two slots).</summary>
        private static readonly (string Fn, long N, string Hex)[] Pins =
        {
            ("chebpts1", 1, "0000000000000000"),
            ("chebgauss.x", 1, "075c143326a6913c"),
            ("chebgauss.w", 1, "182d4454fb210940"),
            ("chebpts1", 2, "cd3b7f669ea0e6bfcd3b7f669ea0e63f"),
            ("chebpts2", 2, "000000000000f0bf000000000000f03f"),
            ("chebgauss.x", 2, "cd3b7f669ea0e63fcc3b7f669ea0e6bf"),
            ("chebgauss.w", 2, "182d4454fb21f93f182d4454fb21f93f"),
            ("chebpts1", 3, "aa4c58e87ab6ebbf0000000000000000aa4c58e87ab6eb3f"),
            ("chebpts2", 3, "000000000000f0bf075c143326a6913c000000000000f03f"),
            ("chebgauss.x", 3, "ab4c58e87ab6eb3f075c143326a6913cab4c58e87ab6ebbf"),
            ("chebgauss.w", 3, "65732d3852c1f03f65732d3852c1f03f65732d3852c1f03f"),
            ("chebpts1", 4, "468d32cf6b90edbf63a9aea6e27dd8bf63a9aea6e27dd83f468d32cf6b90ed3f"),
            ("chebpts2", 4, "000000000000f0bf020000000000e0bffeffffffffffdf3f000000000000f03f"),
            ("chebgauss.x", 4, "468d32cf6b90ed3f64a9aea6e27dd83f62a9aea6e27dd8bf468d32cf6b90edbf"),
            ("chebgauss.w", 4, "182d4454fb21e93f182d4454fb21e93f182d4454fb21e93f182d4454fb21e93f"),
            ("chebpts1", 5, "ff5444130e6feebf5e5a750423cfe2bf00000000000000005e5a750423cfe23fff5444130e6fee3f"),
            ("chebpts2", 5, "000000000000f0bfcc3b7f669ea0e6bf075c143326a6913ccd3b7f669ea0e63f000000000000f03f"),
            ("chebgauss.x", 5, "ff5444130e6fee3f5e5a750423cfe23f075c143326a6913c5d5a750423cfe2bfff5444130e6feebf"),
            ("chebgauss.w", 5, "e0f09c762f1be43fe0f09c762f1be43fe0f09c762f1be43fe0f09c762f1be43fe0f09c762f1be43f"),
            ("chebpts1", 6, "14bf4847dde8eebfcd3b7f669ea0e6bf900693c17d90d0bf900693c17d90d03fcd3b7f669ea0e63f14bf4847dde8ee3f"),
            ("chebpts2", 6, "000000000000f0bfa7f4979b77e3e9bf4ee92f37efc6d3bf50e92f37efc6d33fa8f4979b77e3e93f000000000000f03f"),
            ("chebgauss.x", 6, "15bf4847dde8ee3fcd3b7f669ea0e63f900693c17d90d03f8e0693c17d90d0bfcc3b7f669ea0e6bf14bf4847dde8eebf"),
            ("chebgauss.w", 6, "65732d3852c1e03f65732d3852c1e03f65732d3852c1e03f65732d3852c1e03f65732d3852c1e03f65732d3852c1e03f"),
            ("chebpts1", 7, "69e958059c32efbf4bde0575c304e9bfc1ab714dc0c4dbbf0000000000000000c1ab714dc0c4db3f4bde0575c304e93f69e958059c32ef3f"),
            ("chebpts2", 7, "000000000000f0bfab4c58e87ab6ebbf020000000000e0bf075c143326a6913cfeffffffffffdf3fa94c58e87ab6eb3f000000000000f03f"),
            ("chebgauss.x", 7, "69e958059c32ef3f4bde0575c304e93fc2ab714dc0c4db3f075c143326a6913cc0ab714dc0c4dbbf48de0575c304e9bf6ae958059c32efbf"),
            ("chebgauss.w", 7, "40a1bb3b1fb9dc3f40a1bb3b1fb9dc3f40a1bb3b1fb9dc3f40a1bb3b1fb9dc3f40a1bb3b1fb9dc3f40a1bb3b1fb9dc3f40a1bb3b1fb9dc3f"),
            ("chebpts1", 8, "b05cf7cf9762efbfa3a10e29669beabfc868ae393bc7e1bf0aa6693cb8f8c8bf0aa6693cb8f8c83fc868ae393bc7e13fa3a10e29669bea3fb05cf7cf9762ef3f"),
            ("chebpts2", 8, "000000000000f0bf705ccba9bcd4ecbfd1ed8be2a0f3e3bf804502e3907bccbf844502e3907bcc3fd2ed8be2a0f3e33f715ccba9bcd4ec3f000000000000f03f"),
            ("chebgauss.x", 8, "b05cf7cf9762ef3fa3a10e29669bea3fc968ae393bc7e13f0da6693cb8f8c83f08a6693cb8f8c8bfc668ae393bc7e1bfa4a10e29669beabfb05cf7cf9762efbf"),
            ("chebgauss.w", 8, "182d4454fb21d93f182d4454fb21d93f182d4454fb21d93f182d4454fb21d93f182d4454fb21d93f182d4454fb21d93f182d4454fb21d93f182d4454fb21d93f"),
            ("chebpts1", 13, "6967966645c4efbf4601f22ea7ebedbfd2c3a442e255eabf832f4c024d38e5bf7bc4674206beddbfbe8dc44be5a1cebf0000000000000000be8dc44be5a1ce3f7bc4674206bedd3f832f4c024d38e53fd2c3a442e255ea3f4601f22ea7ebed3f6967966645c4ef3f"),
            ("chebpts2", 13, "000000000000f0bf14bf4847dde8eebfab4c58e87ab6ebbfcc3b7f669ea0e6bf020000000000e0bf920693c17d90d0bf075c143326a6913c900693c17d90d03ffeffffffffffdf3fcd3b7f669ea0e63fa94c58e87ab6eb3f14bf4847dde8ee3f000000000000f03f"),
            ("chebgauss.x", 13, "6967966645c4ef3f4701f22ea7ebed3fd2c3a442e255ea3f852f4c024d38e53f7dc4674206bedd3fc98dc44be5a1ce3ffdd175e6ec2ca7bcb58dc44be5a1cebf7bc4674206beddbf822f4c024d38e5bfd2c3a442e255eabf4601f22ea7ebedbf6967966645c4efbf"),
            ("chebgauss.w", 13, "a7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3fa7ada22cbfeece3f"),
            ("chebpts1", 33, "83ac5aa3b8f6efbfef4238049eacefbfdf5efa921419efbfc553625b723deebf132cee8eb41bedbfaa4c58e87ab6ebbf954145960111eabfa1283abb192fe8bfff173f942015e6bff278b15af5c7e3bf0b58bbf8ed4ce1bfefa83e559553ddbffa930d5351c9d7bfbf8d76fde707d2bf90aca0806239c8bfec5347c59755b8bf0000000000000000ec5347c59755b83f90aca0806239c83fbf8d76fde707d23ffa930d5351c9d73fefa83e559553dd3f0b58bbf8ed4ce13ff278b15af5c7e33fff173f942015e63fa1283abb192fe83f954145960111ea3faa4c58e87ab6eb3f132cee8eb41bed3fc553625b723dee3fdf5efa921419ef3fef4238049eacef3f83ac5aa3b8f6ef3f"),
            ("chebpts2", 33, "000000000000f0bf2525d1a38dd8efbfb05cf7cf9762efbfda2dc656419feebf468d32cf6b90edbfb0bd80f1b238ecbfa4a10e29669beabf4117156b80bce8bfcc3b7f669ea0e6bfd51d0925f34ce4bfc668ae393bc7e1bf3cf606385d2bdebf62a9aea6e27dd8bf029fd52e0694d2bf08a6693cb8f8c8bf2fb429bca617b9bf075c143326a6913c38b429bca617b93f0da6693cb8f8c83f059fd52e0694d23f64a9aea6e27dd83f3ef606385d2bde3fc768ae393bc7e13fd61d0925f34ce43fcd3b7f669ea0e63f4117156b80bce83fa4a10e29669bea3fb0bd80f1b238ec3f468d32cf6b90ed3fdb2dc656419fee3fb05cf7cf9762ef3f2625d1a38dd8ef3f000000000000f03f"),
            ("chebgauss.x", 33, "83ac5aa3b8f6ef3fef4238049eacef3fdf5efa921419ef3fc553625b723dee3f142cee8eb41bed3fab4c58e87ab6eb3f964145960111ea3fa2283abb192fe83f00183f942015e63ff378b15af5c7e33f0b58bbf8ed4ce13ff1a83e559553dd3ffa930d5351c9d73fc18d76fde707d23f91aca0806239c83ff75347c59755b83f075c143326a6913cdf5347c59755b8bf8daca0806239c8bfbb8d76fde707d2bff8930d5351c9d7bfefa83e559553ddbf0b58bbf8ed4ce1bff378b15af5c7e3bffe173f942015e6bfa0283abb192fe8bf964145960111eabfab4c58e87ab6ebbf122cee8eb41bedbfc553625b723deebfdf5efa921419efbfef4238049eacefbf83ac5aa3b8f6efbf"),
            ("chebgauss.w", 33, "d9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83fd9a7703a035fb83f"),
        };

        /// <summary>
        ///     NumPy's outcome per (function, argument kind of <see cref="Inputs"/>), written by the scratchpad probe
        ///     <c>pts_kinds.py</c>: <c>dtype[shape] sha256[:16]</c> of each result (<c> &amp; </c>-joined for chebgauss) or
        ///     <c>!! PythonType: message</c>. The two counts where NumPy returns an empty object array are left out (see
        ///     <see cref="TwoPow63Band_FollowsNumPysArange"/>).
        /// </summary>
        private static readonly (string Fn, string Label, string Outcome)[] KindOutcomes =
        {
            ("chebpts1", "int 3", "float64[3] 905910399692865d"),
            ("chebpts1", "int 0", "!! ValueError: npts must be >= 1"),
            ("chebpts1", "int 1", "float64[1] af5570f5a1810b7a"),
            ("chebpts1", "int 2", "float64[2] 545f002dc0f61c16"),
            ("chebpts1", "int -1", "!! ValueError: npts must be >= 1"),
            ("chebpts1", "long 3", "float64[3] 905910399692865d"),
            ("chebpts1", "byte 3", "float64[3] 905910399692865d"),
            ("chebpts1", "ulong max", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts1", "double 3.0", "float64[3] 905910399692865d"),
            ("chebpts1", "double 3.7", "!! ValueError: npts must be integer"),
            ("chebpts1", "double -0.0", "!! ValueError: npts must be >= 1"),
            ("chebpts1", "float 3.5f", "!! ValueError: npts must be integer"),
            ("chebpts1", "double nan", "!! ValueError: cannot convert float NaN to integer"),
            ("chebpts1", "double inf", "!! OverflowError: cannot convert float infinity to integer"),
            ("chebpts1", "double -inf", "!! OverflowError: cannot convert float infinity to integer"),
            ("chebpts1", "double 1e300", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts1", "double 1e19", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts1", "double 2**63", "float64[0] e3b0c44298fc1c14"),
            ("chebpts1", "bool true", "float64[1] af5570f5a1810b7a"),
            ("chebpts1", "bool false", "!! ValueError: npts must be >= 1"),
            ("chebpts1", "str 3", "!! ValueError: npts must be integer"),
            ("chebpts1", "str 3.0", "!! ValueError: invalid literal for int() with base 10: '3.0'"),
            ("chebpts1", "str sp3sp", "!! ValueError: npts must be integer"),
            ("chebpts1", "str fullwidth3", "!! ValueError: npts must be integer"),
            ("chebpts1", "str 3_0", "!! ValueError: npts must be integer"),
            ("chebpts1", "str 0x3", "!! ValueError: invalid literal for int() with base 10: '0x3'"),
            ("chebpts1", "str -0", "!! ValueError: npts must be integer"),
            ("chebpts1", "str empty", "!! ValueError: invalid literal for int() with base 10: ''"),
            ("chebpts1", "null", "!! TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"),
            ("chebpts1", "complex 3+0j", "!! TypeError: int() argument must be a string, a bytes-like object or a real number, not 'complex'"),
            ("chebpts1", "half 3", "float64[3] 905910399692865d"),
            ("chebpts1", "half 3.5", "!! ValueError: npts must be integer"),
            ("chebpts1", "half nan", "!! ValueError: cannot convert float NaN to integer"),
            ("chebpts1", "char 3", "float64[3] 905910399692865d"),
            ("chebpts1", "nd0 int64 3", "float64[3] 905910399692865d"),
            ("chebpts1", "nd0 double 3.5", "!! ValueError: npts must be integer"),
            ("chebpts1", "nd0 double 3.0", "float64[3] 905910399692865d"),
            ("chebpts1", "nd0 bool true", "float64[1] af5570f5a1810b7a"),
            ("chebpts1", "nd0 half 3.5", "!! ValueError: npts must be integer"),
            ("chebpts1", "nd0 complex 3", "!! TypeError: int() argument must be a string, a bytes-like object or a real number, not 'complex'"),
            ("chebpts1", "nd0 uint64 2**63", "float64[0] e3b0c44298fc1c14"),
            ("chebpts1", "nd0 uint16 3", "float64[3] 905910399692865d"),
            ("chebpts1", "nd1 [3]", "!! TypeError: only 0-dimensional arrays can be converted to Python scalars"),
            ("chebpts1", "int[] {3}", "!! TypeError: only 0-dimensional arrays can be converted to Python scalars"),
            ("chebpts1", "object[] {3}", "!! TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'"),
            ("chebpts1", "tuple (3,)", "!! TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'"),
            ("chebpts1", "big 2**60", "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
            ("chebpts1", "big 2**63-513", "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
            ("chebpts1", "big 2**63-512", "float64[0] e3b0c44298fc1c14"),
            ("chebpts1", "big 2**63-511", "float64[0] e3b0c44298fc1c14"),
            ("chebpts1", "big 2**63-1", "float64[0] e3b0c44298fc1c14"),
            ("chebpts1", "big 2**63", "float64[0] e3b0c44298fc1c14"),
            ("chebpts1", "big 2**63+1", "float64[0] e3b0c44298fc1c14"),
            ("chebpts1", "big 2**63+1025", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts1", "big 2**64", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts1", "big 2**1023", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts1", "big 2**1024", "!! OverflowError: int too large to convert to float"),
            ("chebpts1", "big -2**70", "!! ValueError: npts must be >= 1"),
            ("chebpts2", "int 3", "float64[3] 6d01bdafb6a7c57e"),
            ("chebpts2", "int 0", "!! ValueError: npts must be >= 2"),
            ("chebpts2", "int 1", "!! ValueError: npts must be >= 2"),
            ("chebpts2", "int 2", "float64[2] 723f1c3eac1d2c30"),
            ("chebpts2", "int -1", "!! ValueError: npts must be >= 2"),
            ("chebpts2", "long 3", "float64[3] 6d01bdafb6a7c57e"),
            ("chebpts2", "byte 3", "float64[3] 6d01bdafb6a7c57e"),
            ("chebpts2", "ulong max", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts2", "double 3.0", "float64[3] 6d01bdafb6a7c57e"),
            ("chebpts2", "double 3.7", "!! ValueError: npts must be integer"),
            ("chebpts2", "double -0.0", "!! ValueError: npts must be >= 2"),
            ("chebpts2", "float 3.5f", "!! ValueError: npts must be integer"),
            ("chebpts2", "double nan", "!! ValueError: cannot convert float NaN to integer"),
            ("chebpts2", "double inf", "!! OverflowError: cannot convert float infinity to integer"),
            ("chebpts2", "double -inf", "!! OverflowError: cannot convert float infinity to integer"),
            ("chebpts2", "double 1e300", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts2", "double 1e19", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts2", "double 2**63", "!! IndexError: index -1 is out of bounds for axis 0 with size 0"),
            ("chebpts2", "bool true", "!! ValueError: npts must be >= 2"),
            ("chebpts2", "bool false", "!! ValueError: npts must be >= 2"),
            ("chebpts2", "str 3", "!! ValueError: npts must be integer"),
            ("chebpts2", "str 3.0", "!! ValueError: invalid literal for int() with base 10: '3.0'"),
            ("chebpts2", "str sp3sp", "!! ValueError: npts must be integer"),
            ("chebpts2", "str fullwidth3", "!! ValueError: npts must be integer"),
            ("chebpts2", "str 3_0", "!! ValueError: npts must be integer"),
            ("chebpts2", "str 0x3", "!! ValueError: invalid literal for int() with base 10: '0x3'"),
            ("chebpts2", "str -0", "!! ValueError: npts must be integer"),
            ("chebpts2", "str empty", "!! ValueError: invalid literal for int() with base 10: ''"),
            ("chebpts2", "null", "!! TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"),
            ("chebpts2", "complex 3+0j", "!! TypeError: int() argument must be a string, a bytes-like object or a real number, not 'complex'"),
            ("chebpts2", "half 3", "float64[3] 6d01bdafb6a7c57e"),
            ("chebpts2", "half 3.5", "!! ValueError: npts must be integer"),
            ("chebpts2", "half nan", "!! ValueError: cannot convert float NaN to integer"),
            ("chebpts2", "char 3", "float64[3] 6d01bdafb6a7c57e"),
            ("chebpts2", "nd0 int64 3", "float64[3] 6d01bdafb6a7c57e"),
            ("chebpts2", "nd0 double 3.5", "!! ValueError: npts must be integer"),
            ("chebpts2", "nd0 double 3.0", "float64[3] 6d01bdafb6a7c57e"),
            ("chebpts2", "nd0 bool true", "!! ValueError: npts must be >= 2"),
            ("chebpts2", "nd0 half 3.5", "!! ValueError: npts must be integer"),
            ("chebpts2", "nd0 complex 3", "!! TypeError: int() argument must be a string, a bytes-like object or a real number, not 'complex'"),
            ("chebpts2", "nd0 uint64 2**63", "!! IndexError: index -1 is out of bounds for axis 0 with size 0"),
            ("chebpts2", "nd0 uint16 3", "float64[3] 6d01bdafb6a7c57e"),
            ("chebpts2", "nd1 [3]", "!! TypeError: only 0-dimensional arrays can be converted to Python scalars"),
            ("chebpts2", "int[] {3}", "!! TypeError: only 0-dimensional arrays can be converted to Python scalars"),
            ("chebpts2", "object[] {3}", "!! TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'"),
            ("chebpts2", "tuple (3,)", "!! TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'"),
            ("chebpts2", "big 2**60", "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
            ("chebpts2", "big 2**63-513", "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
            ("chebpts2", "big 2**63-512", "!! IndexError: index -1 is out of bounds for axis 0 with size 0"),
            ("chebpts2", "big 2**63-511", "!! IndexError: index -1 is out of bounds for axis 0 with size 0"),
            ("chebpts2", "big 2**63-1", "!! IndexError: index -1 is out of bounds for axis 0 with size 0"),
            ("chebpts2", "big 2**63", "!! IndexError: index -1 is out of bounds for axis 0 with size 0"),
            ("chebpts2", "big 2**63+1", "!! IndexError: index -1 is out of bounds for axis 0 with size 0"),
            ("chebpts2", "big 2**63+2", "!! IndexError: index -1 is out of bounds for axis 0 with size 0"),
            ("chebpts2", "big 2**63+1024", "!! IndexError: index -1 is out of bounds for axis 0 with size 0"),
            ("chebpts2", "big 2**63+1025", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts2", "big 2**64", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts2", "big 2**1023", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts2", "big 2**1024", "!! ValueError: Maximum allowed size exceeded"),
            ("chebpts2", "big -2**70", "!! ValueError: npts must be >= 2"),
            ("chebgauss", "int 3", "float64[3] aafef330b0a8312c & float64[3] c509a47831a0ce53"),
            ("chebgauss", "int 0", "!! ValueError: deg must be a positive integer"),
            ("chebgauss", "int 1", "float64[1] 7446ff0c35e2f974 & float64[1] 8b5319c77d1df2dc"),
            ("chebgauss", "int 2", "float64[2] 6505a43835d56b45 & float64[2] c06ce994927eca06"),
            ("chebgauss", "int -1", "!! ValueError: deg must be a positive integer"),
            ("chebgauss", "long 3", "float64[3] aafef330b0a8312c & float64[3] c509a47831a0ce53"),
            ("chebgauss", "byte 3", "float64[3] aafef330b0a8312c & float64[3] c509a47831a0ce53"),
            ("chebgauss", "ulong max", "!! ValueError: Maximum allowed size exceeded"),
            ("chebgauss", "double 3.0", "!! TypeError: deg must be an integer, received 3.0"),
            ("chebgauss", "double 3.7", "!! TypeError: deg must be an integer, received 3.7"),
            ("chebgauss", "double -0.0", "!! TypeError: deg must be an integer, received -0.0"),
            ("chebgauss", "float 3.5f", "!! TypeError: deg must be an integer, received 3.5"),
            ("chebgauss", "double nan", "!! TypeError: deg must be an integer, received nan"),
            ("chebgauss", "double inf", "!! TypeError: deg must be an integer, received inf"),
            ("chebgauss", "double -inf", "!! TypeError: deg must be an integer, received -inf"),
            ("chebgauss", "double 1e300", "!! TypeError: deg must be an integer, received 1e+300"),
            ("chebgauss", "double 1e19", "!! TypeError: deg must be an integer, received 1e+19"),
            ("chebgauss", "double 2**63", "!! TypeError: deg must be an integer, received 9.223372036854776e+18"),
            ("chebgauss", "bool true", "float64[1] 7446ff0c35e2f974 & float64[1] 8b5319c77d1df2dc"),
            ("chebgauss", "bool false", "!! ValueError: deg must be a positive integer"),
            ("chebgauss", "str 3", "!! TypeError: deg must be an integer, received 3"),
            ("chebgauss", "str 3.0", "!! TypeError: deg must be an integer, received 3.0"),
            ("chebgauss", "str sp3sp", "!! TypeError: deg must be an integer, received  3 "),
            ("chebgauss", "str fullwidth3", "!! TypeError: deg must be an integer, received \uff13"),
            ("chebgauss", "str 3_0", "!! TypeError: deg must be an integer, received 3_0"),
            ("chebgauss", "str 0x3", "!! TypeError: deg must be an integer, received 0x3"),
            ("chebgauss", "str -0", "!! TypeError: deg must be an integer, received -0"),
            ("chebgauss", "str empty", "!! TypeError: deg must be an integer, received "),
            ("chebgauss", "null", "!! TypeError: deg must be an integer, received None"),
            ("chebgauss", "complex 3+0j", "!! TypeError: deg must be an integer, received (3+0j)"),
            ("chebgauss", "half 3", "!! TypeError: deg must be an integer, received 3.0"),
            ("chebgauss", "half 3.5", "!! TypeError: deg must be an integer, received 3.5"),
            ("chebgauss", "half nan", "!! TypeError: deg must be an integer, received nan"),
            ("chebgauss", "char 3", "float64[3] aafef330b0a8312c & float64[3] c509a47831a0ce53"),
            ("chebgauss", "nd0 int64 3", "float64[3] aafef330b0a8312c & float64[3] c509a47831a0ce53"),
            ("chebgauss", "nd0 double 3.5", "!! TypeError: deg must be an integer, received 3.5"),
            ("chebgauss", "nd0 double 3.0", "!! TypeError: deg must be an integer, received 3.0"),
            ("chebgauss", "nd0 bool true", "!! TypeError: deg must be an integer, received True"),
            ("chebgauss", "nd0 half 3.5", "!! TypeError: deg must be an integer, received 3.5"),
            ("chebgauss", "nd0 complex 3", "!! TypeError: deg must be an integer, received (3+0j)"),
            ("chebgauss", "nd0 uint64 2**63", "!! ValueError: Maximum allowed dimension exceeded"),
            ("chebgauss", "nd0 uint16 3", "float64[3] aafef330b0a8312c & float64[3] c509a47831a0ce53"),
            ("chebgauss", "nd1 [3]", "!! TypeError: deg must be an integer, received [3]"),
            ("chebgauss", "int[] {3}", "!! TypeError: deg must be an integer, received [3]"),
            ("chebgauss", "object[] {3}", "!! TypeError: deg must be an integer, received [3]"),
            ("chebgauss", "tuple (3,)", "!! TypeError: deg must be an integer, received (3,)"),
            ("chebgauss", "big 2**60", "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
            ("chebgauss", "big 2**63-513", "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
            ("chebgauss", "big 2**63-512", "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
            ("chebgauss", "big 2**63-511", "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
            ("chebgauss", "big 2**63-1", "!! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."),
            ("chebgauss", "big 2**63", "!! ValueError: Maximum allowed dimension exceeded"),
            ("chebgauss", "big 2**63+1", "!! ValueError: Maximum allowed dimension exceeded"),
            ("chebgauss", "big 2**63+2", "!! ValueError: Maximum allowed dimension exceeded"),
            ("chebgauss", "big 2**63+1024", "!! ValueError: Maximum allowed dimension exceeded"),
            ("chebgauss", "big 2**63+1025", "!! ValueError: Maximum allowed size exceeded"),
            ("chebgauss", "big 2**64", "!! ValueError: Maximum allowed size exceeded"),
            ("chebgauss", "big 2**1023", "!! ValueError: Maximum allowed size exceeded"),
            ("chebgauss", "big 2**1024", "!! ValueError: Maximum allowed size exceeded"),
            ("chebgauss", "big -2**70", "!! ValueError: deg must be a positive integer"),
        };

        // ---------------------------------------------------------------------------------------------
        //  NumPy's bytes and outcomes
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     chebpts1 / chebpts2 / chebgauss for counts 1-8, 13 and 33 have NumPy's float64 bytes (2 ULP off win-amd64, whose
        ///     C runtime made them), shape and dtype.
        /// </summary>
        [TestMethod]
        public void NumPyPins_AreReproduced()
        {
            foreach (var (fn, n, hex) in Pins)
            {
                string label = $"{fn}({n})";
                NDArray r = fn switch
                {
                    "chebpts1" => np.polynomial.chebyshev.chebpts1(n),
                    "chebpts2" => np.polynomial.chebyshev.chebpts2(n),
                    "chebgauss.x" => np.polynomial.chebyshev.chebgauss(n).x,
                    _ => np.polynomial.chebyshev.chebgauss(n).w,
                };
                Assert.AreEqual("float64", r.dtype.name, $"{label}: dtype");
                CollectionAssert.AreEqual(new[] { n }, r.Shape.Dimensions, $"{label}: shape");
                AssertBytes(hex, Hex(r), label);
            }
        }

        /// <summary>
        ///     Every argument kind of <see cref="Inputs"/> gives NumPy's outcome for each function: its error type and text
        ///     (int()'s parser, <c>_npts != npts</c>, operator.index's TypeError, np.arange's size errors, linspace's IndexError,
        ///     np.ones' dimension error) on every host, or its result's dtype and shape — and, on win-amd64, its bytes.
        /// </summary>
        [TestMethod]
        public void ArgumentKinds_MatchNumPy()
        {
            foreach (var (fn, label, outcome) in KindOutcomes)
            {
                string got = Outcome(() => Functions[fn](Inputs[label]()));
                string context = $"{fn}({label})";
                if (outcome.StartsWith("!!", StringComparison.Ordinal) || OperatingSystem.IsWindows())
                {
                    Assert.AreEqual(outcome, got, context);
                    continue;
                }
                // Off win-amd64 the digests of a value result come from another C runtime: compare dtype and shape only.
                Assert.AreEqual(StripDigests(outcome), StripDigests(got), context);
            }
        }

        /// <summary>
        ///     The counts whose np.arange length rounds to exactly <c>2**63</c> (and so casts to INT64_MIN on x86, an empty
        ///     arange): chebpts1 returns NumPy's empty float64 array while the ramp's start fits int64
        ///     (<c>[2**63 - 512, 2**63 + 1]</c>) and refuses the counts where NumPy's empty result is an OBJECT array
        ///     (<c>[2**63 + 2, 2**63 + 1024]</c>); one past the band is NumPy's <c>Maximum allowed size exceeded</c>.
        /// </summary>
        [TestMethod]
        public void TwoPow63Band_FollowsNumPysArange()
        {
            foreach (var n in new[] { T - 512, T - 2, T, T + 1 })
            {
                var r = np.polynomial.chebyshev.chebpts1(n);
                Assert.AreEqual("float64", r.dtype.name, $"chebpts1({n})");
                Assert.AreEqual(0L, r.size, $"chebpts1({n})");
                Assert.IsTrue(r.flags.owndata && r.flags.writeable, $"chebpts1({n}): a fresh array");
            }
            foreach (var n in new[] { T + 2, T + 3, T + 1024 })
            {
                var e = Throws(() => np.polynomial.chebyshev.chebpts1(n), $"chebpts1({n})");
                Assert.IsInstanceOfType(e, typeof(NotSupportedException), $"chebpts1({n}): {e.Message}");
                StringAssert.Contains(e.Message, "object array");
            }
            AssertValueError(() => np.polynomial.chebyshev.chebpts1(T - 513), AllocationGuard.TooBigMessage, "chebpts1(2**63-513)");
            AssertValueError(() => np.polynomial.chebyshev.chebpts1(T + 1025), "Maximum allowed size exceeded", "chebpts1(2**63+1025)");
            foreach (var n in new[] { T - 512, T, T + 1024 })
            {
                var e = Throws(() => np.polynomial.chebyshev.chebpts2(n), $"chebpts2({n})");
                Assert.IsInstanceOfType(e, typeof(IndexError), $"chebpts2({n}): {e.Message}");
                Assert.AreEqual("index -1 is out of bounds for axis 0 with size 0", e.Message);
            }
            AssertValueError(() => np.polynomial.chebyshev.chebgauss(T - 511), AllocationGuard.TooBigMessage, "chebgauss(2**63-511)");
            AssertValueError(() => np.polynomial.chebyshev.chebgauss(T), "Maximum allowed dimension exceeded", "chebgauss(2**63)");
            AssertValueError(() => np.polynomial.chebyshev.chebgauss(T + 1024), "Maximum allowed dimension exceeded", "chebgauss(2**63+1024)");
            AssertValueError(() => np.polynomial.chebyshev.chebgauss(T + 1025), "Maximum allowed size exceeded", "chebgauss(2**63+1025)");
        }

        /// <summary>
        ///     <see cref="NDPolyChebPoints.ArangeLength"/> is NumPy's <c>len(np.arange(start, stop, step))</c> for Python-int
        ///     arguments (probed against 2.4.2): the ceiling of a correctly rounded double quotient, the underflow rule (one
        ///     element for a positive zero quotient of a nonzero difference, none for a negative one), the <c>2**63</c> band's
        ///     empty range on both sides, and <c>Maximum allowed size exceeded</c> past npy_intp or past the float range.
        /// </summary>
        [TestMethod]
        public void ArangeLength_IsNumPysRule()
        {
            var big = BigInteger.One << 1100;
            var cases = new (BigInteger Start, BigInteger Stop, BigInteger Step, long Length)[]
            {
                (0, 10, 1, 10), (5, 0, 1, 0), (0, 0, 1, 0), (0, 7, 2, 4), (0, -7, -2, 4), (3, 3, -1, 0), (0, 1, big, 1),
                (0, -1, big, 0), (0, T, 1, 0), (1, BigInteger.One << 64, 2, 0), (1 - T, T + 1, 2, 0), (0, -T, 1, 0),
                (0, (BigInteger.One << 53) + 1, 1, 1L << 53), (0, T - 513, 1, (long)(T - 1024)),
            };
            foreach (var (start, stop, step, length) in cases)
                Assert.AreEqual(length, NDPolyChebPoints.ArangeLength(start, stop, step), $"arange({start}, {stop}, {step})");
            foreach (var (start, stop, step) in new (BigInteger, BigInteger, BigInteger)[]
                     {
                         (0, T + 1025, 1), (0, -T - 1025, 1), (0, -(BigInteger.One << 64), 1), (0, big, 1), (1, 2 * big, 2),
                     })
                AssertValueError(() => NDPolyChebPoints.ArangeLength(start, stop, step), "Maximum allowed size exceeded",
                    $"arange({start}, {stop}, {step})");
        }

        // ---------------------------------------------------------------------------------------------
        //  The formulas and the kernel
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Every node is NumPy's statement chain on the exact integer ramp, in NumPy's order, under this host's
        ///     <see cref="System.Math.Sin"/> / <see cref="System.Math.Cos"/> (the C runtime NumPy's float64 loops call): chebpts1's
        ///     <c>sin(c * (1 - n + 2k))</c> with <c>c = 0.5*pi/n</c>, chebpts2's <c>cos(k*(pi/(n - 1)) + -pi)</c> and its endpoint
        ///     <c>cos(0)</c>, chebgauss' <c>cos((pi * (2k + 1)) / (2.0*n))</c> and its constant weight <c>pi/n</c> — bit for bit
        ///     on every host, for counts across the fill's whole length range.
        /// </summary>
        [TestMethod]
        public void Nodes_AreNumPysStatementChain()
        {
            var counts = new List<long>();
            for (long n = 1; n <= 130; n++)
                counts.Add(n);
            counts.AddRange(new long[] { 1000, 1001, 4097, 65537 });
            foreach (long n in counts)
            {
                var p1 = np.polynomial.chebyshev.chebpts1(n).ToArray<double>();
                double c = 0.5 * System.Math.PI / n;
                for (long k = 0; k < n; k++)
                    AssertSame(System.Math.Sin(c * (double)(1 - n + 2 * k)), p1[k], $"chebpts1({n})[{k}]");
                if (n >= 2)
                {
                    var p2 = np.polynomial.chebyshev.chebpts2(n).ToArray<double>();
                    double step = System.Math.PI / (double)(n - 1);
                    for (long k = 0; k < n - 1; k++)
                        AssertSame(System.Math.Cos((double)k * step + -System.Math.PI), p2[k], $"chebpts2({n})[{k}]");
                    AssertSame(System.Math.Cos(0.0), p2[n - 1], $"chebpts2({n})[-1]");
                }
                var (x, w) = np.polynomial.chebyshev.chebgauss(n);
                var xs = x.ToArray<double>();
                var ws = w.ToArray<double>();
                for (long k = 0; k < n; k++)
                {
                    AssertSame(System.Math.Cos(System.Math.PI * (double)(2 * k + 1) / (2.0 * n)), xs[k], $"chebgauss({n}).x[{k}]");
                    AssertSame(System.Math.PI / n, ws[k], $"chebgauss({n}).w[{k}]");
                }
            }
        }

        /// <summary>
        ///     The fill kernel writes exactly its count — nothing for zero or a negative count, even and odd counts alike for
        ///     the mirrored chebpts1 fill — and leaves the slots around it untouched (each formula, with sentinel slots on both
        ///     sides; chebpts1's kind is given the antisymmetric ramp its contract requires).
        /// </summary>
        [TestMethod]
        public unsafe void Kernel_WritesExactlyItsCount()
        {
            foreach (PolyChebNodes kind in Enum.GetValues(typeof(PolyChebNodes)))
            {
                var kernel = DirectILKernelGenerator.GetPolyChebNodesKernel(kind);
                foreach (long count in new long[] { -3, -2, 0, 1, 2, 3, 7, 8, 64, 65 })
                {
                    var buf = new double[count > 0 ? count + 2 : 2];
                    Array.Fill(buf, 12345.0);
                    double first = kind == PolyChebNodes.Pts1 ? 1.0 - count : 1.0;
                    fixed (double* p = buf)
                        kernel(p + 1, count, first, 2.0, 3.0);
                    string label = $"{kind} count {count}";
                    Assert.AreEqual(12345.0, buf[0], $"{label}: the slot before");
                    Assert.AreEqual(12345.0, buf[^1], $"{label}: the slot after");
                    for (long k = 0; k < count; k++)
                    {
                        double v = kind == PolyChebNodes.Pts2 ? k : first + 2.0 * k;
                        double expected = kind switch
                        {
                            PolyChebNodes.Pts1 => System.Math.Sin(3.0 * v),
                            PolyChebNodes.Pts2 => System.Math.Cos(v * 2.0 + 1.0),
                            _ => System.Math.Cos(System.Math.PI * v / 3.0),
                        };
                        AssertSame(expected, buf[k + 1], $"{label}: element {k}");
                    }
                }
            }
        }

        /// <summary>
        ///     Each result is a fresh, owning, writeable, C- and F-contiguous float64 vector, as NumPy's ufunc results are
        ///     (<c>chebgauss</c>'s two slots are separate arrays).
        /// </summary>
        [TestMethod]
        public void Results_AreFreshOwningVectors()
        {
            var (x, w) = np.polynomial.chebyshev.chebgauss(5);
            foreach (var (label, r) in new (string, NDArray)[]
                     {
                         ("chebpts1", np.polynomial.chebyshev.chebpts1(5)), ("chebpts2", np.polynomial.chebyshev.chebpts2(5)),
                         ("chebgauss.x", x), ("chebgauss.w", w), ("chebpts1(1)", np.polynomial.chebyshev.chebpts1(1)),
                     })
            {
                Assert.IsTrue(r.flags.c_contiguous && r.flags.f_contiguous, $"{label}: contiguous");
                Assert.IsTrue(r.flags.owndata && r.flags.writeable, $"{label}: owning and writeable");
                Assert.AreEqual(1, r.ndim, label);
            }
            Assert.IsFalse(np.shares_memory(x, w), "chebgauss: x and w are separate arrays");
        }

        // ---------------------------------------------------------------------------------------------
        //  NumPy's own tests (numpy/polynomial/tests/test_chebyshev.py) and the NumSharp-only kinds
        // ---------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>TestMisc.test_chebpts1</c> / <c>test_chebpts2</c> and <c>TestGauss.test_100</c>, ported.</summary>
        [TestMethod]
        public void NumPysOwnTests()
        {
            Assert.ThrowsException<ValueError>(() => np.polynomial.chebyshev.chebpts1(1.5));
            Assert.ThrowsException<ValueError>(() => np.polynomial.chebyshev.chebpts1(0));
            AssertAlmostEqual(new[] { 0.0 }, np.polynomial.chebyshev.chebpts1(1));
            AssertAlmostEqual(new[] { -0.70710678118654746, 0.70710678118654746 }, np.polynomial.chebyshev.chebpts1(2));
            AssertAlmostEqual(new[] { -0.86602540378443871, 0, 0.86602540378443871 }, np.polynomial.chebyshev.chebpts1(3));
            AssertAlmostEqual(new[] { -0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325 }, np.polynomial.chebyshev.chebpts1(4));

            Assert.ThrowsException<ValueError>(() => np.polynomial.chebyshev.chebpts2(1.5));
            Assert.ThrowsException<ValueError>(() => np.polynomial.chebyshev.chebpts2(1));
            AssertAlmostEqual(new[] { -1.0, 1 }, np.polynomial.chebyshev.chebpts2(2));
            AssertAlmostEqual(new[] { -1.0, 0, 1 }, np.polynomial.chebyshev.chebpts2(3));
            AssertAlmostEqual(new[] { -1, -0.5, .5, 1 }, np.polynomial.chebyshev.chebpts2(4));
            AssertAlmostEqual(new[] { -1.0, -0.707106781187, 0, 0.707106781187, 1.0 }, np.polynomial.chebyshev.chebpts2(5));

            // test_100: the nodes and weights make the first 100 Chebyshev polynomials orthogonal, and the weights sum to pi.
            var (x, w) = np.polynomial.chebyshev.chebgauss(100);
            var v = np.polynomial.chebyshev.chebvander(x, 99);
            var vv = np.dot(v.T * w, v);
            var vd = 1 / np.sqrt(vv.diagonal());
            vv = np.expand_dims(vd, 1) * vv * vd;
            var eye = np.eye(100);
            var diff = np.abs(vv - eye);
            Assert.IsTrue(diff.max().GetDouble() < 1.5e-7, $"orthogonality: max deviation {diff.max().GetDouble()}");
            Assert.IsTrue(System.Math.Abs(w.sum().GetDouble() - System.Math.PI) < 1.5e-7, "the weights sum to pi");
        }

        /// <summary>
        ///     The kinds NumPy has no spelling for, under the house map: a <c>decimal</c> count is a NumPy-like scalar that
        ///     <c>int()</c> truncates (3m is 3 points, 3.5m is <c>npts must be integer</c>) and <c>operator.index</c> refuses; a
        ///     <c>char</c> is np.uint16, an integer for both.
        /// </summary>
        [TestMethod]
        public void DecimalAndChar_FollowTheHouseMap()
        {
            Assert.AreEqual(Hex(np.polynomial.chebyshev.chebpts1(3)), Hex(np.polynomial.chebyshev.chebpts1(3m)), "chebpts1(3m)");
            Assert.AreEqual(Hex(np.polynomial.chebyshev.chebpts2(3)), Hex(np.polynomial.chebyshev.chebpts2(3m)), "chebpts2(3m)");
            AssertValueError(() => np.polynomial.chebyshev.chebpts1(3.5m), "npts must be integer", "chebpts1(3.5m)");
            var e = Throws(() => np.polynomial.chebyshev.chebgauss(3m), "chebgauss(3m)");
            Assert.IsInstanceOfType(e, typeof(TypeError), $"chebgauss(3m): {e.Message}");
            StringAssert.StartsWith(e.Message, "deg must be an integer, received ");
            Assert.AreEqual(Hex(np.polynomial.chebyshev.chebpts1(3)), Hex(np.polynomial.chebyshev.chebpts1((char)3)), "chebpts1((char)3)");
            var (x, w) = np.polynomial.chebyshev.chebgauss((char)3);
            var (x3, w3) = np.polynomial.chebyshev.chebgauss(3);
            Assert.AreEqual(Hex(x3) + Hex(w3), Hex(x) + Hex(w), "chebgauss((char)3)");
        }

        // ---------------------------------------------------------------------------------------------
        //  Helpers
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     One call's outcome in the probe's format: <c>dtype[shape] sha256[:16]</c> per array (<c> &amp; </c>-joined), or
        ///     <c>!! PythonType: message</c>.
        /// </summary>
        /// <param name="call">The call.</param>
        /// <returns>The outcome text.</returns>
        private static string Outcome(Func<NDArray[]> call)
        {
            try
            {
                var parts = new List<string>();
                foreach (var a in call())
                {
                    string digest = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(a.tobytes())).ToLowerInvariant()[..16];
                    parts.Add($"{a.dtype.name}[{string.Join(", ", a.shape)}] {digest}");
                }
                return string.Join(" & ", parts);
            }
            catch (Exception e)
            {
                string name = e switch
                {
                    OverflowException => "OverflowError",
                    OutOfMemoryException => "MemoryError",
                    _ => e.GetType().Name,
                };
                return $"!! {name}: {e.Message}";
            }
        }

        /// <summary>An outcome text with the digests removed (dtype and shape kept).</summary>
        /// <param name="outcome">The outcome.</param>
        /// <returns>The text without its digests.</returns>
        private static string StripDigests(string outcome)
        {
            var parts = outcome.Split(" & ");
            for (int i = 0; i < parts.Length; i++)
            {
                int space = parts[i].LastIndexOf(' ');
                if (space > 0)
                    parts[i] = parts[i][..space];
            }
            return string.Join(" & ", parts);
        }

        /// <summary>
        ///     Asserts NumPy's bytes: exact on win-amd64, each float64 within 2 ULP elsewhere (another C runtime's sin / cos).
        /// </summary>
        /// <param name="expectedHex">NumPy's bytes.</param>
        /// <param name="actualHex">NumSharp's bytes.</param>
        /// <param name="label">The case.</param>
        private static void AssertBytes(string expectedHex, string actualHex, string label)
        {
            if (OperatingSystem.IsWindows())
            {
                Assert.AreEqual(expectedHex, actualHex, $"{label}: bytes");
                return;
            }
            byte[] e = Convert.FromHexString(expectedHex), a = Convert.FromHexString(actualHex);
            Assert.AreEqual(e.Length, a.Length, $"{label}: length");
            for (int i = 0; i < e.Length; i += 8)
                Assert.IsTrue(UlpDistance(BitConverter.ToInt64(a, i), BitConverter.ToInt64(e, i)) <= 2,
                    $"{label}: element {i / 8} differs by more than 2 ULP");
        }

        /// <summary>The distance in units in the last place between two float64 bit patterns read as signed integers.</summary>
        /// <param name="a">One bit pattern.</param>
        /// <param name="b">The other.</param>
        /// <returns>The distance (a sign difference is the sum of both distances from zero).</returns>
        private static long UlpDistance(long a, long b)
        {
            if (a == b)
                return 0;
            static long Ordered(long v) => v < 0 ? long.MinValue - v : v;   // two's-complement order of sign-magnitude
            long d = Ordered(a) - Ordered(b);
            return d < 0 ? -d : d;
        }

        /// <summary>Asserts two doubles are the same bit pattern.</summary>
        /// <param name="expected">The expected value.</param>
        /// <param name="actual">The actual value.</param>
        /// <param name="label">The case.</param>
        private static void AssertSame(double expected, double actual, string label)
            => Assert.AreEqual(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual),
                $"{label}: {expected:R} vs {actual:R}");

        /// <summary>NumPy's <c>assert_almost_equal</c> at its default <c>decimal=7</c>: <c>|a - b| &lt; 1.5e-7</c> elementwise.</summary>
        /// <param name="expected">NumPy's target.</param>
        /// <param name="actual">NumSharp's result.</param>
        private static void AssertAlmostEqual(double[] expected, NDArray actual)
        {
            var a = actual.ToArray<double>();
            Assert.AreEqual(expected.Length, a.Length);
            for (int i = 0; i < a.Length; i++)
                Assert.IsTrue(System.Math.Abs(a[i] - expected[i]) < 1.5e-7, $"element {i}: {a[i]:R} vs {expected[i]:R}");
        }

        /// <summary>The C-order bytes of <paramref name="a"/> as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The hex text.</returns>
        private static string Hex(NDArray a) => Convert.ToHexString(a.tobytes()).ToLowerInvariant();

        /// <summary>Runs <paramref name="call"/> and returns the exception it throws.</summary>
        /// <param name="call">The call.</param>
        /// <param name="label">The case, for the failure message.</param>
        /// <returns>The exception.</returns>
        /// <exception cref="AssertFailedException">The call returned.</exception>
        private static Exception Throws(Func<object> call, string label)
        {
            try
            {
                call();
            }
            catch (Exception e)
            {
                return e;
            }
            Assert.Fail($"{label}: expected an exception");
            return null;
        }

        /// <summary>Asserts that <paramref name="call"/> raises a <see cref="ValueError"/> with exactly <paramref name="text"/>.</summary>
        /// <param name="call">The call.</param>
        /// <param name="text">NumPy's text.</param>
        /// <param name="label">The case.</param>
        private static void AssertValueError(Func<object> call, string text, string label)
        {
            var e = Throws(call, label);
            Assert.IsInstanceOfType(e, typeof(ValueError), $"{label}: {e.GetType().Name}: {e.Message}");
            Assert.AreEqual(text, e.Message, label);
        }
    }
}
