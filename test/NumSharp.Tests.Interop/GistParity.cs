using System;
using System.Buffers.Binary;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace NumSharp.Tests.Interop;

// Gist algorithm parity is separate from the strict existing BLAS/operator gates.
// A test chooses EXACT or a stated ULP budget before comparison; failures never auto-fall back.
internal static class GistParity
{
    internal static void AssertExact(NDArray actual, PyObject expected, string label)
    {
        AssertShapeAndDtype(actual, expected, label);
        ByteContract.AssertSameBytes(actual, expected, label);
    }

    internal static void AssertUlps(NDArray actual, PyObject expected, ulong maxUlps, string label)
    {
        AssertShapeAndDtype(actual, expected, label);
        Assert.IsTrue(actual.typecode is NPTypeCode.Double or NPTypeCode.Single, "ULP gate requires float32/64.");
        byte[] a = ByteContract.NsBytes(actual), b = expected.bytes_c();
        Assert.AreEqual(a.Length, b.Length, label);
        int width = actual.dtypesize;
        ulong largest = 0;
        int unequal = 0;
        for (int i = 0; i < a.Length; i += width)
        {
            if (a.AsSpan(i, width).SequenceEqual(b.AsSpan(i, width))) continue;
            unequal++;
            ulong distance;
            if (width == 8)
            {
                ulong x = BinaryPrimitives.ReadUInt64LittleEndian(a.AsSpan(i, 8));
                ulong y = BinaryPrimitives.ReadUInt64LittleEndian(b.AsSpan(i, 8));
                double vx = BitConverter.UInt64BitsToDouble(x), vy = BitConverter.UInt64BitsToDouble(y);
                Assert.IsTrue(double.IsFinite(vx) && double.IsFinite(vy) && !(vx == 0 && vy == 0),
                    $"{label}: nonfinite values and signed zero must match bits at element {i / width}.");
                distance = Distance(Ordered64(x), Ordered64(y));
            }
            else
            {
                uint x = BinaryPrimitives.ReadUInt32LittleEndian(a.AsSpan(i, 4));
                uint y = BinaryPrimitives.ReadUInt32LittleEndian(b.AsSpan(i, 4));
                float vx = BitConverter.UInt32BitsToSingle(x), vy = BitConverter.UInt32BitsToSingle(y);
                Assert.IsTrue(float.IsFinite(vx) && float.IsFinite(vy) && !(vx == 0 && vy == 0),
                    $"{label}: nonfinite values and signed zero must match bits at element {i / width}.");
                distance = Distance(Ordered32(x), Ordered32(y));
            }
            largest = Math.Max(largest, distance);
            Assert.IsTrue(distance <= maxUlps,
                $"{label}: element {i / width} differs by {distance} ULP (budget {maxUlps}); " +
                $"C#={Convert.ToHexString(a.AsSpan(i, width))}, Python={Convert.ToHexString(b.AsSpan(i, width))}.");
        }
        Console.WriteLine($"GIST_PARITY {label}: elements={actual.size}, unequal={unequal}, max_ulp={largest}, budget={maxUlps}");
    }

    private static void AssertShapeAndDtype(NDArray actual, PyObject expected, string label)
    {
        Assert.IsTrue(BitConverter.IsLittleEndian, "This raw ULP diagnostic expects a little-endian host.");
        Assert.AreEqual(actual.dtype.str, expected.dtype_str(), label + " dtype");
        CollectionAssert.AreEqual(actual.shape, expected.shape_dims(), label + " shape");
    }

    private static ulong Ordered64(ulong bits) => (bits & 0x8000000000000000UL) != 0 ? ~bits : bits | 0x8000000000000000UL;
    private static uint Ordered32(uint bits) => (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
    private static ulong Distance(ulong a, ulong b) => a >= b ? a - b : b - a;
}
