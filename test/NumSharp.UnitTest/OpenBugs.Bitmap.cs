using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using FluentAssertions;
using NumSharp;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Open bugs in the NumSharp.Bitmap extension library (np_.extensions.cs).
    ///
    ///     Each test asserts the CORRECT expected behavior.
    ///     Tests FAIL while the bug exists, and PASS when the bug is fixed.
    ///
    ///     When a bug is fixed, the test starts passing. Move passing tests to
    ///     BitmapExtensionsTests.cs.
    ///
    ///     =====================================================================
    ///     ROOT CAUSE ANALYSIS
    ///     =====================================================================
    ///
    ///     The majority of Bitmap bugs stem from TWO architectural problems:
    ///
    ///     1. STRIDE PADDING NOT HANDLED IN COPY PATH
    ///        GDI+ aligns bitmap scan lines to 4-byte boundaries, so
    ///        stride >= width * bytesPerPixel. The no-copy path uses
    ///        ReshapeFlatData() which correctly handles stride != width * bpp
    ///        by reshaping to strideWidth then slicing. But the copy path
    ///        (lines 44-54) copies stride*height bytes then reshapes using
    ///        stride/width as bpp — which is WRONG when stride includes
    ///        padding. For a 3px wide 24bpp image: stride=12, width=3,
    ///        stride/width=4, so it reshapes as 4-channel instead of 3.
    ///
    ///     2. ToBitmap STRIDE PADDING CONCAT LOGIC IS BROKEN
    ///        When bitdata.Stride != width * bpp (line 242), ToBitmap tries
    ///        to fix it by concatenating a zeros column. But this changes
    ///        nd.size while the earlier size check (line 237) already passed
    ///        for the original size. The concat also doesn't match the actual
    ///        stride padding needed — GDI+ pads to 4-byte boundaries per row,
    ///        not by adding a whole extra pixel column.
    ///
    ///     Bug categories:
    ///       Bugs 1, 2:     Stride padding not handled in copy path (ToNDArray)
    ///       Bug 3:         AsNDArray accesses shape[3] before reshaping flat array
    ///       Bug 4:         ToBitmap non-contiguous sliced array fails in MultiIterator
    ///       Bug 5:         ToBitmap no dtype validation (only byte arrays work)
    ///       Bugs 6, 7:     ToBitmap stride padding concat logic crashes
    ///       Bug 8:         ToBitmap 24bpp round-trip with even narrow widths corrupts data
    ///
    ///     GitHub issues: #396, #440, #475, #491
    ///
    ///     Total: 8 distinct bugs, 10 test methods.
    /// </summary>
    [TestClass]
    public class OpenBugsBitmap : TestClass
    {
        [ClassInitialize]
        public static void RequireWindows(TestContext _)
        {
            if (!OperatingSystem.IsWindows())
                Assert.Inconclusive("System.Drawing.Common requires Windows (GDI+).");
        }

        // ================================================================
        //
        //  BUG 1: ToNDArray(copy:true) on odd-width 24bpp images produces
        //         wrong shape — treats stride padding as extra channels
        //
        //  SEVERITY: High — silently returns wrong shape and data layout.
        //
        //  GITHUB: #396 (Bitmap.ToNDArray problem with odd bitmap width)
        //
        //  ROOT CAUSE: The copy path (line 54 of np_.extensions.cs) computes
        //  bytes-per-pixel as `bmpData.Stride / bmpData.Width`. For a 3px
        //  wide 24bpp image: stride=12 (9 bytes padded to 4-byte boundary),
        //  so stride/width = 12/3 = 4. The reshape then creates shape
        //  (1, H, W, 4) instead of (1, H, W, 3). The no-copy path uses
        //  ReshapeFlatData() which handles this correctly.
        //
        //  For wider odd-width images (e.g. 227px), the stride/width division
        //  truncates to 2 instead of 3, causing the reshape to fail entirely
        //  with IncorrectShapeException because stride*height bytes don't
        //  divide evenly into (1, H, W, 2).
        //
        //  VERIFICATION:
        //    3px wide 24bpp: stride=12, stride/width=4, shape becomes (1,H,3,4)
        //    227px wide 24bpp: stride=684, stride/width=3 (happens to work)
        //    5px wide 24bpp: stride=16, stride/width=3 (but 16*H != H*5*3)
        //
        // ================================================================

        /// <summary>
        ///     BUG 1a: ToNDArray(copy:true) on 3px-wide 24bpp image.
        ///     stride=12, width=3, stride/width=4 → shape (1,2,3,4) instead of (1,2,3,3).
        ///     The image has 3 channels (RGB) but gets reshaped as 4 channels.
        /// </summary>
        [TestMethod]
        public void Bug1a_ToNDArray_CopyTrue_OddWidth24bpp_WrongShape()
        {
            var bmp = new Bitmap(3, 2, PixelFormat.Format24bppRgb);
            bmp.SetPixel(0, 0, Color.FromArgb(10, 20, 30));
            bmp.SetPixel(1, 0, Color.FromArgb(40, 50, 60));
            bmp.SetPixel(2, 0, Color.FromArgb(70, 80, 90));

            var nd = bmp.ToNDArray(copy: true, discardAlpha: false);

            // Expected: shape should be (1, 2, 3, 3) for 24bpp RGB
            // Actual: shape is (1, 2, 3, 4) because stride/width = 12/3 = 4
            nd.shape[3].Should().Be(3,
                "a 24bpp RGB image has 3 bytes per pixel, not 4. " +
                "The copy path uses stride/width which includes padding bytes.");
        }

        /// <summary>
        ///     BUG 1b: ToNDArray(copy:true) on 5px-wide 24bpp image.
        ///     stride=16 (5*3=15 padded to 16), width=5, stride/width=3.
        ///     The division happens to give 3 (correct bpp), but stride*height=32
        ///     does not equal height*width*3=30, so the reshape may produce wrong data.
        /// </summary>
        [TestMethod]
        public void Bug1b_ToNDArray_CopyTrue_5pxWide24bpp_ExtraPaddingBytes()
        {
            var bmp = new Bitmap(5, 2, PixelFormat.Format24bppRgb);
            for (int x = 0; x < 5; x++)
            {
                bmp.SetPixel(x, 0, Color.FromArgb(10 + x, 20 + x, 30 + x));
                bmp.SetPixel(x, 1, Color.FromArgb(50 + x, 60 + x, 70 + x));
            }

            var nd = bmp.ToNDArray(copy: true, discardAlpha: false);

            // stride=16, height=2 → 32 bytes copied. But 1*2*5*3 = 30.
            // The reshape(1, 2, 5, 3) should work since stride/width = 16/5 = 3.
            // But the NDArray has 32 bytes, and reshape expects 30.
            nd.Should().BeShaped(1, 2, 5, 3);
            nd.size.Should().Be(30,
                "5x2 24bpp image has 30 pixels total (5*2*3), " +
                "but copy path copies stride*height=32 bytes including padding.");
        }

        // ================================================================
        //
        //  BUG 2: ToNDArray(copy:true) on 24bpp with even narrow widths
        //         produces wrong pixel data — stride padding corrupts layout
        //
        //  SEVERITY: High — pixel data is silently wrong.
        //
        //  GITHUB: #440 (ToBitmap has critical issue with 24bpp vertical images)
        //
        //  ROOT CAUSE: Same stride/width issue. For 2px wide 24bpp:
        //  stride=8 (2*3=6 padded to 8), stride/width=8/2=4.
        //  The reshape produces (1, H, 2, 4) — 4 channels per pixel when
        //  there are only 3. The round-trip then fails because ToBitmap
        //  reads back the wrong channel layout.
        //
        // ================================================================

        /// <summary>
        ///     BUG 2: ToNDArray(copy:true) on 2px-wide 24bpp → wrong bpp.
        ///     stride=8, width=2, stride/width=4 → shape (1,2,2,4) instead of (1,2,2,3).
        ///     Round-trip pixel values are corrupted because channel boundaries shift.
        /// </summary>
        [TestMethod]
        public void Bug2_ToNDArray_CopyTrue_2pxWide24bpp_WrongBpp()
        {
            var bmp = new Bitmap(2, 2, PixelFormat.Format24bppRgb);
            bmp.SetPixel(0, 0, Color.FromArgb(10, 20, 30));
            bmp.SetPixel(1, 0, Color.FromArgb(40, 50, 60));
            bmp.SetPixel(0, 1, Color.FromArgb(70, 80, 90));
            bmp.SetPixel(1, 1, Color.FromArgb(100, 110, 120));

            var nd = bmp.ToNDArray(copy: true, discardAlpha: false);

            // Expected: 3 channels for 24bpp
            // Actual: 4 channels because stride/width = 8/2 = 4
            nd.shape[3].Should().Be(3,
                "a 24bpp image has 3 bytes per pixel. The copy path computes " +
                "stride/width = 8/2 = 4, treating stride padding as a 4th channel.");
        }

        // ================================================================
        //
        //  BUG 3: AsNDArray(flat:true) crashes with IndexOutOfRangeException
        //
        //  SEVERITY: Medium — crashes instead of returning flat data.
        //
        //  ROOT CAUSE: AsNDArray (line 187-189) checks `ret.shape[3]` to
        //  decide whether to discard alpha. But at that point, `ret` is a
        //  flat 1-d array (just wrapped from Scan0), so it only has 1
        //  dimension. Accessing shape[3] on a 1-d array throws
        //  IndexOutOfRangeException. The reshape to 4-d should happen
        //  BEFORE the shape[3] check, not after.
        //
        // ================================================================

        /// <summary>
        ///     BUG 3a: AsNDArray(flat:true) crashes accessing shape[3] on 1-d array.
        /// </summary>
        [TestMethod]
        public void Bug3a_AsNDArray_FlatTrue_IndexOutOfRange()
        {
            var bmp = new Bitmap(4, 4, PixelFormat.Format32bppArgb);
            var bmpData = bmp.LockBits(
                new Rectangle(0, 0, 4, 4),
                ImageLockMode.ReadOnly,
                bmp.PixelFormat);

            try
            {
                var nd = bmpData.AsNDArray(flat: true, discardAlpha: false);

                // Expected: flat 1-d array with all pixel bytes
                nd.ndim.Should().Be(1, "flat:true should return a 1-d array");
                nd.size.Should().Be(4 * 4 * 4, "4x4 32bpp = 64 bytes");
            }
            finally
            {
                bmp.UnlockBits(bmpData);
            }
        }

        /// <summary>
        ///     BUG 3b: AsNDArray(flat:true, discardAlpha:true) crashes the same way.
        ///     The discardAlpha path also accesses shape[3] before reshaping.
        /// </summary>
        [TestMethod]
        public void Bug3b_AsNDArray_FlatTrue_DiscardAlpha_IndexOutOfRange()
        {
            var bmp = new Bitmap(4, 4, PixelFormat.Format32bppArgb);
            var bmpData = bmp.LockBits(
                new Rectangle(0, 0, 4, 4),
                ImageLockMode.ReadOnly,
                bmp.PixelFormat);

            try
            {
                var nd = bmpData.AsNDArray(flat: true, discardAlpha: true);

                // Expected: flat 1-d array with RGB bytes (alpha discarded)
                nd.ndim.Should().Be(1, "flat:true should return a 1-d array");
                nd.size.Should().Be(4 * 4 * 3, "4x4 with alpha discarded = 48 bytes");
            }
            finally
            {
                bmp.UnlockBits(bmpData);
            }
        }

        // ================================================================
        //
        //  BUG 4: ToBitmap fails on non-contiguous (sliced) NDArray
        //
        //  SEVERITY: Medium — slicing before ToBitmap always crashes.
        //
        //  GITHUB: #475 (ToBitmap fails if not contiguous because of
        //          Broadcast mismatch)
        //
        //  ROOT CAUSE: When Shape.IsContiguous is false (line 248), ToBitmap
        //  falls through to MultiIterator.Assign(). This creates an
        //  UnmanagedStorage with Shape.Vector(stride*height) and tries to
        //  broadcast it against the sliced NDArray's storage. The shapes
        //  don't broadcast: a flat (N,) vector can't broadcast against a
        //  (1, H, W, C) array, causing IncorrectShapeException.
        //
        //  The fix should either copy the sliced NDArray to contiguous memory
        //  first, or use an iterator that doesn't require broadcasting.
        //
        // ================================================================

        /// <summary>
        ///     BUG 4: ToBitmap with sliced (non-contiguous) NDArray throws
        ///     IncorrectShapeException from MultiIterator broadcast.
        /// </summary>
        [TestMethod]
        public void Bug4_ToBitmap_SlicedNDArray_BroadcastMismatch()
        {
            var full = np.arange(0, 4 * 4 * 4).reshape(1, 4, 4, 4).astype(NPTypeCode.Byte);
            var sliced = full[":, :2, :2, :"];

            sliced.shape.Should().BeEquivalentTo(new[] { 1, 2, 2, 4 });
            sliced.Shape.IsContiguous.Should().BeFalse("slicing makes it non-contiguous");

            // Expected: ToBitmap should handle non-contiguous arrays
            // Actual: throws IncorrectShapeException from MultiIterator.Assign
            var bmp = sliced.ToBitmap();
            bmp.Width.Should().Be(2);
            bmp.Height.Should().Be(2);
        }

        // ================================================================
        //
        //  BUG 5: ToBitmap with non-byte dtype throws InvalidCastException
        //
        //  SEVERITY: Medium — common user error with no helpful message.
        //
        //  GITHUB: #491 (ToBitmap() - datatype mismatch)
        //
        //  ROOT CAUSE: ToBitmap creates an ArraySlice<byte> for the
        //  destination (line 245), then calls nd.CopyTo(dst) which does a
        //  typed copy. If nd.dtype is int32 (or any non-byte type), the
        //  CopyTo fails with InvalidCastException. There's no dtype check
        //  at the top of ToBitmap to give a clear error message, and no
        //  automatic .astype(NPTypeCode.Byte) conversion.
        //
        //  At minimum, ToBitmap should throw ArgumentException with a
        //  message saying the NDArray must be of byte dtype. Ideally it
        //  should auto-convert with .astype(NPTypeCode.Byte).
        //
        // ================================================================

        /// <summary>
        ///     BUG 5: ToBitmap with int32 NDArray throws cryptic InvalidCastException
        ///     instead of a helpful error or auto-converting to byte.
        /// </summary>
        [TestMethod]
        public void Bug5_ToBitmap_NonByteDtype_InvalidCastException()
        {
            var nd = np.arange(0, 1 * 3 * 4 * 3).reshape(1, 3, 4, 3); // int32
            nd.dtype.Should().Be(typeof(int));

            // Expected: either auto-convert to byte, or throw helpful ArgumentException
            // Actual: throws InvalidCastException "Unable to perform CopyTo when T does not match dtype"
            Action act = () => nd.ToBitmap();
            act.Should().NotThrow<InvalidCastException>(
                "ToBitmap should either auto-convert to byte or throw a clear " +
                "ArgumentException explaining that only byte arrays are supported.");
        }

        // ================================================================
        //
        //  BUG 6: ToBitmap crashes on images where stride != width * bpp
        //         (1px wide, 5px wide, and other widths where GDI+ pads)
        //
        //  SEVERITY: High — many common image sizes crash.
        //
        //  GITHUB: #440 (ToBitmap critical issue with 24bpp vertical images)
        //
        //  ROOT CAUSE: When bitdata.Stride != width * format.ToBytesPerPixel()
        //  (line 242), ToBitmap tries to "fix" it by concatenating a zeros
        //  column: np.concatenate((nd, np.zeros(...)), axis=2). This is
        //  fundamentally wrong:
        //
        //    1. GDI+ pads rows to 4-byte boundaries, not by adding pixel columns.
        //       A 5px wide 24bpp row is 15 bytes, padded to 16 — that's 1 byte
        //       of padding, not 3 bytes (one pixel column).
        //
        //    2. The concatenation changes nd.size, but the earlier size check
        //       (line 237) already passed for the original unpadded size.
        //
        //    3. After concatenation, the CopyTo source (now larger) overflows
        //       the destination buffer, causing ArgumentOutOfRangeException.
        //
        //  The correct fix: copy row-by-row, writing width*bpp bytes per row
        //  to a destination with stride bytes per row (padding the gap with zeros).
        //
        // ================================================================

        /// <summary>
        ///     BUG 6a: ToBitmap crashes on 1px wide 24bpp image.
        ///     stride=4 (1*3=3, padded to 4), width*bpp=3 → concat triggers.
        /// </summary>
        [TestMethod]
        public void Bug6a_ToBitmap_1pxWide24bpp_StridePaddingCrash()
        {
            var nd = np.ones(1, 2, 1, 3).astype(NPTypeCode.Byte);
            nd.size.Should().Be(6);

            // Expected: creates a 1x2 24bpp bitmap with all-ones pixels
            // Actual: ArgumentOutOfRangeException from CopyTo after broken concat
            var bmp = nd.ToBitmap();
            bmp.Width.Should().Be(1);
            bmp.Height.Should().Be(2);

            var p0 = bmp.GetPixel(0, 0);
            var p1 = bmp.GetPixel(0, 1);
            // GDI+ stores BGR, so RGB(1,1,1) → GetPixel returns (1,1,1)
            p0.B.Should().Be(1);
            p1.B.Should().Be(1);
        }

        /// <summary>
        ///     BUG 6b: ToBitmap crashes on 5px wide 24bpp image.
        ///     stride=16 (5*3=15, padded to 16), width*bpp=15 → concat triggers.
        /// </summary>
        [TestMethod]
        public void Bug6b_ToBitmap_5pxWide24bpp_StridePaddingCrash()
        {
            var nd = np.ones(1, 2, 5, 3).astype(NPTypeCode.Byte);
            nd.size.Should().Be(30);

            // Expected: creates a 5x2 24bpp bitmap
            // Actual: ArgumentOutOfRangeException from CopyTo after broken concat
            var bmp = nd.ToBitmap();
            bmp.Width.Should().Be(5);
            bmp.Height.Should().Be(2);

            for (int x = 0; x < 5; x++)
            {
                var p = bmp.GetPixel(x, 0);
                p.B.Should().Be(1, $"pixel ({x},0) blue channel should be 1");
            }
        }

        // ================================================================
        //
        //  BUG 7: ToNDArray(copy:true) 24bpp round-trip corrupts pixel data
        //         on narrow even-width images (2px wide)
        //
        //  SEVERITY: High — silent data corruption.
        //
        //  ROOT CAUSE: For 2px wide 24bpp: stride=8 (2*3=6, padded to 8).
        //  stride/width = 8/2 = 4. The copy path reshapes to (1, H, W, 4),
        //  treating the 2 padding bytes as a 4th channel. When this is
        //  round-tripped through ToBitmap, the pixel data boundaries are
        //  shifted by the phantom 4th channel, producing wrong colors.
        //
        //  Example: SetPixel(1,0) = RGB(40,50,60). In memory (BGR): 60,50,40.
        //  With stride padding, row 0 = [30,20,10, 60,50,40, pad,pad].
        //  Reshape as 4-channel: pixel(0,0)=[30,20,10,60], pixel(0,1)=[50,40,pad,pad].
        //  Round-trip reads pixel(0,1) as RGB(pad,50,40) → wrong.
        //
        // ================================================================

        /// <summary>
        ///     BUG 7: Round-trip 24bpp 2px wide — pixel values corrupted.
        /// </summary>
        [TestMethod]
        public void Bug7_ToNDArray_CopyTrue_2pxWide24bpp_RoundTripCorruption()
        {
            var bmp = new Bitmap(2, 2, PixelFormat.Format24bppRgb);
            bmp.SetPixel(0, 0, Color.FromArgb(10, 20, 30));
            bmp.SetPixel(1, 0, Color.FromArgb(40, 50, 60));
            bmp.SetPixel(0, 1, Color.FromArgb(70, 80, 90));
            bmp.SetPixel(1, 1, Color.FromArgb(100, 110, 120));

            var nd = bmp.ToNDArray(copy: true, discardAlpha: false);
            var bmp2 = nd.ToBitmap();

            // Verify round-trip preserves pixel values
            var orig_1_0 = bmp.GetPixel(1, 0);  // (40, 50, 60)
            var recovered_1_0 = bmp2.GetPixel(1, 0);

            recovered_1_0.R.Should().Be(orig_1_0.R, "R channel of pixel (1,0) should survive round-trip");
            recovered_1_0.G.Should().Be(orig_1_0.G, "G channel of pixel (1,0) should survive round-trip");
            recovered_1_0.B.Should().Be(orig_1_0.B, "B channel of pixel (1,0) should survive round-trip");
        }

        // ================================================================
        //
        //  BUG 8: extractFormatNumber() silently falls through for
        //         unsupported bbp values, leaving format as DontCare
        //
        //  SEVERITY: Low — the Bitmap constructor catches this, but the
        //  error message is unhelpful ("Parameter is not valid" from GDI+).
        //
        //  ROOT CAUSE: The switch in extractFormatNumber() (line 267-286)
        //  only handles bbp 3, 4, 6, 8. For any other value (1, 2, 5, etc.),
        //  format stays DontCare and the function returns bbp. Then
        //  `new Bitmap(width, height, format)` with DontCare throws a
        //  generic GDI+ error. The code should throw a descriptive
        //  ArgumentException before reaching the Bitmap constructor.
        //
        // ================================================================

        /// <summary>
        ///     BUG 8: ToBitmap with 2-channel NDArray throws unhelpful GDI+ error.
        ///     extractFormatNumber() doesn't handle bbp=2, leaving format as DontCare.
        /// </summary>
        [TestMethod]
        public void Bug8_ToBitmap_UnsupportedBpp_UnhelpfulError()
        {
            var nd = np.zeros(1, 2, 2, 2).astype(NPTypeCode.Byte);

            // Expected: descriptive ArgumentException about unsupported channel count
            // Actual: "Parameter is not valid" from new Bitmap(w, h, DontCare)
            Action act = () => nd.ToBitmap();
            act.Should().Throw<ArgumentException>()
                .And.Message.Should().Contain("2",
                    "the error message should mention the unsupported byte-per-pixel " +
                    "count to help the user understand what went wrong, instead of " +
                    "a generic GDI+ 'Parameter is not valid' error.");
        }
    }
}
