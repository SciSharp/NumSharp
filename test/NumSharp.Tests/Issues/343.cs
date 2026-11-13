using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Reflection;
using System.Runtime.Versioning;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #343, "Built-in System.Drawing.Image and Bitmap methods"
    ///     (https://github.com/SciSharp/NumSharp/issues/343): the methods ship as NumSharp.Bitmap, for .NET 8 / .NET 10 on
    ///     Windows, with <c>Bitmap.ToNDArray</c>, <c>Image.ToNDArray</c>, <c>BitmapData.AsNDArray</c> (no copy) and
    ///     <c>NDArray.ToBitmap</c>. Its known bugs (#396, #440, #475, #491) are pinned in <see cref="OpenBugsBitmap"/>.
    /// </summary>
    [TestClass]
    [WindowsOnly]
    public class Issue343
    {
        /// <summary>
        ///     The four extension methods the answer names exist on NumSharp.Bitmap's <c>np_</c> class, which is marked
        ///     Windows-only, in an assembly built for .NET 8 or .NET 10.
        /// </summary>
        [TestMethod]
        public void NumSharpBitmap_ShipsTheFourMethods_WindowsOnly()
        {
            var t = typeof(np_);
            var methods = t.GetMethods(BindingFlags.Public | BindingFlags.Static);
            bool Has(string name, System.Type receiver) =>
                methods.Any(m => m.Name == name && m.GetParameters().FirstOrDefault()?.ParameterType == receiver);

            Has("ToNDArray", typeof(Bitmap)).Should().BeTrue();
            Has("ToNDArray", typeof(Image)).Should().BeTrue();
            Has("AsNDArray", typeof(BitmapData)).Should().BeTrue();
            Has("ToBitmap", typeof(NDArray)).Should().BeTrue();

            t.GetCustomAttribute<SupportedOSPlatformAttribute>()?.PlatformName.Should().Be("windows");
            t.Assembly.GetCustomAttribute<TargetFrameworkAttribute>().FrameworkName
                .Should().Match(f => f == ".NETCoreApp,Version=v8.0" || f == ".NETCoreApp,Version=v10.0");
        }

        /// <summary>
        ///     <c>BitmapData.AsNDArray</c> wraps the locked pixels without copying: a write through the array is a write into
        ///     the bitmap's buffer.
        /// </summary>
        [TestMethod]
        public void AsNDArray_ViewsTheLockedPixels()
        {
            using var bmp = new Bitmap(4, 2, PixelFormat.Format32bppArgb);
            var data = bmp.LockBits(new Rectangle(0, 0, 4, 2), ImageLockMode.ReadWrite, bmp.PixelFormat);
            try
            {
                var nd = data.AsNDArray();
                nd.shape.Should().Equal(1L, 2L, 4L, 4L);
                nd.SetByte(77, 0, 1, 3, 2);
                unsafe
                {
                    // Row 1, pixel 3, byte 2 of a 32bpp image: offset stride*1 + 3*4 + 2.
                    ((byte*)data.Scan0)[data.Stride + 3 * 4 + 2].Should().Be(77);
                }
            }
            finally
            {
                bmp.UnlockBits(data);
            }
        }
    }
}
