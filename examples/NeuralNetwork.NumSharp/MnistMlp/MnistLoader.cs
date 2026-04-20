using System;
using System.IO;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.MnistMlp
{
    /// <summary>
    /// Loads MNIST from the standard IDX file format, with a synthetic fallback.
    ///
    /// The IDX format is big-endian:
    ///   images: [magic=0x00000803][count][rows][cols][row-major uint8 pixels]
    ///   labels: [magic=0x00000801][count][uint8 labels]
    ///
    /// If either file is missing, a deterministic synthetic dataset is returned
    /// so the experiment stays self-contained. Synthetic accuracy will be near
    /// chance (~10%); place real t10k-images.idx3-ubyte / t10k-labels.idx1-ubyte
    /// in the provided directory to evaluate trained weights against actual data.
    /// </summary>
    public static class MnistLoader
    {
        public const int ImageRows = 28;
        public const int ImageCols = 28;
        public const int ImageSize = ImageRows * ImageCols; // 784

        /// <summary>
        /// Reads images + labels from IDX files. Images are returned as
        /// (count, 784) float32 normalized to [0, 1]. Labels are (count,) uint8.
        /// Falls back to deterministic synthetic data if either file is absent.
        /// </summary>
        public static (NDArray images, NDArray labels, bool isSynthetic) LoadOrSynthesize(
            string imagePath, string labelPath, int syntheticCount, int seed)
        {
            bool realData = File.Exists(imagePath) && File.Exists(labelPath);
            if (realData)
            {
                var images = LoadImages(imagePath);
                var labels = LoadLabels(labelPath);
                return (images, labels, false);
            }

            return (Synthesize(syntheticCount, seed), SynthesizeLabels(syntheticCount, seed + 1), true);
        }

        private static NDArray LoadImages(string path)
        {
            byte[] raw = File.ReadAllBytes(path);
            if (raw.Length < 16)
                throw new InvalidDataException($"{path}: file too short to be an MNIST image IDX.");

            int magic = BigEndianInt32(raw, 0);
            if (magic != 0x00000803)
                throw new InvalidDataException($"{path}: bad IDX magic 0x{magic:X8} (expected 0x00000803).");

            int count = BigEndianInt32(raw, 4);
            int rows  = BigEndianInt32(raw, 8);
            int cols  = BigEndianInt32(raw, 12);
            int px    = rows * cols;
            int need  = 16 + count * px;
            if (raw.Length < need)
                throw new InvalidDataException($"{path}: truncated (have {raw.Length}, need {need}).");

            // Allocate contiguous float32 (count, rows*cols) and normalize to [0, 1].
            var arr = new NDArray(NPTypeCode.Single, new Shape(count, px), fillZeros: false);
            unsafe
            {
                float* dst = (float*)arr.Address;
                for (int i = 0; i < count; i++)
                {
                    int srcBase = 16 + i * px;
                    int dstBase = i * px;
                    for (int j = 0; j < px; j++)
                        dst[dstBase + j] = raw[srcBase + j] * (1f / 255f);
                }
            }
            return arr;
        }

        private static NDArray LoadLabels(string path)
        {
            byte[] raw = File.ReadAllBytes(path);
            if (raw.Length < 8)
                throw new InvalidDataException($"{path}: file too short to be an MNIST label IDX.");

            int magic = BigEndianInt32(raw, 0);
            if (magic != 0x00000801)
                throw new InvalidDataException($"{path}: bad IDX magic 0x{magic:X8} (expected 0x00000801).");

            int count = BigEndianInt32(raw, 4);
            int need  = 8 + count;
            if (raw.Length < need)
                throw new InvalidDataException($"{path}: truncated (have {raw.Length}, need {need}).");

            var arr = new NDArray(NPTypeCode.Byte, new Shape(count), fillZeros: false);
            unsafe
            {
                byte* dst = (byte*)arr.Address;
                for (int i = 0; i < count; i++)
                    dst[i] = raw[8 + i];
            }
            return arr;
        }

        private static NDArray Synthesize(int count, int seed)
        {
            var arr = new NDArray(NPTypeCode.Single, new Shape(count, ImageSize), fillZeros: false);
            var rng = new Random(seed);
            unsafe
            {
                float* dst = (float*)arr.Address;
                long n = (long)count * ImageSize;
                for (long i = 0; i < n; i++)
                    dst[i] = (float)rng.NextDouble();
            }
            return arr;
        }

        private static NDArray SynthesizeLabels(int count, int seed)
        {
            var arr = new NDArray(NPTypeCode.Byte, new Shape(count), fillZeros: false);
            var rng = new Random(seed);
            unsafe
            {
                byte* dst = (byte*)arr.Address;
                for (int i = 0; i < count; i++)
                    dst[i] = (byte)rng.Next(10);
            }
            return arr;
        }

        private static int BigEndianInt32(byte[] buf, int offset)
            => (buf[offset] << 24) | (buf[offset + 1] << 16) | (buf[offset + 2] << 8) | buf[offset + 3];
    }
}
