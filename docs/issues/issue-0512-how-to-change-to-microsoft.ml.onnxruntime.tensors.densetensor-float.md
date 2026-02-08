# #512: how to change to Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>?

- **URL:** https://github.com/SciSharp/NumSharp/issues/512
- **State:** OPEN
- **Author:** @BingGitCn
- **Created:** 2024-04-01T09:13:01Z
- **Updated:** 2024-04-01T09:13:01Z

## Description

 public static void ExtractPixelsArgb(DenseTensor<float> tensor, Span<byte> data, int pixelCount)
 {
     Span<float> spanR = tensor.Buffer.Span;
     Span<float> spanG = spanR[pixelCount..];
     Span<float> spanB = spanG[pixelCount..];

     int sidx = 0;
     for (int i = 0; i < pixelCount; i++)
     {
         spanR[i] = data[sidx + 2] * 0.0039215686274509803921568627451f;
         spanG[i] = data[sidx + 1] * 0.0039215686274509803921568627451f;
         spanB[i] = data[sidx] * 0.0039215686274509803921568627451f;
         sidx += 4;
     }
 }
Is there a faster way than this?
