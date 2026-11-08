namespace NumSharpOnnxRuntimeExample;

/// <summary>
///     The model, the class names and the sample photo the example runs on, fetched on the first run into a
///     <c>data</c> folder under the working directory and reused afterwards.
/// </summary>
internal static class Downloads
{
    /// <summary>MobileNetV2 (opset 12) from the ONNX model zoo: input <c>(batch_size, 3, 224, 224)</c> float32, output <c>(batch_size, 1000)</c> logits.</summary>
    public const string MobileNetV2 = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx";

    /// <summary>The 1,000 ImageNet class names, one per line, in the order of the models' outputs.</summary>
    public const string ImageNetLabels = "https://github.com/onnx/models/raw/main/validated/vision/classification/synset.txt";

    /// <summary>A sample photo (a Samoyed) from PyTorch Hub's examples.</summary>
    public const string SamplePhoto = "https://github.com/pytorch/hub/raw/master/images/dog.jpg";

    /// <summary>
    ///     The local path of <paramref name="fileName"/> in <c>./data</c>, downloading it from <paramref name="url"/>
    ///     first when it is not there yet.
    /// </summary>
    /// <remarks>
    ///     The download goes to a temporary name and is renamed only once it is complete, so an interrupted run leaves
    ///     no truncated model behind that a later run would try to load.
    /// </remarks>
    /// <param name="fileName">The file's name in the cache folder.</param>
    /// <param name="url">Where to download it from.</param>
    /// <returns>The full path of the cached file.</returns>
    /// <exception cref="HttpRequestException">The download failed (no network, or the server refused it).</exception>
    public static async Task<string> FetchAsync(string fileName, string url)
    {
        string folder = Path.Combine(Directory.GetCurrentDirectory(), "data");
        string path = Path.Combine(folder, fileName);
        if (File.Exists(path))
            return path;

        Directory.CreateDirectory(folder);
        Console.WriteLine($"downloading {fileName} ...");
        using var http = new HttpClient();
        byte[] content = await http.GetByteArrayAsync(url);
        string partial = path + ".part";
        await File.WriteAllBytesAsync(partial, content);
        File.Move(partial, path, overwrite: true);
        return path;
    }
}
