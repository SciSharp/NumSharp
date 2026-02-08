# #467: NumSharp and Tensorflow.NET works on Desktop but fails on Cloud Web Service (.NET 5)

- **URL:** https://github.com/SciSharp/NumSharp/issues/467
- **State:** OPEN
- **Author:** @marsousi
- **Created:** 2021-12-06T04:56:23Z
- **Updated:** 2021-12-06T04:56:23Z

## Description

NumSharp and Tensorflow.NET work fine on my desktop computer. But once I publish it on the cloud service (Azure Web Service using ASP.NET Core - .NET 5), even running a simple code to define an NDArray gives the following error:

<b>_DllNotFoundException: Unable to load DLL 'tensorflow' or one of its dependencies: The specified module could not be found. (0x8007007E)
Tensorflow.c_api.TF_NewStatus()

TypeInitializationException: The type initializer for 'Tensorflow.Binding' threw an exception.
Tensorflow.Binding.get_tf()_ </b>

The following packages are installed in Visual Studio Solution:

<b>
SciSharp.TensorFlow.Redist (2.3.1) <br>
NumSharp (0.30.0)<br> 
NumSharp.Bitmap (0.30.0)<br>
SharpCV (0.10.1)<br>
Rensorflow.Net (0.60.5)<br>
Tensorflow.Keras (0.6.5)<br>
Google.Protobuf (3.19.1)<br>
Protobuf.Text (0.5.0)<br>
Serilog.Sinks.Console (4.0.1)<br>

<i>and some more but I don't think they would interfere with the above packages</i>
</b>

Besides, I tried to change the Release CPU from AnyCPU to x64, but then the cloud service fails running. 

Am I missing something? 


