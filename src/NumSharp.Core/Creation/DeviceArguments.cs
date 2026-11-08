using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>PyArray_DeviceConverterOptional</c> (<c>conversion_utils.c</c>): the check of the Array-API
    ///     <c>device</c> argument that the creation functions, np.asarray / np.asanyarray, the engine's linspace /
    ///     logspace / geomspace and np.fft.fftfreq / rfftfreq share. One function: it lives outside the np facade class
    ///     so the engine calls no np.* member (ratchet R2 of the np-layer dtype plan).
    /// </summary>
    internal static class DeviceArguments
    {
        /// <summary>
        ///     Validates a <c>device</c> argument for the Array-API creation functions (matches NumPy 2.x).
        ///     NumSharp — like NumPy — is single-device: only <c>"cpu"</c> and <c>null</c> (the default,
        ///     meaning "leave it to the library") are accepted. Anything else raises NumPy's ValueError with its verbatim
        ///     creation-path message.
        /// </summary>
        /// <param name="device">The requested device. <c>null</c> is accepted and means "default".</param>
        /// <exception cref="ValueError">If <paramref name="device"/> is neither <c>null</c> nor <c>"cpu"</c>:
        ///     <c>Device not understood. Only "cpu" is allowed, but received: gpu</c>, the text alone (an
        ///     <see cref="ArgumentException"/> given a parameter name appends <c> (Parameter 'device')</c>, which NumPy's text
        ///     does not have).</exception>
        /// <remarks>
        ///     Port of NumPy's <c>PyArray_DeviceConverterOptional</c> (conversion_utils.c). The message quotes
        ///     <c>"cpu"</c> with DOUBLE quotes here; the <see cref="NDArray.to_device"/> path quotes it with
        ///     single quotes — the two messages are deliberately different, matching NumPy verbatim.
        /// </remarks>
        internal static void ValidateDevice(string device)
        {
            if (device != null && !string.Equals(device, "cpu", StringComparison.Ordinal))
                throw new ValueError($"Device not understood. Only \"cpu\" is allowed, but received: {device}");
        }
    }
}
