using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Validates a <c>device</c> argument for the Array-API creation functions (matches NumPy 2.x).
        ///     NumSharp — like NumPy — is single-device: only <c>"cpu"</c> and <c>null</c> (the default,
        ///     meaning "leave it to the library") are accepted. Anything else raises with NumPy's verbatim
        ///     creation-path message.
        /// </summary>
        /// <param name="device">The requested device. <c>null</c> is accepted and means "default".</param>
        /// <exception cref="ArgumentException">If <paramref name="device"/> is neither <c>null</c> nor <c>"cpu"</c>.</exception>
        /// <remarks>
        ///     Port of NumPy's <c>PyArray_DeviceConverterOptional</c> (conversion_utils.c). The message quotes
        ///     <c>"cpu"</c> with DOUBLE quotes here; the <see cref="NDArray.to_device"/> path quotes it with
        ///     single quotes — the two messages are deliberately different, matching NumPy verbatim.
        /// </remarks>
        internal static void ValidateDevice(string device)
        {
            if (device != null && !string.Equals(device, "cpu", StringComparison.Ordinal))
                throw new ArgumentException(
                    $"Device not understood. Only \"cpu\" is allowed, but received: {device}",
                    nameof(device));
        }
    }
}
