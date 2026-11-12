using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Interop.TorchSharp
{
    /// <summary>
    ///     The LibTorchSharp natives the bridge calls itself: TorchSharp exposes no public way to wrap memory it did not
    ///     allocate (its sharing factories take a managed <see cref="Array"/> only, and its <c>Memory&lt;T&gt;</c> overloads
    ///     copy), nor to take a reference on a tensor's storage without a managed wrapper that a <c>DisposeScope</c> could
    ///     dispose under a NumSharp view.
    /// </summary>
    /// <remarks>
    ///     <para><b>Which library.</b> <c>LibTorchSharp</c> is the C shim the managed <c>TorchSharp</c> package ships (its
    ///     <c>runtimes/&lt;rid&gt;/native</c> asset), the same library TorchSharp's own <c>[DllImport("LibTorchSharp")]</c>
    ///     declarations bind. <see cref="EnsureLoaded"/> lets TorchSharp load it first, with its own probing (it preloads
    ///     libtorch and, for CUDA, its dependencies), so the name binds to the module already in the process.</para>
    ///     <para><b>One ABI changed.</b> <c>THSTensor_new</c> (libtorch's <c>from_blob</c> with a deleter) took six arguments in
    ///     TorchSharp 0.100.x and takes nine from 0.101.0 on (<c>dtype</c>, <c>device_type</c> and <c>device_index</c> were
    ///     inserted before <c>requires_grad</c>). Calling the wrong form is not an error the runtime can catch: the six-argument
    ///     library read <c>dtype</c> as <c>requires_grad</c>, and the nine-argument library read garbage device arguments.
    ///     <see cref="Abi"/> reads the argument count of TorchSharp's OWN declaration, which always ships with the native
    ///     library it binds, and <see cref="NewTensor"/> calls the matching form. The other natives kept their signatures
    ///     from 0.100.0 to 0.107.0 (read from each package's metadata).</para>
    /// </remarks>
    internal static unsafe class LibTorchSharpNative
    {
        /// <summary>The native library's name, exactly as TorchSharp's own declarations spell it.</summary>
        private const string Library = "LibTorchSharp";

        /// <summary>
        ///     libtorch's <c>void (*deleter)(void*)</c>: called with the data pointer once, on whichever thread drops the
        ///     storage's last reference (the caller's <c>Dispose</c>, a <c>DisposeScope</c>, the GC finalizer thread through
        ///     TorchSharp's <c>~Tensor</c>, or a libtorch worker).
        /// </summary>
        /// <param name="data">The pointer the storage was created over.</param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void Deleter(IntPtr data);

        /// <summary>
        ///     The six-argument <c>THSTensor_new</c> of TorchSharp 0.100.x: <c>from_blob(data, sizes, deleter, dtype)</c> with
        ///     <c>requires_grad</c>.
        /// </summary>
        /// <param name="data">The first element's address.</param>
        /// <param name="deleter">The function pointer libtorch calls when the storage dies.</param>
        /// <param name="sizes">The dimensions (<c>const int64_t*</c>).</param>
        /// <param name="szlength">The number of dimensions (0 for a 0-d tensor).</param>
        /// <param name="scalarType">The element type (c10's <c>ScalarType</c> ordinal).</param>
        /// <param name="requiresGrad">Whether the tensor tracks gradients.</param>
        /// <returns>An owned <c>at::Tensor*</c>, or <see cref="IntPtr.Zero"/> with the error recorded for <c>torch.CheckForErrors</c>.</returns>
        [DllImport(Library, EntryPoint = "THSTensor_new")]
        private static extern IntPtr THSTensor_new_Args6(IntPtr data, IntPtr deleter, IntPtr sizes, int szlength, sbyte scalarType,
            [MarshalAs(UnmanagedType.U1)] bool requiresGrad);

        /// <summary>
        ///     The nine-argument <c>THSTensor_new</c> of TorchSharp 0.101.0 and later: <c>from_blob</c>, then a native conversion
        ///     to <paramref name="dtype"/> and a move to the device when they differ (the bridge never asks for either).
        /// </summary>
        /// <param name="data">The first element's address.</param>
        /// <param name="deleter">The function pointer libtorch calls when the storage dies.</param>
        /// <param name="sizes">The dimensions (<c>const int64_t*</c>).</param>
        /// <param name="szlength">The number of dimensions (0 for a 0-d tensor).</param>
        /// <param name="scalarType">The element type of the memory.</param>
        /// <param name="dtype">The element type of the tensor (the bridge passes <paramref name="scalarType"/>: no conversion).</param>
        /// <param name="deviceType">The target device type (the bridge passes 0, CPU: no move).</param>
        /// <param name="deviceIndex">The target device index (-1, TorchSharp's CPU index).</param>
        /// <param name="requiresGrad">Whether the tensor tracks gradients.</param>
        /// <returns>An owned <c>at::Tensor*</c>, or <see cref="IntPtr.Zero"/> with the error recorded for <c>torch.CheckForErrors</c>.</returns>
        [DllImport(Library, EntryPoint = "THSTensor_new")]
        private static extern IntPtr THSTensor_new_Args9(IntPtr data, IntPtr deleter, IntPtr sizes, int szlength, sbyte scalarType,
            sbyte dtype, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requiresGrad);

        /// <summary>
        ///     <c>tensor-&gt;data_ptr()</c>: the address of the tensor's FIRST element (the storage's base plus
        ///     <c>storage_offset</c> elements). Null for a tensor without storage (sparse, meta), with the error recorded for a
        ///     sparse one and NOT for a meta one (measured on 0.107.0), so callers decide by the tensor's kind, never by this.
        /// </summary>
        /// <param name="tensor">An <c>at::Tensor*</c>.</param>
        /// <returns>The first element's address, or <see cref="IntPtr.Zero"/>.</returns>
        [DllImport(Library)]
        internal static extern IntPtr THSTensor_data(IntPtr tensor);

        /// <summary>
        ///     <c>new at::Tensor(tensor-&gt;detach())</c>: a NEW <c>TensorImpl</c> over the same storage. Unlike
        ///     <c>THSTensor_alias</c> (a second reference to the SAME <c>TensorImpl</c>), it keeps the memory alive through a
        ///     <c>set_</c> that rebinds the caller's tensor to other storage (measured: an alias's data pointer moved with it).
        /// </summary>
        /// <param name="tensor">An <c>at::Tensor*</c>.</param>
        /// <returns>An owned <c>at::Tensor*</c> to release with <see cref="THSTensor_dispose"/>, or <see cref="IntPtr.Zero"/> on error.</returns>
        [DllImport(Library)]
        internal static extern IntPtr THSTensor_detach(IntPtr tensor);

        /// <summary><c>delete tensor</c>: drops this reference; the storage dies with its last one.</summary>
        /// <param name="tensor">An owned <c>at::Tensor*</c>; never one a managed <c>torch.Tensor</c> still holds.</param>
        [DllImport(Library)]
        internal static extern void THSTensor_dispose(IntPtr tensor);

        /// <summary>The byte size of the tensor's whole storage (all of it stays alive while any view over it lives).</summary>
        /// <param name="tensor">An <c>at::Tensor*</c> with storage.</param>
        /// <returns><c>tensor-&gt;storage().nbytes()</c>.</returns>
        [DllImport(Library)]
        internal static extern ulong THSStorage_nbytes(IntPtr tensor);

        /// <summary>The <c>THSTensor_new</c> forms the bridge can call.</summary>
        internal enum NewTensorAbi
        {
            /// <summary>The loaded TorchSharp declares a form the bridge does not know: every export is refused.</summary>
            Unknown = 0,

            /// <summary>TorchSharp 0.100.x: <c>(data, deleter, sizes, szlength, scalar_type, requires_grad)</c>.</summary>
            Args6 = 6,

            /// <summary>TorchSharp 0.101.0 and later: <c>(data, deleter, sizes, szlength, scalar_type, dtype, device_type, device_index, requires_grad)</c>.</summary>
            Args9 = 9,
        }

        /// <summary>The detected form, or -1 before the first detection (racing detections compute the same answer).</summary>
        private static int _abi = -1;

        /// <summary>1 once <see cref="EnsureLoaded"/> ran TorchSharp's native initialization.</summary>
        private static int _loaded;

        /// <summary>
        ///     The <c>THSTensor_new</c> form the loaded TorchSharp binds, read once from TorchSharp's own declaration.
        /// </summary>
        internal static NewTensorAbi Abi
        {
            get
            {
                int abi = Volatile.Read(ref _abi);
                if (abi < 0)
                {
                    abi = (int)DetectAbi();
                    Volatile.Write(ref _abi, abi);
                }
                return (NewTensorAbi)abi;
            }
        }

        /// <summary>
        ///     Reads the parameter list of TorchSharp's internal <c>THSTensor_new</c> declaration (in
        ///     <c>TorchSharp.PInvoke.NativeMethods</c> from 0.101.0 on, in <c>TorchSharp.PInvoke.LibTorchSharp</c> in 0.100.x).
        /// </summary>
        /// <returns>
        ///     <see cref="NewTensorAbi.Args9"/> or <see cref="NewTensorAbi.Args6"/> when the declaration has exactly that
        ///     parameter list, otherwise <see cref="NewTensorAbi.Unknown"/>: a form with the same count but other types is not
        ///     guessed at, because calling a mismatched native signature corrupts arguments silently.
        /// </returns>
        internal static NewTensorAbi DetectAbi()
        {
            Assembly torchSharp = typeof(torch).Assembly;
            foreach (string typeName in new[] { "TorchSharp.PInvoke.NativeMethods", "TorchSharp.PInvoke.LibTorchSharp" })
            {
                Type declaring = torchSharp.GetType(typeName, throwOnError: false);
                if (declaring is null)
                    continue;
                foreach (MethodInfo method in declaring.GetMethods(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static))
                {
                    if (method.Name != "THSTensor_new")
                        continue;
                    ParameterInfo[] p = method.GetParameters();
                    // The deleter parameter is TorchSharp's internal GCHandleDeleter delegate: checked as "a delegate", since its
                    // name is TorchSharp's business; every other position must be exactly the type the native form takes.
                    bool head = p.Length >= 6 && p[0].ParameterType == typeof(IntPtr) && typeof(Delegate).IsAssignableFrom(p[1].ParameterType)
                                && p[2].ParameterType == typeof(IntPtr) && p[3].ParameterType == typeof(int) && p[4].ParameterType == typeof(sbyte);
                    if (!head)
                        continue;
                    if (p.Length == 9 && p[5].ParameterType == typeof(sbyte) && p[6].ParameterType == typeof(int)
                        && p[7].ParameterType == typeof(int) && p[8].ParameterType == typeof(bool))
                        return NewTensorAbi.Args9;
                    if (p.Length == 6 && p[5].ParameterType == typeof(bool))
                        return NewTensorAbi.Args6;
                }
            }
            return NewTensorAbi.Unknown;
        }

        /// <summary>
        ///     Lets TorchSharp load its native libraries before the bridge's first native call: touching <c>torch</c> runs its
        ///     static constructor (which tries the CUDA backend), and <c>InitializeDeviceType(CPU)</c> loads the CPU backend when
        ///     nothing is loaded yet - what TorchSharp's own factories do before their first native call. Cheap after the first
        ///     call (TorchSharp keeps a loaded flag).
        /// </summary>
        /// <exception cref="NotSupportedException">TorchSharp could not load libtorch (no runtime package is referenced, or a
        ///     32-bit process): TorchSharp's own message, with its load trace.</exception>
        internal static void EnsureLoaded()
        {
            if (Volatile.Read(ref _loaded) != 0)
                return;
            torch.InitializeDeviceType(DeviceType.CPU);
            Volatile.Write(ref _loaded, 1);
        }

        /// <summary>
        ///     <c>from_blob</c> over memory the caller keeps alive until <paramref name="deleter"/> runs: a CPU tensor of
        ///     <paramref name="scalarType"/> with the C-contiguous layout of <paramref name="dims"/>, no conversion, no device
        ///     move, no gradient tracking - the same call in both ABIs.
        /// </summary>
        /// <param name="data">The first element's address (any address for an empty tensor).</param>
        /// <param name="deleter">The native function pointer of a delegate the caller keeps alive until libtorch calls it.</param>
        /// <param name="dims">The dimensions; empty for a 0-d tensor.</param>
        /// <param name="scalarType">The element type the memory holds.</param>
        /// <returns>
        ///     An owned <c>at::Tensor*</c>, or <see cref="IntPtr.Zero"/> when libtorch refused (the error is then recorded for
        ///     <c>torch.CheckForErrors</c>). On a zero return the deleter may or may not have run: libtorch validates the sizes
        ///     before it takes ownership, but an allocation failure after it destroys the storage during unwinding.
        /// </returns>
        /// <exception cref="NotSupportedException">The loaded TorchSharp declares a <c>THSTensor_new</c> form this bridge does not know.</exception>
        internal static IntPtr NewTensor(void* data, IntPtr deleter, long[] dims, ScalarType scalarType)
        {
            NewTensorAbi abi = Abi;
            // A 0-d tensor passes szlength 0; the pointer must still be valid for `fixed`, so an empty array pins a dummy.
            fixed (long* sizes = dims.Length == 0 ? ZeroDims : dims)
            {
                switch (abi)
                {
                    case NewTensorAbi.Args9:
                        // dtype == scalar_type and device_type 0 (CPU): THSTensor_new then neither converts nor moves, so the
                        // returned tensor IS the from_blob tensor over `data`.
                        return THSTensor_new_Args9((IntPtr)data, deleter, (IntPtr)sizes, dims.Length, (sbyte)scalarType,
                            (sbyte)scalarType, 0, -1, false);
                    case NewTensorAbi.Args6:
                        return THSTensor_new_Args6((IntPtr)data, deleter, (IntPtr)sizes, dims.Length, (sbyte)scalarType, false);
                    default:
                        throw new NotSupportedException(
                            $"The loaded TorchSharp ({typeof(torch).Assembly.GetName().Version}) declares a THSTensor_new this bridge " +
                            "does not know (it knows the 6-argument form of 0.100.x and the 9-argument form of 0.101.0 and later), so " +
                            "it cannot hand libtorch a NumSharp buffer. Use a TorchSharp the bridge supports, or cross through torch's " +
                            "own factories (torch.tensor(nd.ToArray<T>(), shape)).");
                }
            }
        }

        /// <summary>A one-element array to pin for a 0-d tensor's sizes pointer (never read: szlength is 0).</summary>
        private static readonly long[] ZeroDims = { 0 };
    }
}
