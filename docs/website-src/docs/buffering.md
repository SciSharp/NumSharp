# Buffering, Arrays and Unmanaged Memory

NumSharp stores all array data in unmanaged memory for maximum performance. This design choice—borrowed from NumPy's architecture—enables zero-copy interop with native libraries, efficient memory-mapped file access, and predictable memory layout for SIMD operations.

This page explains how to create arrays from existing buffers without copying, how to control who owns and frees the memory, and how memory flows through NumSharp.

---

## Why Unmanaged Memory?

.NET's managed heap is excellent for general-purpose programming, but scientific computing has different requirements:

**Predictable Layout.** Managed arrays can be moved by the garbage collector at any time. Unmanaged memory stays put, which is essential when passing pointers to native libraries or GPU drivers.

**Reduced GC Overhead.** Large managed arrays cause GC pressure and can trigger expensive collections. Unmanaged memory avoids this—though NumSharp still informs the GC about allocation sizes so it can schedule collections appropriately.

**Interop Efficiency.** When calling into native code (BLAS, CUDA, image processing libraries), unmanaged memory can be passed directly without marshaling.

**Memory Mapping.** Memory-mapped files work naturally with unmanaged memory, enabling datasets larger than RAM.

The tradeoff is complexity: you need to understand when memory is shared, who owns it, and when it gets freed.

---

## Memory Architecture

NumSharp uses a layered architecture for memory management. Understanding these layers helps you reason about what happens under the hood.

```
User Code
    │
    ▼
┌─────────────────────────────────────────────┐
│  External APIs (user-facing)                │
│  np.frombuffer(), np.array(), NDArray()     │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Internal Infrastructure                    │
│  ArraySlice, UnmanagedMemoryBlock,          │
│  UnmanagedStorage                           │
└─────────────────────────────────────────────┘
    │
    ▼
  Native Memory (pinned arrays, allocated blocks, external pointers)
```

**External APIs** are what you interact with: `np.frombuffer()`, `np.array()`, and the `NDArray` constructors. These APIs hide the complexity of memory management behind sensible defaults.

**Internal Infrastructure** handles the low-level details: pinning managed arrays so the GC won't move them, tracking ownership so memory gets freed at the right time, and managing the raw pointers. You don't need to interact with these directly—the external APIs handle it for you.

### GC Pressure Tracking

Although NumSharp uses unmanaged memory, the .NET garbage collector still needs to know about it. Otherwise, the GC sees only the small managed wrappers (~100 bytes each) and doesn't realize there's megabytes of unmanaged data attached. This can cause memory to grow unbounded before the GC kicks in.

NumSharp solves this by calling `GC.AddMemoryPressure()` when allocating native memory and `GC.RemoveMemoryPressure()` when freeing it. This applies to arrays created with `np.array()`, `np.zeros()`, `np.empty()`, and similar functions.

For external memory (via `np.frombuffer()` with a dispose callback), the caller is responsible for pressure tracking since NumSharp doesn't know how the memory was allocated.

---

## Creating Arrays from Buffers

The most common scenario is interpreting an existing buffer as a NumSharp array. Maybe you've read a binary file, received a network packet, or have data from a native library. You want to work with it as an NDArray without copying gigabytes of data.

### From byte[]

`np.frombuffer()` is the primary tool for this. Given a byte array, it interprets the bytes as a typed array:

```csharp
byte[] buffer = File.ReadAllBytes("sensor_data.bin");

// Interpret the bytes as 32-bit floats
var readings = np.frombuffer(buffer, typeof(float));

Console.WriteLine($"Read {readings.size} float values");
Console.WriteLine($"First reading: {readings.GetSingle(0)}");
Console.WriteLine($"Last reading: {readings.GetSingle(readings.size - 1)}");
```

This creates a **view** of the buffer, not a copy. The `readings` array points directly into `buffer`'s memory. This is fast (no allocation, no copying) but has implications: if you modify `readings`, you're modifying `buffer`, and vice versa.

Often binary files have a header followed by data. Use the `offset` parameter to skip the header:

```csharp
// File format: 16-byte header, then float data
var data = np.frombuffer(buffer, typeof(float), offset: 16);
```

If you only want part of the data, use `count`:

```csharp
// Read only the first 1000 floats, starting at byte 16
var subset = np.frombuffer(buffer, typeof(float), count: 1000, offset: 16);
```

The `count` parameter specifies the number of *elements* (floats, in this case), not bytes. If you ask for more elements than are available after the offset, NumSharp throws an exception.

### From Typed Arrays

Sometimes you have a .NET array and want to use it with NumSharp. There are two approaches with different tradeoffs.

**Copying (safe, independent):**

```csharp
int[] scores = { 85, 92, 78, 95, 88 };
var arr = np.array(scores);  // Copies the data

scores[0] = 0;               // Original changed
Console.WriteLine(arr[0]);   // Still 85 - arr has its own copy
```

`np.array()` copies by default. This is safe: you can modify the original array or the NDArray independently. But for large arrays, copying is expensive.

**Viewing (fast, shared memory):**

```csharp
int[] scores = { 85, 92, 78, 95, 88 };
var arr = new NDArray(scores);  // Creates a view

scores[0] = 0;                  // Original changed
Console.WriteLine(arr[0]);      // 0 - they share memory!
```

The `NDArray` constructor creates a view. This is fast but couples their lifetimes: modifications to one affect the other.

**Reinterpreting types:**

A powerful technique is viewing an array as a different type. This lets you inspect the raw bytes of any data:

```csharp
int[] values = { 1, 2, 3, 4 };

// View the same memory as bytes (16 bytes total for 4 ints)
var asBytes = np.frombuffer<int>(values, typeof(byte));
Console.WriteLine($"First int as bytes: {asBytes[":4"]}");
// Output: [1, 0, 0, 0] on little-endian systems

// View as floats (same bits, different interpretation)
var asFloats = np.frombuffer<int>(values, typeof(float));
```

This is useful for binary serialization, network protocols, or understanding how data is represented in memory.

### From Native Pointers

When working with native code, you often have a pointer to memory that was allocated outside .NET. NumSharp can wrap this memory without copying.

The critical question is: **who frees the memory?**

**View only (caller manages lifetime):**

```csharp
// Native library allocated this memory
IntPtr nativeBuffer = NativeLib.GetData(out int byteSize);

// Create a view - NumSharp does NOT own this memory
var arr = np.frombuffer(nativeBuffer, byteSize, typeof(float));

// Use the array...
ProcessData(arr);

// Caller must free when done
NativeLib.FreeData(nativeBuffer);
```

This is appropriate when you're borrowing memory temporarily. You must ensure the native buffer outlives the NDArray—if the native code frees the memory while NumSharp is using it, you'll get crashes or corruption.

**Transfer ownership (NumSharp frees):**

```csharp
// We allocate native memory
int bytes = 1024 * sizeof(float);
IntPtr ptr = Marshal.AllocHGlobal(bytes);
GC.AddMemoryPressure(bytes);  // Tell GC about this allocation

// Transfer ownership to NumSharp
var arr = np.frombuffer(ptr, bytes, typeof(float),
    dispose: () => {
        Marshal.FreeHGlobal(ptr);
        GC.RemoveMemoryPressure(bytes);
    });

// When arr is garbage collected, the dispose action runs
```

The `dispose` parameter takes an action that NumSharp calls when the array is no longer needed. For large allocations, pair `GC.AddMemoryPressure()` with `GC.RemoveMemoryPressure()` so the GC knows about your memory. Be careful: if you free the memory yourself AND provide a dispose action, you'll double-free.

### From .NET Buffer Types

Modern .NET code often uses `ArraySegment<byte>`, `Memory<byte>`, or `Span<byte>` instead of raw arrays. NumSharp supports these too.

**ArraySegment** is common in network code where you're working with a slice of a larger buffer:

```csharp
byte[] networkBuffer = new byte[65536];
int bytesReceived = socket.Receive(networkBuffer);

// Work with just the received portion
var segment = new ArraySegment<byte>(networkBuffer, 0, bytesReceived);
var packet = np.frombuffer(segment, typeof(byte));
```

ArraySegment already carries offset and count, so NumSharp uses them automatically.

**Memory<byte>** is the modern way to represent a contiguous region of memory:

```csharp
Memory<byte> memory = GetDataFromSomewhere();
var arr = np.frombuffer(memory, typeof(float));
```

If the Memory is backed by an array (the common case), NumSharp creates a view. If it's backed by something else, NumSharp copies.

**ReadOnlySpan<byte>** always requires a copy because spans can't be pinned—they might be stack-allocated:

```csharp
ReadOnlySpan<byte> span = stackalloc byte[16];
var arr = np.frombuffer(span, typeof(int));  // Must copy
```

---

## View vs Copy: When Memory is Shared

Understanding when NumSharp shares memory versus copying is crucial for correctness and performance.

### The Rule of Thumb

- **Mutable, pinnable sources → View.** `byte[]`, `T[]`, `ArraySegment`, array-backed `Memory`
- **Immutable or unpinnable sources → Copy.** `ReadOnlySpan`, non-array `Memory`, big-endian conversion

### Why Views Matter

Views are fast but create hidden coupling. Consider this bug:

```csharp
byte[] buffer = new byte[1024];
var arr = np.frombuffer(buffer, typeof(float));

// Later, somewhere else in the code...
Array.Clear(buffer, 0, buffer.Length);

// arr now contains all zeros!
// If you expected arr to preserve the original data, this is a bug.
```

Views also mean modifications propagate both ways:

```csharp
var buffer = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
var shorts = np.frombuffer(buffer, typeof(short));

// Modify through NumSharp
shorts.SetInt16(9999, 0);

// buffer changed too
Console.WriteLine(buffer[0]);  // 15 (low byte of 9999)
Console.WriteLine(buffer[1]);  // 39 (high byte of 9999)
```

### When to Force a Copy

If you need independent data, call `.copy()`:

```csharp
var buffer = new byte[] { 1, 2, 3, 4 };
var view = np.frombuffer(buffer, typeof(byte));
var independent = view.copy();

buffer[0] = 99;
Console.WriteLine(view[0]);        // 99 - shared
Console.WriteLine(independent[0]); // 1 - independent copy
```

### Memory Lifetime

When you create a view, you're responsible for keeping the source alive:

```csharp
NDArray GetData()
{
    byte[] localBuffer = new byte[1024];
    FillWithData(localBuffer);
    return np.frombuffer(localBuffer, typeof(float));
    // BUG: localBuffer can be GC'd after this returns!
}
```

The returned NDArray points into `localBuffer`, but once `GetData()` returns, nothing holds a reference to `localBuffer`. The GC might collect it, and the NDArray becomes a dangling pointer.

The fix is either to copy:

```csharp
return np.frombuffer(localBuffer, typeof(float)).copy();
```

Or to let NumSharp own the memory by using a different approach:

```csharp
var arr = np.zeros<float>(256);  // NumSharp owns this memory
FillWithData(arr);
return arr;
```

---

## Ownership: Who Frees the Memory?

Memory ownership is the most subtle aspect of NumSharp's memory model. Getting it wrong causes either memory leaks (memory never freed) or crashes (memory freed while still in use).

### Managed Arrays (byte[], T[])

When you view a managed array, NumSharp "pins" it internally. This prevents the GC from moving the array. When the NDArray is garbage collected, NumSharp unpins the array, and normal GC takes over.

```csharp
var buffer = new byte[1024];
var arr = np.frombuffer(buffer, typeof(float));
// buffer is now pinned internally

arr = null;
GC.Collect();
// Eventually: buffer is unpinned, can be GC'd normally
```

You don't need to do anything special—the .NET garbage collector handles it. But be aware that pinned memory can cause heap fragmentation if you have many long-lived pinned arrays.

### Native Memory Without Ownership

When you wrap native memory without a dispose action, NumSharp creates a "borrowed" view:

```csharp
IntPtr ptr = GetNativePointer();
var arr = np.frombuffer(ptr, 1024, typeof(float));
// NumSharp does NOT own this memory
```

The NDArray will use this memory, but when the NDArray is garbage collected, nothing happens to the native memory. It's your responsibility to free it at the appropriate time—which must be after all NDArrays viewing it are gone.

### Native Memory With Ownership Transfer

The `dispose` parameter lets you transfer ownership to NumSharp:

```csharp
IntPtr ptr = Marshal.AllocHGlobal(1024);
var arr = np.frombuffer(ptr, 1024, typeof(float),
    dispose: () => Marshal.FreeHGlobal(ptr));
```

Now when `arr` is garbage collected, NumSharp calls your dispose action. This happens during finalization, which means:

1. It's non-deterministic—you don't know exactly when
2. It will eventually happen (unless the process exits first)
3. Don't rely on order between multiple finalizers

### Common Ownership Patterns

**ArrayPool integration:**

```csharp
var rental = ArrayPool<byte>.Shared.Rent(4096);
var arr = np.frombuffer(rental, typeof(float),
    dispose: () => ArrayPool<byte>.Shared.Return(rental));
// When arr is GC'd, the array returns to the pool
```

**P/Invoke with native allocator:**

```csharp
[DllImport("mylib")]
static extern IntPtr alloc_buffer(int size);

[DllImport("mylib")]
static extern void free_buffer(IntPtr ptr);

IntPtr ptr = alloc_buffer(1024);
var arr = np.frombuffer(ptr, 1024, typeof(float),
    dispose: () => free_buffer(ptr));
```

**COM memory:**

```csharp
IntPtr ptr = Marshal.AllocCoTaskMem(1024);
var arr = np.frombuffer(ptr, 1024, typeof(int),
    dispose: () => Marshal.FreeCoTaskMem(ptr));
```

---

## Endianness

Binary data from files or networks may be in a different byte order than your CPU expects. x86/x64 processors are little-endian, but network protocols and some file formats use big-endian.

NumSharp handles this through dtype strings:

```csharp
byte[] networkData = ReceivePacket();

// Big-endian int32 (network byte order)
var values = np.frombuffer(networkData, ">i4");

// Little-endian int32 (native on x86/x64)
var values = np.frombuffer(networkData, "<i4");
```

The prefix indicates byte order:
- `>` or `!` — Big-endian (most significant byte first)
- `<` — Little-endian (least significant byte first)
- `=` — Native endian (whatever the CPU uses)
- `|` — Not applicable (single-byte types)

**Important:** Big-endian conversion requires a copy because NumSharp must swap the bytes. Little-endian on a little-endian system creates a view.

Common dtype strings:

| String | Type | Bytes |
|--------|------|-------|
| `i1`, `b` | int8 | 1 |
| `u1`, `B` | uint8 | 1 |
| `i2`, `h` | int16 | 2 |
| `u2`, `H` | uint16 | 2 |
| `i4`, `i`, `l` | int32 | 4 |
| `u4`, `I`, `L` | uint32 | 4 |
| `i8`, `q` | int64 | 8 |
| `u8`, `Q` | uint64 | 8 |
| `f4`, `f` | float32 | 4 |
| `f8`, `d` | float64 | 8 |

---

## Common Patterns

### Reading Binary Files

```csharp
// Simple case: entire file as one type
byte[] data = File.ReadAllBytes("measurements.bin");
var readings = np.frombuffer(data, typeof(double));

// With header
// File format: 4-byte int (count), then float data
int count = BitConverter.ToInt32(data, 0);
var values = np.frombuffer(data, typeof(float), count: count, offset: 4);
```

### Memory-Mapped Large Files

For files larger than RAM, memory-map them:

```csharp
using var mmf = MemoryMappedFile.CreateFromFile(
    "huge_dataset.bin", FileMode.Open);
using var accessor = mmf.CreateViewAccessor();

// Read chunks as needed
byte[] chunk = new byte[1_000_000];
accessor.ReadArray(offset, chunk, 0, chunk.Length);
var arr = np.frombuffer(chunk, typeof(float));
```

### Network Protocol Parsing

```csharp
// Packet format:
// - 4 bytes: message type (int32)
// - 4 bytes: payload length (int32)
// - N bytes: payload (float32[])

byte[] packet = socket.Receive();

int messageType = BitConverter.ToInt32(packet, 0);
int payloadLength = BitConverter.ToInt32(packet, 4);
int floatCount = payloadLength / sizeof(float);

var payload = np.frombuffer(packet, typeof(float),
    count: floatCount, offset: 8);
```

### Interop with Native Libraries

```csharp
[DllImport("imagelib")]
static extern IntPtr process_image(
    IntPtr input, int width, int height, out IntPtr output);

[DllImport("imagelib")]
static extern void free_image(IntPtr ptr);

// Process an image
var inputArr = np.array(imageBytes).reshape(height, width, 3);
IntPtr outputPtr = process_image(
    inputArr.data, width, height, out IntPtr resultPtr);

// Wrap output with ownership transfer
var result = np.frombuffer(resultPtr, width * height * 3, typeof(byte),
    dispose: () => free_image(resultPtr));
var outputImage = result.reshape(height, width, 3);
```

### Sharing Data with GPU Libraries

```csharp
// Create data on CPU
var cpuData = np.random.rand(1000, 1000).astype(np.float32);

// Get pointer for GPU transfer
IntPtr dataPtr = cpuData.data;

// Transfer to GPU (CUDA example)
cudaMemcpy(devicePtr, dataPtr,
    cpuData.size * sizeof(float), cudaMemcpyHostToDevice);

// ... GPU computation ...

// Transfer back
cudaMemcpy(dataPtr, devicePtr,
    cpuData.size * sizeof(float), cudaMemcpyDeviceToHost);
```

---

## Troubleshooting

### "buffer size must be a multiple of element size"

The byte buffer's length (after offset) must be divisible by the element size:

```csharp
byte[] buf = new byte[7];
var arr = np.frombuffer(buf, typeof(int));  // Error: 7 not divisible by 4

// Fix: use count to read only complete elements
var arr = np.frombuffer(buf, typeof(int), count: 1);  // Reads 4 bytes
```

### "offset must be non-negative and no greater than buffer length"

The offset is in bytes and must be within the buffer:

```csharp
byte[] buf = new byte[100];
var arr = np.frombuffer(buf, typeof(int), offset: 200);  // Error

// Fix: check your offset calculation
var arr = np.frombuffer(buf, typeof(int), offset: 96);  // OK, reads 1 int
```

### Access Violation / Segmentation Fault

Usually means you're accessing freed memory:

```csharp
NDArray arr;
{
    byte[] localBuffer = new byte[1024];
    arr = np.frombuffer(localBuffer, typeof(float));
}
// localBuffer may be GC'd here

arr[0] = 1.0f;  // CRASH: accessing freed memory
```

Fix: either copy the data or ensure the buffer outlives the NDArray.

### Memory Leak

If you transfer ownership but the NDArray never gets garbage collected:

```csharp
static List<NDArray> cache = new List<NDArray>();

void ProcessData()
{
    IntPtr ptr = Marshal.AllocHGlobal(1024);
    var arr = np.frombuffer(ptr, 1024, typeof(float),
        dispose: () => Marshal.FreeHGlobal(ptr));

    cache.Add(arr);  // arr never gets GC'd!
}
```

The dispose action only runs on GC. If you hold references forever, the memory leaks. Clear your caches or use weak references if needed.

---

## API Reference

### np.frombuffer Overloads

| Signature | View/Copy | Notes |
|-----------|-----------|-------|
| `frombuffer(byte[], dtype, count, offset)` | View | Pins array |
| `frombuffer(byte[], string dtype, count, offset)` | Copy if big-endian | Handles endianness |
| `frombuffer(ReadOnlySpan<byte>, dtype, count, offset)` | Copy | Spans can't be pinned |
| `frombuffer(ArraySegment<byte>, dtype, count)` | View | Uses segment's offset |
| `frombuffer(Memory<byte>, dtype, count, offset)` | View if array-backed | Fallback to copy |
| `frombuffer(IntPtr, byteLength, dtype, count, offset, dispose)` | View | Optional ownership |
| `frombuffer<TSource>(TSource[], dtype, count, offset)` | View | Reinterpret typed array |

### np.array vs NDArray Constructor

| API | Default Behavior | Use When |
|-----|------------------|----------|
| `np.array(T[])` | Copy | You want independent data |
| `new NDArray(Array)` | View | You want shared memory, better performance |

---

## Summary

NumSharp's memory system is designed for performance and interoperability. The key concepts are:

1. **Views share memory** with the source. Fast but coupled.
2. **Copies are independent** but require allocation.
3. **Ownership determines who frees.** Managed arrays are GC'd; native memory needs explicit handling.
4. **The dispose callback** transfers ownership to NumSharp for native memory.
5. **Use `.copy()` when in doubt** to avoid lifetime bugs.

For most code, `np.frombuffer()` with default settings does the right thing. When you need more control, the parameters are there.
