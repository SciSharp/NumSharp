* We should avoid `Parallel.For(n)` and instead chunk into the 
amount of cores available (d) and instead run `Parallel.for(n/d)` 
where inside it'll run a for-loop to it's chunked data.<br>
Not to forget to search the entire project for these.

* `Span<T>` uses a clever trick to avoid `GCPin` by referencing the array in a `System.Pinnable<T>` which allows the GC to move the array around but still have the address of it.
Learn more on how they handle it and implement a similar structure in `MemoryBlock`.
Refer to `internal static Span<T> Create(T[] array, int start)`<br>
It does that by forcing `ref Span<T>` that makes the ref to the array to be non-changing.<br>

