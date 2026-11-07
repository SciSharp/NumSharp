"""NumPy twin of subclass_probe.cs: the class an ndarray subclass gets back from each operation (NumPy 2.4.2)."""
import numpy as np


class MyArray(np.ndarray):
    """A user's ndarray subclass with a priority above numpy.matrix's 10.0."""

    __array_priority__ = 20.0


x = np.arange(6.0).reshape(2, 3).view(MyArray)
for what, result in [
    ("x itself", x),
    ("np.add(x, x)", np.add(x, x)),
    ("x + 1", x + 1),
    ("-x", -x),
    ("x[0] (row view)", x[0]),
    ("x[:, 1:] (slice view)", x[:, 1:]),
    ("x.T", x.T),
    ("x.reshape(3, 2)", x.reshape(3, 2)),
    ("np.sum(x, 0)", np.sum(x, 0)),
    ("np.concatenate([x, x])", np.concatenate([x, x])),
    ("np.sqrt(x)", np.sqrt(x)),
    ("x.copy()", x.copy()),
    ("np.asanyarray(x)", np.asanyarray(x)),
]:
    print(f"{what:<28} -> {type(result).__name__}")
print("same object from np.asanyarray:", np.asanyarray(x) is x, "| numpy", np.__version__)
