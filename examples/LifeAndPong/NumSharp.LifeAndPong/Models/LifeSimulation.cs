using NumSharp.Backends.Unmanaged;

namespace NumSharp.LifeAndPong.Models;

/// <summary>
/// Conway's Game of Life backed by a contiguous NumSharp byte array.
/// </summary>
public sealed class LifeSimulation : IDisposable
{
    private readonly Random _random;
    private NDArray _cells;
    private bool _disposed;

    public LifeSimulation(int rows = 40, int columns = 48, int seed = 73021)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(rows, 3);
        ArgumentOutOfRangeException.ThrowIfLessThan(columns, 3);

        Rows = rows;
        Columns = columns;
        _random = new Random(seed);
        _cells = np.zeros(new Shape(rows, columns), NPTypeCode.Byte);
        Reseed();
    }

    public int Rows { get; }

    public int Columns { get; }

    public long Generation { get; private set; }

    public int LiveCount { get; private set; }

    public bool IsAlive(int row, int column)
    {
        ThrowIfDisposed();
        ValidateCoordinates(row, column);
        return _cells.GetData<byte>()[IndexOf(row, column)] != 0;
    }

    public void SetCell(int row, int column, bool alive)
    {
        ThrowIfDisposed();
        ValidateCoordinates(row, column);

        var data = _cells.GetData<byte>();
        var index = IndexOf(row, column);
        var wasAlive = data[index] != 0;
        if (wasAlive == alive)
            return;

        data[index] = alive ? (byte)1 : (byte)0;
        LiveCount += alive ? 1 : -1;
    }

    public bool ToggleCell(int row, int column)
    {
        var next = !IsAlive(row, column);
        SetCell(row, column, next);
        return next;
    }

    public void Clear()
    {
        ThrowIfDisposed();
        _cells.GetData<byte>().Fill(0);
        Generation = 0;
        LiveCount = 0;
    }

    public void Reseed(double density = 0.22)
    {
        ThrowIfDisposed();
        if (density is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(density));

        var data = _cells.GetData<byte>();
        var live = 0;
        for (long i = 0; i < data.Count; i++)
        {
            var value = _random.NextDouble() < density ? (byte)1 : (byte)0;
            data[i] = value;
            live += value;
        }

        // A recognizable glider makes the seed feel intentional even in a busy field.
        var centerRow = Rows / 2;
        var centerColumn = Columns / 2;
        SetRaw(data, centerRow - 1, centerColumn, true, ref live);
        SetRaw(data, centerRow, centerColumn + 1, true, ref live);
        SetRaw(data, centerRow + 1, centerColumn - 1, true, ref live);
        SetRaw(data, centerRow + 1, centerColumn, true, ref live);
        SetRaw(data, centerRow + 1, centerColumn + 1, true, ref live);

        Generation = 0;
        LiveCount = live;
    }

    public void Step()
    {
        ThrowIfDisposed();

        var next = np.zeros(new Shape(Rows, Columns), NPTypeCode.Byte);
        var source = _cells.GetData<byte>();
        var destination = next.GetData<byte>();
        var live = 0;

        for (var row = 0; row < Rows; row++)
        {
            var north = row == 0 ? Rows - 1 : row - 1;
            var south = row == Rows - 1 ? 0 : row + 1;

            for (var column = 0; column < Columns; column++)
            {
                var west = column == 0 ? Columns - 1 : column - 1;
                var east = column == Columns - 1 ? 0 : column + 1;

                var neighbors =
                    source[IndexOf(north, west)] + source[IndexOf(north, column)] + source[IndexOf(north, east)] +
                    source[IndexOf(row, west)] + source[IndexOf(row, east)] +
                    source[IndexOf(south, west)] + source[IndexOf(south, column)] + source[IndexOf(south, east)];

                var alive = source[IndexOf(row, column)] != 0;
                var value = neighbors == 3 || (alive && neighbors == 2) ? (byte)1 : (byte)0;
                destination[IndexOf(row, column)] = value;
                live += value;
            }
        }

        var previous = _cells;
        _cells = next;
        previous.Dispose();
        Generation++;
        LiveCount = live;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _cells.Dispose();
        _disposed = true;
    }

    private long IndexOf(int row, int column) => (long)row * Columns + column;

    private void SetRaw(ArraySlice<byte> data, int row, int column, bool alive, ref int live)
    {
        var index = IndexOf(row, column);
        var wasAlive = data[index] != 0;
        if (wasAlive == alive)
            return;

        data[index] = alive ? (byte)1 : (byte)0;
        live += alive ? 1 : -1;
    }

    private void ValidateCoordinates(int row, int column)
    {
        if ((uint)row >= (uint)Rows)
            throw new ArgumentOutOfRangeException(nameof(row));
        if ((uint)column >= (uint)Columns)
            throw new ArgumentOutOfRangeException(nameof(column));
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
}
