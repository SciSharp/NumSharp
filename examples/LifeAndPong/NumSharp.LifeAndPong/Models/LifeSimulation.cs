using NumSharp.Backends.Unmanaged;

namespace NumSharp.LifeAndPong.Models;

/// <summary>
/// Conway's Game of Life backed by a contiguous NumSharp byte array.
/// </summary>
public sealed class LifeSimulation : IDisposable
{
    private readonly Random _random;
    private NDArray _cells;
    private NDArray _next;
    private bool _disposed;

    public LifeSimulation(int rows = 40, int columns = 48, int seed = 73021)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(rows, 3);
        ArgumentOutOfRangeException.ThrowIfLessThan(columns, 3);

        Rows = rows;
        Columns = columns;
        _random = new Random(seed);
        _cells = np.zeros(new Shape(rows, columns), NPTypeCode.Byte);
        _next = np.zeros(new Shape(rows, columns), NPTypeCode.Byte);
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
        if (!double.IsFinite(density) || density is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(density));

        var data = _cells.GetData<byte>();
        var live = 0;
        for (long i = 0; i < data.Count; i++)
        {
            var value = _random.NextDouble() < density ? (byte)1 : (byte)0;
            data[i] = value;
            live += value;
        }

        Generation = 0;
        LiveCount = live;
    }

    public void Step()
    {
        ThrowIfDisposed();

        var source = _cells.GetData<byte>();
        var destination = _next.GetData<byte>();
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

        (_cells, _next) = (_next, _cells);
        Generation++;
        LiveCount = live;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _cells.Dispose();
        _next.Dispose();
        _disposed = true;
    }

    private long IndexOf(int row, int column) => (long)row * Columns + column;

    public void LoadPattern(bool pulsar)
    {
        string[] pattern = pulsar
            ? ["..###...###..", ".............", "#....#.#....#", "#....#.#....#", "#....#.#....#", "..###...###..", ".............", "..###...###..", "#....#.#....#", "#....#.#....#", "#....#.#....#", ".............", "..###...###.."]
            : [".#.", "..#", "###"];
        if (Rows < pattern.Length || Columns < pattern[0].Length)
            throw new InvalidOperationException("The pattern does not fit the field.");
        Clear();
        for (var row = 0; row < pattern.Length; row++)
            for (var column = 0; column < pattern[row].Length; column++)
                if (pattern[row][column] == '#')
                    SetCell((Rows - pattern.Length) / 2 + row, (Columns - pattern[0].Length) / 2 + column, true);
    }

    public void PaintLine(int fromRow, int fromColumn, int toRow, int toColumn, bool alive)
    {
        ThrowIfDisposed();
        ValidateCoordinates(fromRow, fromColumn);
        ValidateCoordinates(toRow, toColumn);
        var dx = Math.Abs(toColumn - fromColumn);
        var dy = -Math.Abs(toRow - fromRow);
        var sx = fromColumn < toColumn ? 1 : -1;
        var sy = fromRow < toRow ? 1 : -1;
        var error = dx + dy;
        while (true)
        {
            SetCell(fromRow, fromColumn, alive);
            if (fromRow == toRow && fromColumn == toColumn)
                break;
            var twice = error * 2;
            if (twice >= dy) { error += dy; fromColumn += sx; }
            if (twice <= dx) { error += dx; fromRow += sy; }
        }
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
