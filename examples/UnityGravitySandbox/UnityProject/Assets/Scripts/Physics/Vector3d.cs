using System;
using System.Globalization;

namespace NumSharp.Examples.GravitySandbox.Physics
{
    /// <summary>
    /// A minimal double-precision 3-vector. It exists so the physics layer can hand back scalar-vector
    /// results (momentum, a spawned body's position) WITHOUT taking a dependency on
    /// <c>UnityEngine.Vector3</c>, which is single precision and only available inside Unity. Keeping the
    /// engine free of <c>UnityEngine</c> is what lets the exact same physics source compile in the
    /// standalone verification harness. The Unity view layer converts to <c>Vector3</c> at the boundary.
    /// </summary>
    /// <remarks>
    /// This is a deliberately tiny convenience type, not a general linear-algebra facility — the heavy
    /// vector math all happens inside NumSharp on whole <c>(N,3)</c> arrays. Only the few per-body scalar
    /// reads and reductions that cross back into managed code use it.
    /// </remarks>
    public readonly struct Vector3d : IEquatable<Vector3d>
    {
        /// <summary>The x component.</summary>
        public readonly double X;
        /// <summary>The y component.</summary>
        public readonly double Y;
        /// <summary>The z component.</summary>
        public readonly double Z;

        /// <summary>Constructs a vector from its three components.</summary>
        /// <param name="x">The x component.</param>
        /// <param name="y">The y component.</param>
        /// <param name="z">The z component.</param>
        public Vector3d(double x, double y, double z) { X = x; Y = y; Z = z; }

        /// <summary>The zero vector, a convenient additive identity for accumulations.</summary>
        public static Vector3d Zero => new Vector3d(0, 0, 0);

        /// <summary>Component-wise sum.</summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>The sum <c>a + b</c>.</returns>
        public static Vector3d operator +(Vector3d a, Vector3d b) => new Vector3d(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

        /// <summary>Component-wise difference.</summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>The difference <c>a - b</c>.</returns>
        public static Vector3d operator -(Vector3d a, Vector3d b) => new Vector3d(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

        /// <summary>Scales a vector by a scalar.</summary>
        /// <param name="a">The vector.</param>
        /// <param name="s">The scalar factor.</param>
        /// <returns>The scaled vector <c>a·s</c>.</returns>
        public static Vector3d operator *(Vector3d a, double s) => new Vector3d(a.X * s, a.Y * s, a.Z * s);

        /// <summary>The Euclidean length (magnitude) of the vector.</summary>
        public double Length => Math.Sqrt(X * X + Y * Y + Z * Z);

        /// <summary>The squared length — cheaper than <see cref="Length"/> when only comparisons are needed.</summary>
        public double LengthSquared => X * X + Y * Y + Z * Z;

        /// <summary>Dot product with another vector.</summary>
        /// <param name="o">The other vector.</param>
        /// <returns>The scalar dot product.</returns>
        public double Dot(Vector3d o) => X * o.X + Y * o.Y + Z * o.Z;

        /// <summary>Cross product with another vector (right-handed).</summary>
        /// <param name="o">The other vector.</param>
        /// <returns>The vector <c>this × o</c>.</returns>
        public Vector3d Cross(Vector3d o) =>
            new Vector3d(Y * o.Z - Z * o.Y, Z * o.X - X * o.Z, X * o.Y - Y * o.X);

        /// <inheritdoc/>
        public bool Equals(Vector3d other) => X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z);

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is Vector3d v && Equals(v);

        /// <inheritdoc/>
        public override int GetHashCode() => HashCode.Combine(X, Y, Z);

        /// <summary>Formats the vector as <c>(x, y, z)</c> with invariant, round-trippable components.</summary>
        /// <returns>A culture-invariant string representation.</returns>
        public override string ToString() =>
            string.Format(CultureInfo.InvariantCulture, "({0:R}, {1:R}, {2:R})", X, Y, Z);
    }
}
