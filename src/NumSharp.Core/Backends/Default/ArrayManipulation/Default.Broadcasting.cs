using System;
using System.Linq;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // Broadcasting methods - all delegate to Shape struct
        // Shape is the canonical location for broadcasting logic.
        // These methods remain for backward compatibility within Backends.
        // =============================================================================

        /// <summary>
        /// Resolves the output shape when broadcasting two shapes together.
        /// </summary>
        /// <remarks>Delegates to Shape.ResolveReturnShape</remarks>
        public static Shape ResolveReturnShape(Shape leftShape, Shape rightShape)
            => Shape.ResolveReturnShape(leftShape, rightShape);

        /// <summary>
        /// Resolves the output shape when broadcasting multiple shapes together.
        /// </summary>
        /// <remarks>Delegates to Shape.ResolveReturnShape</remarks>
        public static Shape ResolveReturnShape(params Shape[] shapes)
            => Shape.ResolveReturnShape(shapes);

        /// <summary>
        /// Resolves the output shape when broadcasting multiple arrays together.
        /// </summary>
        /// <remarks>Extracts shapes and delegates to Shape.ResolveReturnShape</remarks>
        public static Shape ResolveReturnShape(params NDArray[] arrays)
        {
            if (arrays.Length == 0)
                return default;
            if (arrays.Length == 1)
                return arrays[0].Shape.Clean();
            return Shape.ResolveReturnShape(arrays.Select(a => a.Shape).ToArray());
        }

        /// <summary>
        /// Broadcasts multiple shapes and returns the broadcasted shapes with computed strides.
        /// </summary>
        /// <remarks>Delegates to Shape.Broadcast</remarks>
        public static Shape[] Broadcast(params Shape[] shapes)
            => Shape.Broadcast(shapes);

        /// <summary>
        /// Broadcasts two shapes and returns the broadcasted shapes with computed strides.
        /// </summary>
        /// <remarks>Delegates to Shape.Broadcast</remarks>
        public static (Shape LeftShape, Shape RightShape) Broadcast(Shape leftShape, Shape rightShape)
            => Shape.Broadcast(leftShape, rightShape);

        /// <summary>
        /// Broadcasts multiple arrays and returns new NDArrays with broadcasted shapes.
        /// </summary>
        /// <remarks>Extracts shapes, delegates to Shape.Broadcast, wraps results in NDArray</remarks>
        public static NDArray[] Broadcast(params NDArray[] arrays)
        {
            if (arrays.Length == 0)
                return Array.Empty<NDArray>();

            if (arrays.Length == 1)
                return arrays;

            var shapes = Shape.Broadcast(arrays.Select(a => a.Shape).ToArray());

            for (int i = 0; i < arrays.Length; i++)
                arrays[i] = new NDArray(arrays[i].Storage, shapes[i]);

            return arrays;
        }

        /// <summary>
        /// Checks if the given shapes can be broadcast together.
        /// </summary>
        /// <remarks>Delegates to Shape.AreBroadcastable</remarks>
        public static bool AreBroadcastable(params Shape[] shapes)
            => Shape.AreBroadcastable(shapes);

        /// <summary>
        /// Checks if the given dimension arrays can be broadcast together.
        /// </summary>
        /// <remarks>Delegates to Shape.AreBroadcastable</remarks>
        public static bool AreBroadcastable(params int[][] shapes)
            => Shape.AreBroadcastable(shapes);

        /// <summary>
        /// Checks if the given arrays can be broadcast together.
        /// </summary>
        /// <remarks>Extracts shapes and delegates to Shape.AreBroadcastable</remarks>
        public static bool AreBroadcastable(params NDArray[] arrays)
        {
            if (arrays.Length <= 1)
                return true;
            return Shape.AreBroadcastable(arrays.Select(a => a.Shape).ToArray());
        }
    }
}
