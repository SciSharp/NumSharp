using System;

namespace NumSharp
{
    class IncorrectShapeException : System.Exception
    {
        public IncorrectShapeException() : base("This method does not work with this shape or was not already implemented.")
        { }

        public IncorrectShapeException(string message) : base(message)
        { }
    }
}
