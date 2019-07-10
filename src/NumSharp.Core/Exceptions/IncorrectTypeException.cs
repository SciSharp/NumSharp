using System;
using NumSharp;

namespace NumSharp
{
    class IncorrectTypeException : System.Exception
    {
        public IncorrectTypeException() : base("This method does not work with this dtype or was not already implemented.")
        { }
    }
}
