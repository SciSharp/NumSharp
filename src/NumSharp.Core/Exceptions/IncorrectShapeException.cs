using System;
using NumSharp.Core;

namespace NumSharp.Core
{
    class IncorrectShapeException : System.Exception
    {
        public IncorrectShapeException() : base("This method does not work with this shape or was not already implemented.")
        {
            
        }
    }    
}