using System;
using NumSharp.Core;

namespace NumSharp.Core
{
    class IncorrectTypeException : System.Exception
    {
        public IncorrectTypeException() : base("This method does not work with this dtype or was not already implemented.")
        {
            
        }
    }    
}