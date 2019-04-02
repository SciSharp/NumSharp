using System;

namespace NumSharp
{
    class IncorrectSizeException : System.Exception
    {
        public IncorrectSizeException(string message) : base(message)
        {
            
        }
    }    
}