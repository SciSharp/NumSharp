using System;
using NumSharp;
using System.Linq;
using System.Collections.Generic;

public static partial class numpy 
{
    public static dynamic arange(object start, object stop, object step)
    {
        dynamic returnValue = null;

        switch (start)
        {
            case int startCast : 
            {
                int stopCast = (int) stop;
                int stepCast = (int) step;

                returnValue = new NumSharp.NDArray<int>().arange(stopCast,startCast,stepCast);

                break;
            }
            case float startCast_ : 
            {
                int startCast = (int) start;
                int stopCast = (int) stop;
                int stepCast = (int) step;

                returnValue = new NumSharp.NDArray<float>().arange(stopCast,startCast,stepCast);

                break;
            }

        }
        return returnValue;
    }    
}
