using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp.Utilities
{
    public static class SteppingExtension
    {
        // this will step an array ... [1,2,3,4].Step(2) => [1,3]
        // if step is 1 the original array is returned without copying it
        public static T[] Step<T>(this T[] array, int step)
        {
            // todo: optimize performance? avoid array copying as much as possible, etc.
            if (step == 0)
                throw new ArgumentException("Step of 0 is not allowed!");
            if (step == 1)
                return array;
            if (step == -1)
                return array.AsEnumerable().Reverse().ToArray();
            var stepped_enumerable = Step(step < 0 ? array.AsEnumerable().Reverse().GetEnumerator() : array.OfType<T>().GetEnumerator(), Math.Abs(step));
            return stepped_enumerable.ToArray();
        }

        private static IEnumerable<T> Step<T>(IEnumerator<T> enumerator, int step)
        {
            while (enumerator.MoveNext())
            {
                yield return enumerator.Current;
                for (int i = 0; i < step - 1; i++)
                {
                    if (!enumerator.MoveNext())
                        yield break;
                }
            }
        }
    }
}
