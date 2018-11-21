using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArrayGeneric<T> : IEnumerable, IEnumerator
    {
        private int pos = -1;
        public object Current
        {
            get
            {
                if (shape.Length == 1)
                {
                    return Data[pos];
                }
                else
                {
                    return this[new Shape(pos)];
                }
            }
        }

        public IEnumerator GetEnumerator()
        {
            return this;
        }

        public bool MoveNext()
        {
            pos++;
            return pos < Shape[0];
        }

        public void Reset()
        {
            pos = -1;
        }
    }
}
