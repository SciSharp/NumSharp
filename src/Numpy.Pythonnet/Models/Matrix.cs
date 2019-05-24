using System;
using System.Collections.Generic;
using System.Text;
using Python.Runtime;

namespace Numpy.Models
{
    public class Matrix : PythonObject
    {
        public Matrix(PyObject pyobject) : base(pyobject)
        {
        }

    }
}
