using System;
using System.Collections.Generic;
using System.Text;
using Python.Runtime;

namespace Numpy
{
    public partial class NDarray
    {
        //------------------------------
        // Comparison operators:
        //------------------------------

        // Return self<value.
        public static NDarray<bool> operator <(NDarray a, ValueType obj)
        {
            return new NDarray<bool>(a.self.InvokeMethod("__lt__", obj.ToPython()));
        }

        // Return self<=value.
        public static NDarray<bool> operator <=(NDarray a, ValueType obj)
        {
            return new NDarray<bool>(a.self.InvokeMethod("__le__", obj.ToPython()));
        }

        // Return self>value.
        public static NDarray<bool> operator >(NDarray a, ValueType obj)
        {
            return new NDarray<bool>(a.self.InvokeMethod("__gt__", obj.ToPython()));
        }

        // Return self>=value.
        public static NDarray<bool> operator >=(NDarray a, ValueType obj)
        {
            return new NDarray<bool>(a.self.InvokeMethod("__ge__", obj.ToPython()));
        }

        /// <summary>
        /// Returns an array of bool where the elements of the array are == value
        /// </summary>
        public static NDarray<bool> equals(NDarray a, ValueType obj)
        {
            return new NDarray<bool>(a.self.InvokeMethod("__eq__", obj.ToPython()));
        }

        /// <summary>
        /// Returns an array of bool where the elements of the array are == value
        /// </summary>
        public static NDarray<bool> not_equals(NDarray a, ValueType obj)
        {
            return new NDarray<bool>(a.self.InvokeMethod("__ne__", obj.ToPython()));
        }


        //------------------------------
        // Truth value of an array(bool) :
        //------------------------------

        /// <summary>
        /// Note
        /// Truth-value testing of an array invokes ndarray.__nonzero__, which raises an error if the
        /// number of elements in the array is larger than 1, because the truth value of such arrays is
        /// ambiguous.Use.any() and.all() instead to be clear about what is meant in such cases.
        /// (If the number of elements is 0, the array evaluates to False.)
        /// </summary>
        public static NDarray<bool> nonzero(NDarray a)
        {
            return new NDarray<bool>(a.self.InvokeMethod("__nonzero__"));
        }

        //------------------------------
        // Unary operations:
        //------------------------------

        // Return 	-self
        public static NDarray operator -(NDarray a)
        {
            return new NDarray(a.self.InvokeMethod("__neg__"));
        }

        // Return 	+self
        public static NDarray operator +(NDarray a)
        {
            return new NDarray(a.self.InvokeMethod("__pos__"));
        }

        // ndarray.__abs__(self)  // C# doesn't have an operator for that

        // Return 	~self
        public static NDarray operator ~(NDarray a)
        {
            return new NDarray(a.self.InvokeMethod("__invert__"));
        }

        //------------------------------
        // Arithmetic operators:
        //------------------------------

        // Return self+value.
        public static NDarray operator +(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__add__", obj.ToPython()));
        }

        // Return self-value.
        public static NDarray operator -(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__sub__", obj.ToPython()));
        }

        // Return self*value.
        public static NDarray operator *(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__mul__", obj.ToPython()));
        }

        // Return self/value.
        public static NDarray operator /(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__div__", obj.ToPython()));
        }

        /// <summary>
        /// Return self/value.
        /// </summary>
        public static NDarray truediv(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__truediv__", obj.ToPython()));
        }

        /// <summary>
        /// Return self//value. 
        /// </summary>
        public static NDarray floordiv(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__floordiv__", obj.ToPython()));
        }

        // Return self%value.
        public static NDarray operator %(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__mod__", obj.ToPython()));
        }

        /// <summary>
        /// Return divmod(value). 
        /// </summary>
        public static NDarray divmod(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__divmod__", obj.ToPython()));
        }

        /// <summary>
        /// Return pow(value). 
        /// </summary>
        public static NDarray pow(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__pow__", obj.ToPython()));
        }

        /// <summary>
        /// Return self&lt;&lt;value.
        /// </summary>
        public static NDarray operator <<(NDarray a, int obj)
        {
            return new NDarray(a.self.InvokeMethod("__lshift__", obj.ToPython()));
        }

        /// <summary>
        /// Return self&gt;&gt;value.
        /// </summary>
        public static NDarray operator >>(NDarray a, int obj)
        {
            return new NDarray(a.self.InvokeMethod("__rshift__", obj.ToPython()));
        }

        /// <summary>
        /// Return self&value.
        /// </summary>
        public static NDarray operator &(NDarray a, int obj)
        {
            return new NDarray(a.self.InvokeMethod("__and__", obj.ToPython()));
        }

        /// <summary>
        /// Return self|value.
        /// </summary>
        public static NDarray operator |(NDarray a, int obj)
        {
            return new NDarray(a.self.InvokeMethod("__or__", obj.ToPython()));
        }

        /// <summary>
        /// Return self^value.
        /// </summary>
        public static NDarray operator ^(NDarray a, int obj)
        {
            return new NDarray(a.self.InvokeMethod("__xor__", obj.ToPython()));
        }

        //------------------------------
        // Arithmetic, in-place:
        //------------------------------

        /// <summary>
        /// Return self+=value.
        /// </summary>
        public static NDarray iadd(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__iadd__", obj.ToPython()));
        }

        /// <summary>
        /// Return self-=value.
        /// </summary>
        public static NDarray isub(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__isub__", obj.ToPython()));
        }

        /// <summary>
        /// Return self*=value.
        /// </summary>
        public static NDarray imul(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__imul__", obj.ToPython()));
        }

        /// <summary>
        /// Return self/=value.
        /// </summary>
        public static NDarray idiv(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__idiv__", obj.ToPython()));
        }

        /// <summary>
        /// Return self/=value.
        /// </summary>
        public static NDarray itruediv(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__itruediv__", obj.ToPython()));
        }

        /// <summary>
        /// Return self//=value. 
        /// </summary>
        public static NDarray ifloordiv(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__floordiv__", obj.ToPython()));
        }

        /// <summary>
        /// Return self%value.
        /// </summary>
        public static NDarray imod(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__imod__", obj.ToPython()));
        }

        /// <summary>
        /// Return inplace pow(value). 
        /// </summary>
        public static NDarray ipow(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__ipow__", obj.ToPython()));
        }

        /// <summary>
        /// Return inplace self&lt;&lt;value.
        /// </summary>
        public static NDarray ilshift(NDarray a, int obj)
        {
            return new NDarray(a.self.InvokeMethod("__ilshift__", obj.ToPython()));
        }

        /// <summary>
        /// Return inplace self&gt;&gt;value.
        /// </summary>
        public static NDarray irshift(NDarray a, int obj)
        {
            return new NDarray(a.self.InvokeMethod("__irshift__", obj.ToPython()));
        }

        /// <summary>
        /// Return self&=value.
        /// </summary>
        public static NDarray iand(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__iand__", obj.ToPython()));
        }

        /// <summary>
        /// Return self|=value.
        /// </summary>
        public static NDarray ior(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__ior__", obj.ToPython()));
        }

        /// <summary>
        /// Return self^=value.
        /// </summary>
        public static NDarray ixor(NDarray a, ValueType obj)
        {
            return new NDarray(a.self.InvokeMethod("__ixor__", obj.ToPython()));
        }

        // TODO:
        // ndarray.__matmul__($self, value, /)	Return self@value.
        //ndarray.__copy__() Used if copy.copy is called on an array.
        //ndarray.__deepcopy__(memo, /)   Used if copy.deepcopy is called on an array.
        //ndarray.__reduce__()    For pickling.
        //ndarray.__setstate__(state, /)  For unpickling.
        //ndarray.__contains__($self, key, /)	Return key in self.

        //ndarray.__int__(self)
        //ndarray.__long__
        //ndarray.__float__(self)
        //ndarray.__oct__
        //ndarray.__hex__
    }
}
