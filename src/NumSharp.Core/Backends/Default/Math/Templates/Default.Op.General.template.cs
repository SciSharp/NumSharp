#if _REGEN_TEMPLATE
%template "../Add/Default.Add.#1.cs" for every supported_dtypes, supported_dtypes_lowercase, repeatelement("Add", supported_dtypes.Count)
%template "../Subtract/Default.Subtract.#1.cs" for every supported_dtypes, supported_dtypes_lowercase, repeatelement("Subtract", supported_dtypes.Count)
%template "../Multiply/Default.Multiply.#1.cs" for every supported_dtypes, supported_dtypes_lowercase, repeatelement("Multiply", supported_dtypes.Count)
%template "../Divide/Default.Divide.#1.cs" for every supported_dtypes, supported_dtypes_lowercase, repeatelement("Divide", supported_dtypes.Count)
%template "../Mod/Default.Mod.#1.cs" for every supported_dtypes, supported_dtypes_lowercase, repeatelement("Mod", supported_dtypes.Count)
#endif

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;
using NumSharp.Utilities.Maths;

namespace NumSharp.Backends
{
    //v2
    public partial class DefaultEngine
    {
        [MethodImpl((MethodImplOptions)768)]
        [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
        [SuppressMessage("ReSharper", "CompareOfFloatsByEqualityOperator")]
        public unsafe NDArray __3____1__(in NDArray lhs, in NDArray rhs)
        {
            //lhs is NDArray of __2__
            switch (rhs.GetTypeCode)
            {
#if _REGEN
                %op = "__3__"
                %op_bool = "*"
	            %foreach supported_dtypes, supported_dtypes_lowercase%
                case NPTypeCode.#1: {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                        return NDArray.Scalar(Converts.ChangeType(Operator.#(op)(*((__2__*)lhs.Address), *((#2*)rhs.Address)), ret_type));
                    
                    (Shape leftshape, Shape rightshape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (#2*)rhs.Address;
                    var retShape = leftshape.Clean();
                    var ret = new NDArray(ret_type, retShape, false);
                    var leftLinear = !leftshape.IsBroadcasted && !leftshape.IsSliced;
                    var rightLinear = !rightshape.IsBroadcasted && !rightshape.IsSliced;
                    switch (ret_type) {
                        %foreach supported_dtypes,supported_dtypes_lowercase%
                        |#normalcast = ("("+str("#102")+")")
                        |#caster = ( "#102"=="bool" | ("Converts.To" + str("#101")) | ("__2__"=="bool"|("#2"=="bool"|("Converts.To" + str("#101"))|normalcast)| normalcast) )
	                    case NPTypeCode.#101: {
		                    var ret_address = (#102*)ret.Address;
                            if (leftLinear && rightLinear) {
                                var len = ret.size;
                                Debug.Assert(leftshape.size == len && rightshape.size == len);
                                if (rightshape.IsBroadcasted && rightshape.BroadcastInfo.OriginalShape.IsScalar) {
                                    var rval =  *rhs_address;
                                    Parallel.For(0, len, i => ret_address[i] = #(caster)(Operator.__3__((lhs_address[i]), rval)));
                                } else if (leftshape.IsBroadcasted && leftshape.BroadcastInfo.OriginalShape.IsScalar) {
                                    var lval =  *lhs_address;
                                    Parallel.For(0, len, i => ret_address[i] = #(caster)(Operator.__3__(lval, (rhs_address[i]))));
                                } else {
                                    Parallel.For(0, len, i => ret_address[i] = #(caster)(Operator.__3__((lhs_address[i]), (rhs_address[i]))));
                                }
                            } else if (leftLinear) { // && !rightLinear
                                if (rightshape.IsBroadcasted && rightshape.BroadcastInfo.OriginalShape.IsScalar) {
                                    var rval =  *rhs_address;
                                    Parallel.For(0, ret.size, i => ret_address[i] = #(caster)(Operator.__3__((lhs_address[i]), rval)));
                                } else {
                                    int leftOffset = 0;
                                    int retOffset = 0;
                                    var incr = new NDCoordinatesIncrementor(ref retShape);
                                    int[] current = incr.Index;
                                    Func<int[], int> rightOffset = rightshape.GetOffset;
                                    do {
                                        *(ret_address + retOffset++) = #(caster)(Operator.__3__((*(lhs_address + leftOffset++)), (*(rhs_address + rightOffset(current)))));
                                    } while (incr.Next() != null);
                                }
                            } else if (rightLinear) { // !leftLinear && 
                                if (leftshape.IsBroadcasted && leftshape.BroadcastInfo.OriginalShape.IsScalar) {
                                    var lval =  *lhs_address;
                                    Parallel.For(0, ret.size, i => ret_address[i] = #(caster)(Operator.__3__(lval, (rhs_address[i]))));
                                } else {
                                    int rightOffset = 0;
                                    int retOffset = 0;
                                    var incr = new NDCoordinatesIncrementor(ref retShape);
                                    int[] current = incr.Index;
                                    Func<int[], int> leftOffset = leftshape.GetOffset;
                                    do {
                                        *(ret_address + retOffset++) = #(caster)(Operator.__3__((*(lhs_address + leftOffset(current))), (*(rhs_address + rightOffset++))));
                                    } while (incr.Next() != null);
                                }
                            } else {
                                int retOffset = 0;
                                var incr = new NDCoordinatesIncrementor(ref retShape);
                                int[] current = incr.Index;
                                Func<int[], int> rightOffset = rightshape.GetOffset;
                                Func<int[], int> leftOffset = leftshape.GetOffset;
                                do {
                                    *(ret_address + retOffset++) = #(caster)(Operator.__3__((*(lhs_address + leftOffset(current))), (*(rhs_address + rightOffset(current)))));
                                } while (incr.Next() != null);
                            }

                            return ret;
	                    }
                        %
	                    default:
		                    throw new NotSupportedException();
                    }
                }
                %
                default:
		            throw new NotSupportedException();
#else

#endif
            }
        }
    }
}
