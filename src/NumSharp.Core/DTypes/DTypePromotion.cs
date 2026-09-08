using System;
using System.Collections.Generic;
using System.Numerics;

namespace NumSharp
{
    /// <summary>
    ///     The NEP 42 / NEP 50 promotion engine over DType classes and descriptors — the ports of NumPy's
    ///     <c>common_dtype.c</c> and <c>convert_datatype.c</c>: <see cref="CommonDType"/> (<c>PyArray_CommonDType</c>),
    ///     <see cref="PromoteDTypeSequence"/> (<c>PyArray_PromoteDTypeSequence</c> with its
    ///     <c>reduce_dtypes_to_most_knowledgeable</c> pass), <see cref="PromoteTypes"/> (<c>PyArray_PromoteTypes</c>),
    ///     <see cref="CastDescrToDType"/> (<c>PyArray_CastDescrToDType</c>),
    ///     <see cref="CastToDTypeAndPromoteDescriptors"/> and <see cref="ResultType"/> (<c>PyArray_ResultType</c>, the
    ///     NEP 50 entry point where C# literals are weak and arrays — 0-d included — are strong).
    /// </summary>
    /// <remarks>
    ///     For the storage-backed builtins the answers ARE NumSharp's frozen promotion table
    ///     (<see cref="np._nptypemap_arr_arr"/>, consulted by <see cref="DefaultBuiltinCommonDType"/>), so every
    ///     two-operand <c>promote_types</c> / <c>result_type</c> answer the <c>dtype_text</c> fuzz tier gates is unchanged.
    ///     Three things the old engine could not do come with the port: order-independent reduction of THREE OR MORE
    ///     operands (the old fold dropped every other pair, so <c>result_type(i1, i1, f8, i1)</c> was <c>int8</c>),
    ///     weak C# literals (<c>result_type(int8_array, 300)</c> is <c>int8</c>, <c>result_type(int8_array, 1.5)</c> is
    ///     <c>float64</c>), and parametric classes (the datetime unit GCD).
    /// </remarks>
    public static class DTypePromotion
    {
        /// <summary>
        ///     NumPy's <c>PyArray_CommonDType</c>: <c>dtype1.common_dtype(dtype2)</c>, then the mirrored question, then
        ///     <see cref="DTypePromotionError"/> with NumPy's verbatim text.
        /// </summary>
        public static DTypeMeta CommonDType(DTypeMeta dtype1, DTypeMeta dtype2)
        {
            if (dtype1 == null) throw new ArgumentNullException(nameof(dtype1));
            if (dtype2 == null) throw new ArgumentNullException(nameof(dtype2));
            if (ReferenceEquals(dtype1, dtype2))
                return dtype1;

            var common = dtype1.CommonDType(dtype2) ?? dtype2.CommonDType(dtype1);
            if (common == null)
                throw new DTypePromotionError(
                    $"The DTypes {dtype1} and {dtype2} do not have a common DType. " +
                    "For example they cannot be stored in a single array unless the dtype is `object`.");
            return common;
        }

        /// <summary>
        ///     NumPy's <c>default_builtin_common_dtype</c> — the <c>common_dtype</c> slot every builtin class shares:
        ///     <list type="number">
        ///       <item><description>a NEP 50 scalar class: a weak complex adopts a complex class and lifts a real float class
        ///       to complex; a weak float adopts a float or complex class; a weak int adopts any numeric class (and
        ///       <c>timedelta64</c>); anything else defers (null);</description></item>
        ///       <item><description>an operand with a LARGER type number is "more generic" and must answer instead (null);</description></item>
        ///       <item><description>otherwise the frozen promotion table decides — for the datetime classes NumPy's table row
        ///       says <c>timedelta64</c> absorbs bool and the integers narrower than uint64, and nothing else promotes.</description></item>
        ///     </list>
        /// </summary>
        internal static DTypeMeta DefaultBuiltinCommonDType(DTypeMeta cls, DTypeMeta other)
        {
            if (other is PyScalarDTypeMeta py)
            {
                /*
                 * Deal with the non-legacy types we understand: python scalars.
                 * These may have lower priority than the concrete inexact types,
                 * but can change the type of the result (complex, float, int).
                 * If our own DType is not numerical or has lower priority (e.g.
                 * integer but abstract one is float), signal not implemented.
                 */
                switch (py.PyKind)
                {
                    case PyScalarKind.Complex:
                        if (cls.Kind == 'c')
                            return cls;
                        if (cls.Kind == 'f')
                            return DTypeRegistry.Complex128; // NumSharp's single complex width
                        break;
                    case PyScalarKind.Float:
                        if (cls.Kind == 'c' || cls.Kind == 'f')
                            return cls;
                        break;
                    case PyScalarKind.Int:
                        if (cls.Kind == 'c' || cls.Kind == 'f' || cls.Kind == 'i' || cls.Kind == 'u'
                            || (cls is DatetimeDTypeMeta dt && dt.IsTimedelta))
                            return cls;
                        break;
                }
                return null;
            }

            if (other.TypeNum > cls.TypeNum)
            {
                /*
                 * Let the more generic (larger type number) DType handle this
                 * (note that half is after all others, which works out here.)
                 */
                return null;
            }

            if (cls.HasStorage && other.HasStorage
                && np._nptypemap_arr_arr.TryGetValue((cls.TypeCode, other.TypeCode), out var promoted))
            {
                return DTypeRegistry.FromTypeCode(promoted);
            }

            if (cls is DatetimeDTypeMeta dtm)
            {
                /*
                 * NumPy's _npy_type_promotion_table rows for NPY_TIMEDELTA: an integer (or bool) casts safely to
                 * timedelta64 — except a 64-bit unsigned one — so those pairs promote to timedelta64; datetime64
                 * promotes with nothing numeric.
                 */
                if (dtm.IsTimedelta && other.IsLegacy && other.IsNumeric
                    && (other.Kind == 'b' || other.Kind == 'i' || (other.Kind == 'u' && other.ItemSize < 8)))
                    return cls;
            }

            return null;
        }

        /// <summary>
        ///     NumPy's <c>reduce_dtypes_to_most_knowledgeable</c>: pairwise low/high reduction that swaps the pair when
        ///     the low class defers (so the more knowledgeable class migrates to the front) and clears the high entry when
        ///     it cannot influence the result. Returns the last pairwise result; <paramref name="notImplemented"/> reports
        ///     whether that result was a deferral.
        /// </summary>
        private static DTypeMeta ReduceDTypesToMostKnowledgeable(int length, DTypeMeta[] dtypes, out bool notImplemented)
        {
            int half = length / 2;
            DTypeMeta res = null;
            notImplemented = false;

            for (int low = 0; low < half; low++)
            {
                int high = length - 1 - low;
                if (ReferenceEquals(dtypes[high], dtypes[low]))
                {
                    /* Fast path for identical dtypes: do not call common_dtype */
                    res = dtypes[low];
                    notImplemented = false;
                }
                else
                {
                    res = dtypes[low].CommonDType(dtypes[high]);
                    notImplemented = res == null;
                }

                if (notImplemented)
                {
                    /* guess at other being more "knowledgeable" */
                    (dtypes[low], dtypes[high]) = (dtypes[high], dtypes[low]);
                }
                else if (ReferenceEquals(res, dtypes[low]))
                {
                    /* `dtypes[high]` cannot influence result: clear */
                    dtypes[high] = null;
                }
            }

            if (length == 2)
                return res;
            return ReduceDTypesToMostKnowledgeable(length - half, dtypes, out notImplemented);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_PromoteDTypeSequence</c>: the common class of any number of classes, computed so the
        ///     answer does not depend on operand order (the "most knowledgeable" class — the highest category — promotes
        ///     every other one; two mutually unknown classes raise).
        /// </summary>
        /// <exception cref="DTypePromotionError">NumPy's verbatim <c>The DType %S could not be promoted by %S. …</c>.</exception>
        public static DTypeMeta PromoteDTypeSequence(IReadOnlyList<DTypeMeta> dtypesIn)
        {
            if (dtypesIn == null) throw new ArgumentNullException(nameof(dtypesIn));
            int length = dtypesIn.Count;
            if (length == 0)
                throw new ValueError("at least one array or dtype is required");
            if (length == 1)
                return dtypesIn[0] ?? throw new ArgumentNullException(nameof(dtypesIn));

            /* Copy dtypes so that we can reorder them */
            var dtypes = new DTypeMeta[length];
            for (int i = 0; i < length; i++)
                dtypes[i] = dtypesIn[i] ?? throw new ArgumentNullException(nameof(dtypesIn));

            /*
             * `result` is the last promotion result, which can usually be reused if
             * it is not NotImplemented.
             * The passed in dtypes are partially sorted (and cleared, when clearly
             * not relevant anymore).
             * `dtypes[0]` will be the most knowledgeable (highest category) which
             * we consider the "main_dtype" here.
             */
            var result = ReduceDTypesToMostKnowledgeable(length, dtypes, out bool notImplemented);
            var mainDType = dtypes[0];

            int reduceStart = 1;
            if (notImplemented)
                result = null;
            else
                reduceStart = 2; /* (new) first value is already taken care of in `result` */

            /*
             * At this point, we have only looked at every DType at most once.
             * The `main_dtype` must know all others (or it will be a failure) and
             * all dtypes returned by its `common_dtype` must be guaranteed to succeed
             * promotion with one another.
             * It is the job of the "main DType" to ensure that at this point order
             * is irrelevant.
             */
            for (int i = reduceStart; i < length; i++)
            {
                if (dtypes[i] == null)
                    continue;
                /*
                 * "Promote" the current dtype with the main one (which should be
                 * a higher category). We assume that the result is not in a lower
                 * category.
                 */
                var promotion = mainDType.CommonDType(dtypes[i]);
                if (promotion == null)
                {
                    throw new DTypePromotionError(
                        $"The DType {dtypes[i]} could not be promoted by {mainDType}. This means that " +
                        "no common DType exists for the given inputs. " +
                        "For example they cannot be stored in a single array unless " +
                        $"the dtype is `object`. The full list of DTypes is: ({string.Join(", ", dtypesIn)})");
                }
                if (result == null)
                {
                    result = promotion;
                    continue;
                }
                /*
                 * The above promoted, now "reduce" with the current result; note that
                 * in the typical cases we expect this step to be a no-op.
                 */
                result = CommonDType(result, promotion);
            }

            return result;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_PromoteTypes</c>: the smallest descriptor both inputs cast to safely. An identical
        ///     native legacy input is returned as is (metadata preserved); otherwise the common CLASS decides — a
        ///     non-parametric class yields its default descriptor, a parametric one casts both inputs to itself
        ///     (<see cref="CastDescrToDType"/>) and takes their <c>common_instance</c>.
        /// </summary>
        public static DType PromoteTypes(DType type1, DType type2)
        {
            if (type1 == null) throw new ArgumentNullException(nameof(type1));
            if (type2 == null) throw new ArgumentNullException(nameof(type2));

            /* Fast path for identical inputs (NOTE: This path preserves metadata!) */
            if (type1.Equals(type2) && type1.Meta.IsLegacy && type1.isnative)
                return type1;

            var commonDType = CommonDType(type1.Meta, type2.Meta);

            if (!commonDType.IsParametric)
            {
                /* Note that this path loses all metadata */
                return commonDType.DefaultDescr();
            }

            /* Cast the input types to the common DType if necessary */
            var d1 = CastDescrToDType(type1, commonDType);
            var d2 = CastDescrToDType(type2, commonDType);

            /*
             * And find the common instance of the two inputs
             * NOTE: Common instance preserves metadata (normally and of one input)
             */
            return commonDType.CommonInstance(d1, d2);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_CastDescrToDType</c>: the instance of <paramref name="givenDType"/> that can represent
        ///     <paramref name="descr"/> — the descriptor itself when already of that class, the class default when the
        ///     class is not parametric, else whatever the registered cast resolves (a datetime keeps its unit when cast
        ///     between <c>datetime64</c> and <c>timedelta64</c>; an integer cast to <c>timedelta64</c> lands on the generic
        ///     unit).
        /// </summary>
        /// <exception cref="TypeError"><c>cannot cast dtype %S to %S.</c></exception>
        public static DType CastDescrToDType(DType descr, DTypeMeta givenDType)
        {
            if (descr == null) throw new ArgumentNullException(nameof(descr));
            if (givenDType == null) throw new ArgumentNullException(nameof(givenDType));

            if (ReferenceEquals(descr.Meta, givenDType))
                return descr;
            if (!givenDType.IsParametric)
            {
                /*
                 * Don't actually do anything, the default is always the result
                 * of any cast.
                 */
                return givenDType.DefaultDescr();
            }

            var impl = descr.Meta.GetCastingImpl(givenDType);
            if (impl == null)
                throw new TypeError($"cannot cast dtype {descr} to {givenDType}.");

            var given = new[] { descr, null };
            var loop = new DType[2];
            try
            {
                impl.ResolveDescriptors(given, loop, out _);
            }
            catch (TypeError e)
            {
                throw new TypeError($"cannot cast dtype {descr} to {givenDType}.", e);
            }
            return loop[1];
        }

        /// <summary>
        ///     NumPy's <c>PyArray_CastToDTypeAndPromoteDescriptors</c>: cast every descriptor to <paramref name="dtype"/>
        ///     and fold their <c>common_instance</c> (the string-length / datetime-unit resolution over N operands).
        /// </summary>
        public static DType CastToDTypeAndPromoteDescriptors(IReadOnlyList<DType> descrs, DTypeMeta dtype)
        {
            if (descrs == null) throw new ArgumentNullException(nameof(descrs));
            if (dtype == null) throw new ArgumentNullException(nameof(dtype));
            if (descrs.Count == 0) throw new ArgumentException("at least one descriptor is required", nameof(descrs));

            var result = CastDescrToDType(descrs[0], dtype);
            if (descrs.Count == 1)
                return result;
            if (!dtype.IsParametric)
            {
                /* Note that this "fast" path loses all metadata */
                return dtype.DefaultDescr();
            }
            for (int i = 1; i < descrs.Count; i++)
            {
                var curr = CastDescrToDType(descrs[i], dtype);
                result = dtype.CommonInstance(result, curr);
            }
            return result;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_ResultType</c> under NEP 50: the descriptor of the result of combining
        ///     <paramref name="arraysAndDtypes"/>, where every <see cref="NDArray"/> (0-d included), <see cref="DType"/>,
        ///     <see cref="NPTypeCode"/>, <see cref="Type"/>, dtype string, <c>bool</c>, <c>char</c>, <c>Half</c> and
        ///     <c>decimal</c> is STRONG (contributes its dtype), while a C# integer, <c>float</c>/<c>double</c> or
        ///     <see cref="Complex"/> literal is WEAK — it contributes only its NEP 50 class (<c>_PyLongDType</c>,
        ///     <c>_PyFloatDType</c>, <c>_PyComplexDType</c>) and no width, so <c>result_type(int8_array, 300)</c> is
        ///     <c>int8</c>. A <see cref="DTypeMeta"/> operand contributes the class alone.
        /// </summary>
        /// <exception cref="ValueError"><c>at least one array or dtype is required</c>.</exception>
        /// <exception cref="DTypePromotionError">No common dtype exists.</exception>
        public static DType ResultType(params object[] arraysAndDtypes)
        {
            if (arraysAndDtypes == null || arraysAndDtypes.Length == 0)
                throw new ValueError("at least one array or dtype is required");

            int n = arraysAndDtypes.Length;
            var allDTypes = new DTypeMeta[n];
            var allDescriptors = new DType[n];
            for (int i = 0; i < n; i++)
                ClassifyOperand(arraysAndDtypes[i], out allDTypes[i], out allDescriptors[i]);

            if (n == 1)
            {
                /* If the input is a single value, skip promotion. */
                return allDescriptors[0] != null ? allDTypes[0].EnsureCanonical(allDescriptors[0]) : allDTypes[0].DefaultDescr();
            }

            var commonDType = PromoteDTypeSequence(allDTypes);

            if (commonDType.IsAbstract)
            {
                /* (ab)use default descriptor to define a default */
                commonDType = commonDType.DefaultDescr().Meta;
            }

            DType result = null;
            if (commonDType.IsParametric)
            {
                for (int i = 0; i < n; i++)
                {
                    if (allDescriptors[i] == null)
                        continue; /* originally a python scalar/literal */
                    var curr = CastDescrToDType(allDescriptors[i], commonDType);
                    result = result == null ? curr : commonDType.CommonInstance(result, curr);
                }
            }

            /*
             * If the DType is not parametric, or all were weak scalars,
             * a result may not yet be set.
             */
            return result ?? commonDType.DefaultDescr();
        }

        /// <summary>
        ///     Sorts one <c>result_type</c> operand into (class, descriptor-or-null): the NEP 50 weak/strong rule for C#.
        /// </summary>
        internal static void ClassifyOperand(object operand, out DTypeMeta meta, out DType descr)
        {
            switch (operand)
            {
                case null:
                    throw new ArgumentNullException(nameof(operand), "result_type operands cannot be null");
                case DType d:
                    meta = d.Meta;
                    descr = d;
                    return;
                case DTypeMeta m:
                    meta = m;
                    descr = null;
                    return;
                case NDArray arr:
                    descr = DType.From(arr.typecode);
                    meta = descr.Meta;
                    return;
                case NPTypeCode tc:
                    descr = DType.From(tc);
                    meta = descr.Meta;
                    return;
                case Type t:
                    descr = DType.From(t);
                    meta = descr.Meta;
                    return;
                case string s:
                    descr = np.dtype(s);
                    meta = descr.Meta;
                    return;
                case bool:
                    descr = DType.Boolean;
                    meta = descr.Meta;
                    return;
                case char:
                    descr = DType.Char;
                    meta = descr.Meta;
                    return;
                case Half:
                    descr = DType.Half;
                    meta = descr.Meta;
                    return;
                case decimal:
                    descr = DType.Decimal;
                    meta = descr.Meta;
                    return;
                case sbyte:
                case byte:
                case short:
                case ushort:
                case int:
                case uint:
                case long:
                case ulong:
                    meta = DTypeRegistry.PyLong;
                    descr = null;
                    return;
                case float:
                case double:
                    meta = DTypeRegistry.PyFloat;
                    descr = null;
                    return;
                case Complex:
                    meta = DTypeRegistry.PyComplex;
                    descr = null;
                    return;
                default:
                    throw new TypeError($"Cannot interpret '{operand}' as a data type");
            }
        }
    }
}
