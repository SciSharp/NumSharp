namespace NumSharp
{
    /// <summary>
    ///     NumPy's legacy promotion table over its builtin TYPE NUMBERS — a port of <c>_npy_can_cast_safely_table</c>
    ///     (<c>can_cast_table.h</c>) and of the table loop of <c>initialize_casting_tables</c> (<c>scalartypes.c.src</c>),
    ///     which NumPy's <c>default_builtin_common_dtype</c> reads as <c>_npy_type_promotion_table[a][b]</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumSharp's frozen promotion table (<see cref="np._nptypemap_arr_arr"/>) is keyed by STORAGE code, which cannot
    ///     tell a lane-sharing C-type class from the owner of its lane (<c>longdouble</c> and <c>float64</c> are both the
    ///     <see cref="NPTypeCode.Double"/> lane; <c>intc</c> and <c>int32</c> both <see cref="NPTypeCode.Int32"/> on
    ///     Windows). NumPy's table is keyed by type number and answers for every pair of its classes, so
    ///     <see cref="DTypePromotion"/> consults it for every pair of NumPy numeric classes: <c>promote(intc, int32)</c> is
    ///     <c>int32</c> (the larger type number of two classes that cast safely to each other), <c>promote(float64,
    ///     longdouble)</c> is <c>longdouble</c>, <c>promote(longdouble, complex128)</c> is <c>complex128</c> — and two
    ///     canonical classes can meet in a C-type class: Windows' <c>promote(int16, uint16)</c> climbs to the next larger
    ///     type number, the C int (<c>'i'</c>), before int32's C long (<c>'l'</c>). The storage lane of every answer is
    ///     the one the frozen table names; the class is what the frozen table cannot say.
    ///     </para>
    ///     <para>
    ///     The sizes are NumSharp's model of the host: C <c>long</c> 4 bytes on LLP64 and 8 on LP64
    ///     (<see cref="DTypeRegistry.CLongIs32Bit"/>), <c>long double</c> 8 bytes on every platform (NumSharp has no
    ///     extended precision, so Linux NumPy's 16-byte answers differ: <c>promote(longdouble, complex128)</c> is
    ///     <c>clongdouble</c> there). Built once, on first use; read-only afterwards.
    ///     </para>
    /// </remarks>
    internal static class NumPyPromotionTable
    {
        // NumPy's legacy type numbers (NPY_TYPES), in NumPy's table order.
        private const int BOOL = 0, BYTE = 1, UBYTE = 2, SHORT = 3, USHORT = 4, INT = 5, UINT = 6, LONG = 7, ULONG = 8,
            LONGLONG = 9, ULONGLONG = 10, FLOAT = 11, DOUBLE = 12, LONGDOUBLE = 13, CFLOAT = 14, CDOUBLE = 15,
            CLONGDOUBLE = 16, OBJECT = 17, STRING = 18, UNICODE = 19, VOID = 20, DATETIME = 21, TIMEDELTA = 22, HALF = 23;

        /// <summary>NumPy's <c>NPY_NTYPES_LEGACY</c>: the builtin type numbers 0 … 23.</summary>
        internal const int NTypesLegacy = 24;

        // NumPy's NPY_SCALARKIND values (ndarraytypes.h): the order a promotion climbs when no type of the larger kind fits.
        private const int NOSCALAR = -1, BOOL_SCALAR = 0, INTPOS_SCALAR = 1, INTNEG_SCALAR = 2, FLOAT_SCALAR = 3,
            COMPLEX_SCALAR = 4, OBJECT_SCALAR = 5, NSCALARKINDS = 6;

        /// <summary>
        ///     <c>_npy_type_promotion_table[a][b]</c>: the type number both <paramref name="a"/> and <paramref name="b"/>
        ///     promote to, or -1 when NumPy's legacy table has none (a string, void or datetime class beside a number).
        /// </summary>
        /// <param name="a">A builtin type number (0 … 23).</param>
        /// <param name="b">A builtin type number (0 … 23).</param>
        /// <returns>The promoted type number, or -1.</returns>
        /// <exception cref="System.IndexOutOfRangeException">A number outside NumPy's builtin range (a caller bug: user classes never reach this table).</exception>
        internal static int Promote(int a, int b) => Promotion[a, b];

        /// <summary><c>_npy_can_cast_safely_table[from][to]</c> under NumSharp's size model (see the remarks).</summary>
        /// <param name="from">A builtin type number.</param>
        /// <param name="to">A builtin type number.</param>
        /// <returns>True when NumPy's legacy rules call the cast safe.</returns>
        internal static bool CanCastSafely(int from, int to) => Safe[from, to];

        /// <summary>The safe-cast table, built before <see cref="Promotion"/> (static fields initialise in declaration order).</summary>
        private static readonly bool[,] Safe = BuildCanCastSafely();

        /// <summary>The promotion table, derived from <see cref="Safe"/>.</summary>
        private static readonly int[,] Promotion = BuildPromotion();

        /// <summary>
        ///     The byte size of each C type in NumSharp's model — the <c>NPY_SIZEOF_*</c> macros the table macros compare.
        ///     A complex type's entry is the size of ONE part (<c>CASTS_SAFELY_FROM_CFLOAT(FROM)</c> is called with the part
        ///     type).
        /// </summary>
        /// <param name="typeNum">A numeric type number.</param>
        /// <returns>The size in bytes.</returns>
        private static int SizeOf(int typeNum) => typeNum switch
        {
            BOOL or BYTE or UBYTE => 1,
            SHORT or USHORT or HALF => 2,
            INT or UINT or FLOAT or CFLOAT => 4,
            LONG or ULONG => DTypeRegistry.CLongIs32Bit ? 4 : 8,
            LONGLONG or ULONGLONG or DOUBLE or CDOUBLE or TIMEDELTA or DATETIME => 8,
            // NumSharp's long double is the 8-byte double of NumPy's win-amd64 build on every platform.
            LONGDOUBLE or CLONGDOUBLE => 8,
            _ => 0,
        };

        /// <summary>
        ///     The port of <c>_npy_can_cast_safely_table</c>: the rows <c>CASTS_SAFELY_FROM_INT</c>,
        ///     <c>CASTS_SAFELY_FROM_UINT</c>, <c>CASTS_SAFELY_FROM_FLOAT</c> and <c>CASTS_SAFELY_FROM_CFLOAT</c> over the
        ///     model sizes, and NumPy's literal rows for bool, object, the strings, void and the datetimes.
        /// </summary>
        /// <returns>The table.</returns>
        private static bool[,] BuildCanCastSafely()
        {
            var t = new bool[NTypesLegacy, NTypesLegacy];

            /* Bool safely casts to anything except datetime (has no zero) */
            for (int to = 0; to < NTypesLegacy; to++)
                t[BOOL, to] = to != DATETIME;

            /* Integers in pairs of signed, unsigned */
            foreach (var (signed, unsigned) in new[] { (BYTE, UBYTE), (SHORT, USHORT), (INT, UINT), (LONG, ULONG), (LONGLONG, ULONGLONG) })
            {
                FromInt(t, signed);
                FromUInt(t, unsigned);
            }

            /* Floats and complex */
            FromFloat(t, FLOAT, FLOAT);
            FromFloat(t, DOUBLE, DOUBLE);
            FromFloat(t, LONGDOUBLE, LONGDOUBLE);
            FromCFloat(t, CFLOAT, FLOAT);
            FromCFloat(t, CDOUBLE, DOUBLE);
            FromCFloat(t, CLONGDOUBLE, LONGDOUBLE);
            FromFloat(t, HALF, HALF);

            /*
             * Following the main numeric types are:
             * object, string, unicode, void, datetime, timedelta (and half)
             */
            /* object casts safely only to itself */
            t[OBJECT, OBJECT] = true;
            /* String casts safely to object, unicode and void */
            t[STRING, OBJECT] = t[STRING, STRING] = t[STRING, UNICODE] = t[STRING, VOID] = true;
            /* Unicode casts safely to object and void */
            t[UNICODE, OBJECT] = t[UNICODE, UNICODE] = t[UNICODE, VOID] = true;
            /* Void cast safely to object */
            t[VOID, OBJECT] = t[VOID, VOID] = true;
            /* datetime cast safely to object, string, unicode, void */
            t[DATETIME, OBJECT] = t[DATETIME, STRING] = t[DATETIME, UNICODE] = t[DATETIME, VOID] = t[DATETIME, DATETIME] = true;
            /* timedelta cast safely to object, string, unicode, void */
            t[TIMEDELTA, OBJECT] = t[TIMEDELTA, STRING] = t[TIMEDELTA, UNICODE] = t[TIMEDELTA, VOID] = t[TIMEDELTA, TIMEDELTA] = true;
            return t;
        }

        /// <summary>NumPy's <c>FITS(FROM, TO)</c>: the source has at most as many bytes as the destination.</summary>
        /// <param name="from">The source type number.</param>
        /// <param name="to">The destination type number (a part type for a complex column).</param>
        /// <returns>True when the source fits.</returns>
        private static bool Fits(int from, int to) => SizeOf(from) <= SizeOf(to);

        /// <summary>NumPy's <c>UFITS(FROM, TO)</c>: an unsigned source fits a signed destination only if truly smaller.</summary>
        /// <param name="from">The unsigned source type number.</param>
        /// <param name="to">The signed destination type number.</param>
        /// <returns>True when the source fits.</returns>
        private static bool UFits(int from, int to) => SizeOf(from) < SizeOf(to);

        /// <summary>
        ///     NumPy's <c>IFITS(FROM, TO)</c>: an integer fits a float only if truly smaller, or of the same size and at least
        ///     a double's (an int64 → float64 cast is "safe" in NumPy's legacy table although it can round).
        /// </summary>
        /// <param name="from">The integer source type number.</param>
        /// <param name="to">The float destination type number (a part type for a complex column).</param>
        /// <returns>True when NumPy's table calls the cast safe.</returns>
        private static bool IFits(int from, int to)
            => SizeOf(from) < SizeOf(to) || (SizeOf(from) == SizeOf(to) && SizeOf(from) >= SizeOf(DOUBLE));

        /// <summary>The row <c>CASTS_SAFELY_FROM_INT(FROM)</c> for a signed integer.</summary>
        /// <param name="t">The table being filled.</param>
        /// <param name="from">The signed integer type number.</param>
        private static void FromInt(bool[,] t, int from)
        {
            t[from, BYTE] = Fits(from, BYTE);
            t[from, SHORT] = Fits(from, SHORT);
            t[from, INT] = Fits(from, INT);
            t[from, LONG] = Fits(from, LONG);
            t[from, LONGLONG] = Fits(from, LONGLONG);
            FloatColumns(t, from);
            t[from, OBJECT] = t[from, STRING] = t[from, UNICODE] = t[from, VOID] = true;
            t[from, TIMEDELTA] = SizeOf(from) <= SizeOf(TIMEDELTA);
            t[from, HALF] = IFits(from, HALF);
        }

        /// <summary>The row <c>CASTS_SAFELY_FROM_UINT(FROM)</c> for an unsigned integer.</summary>
        /// <param name="t">The table being filled.</param>
        /// <param name="from">The unsigned integer type number.</param>
        private static void FromUInt(bool[,] t, int from)
        {
            t[from, BYTE] = UFits(from, BYTE);
            t[from, UBYTE] = Fits(from, UBYTE);
            t[from, SHORT] = UFits(from, SHORT);
            t[from, USHORT] = Fits(from, USHORT);
            t[from, INT] = UFits(from, INT);
            t[from, UINT] = Fits(from, UINT);
            t[from, LONG] = UFits(from, LONG);
            t[from, ULONG] = Fits(from, ULONG);
            t[from, LONGLONG] = UFits(from, LONGLONG);
            t[from, ULONGLONG] = Fits(from, ULONGLONG);
            FloatColumns(t, from);
            t[from, OBJECT] = t[from, STRING] = t[from, UNICODE] = t[from, VOID] = true;
            t[from, TIMEDELTA] = SizeOf(from) < SizeOf(TIMEDELTA);
            t[from, HALF] = IFits(from, HALF);
        }

        /// <summary>The float and complex columns of an integer row: <c>IFITS</c> against each part size.</summary>
        /// <param name="t">The table being filled.</param>
        /// <param name="from">The integer type number.</param>
        private static void FloatColumns(bool[,] t, int from)
        {
            t[from, FLOAT] = t[from, CFLOAT] = IFits(from, FLOAT);
            t[from, DOUBLE] = t[from, CDOUBLE] = IFits(from, DOUBLE);
            t[from, LONGDOUBLE] = t[from, CLONGDOUBLE] = IFits(from, LONGDOUBLE);
        }

        /// <summary>The row <c>CASTS_SAFELY_FROM_FLOAT(FROM)</c>: floats are like ints but cap at double.</summary>
        /// <param name="t">The table being filled.</param>
        /// <param name="row">The float type number whose row this is.</param>
        /// <param name="from">The float type whose size the row compares (the row's own type).</param>
        private static void FromFloat(bool[,] t, int row, int from)
        {
            t[row, FLOAT] = t[row, CFLOAT] = Fits(from, FLOAT);
            t[row, DOUBLE] = t[row, CDOUBLE] = Fits(from, DOUBLE);
            t[row, LONGDOUBLE] = t[row, CLONGDOUBLE] = Fits(from, LONGDOUBLE);
            t[row, OBJECT] = t[row, STRING] = t[row, UNICODE] = t[row, VOID] = true;
            t[row, HALF] = Fits(from, HALF);
        }

        /// <summary>The row <c>CASTS_SAFELY_FROM_CFLOAT(FROM)</c>: a complex type casts safely only to a complex type at least as wide.</summary>
        /// <param name="t">The table being filled.</param>
        /// <param name="row">The complex type number whose row this is.</param>
        /// <param name="part">The float type of one part (the macro's <c>FROM</c>).</param>
        private static void FromCFloat(bool[,] t, int row, int part)
        {
            t[row, CFLOAT] = Fits(part, FLOAT);
            t[row, CDOUBLE] = Fits(part, DOUBLE);
            t[row, CLONGDOUBLE] = Fits(part, LONGDOUBLE);
            t[row, OBJECT] = t[row, STRING] = t[row, UNICODE] = t[row, VOID] = true;
        }

        /// <summary>
        ///     The table loop of <c>initialize_casting_tables</c>: a type promotes with itself to itself; a number never
        ///     promotes with a string, void or datetime class; of two types the one the other casts safely to wins (the
        ///     LARGER type number when both directions are safe — which is how <c>promote(intc, int32)</c> is
        ///     <c>int32</c> on Windows); otherwise the search climbs the next larger type of the larger kind, then the
        ///     smallest type of each larger kind, until both cast safely.
        /// </summary>
        /// <returns>The table.</returns>
        private static int[,] BuildPromotion()
        {
            int[] scalarKind = new int[NTypesLegacy];
            int[] nextLarger = new int[NTypesLegacy];
            for (int i = 0; i < NTypesLegacy; i++)
            {
                /* Default for built-in types is object scalar */
                scalarKind[i] = OBJECT_SCALAR;
                /* Default for next largest type is -1, signalling no bigger */
                nextLarger[i] = -1;
            }
            int[] smallestOfKind = new int[NSCALARKINDS];
            smallestOfKind[BOOL_SCALAR] = BOOL;
            smallestOfKind[INTPOS_SCALAR] = UBYTE;
            smallestOfKind[INTNEG_SCALAR] = BYTE;
            smallestOfKind[FLOAT_SCALAR] = HALF;
            smallestOfKind[COMPLEX_SCALAR] = CFLOAT;
            smallestOfKind[OBJECT_SCALAR] = OBJECT;

            // The repeat block: #NAME / #BIGGERTYPE / #SCKIND.
            scalarKind[BOOL] = BOOL_SCALAR;
            foreach (var (name, bigger, kind) in new[]
                     {
                         (BYTE, SHORT, INTNEG_SCALAR), (UBYTE, USHORT, INTPOS_SCALAR), (SHORT, INT, INTNEG_SCALAR),
                         (USHORT, UINT, INTPOS_SCALAR), (INT, LONG, INTNEG_SCALAR), (UINT, ULONG, INTPOS_SCALAR),
                         (LONG, LONGLONG, INTNEG_SCALAR), (ULONG, ULONGLONG, INTPOS_SCALAR), (LONGLONG, -1, INTNEG_SCALAR),
                         (ULONGLONG, -1, INTPOS_SCALAR), (HALF, FLOAT, FLOAT_SCALAR), (FLOAT, DOUBLE, FLOAT_SCALAR),
                         (DOUBLE, LONGDOUBLE, FLOAT_SCALAR), (LONGDOUBLE, -1, FLOAT_SCALAR), (CFLOAT, CDOUBLE, COMPLEX_SCALAR),
                         (CDOUBLE, CLONGDOUBLE, COMPLEX_SCALAR), (CLONGDOUBLE, -1, COMPLEX_SCALAR),
                     })
            {
                scalarKind[name] = kind;
                nextLarger[name] = bigger;
            }

            var table = new int[NTypesLegacy, NTypesLegacy];
            for (int i = 0; i < NTypesLegacy; ++i)
            {
                table[i, i] = i;
                /* Don't let number promote to string/unicode/void/datetime/timedelta */
                if (i == STRING || i == UNICODE || i == VOID || i == DATETIME || i == TIMEDELTA)
                {
                    /* Promoting these types requires examining their contents */
                    table[i, i] = -1;
                    for (int j = i + 1; j < NTypesLegacy; ++j)
                        table[i, j] = table[j, i] = -1;
                    /* Except they can convert to OBJECT */
                    table[i, OBJECT] = table[OBJECT, i] = OBJECT;
                    continue;
                }

                for (int j = i + 1; j < NTypesLegacy; ++j)
                {
                    /* Don't let number promote to string/unicode/void */
                    if (j == STRING || j == UNICODE || j == VOID)
                    {
                        table[i, j] = table[j, i] = -1;
                    }
                    else if (Safe[i, j])
                    {
                        table[i, j] = table[j, i] = j;
                    }
                    else if (Safe[j, i])
                    {
                        table[i, j] = table[j, i] = i;
                    }
                    else
                    {
                        int k, skind;
                        int iskind = scalarKind[i], jskind = scalarKind[j];
                        /* If there's no kind (void/string/etc) */
                        if (iskind == NOSCALAR || jskind == NOSCALAR)
                        {
                            k = -1;
                        }
                        else
                        {
                            /* Start with the type of larger kind */
                            if (iskind > jskind)
                            {
                                skind = iskind;
                                k = i;
                            }
                            else
                            {
                                skind = jskind;
                                k = j;
                            }
                            for (;;)
                            {
                                /* Try the next larger type of this kind */
                                k = nextLarger[k];

                                /* If there is no larger, try a larger kind */
                                if (k < 0)
                                {
                                    ++skind;
                                    /* Use -1 to signal no promoted type found */
                                    if (skind < NSCALARKINDS)
                                    {
                                        k = smallestOfKind[skind];
                                    }
                                    else
                                    {
                                        k = -1;
                                        break;
                                    }
                                }

                                if (Safe[i, k] && Safe[j, k])
                                    break;
                            }
                        }
                        table[i, j] = table[j, i] = k;
                    }
                }
            }
            return table;
        }
    }
}
