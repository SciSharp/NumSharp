using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     OFFLINE twins of the pythonnet tier's text edge families (<c>TextEdgeLiveParityTests</c>): the same operations
    ///     over the same four texts, held to numpy 2.4.2's answers recorded by
    ///     <c>test/oracle/gen_text_edge_twins.py</c> — so the main suite, which runs without Python, guards the edges too:
    ///     operands of two text classes or NA objects (numpy's promotion, its "no loop" / "cannot use operands" refusals,
    ///     the incompatible-NA refusal), every <c>StringDType</c> NA kind through the functions whose answer depends on it,
    ///     the ufunc keywords (<c>out=</c> of another dtype or width, <c>where=</c>, <c>dtype=</c>, and <c>out=</c>'s
    ///     identity), and <c>numpy.char</c>'s trailing-whitespace comparisons.
    /// </summary>
    /// <remarks>
    ///     <para>Each row is numpy's answer: the result's dtype (<c>dtype.str</c>, or the <c>StringDType</c> repr) and its
    ///     elements in C order (an NA as <c>null</c> / <c>double.NaN</c>, an <c>S</c> element as its Latin-1 text), or the
    ///     exception's class chain and text. A NumSharp exception matches numpy's most specific builtin class by name, or
    ///     one of the two documented house conventions (a <c>UFuncTypeError</c> cast refusal is an
    ///     <see cref="ArgumentException"/>, a shape <c>ValueError</c> an <see cref="IncorrectShapeException"/>); its text,
    ///     .NET's <c>(Parameter …)</c> decoration cut, is numpy's.</para>
    ///     <para>The operands: <c>['ab', 'c ', '', 'xyz']</c> as <c>U3</c>, <c>S3</c>, <c>StringDType()</c> and three NA
    ///     kinds with element 1 the NA (<c>None</c>, NaN, the string <c>'NA'</c>); a binary operation's second operand is the
    ///     reversed array.</para>
    /// </remarks>
    [TestClass]
    public class TextBridgeEdgeTwinTests
    {
        /// <summary>The four texts every operand holds (element 1 is the NA of an NA kind).</summary>
        private static readonly string[] Data = { "ab", "c ", "", "xyz" };

        /// <summary>
        ///     numpy's recorded answers (generated; do not edit by hand — re-run the generator): (label, dtype or null,
        ///     elements or null, exception class chain or null, exception text or null).
        /// </summary>
        private static readonly (string Label, string DType, object[] Values, string Mro, string Message)[] Rows =
        {
            // cross class
            ("U add U", "<U6", new object[] { "abxyz", "c ", "c ", "xyzab" }, null, null),
            ("U equal U", "|b1", new object[] { false, false, false, false }, null, null),
            ("U less U", "|b1", new object[] { true, false, true, false }, null, null),
            ("U add S", null, null, "UFuncTypeError,UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'add' cannot use operands with types dtype('<U3') and dtype('S3')"),
            ("U equal S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("U less S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("U add T", "StringDType()", new object[] { "abxyz", "c ", "c ", "xyzab" }, null, null),
            ("U equal T", "|b1", new object[] { false, false, false, false }, null, null),
            ("U less T", "|b1", new object[] { true, false, true, false }, null, null),
            ("U add TNone", "StringDType(na_object=None)", new object[] { "abxyz", "c ", "", "xyzab" }, null, null),
            ("U equal TNone", "|b1", new object[] { false, false, true, false }, null, null),
            ("U less TNone", "|b1", new object[] { true, false, false, false }, null, null),
            ("U add Tnan", "StringDType(na_object=nan)", new object[] { "abxyz", "c ", "", "xyzab" }, null, null),
            ("U equal Tnan", "|b1", new object[] { false, false, true, false }, null, null),
            ("U less Tnan", "|b1", new object[] { true, false, false, false }, null, null),
            ("U add TNA", "StringDType(na_object='NA')", new object[] { "abxyz", "c ", "", "xyzab" }, null, null),
            ("U equal TNA", "|b1", new object[] { false, false, true, false }, null, null),
            ("U less TNA", "|b1", new object[] { true, false, false, false }, null, null),
            ("S add U", null, null, "UFuncTypeError,UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'add' cannot use operands with types dtype('S3') and dtype('<U3')"),
            ("S equal U", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StrDType'>) -> None"),
            ("S less U", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StrDType'>) -> None"),
            ("S add S", "|S6", new object[] { "abxyz", "c ", "c ", "xyzab" }, null, null),
            ("S equal S", "|b1", new object[] { false, false, false, false }, null, null),
            ("S less S", "|b1", new object[] { true, false, true, false }, null, null),
            ("S add T", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'add' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S equal T", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S less T", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S add TNone", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'add' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S equal TNone", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S less TNone", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S add Tnan", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'add' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S equal Tnan", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S less Tnan", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S add TNA", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'add' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S equal TNA", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("S less TNA", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StringDType'>) -> None"),
            ("T add U", "StringDType()", new object[] { "abxyz", "c ", "c ", "xyzab" }, null, null),
            ("T equal U", "|b1", new object[] { false, false, false, false }, null, null),
            ("T less U", "|b1", new object[] { true, false, true, false }, null, null),
            ("T add S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'add' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("T equal S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("T less S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("T add T", "StringDType()", new object[] { "abxyz", "c ", "c ", "xyzab" }, null, null),
            ("T equal T", "|b1", new object[] { false, false, false, false }, null, null),
            ("T less T", "|b1", new object[] { true, false, true, false }, null, null),
            ("T add TNone", "StringDType(na_object=None)", new object[] { "abxyz", "c ", "", "xyzab" }, null, null),
            ("T equal TNone", "|b1", new object[] { false, false, true, false }, null, null),
            ("T less TNone", "|b1", new object[] { true, false, false, false }, null, null),
            ("T add Tnan", "StringDType(na_object=nan)", new object[] { "abxyz", "c ", "", "xyzab" }, null, null),
            ("T equal Tnan", "|b1", new object[] { false, false, true, false }, null, null),
            ("T less Tnan", "|b1", new object[] { true, false, false, false }, null, null),
            ("T add TNA", "StringDType(na_object='NA')", new object[] { "abxyz", "c ", "", "xyzab" }, null, null),
            ("T equal TNA", "|b1", new object[] { false, false, true, false }, null, null),
            ("T less TNA", "|b1", new object[] { true, false, false, false }, null, null),
            ("TNone add U", null, null, "ValueError,Exception,BaseException,object", "Cannot add null that is not a nan-like value"),
            ("TNone equal U", "|b1", new object[] { false, true, false, false }, null, null),
            ("TNone less U", null, null, "ValueError,Exception,BaseException,object", "'less' not supported for null values that are not nan-like or strings."),
            ("TNone add S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'add' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("TNone equal S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("TNone less S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("TNone add T", null, null, "ValueError,Exception,BaseException,object", "Cannot add null that is not a nan-like value"),
            ("TNone equal T", "|b1", new object[] { false, true, false, false }, null, null),
            ("TNone less T", null, null, "ValueError,Exception,BaseException,object", "'less' not supported for null values that are not nan-like or strings."),
            ("TNone add TNone", null, null, "ValueError,Exception,BaseException,object", "Cannot add null that is not a nan-like value"),
            ("TNone equal TNone", "|b1", new object[] { false, true, true, false }, null, null),
            ("TNone less TNone", null, null, "ValueError,Exception,BaseException,object", "'less' not supported for null values that are not nan-like or strings."),
            ("TNone add Tnan", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'None' and 'nan'"),
            ("TNone equal Tnan", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'None' and 'nan'"),
            ("TNone less Tnan", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'None' and 'nan'"),
            ("TNone add TNA", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'None' and ''NA''"),
            ("TNone equal TNA", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'None' and ''NA''"),
            ("TNone less TNA", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'None' and ''NA''"),
            ("Tnan add U", "StringDType(na_object=nan)", new object[] { "abxyz", double.NaN, "c ", "xyzab" }, null, null),
            ("Tnan equal U", "|b1", new object[] { false, false, false, false }, null, null),
            ("Tnan less U", "|b1", new object[] { true, false, true, false }, null, null),
            ("Tnan add S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'add' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("Tnan equal S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("Tnan less S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("Tnan add T", "StringDType(na_object=nan)", new object[] { "abxyz", double.NaN, "c ", "xyzab" }, null, null),
            ("Tnan equal T", "|b1", new object[] { false, false, false, false }, null, null),
            ("Tnan less T", "|b1", new object[] { true, false, true, false }, null, null),
            ("Tnan add TNone", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'nan' and 'None'"),
            ("Tnan equal TNone", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'nan' and 'None'"),
            ("Tnan less TNone", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'nan' and 'None'"),
            ("Tnan add Tnan", "StringDType(na_object=nan)", new object[] { "abxyz", double.NaN, double.NaN, "xyzab" }, null, null),
            ("Tnan equal Tnan", "|b1", new object[] { false, false, false, false }, null, null),
            ("Tnan less Tnan", "|b1", new object[] { true, false, false, false }, null, null),
            ("Tnan add TNA", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'nan' and ''NA''"),
            ("Tnan equal TNA", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'nan' and ''NA''"),
            ("Tnan less TNA", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings 'nan' and ''NA''"),
            ("TNA add U", "StringDType(na_object='NA')", new object[] { "abxyz", "NA", "c ", "xyzab" }, null, null),
            ("TNA equal U", "|b1", new object[] { false, false, false, false }, null, null),
            ("TNA less U", "|b1", new object[] { true, false, true, false }, null, null),
            ("TNA add S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'add' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("TNA equal S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("TNA less S", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BytesDType'>) -> None"),
            ("TNA add T", "StringDType(na_object='NA')", new object[] { "abxyz", "NA", "c ", "xyzab" }, null, null),
            ("TNA equal T", "|b1", new object[] { false, false, false, false }, null, null),
            ("TNA less T", "|b1", new object[] { true, false, true, false }, null, null),
            ("TNA add TNone", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings ''NA'' and 'None'"),
            ("TNA equal TNone", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings ''NA'' and 'None'"),
            ("TNA less TNone", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings ''NA'' and 'None'"),
            ("TNA add Tnan", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings ''NA'' and 'nan'"),
            ("TNA equal Tnan", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings ''NA'' and 'nan'"),
            ("TNA less Tnan", null, null, "TypeError,Exception,BaseException,object", "Cannot find a compatible null string value for null strings ''NA'' and 'nan'"),
            ("TNA add TNA", "StringDType(na_object='NA')", new object[] { "abxyz", "NA", "NA", "xyzab" }, null, null),
            ("TNA equal TNA", "|b1", new object[] { false, false, false, false }, null, null),
            ("TNA less TNA", "|b1", new object[] { true, false, true, false }, null, null),
            // NA kinds
            ("TNone str_len", null, null, "ValueError,Exception,BaseException,object", "The length of a null string is undefined"),
            ("TNone upper", null, null, "TypeError,Exception,BaseException,object", "descriptor 'upper' for 'str' objects doesn't apply to a 'NoneType' object"),
            ("TNone isalpha", null, null, "ValueError,Exception,BaseException,object", "Cannot use the isalpha function with a null that is not a nan-like value"),
            ("TNone isnan", "|b1", new object[] { false, false, false, false }, null, null),
            ("TNone find l", null, null, "ValueError,Exception,BaseException,object", "'find' not supported for null values that are not strings."),
            ("TNone startswith a", null, null, "ValueError,Exception,BaseException,object", "'startswith' not supported for null values that are not nan-like or strings."),
            ("TNone sort", null, null, "ValueError,Exception,BaseException,object", "Cannot compare null that is not a nan-like value"),
            ("TNone argsort", null, null, "ValueError,Exception,BaseException,object", "Cannot compare null that is not a nan-like value"),
            ("TNone astype U4", "<U4", new object[] { "ab", "None", "", "xyz" }, null, null),
            ("TNone astype float64", null, null, "ValueError,Exception,BaseException,object", "could not convert string to float: 'ab'"),
            ("Tnan str_len", null, null, "ValueError,Exception,BaseException,object", "The length of a null string is undefined"),
            ("Tnan upper", null, null, "TypeError,Exception,BaseException,object", "descriptor 'upper' for 'str' objects doesn't apply to a 'float' object"),
            ("Tnan isalpha", "|b1", new object[] { true, false, false, true }, null, null),
            ("Tnan isnan", "|b1", new object[] { false, true, false, false }, null, null),
            ("Tnan find l", null, null, "ValueError,Exception,BaseException,object", "'find' not supported for null values that are not strings."),
            ("Tnan startswith a", "|b1", new object[] { true, false, false, false }, null, null),
            ("Tnan sort", "StringDType(na_object=nan)", new object[] { "", "ab", "xyz", double.NaN }, null, null),
            ("Tnan argsort", "<i8", new object[] { 2L, 0L, 3L, 1L }, null, null),
            ("Tnan astype U4", "<U4", new object[] { "ab", "nan", "", "xyz" }, null, null),
            ("Tnan astype float64", null, null, "ValueError,Exception,BaseException,object", "could not convert string to float: 'ab'"),
            ("TNA str_len", "<i8", new object[] { 2L, 2L, 0L, 3L }, null, null),
            ("TNA upper", "StringDType(na_object='NA')", new object[] { "AB", "NA", "", "XYZ" }, null, null),
            ("TNA isalpha", "|b1", new object[] { true, true, false, true }, null, null),
            ("TNA isnan", "|b1", new object[] { false, false, false, false }, null, null),
            ("TNA find l", "<i8", new object[] { 1L, -1L, -1L, -1L }, null, null),
            ("TNA startswith a", "|b1", new object[] { true, false, false, false }, null, null),
            ("TNA sort", "StringDType(na_object='NA')", new object[] { "", "NA", "ab", "xyz" }, null, null),
            ("TNA argsort", "<i8", new object[] { 2L, 1L, 0L, 3L }, null, null),
            ("TNA astype U4", "<U4", new object[] { "ab", "NA", "", "xyz" }, null, null),
            ("TNA astype float64", null, null, "ValueError,Exception,BaseException,object", "could not convert string to float: 'ab'"),
            // keywords
            ("str_len out int64 where", "<i8", new object[] { 2L, -1L, 0L, -1L }, null, null),
            ("str_len out int8", "|i1", new object[] { 2L, 2L, 0L, 3L }, null, null),
            ("str_len out float64", "<f8", new object[] { 2.0, 2.0, 0.0, 3.0 }, null, null),
            ("str_len dtype=int32", null, null, "UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "ufunc 'str_len' did not contain a loop with signature matching types <class 'numpy.dtypes.StrDType'> -> <class 'numpy.dtypes.Int32DType'>"),
            ("equal out int64", "<i8", new object[] { 0L, 0L, 0L, 0L }, null, null),
            ("add out U10 where", "<U10", new object[] { "abab", "Q", "", "Q" }, null, null),
            ("add out U2", "<U2", new object[] { "ab", "c ", "", "xy" }, null, null),
            ("add U out T", null, null, "TypeError,Exception,BaseException,object", "cannot cast dtype StringDType() to <class 'numpy.dtypes.StrDType'>."),
            ("add T out T", "StringDType()", new object[] { "abab", "c c ", "", "xyzxyz" }, null, null),
            ("add out S10 from U", null, null, "UFuncTypeError,UFuncTypeError,UFuncTypeError,TypeError,Exception,BaseException,object", "Cannot cast ufunc 'add' output from dtype('<U6') to dtype('S10') with casting rule 'same_kind'"),
            // numpy.char
            ("char.equal", "|b1", new object[] { true, true, false, true }, null, null),
            ("char.less", "|b1", new object[] { false, false, true, false }, null, null),
            ("np.equal", "|b1", new object[] { false, false, false, false }, null, null),
            ("chararray ==", "|b1", new object[] { true, true, false, true }, null, null),
            ("compare_chararrays == True", "|b1", new object[] { true, true, false, true }, null, null),
            ("compare_chararrays == False", "|b1", new object[] { false, false, false, false }, null, null),
            ("compare_chararrays >= True", "|b1", new object[] { true, true, false, true }, null, null),
        };

        /// <summary>The rows by label.</summary>
        private static readonly Dictionary<string, (string Label, string DType, object[] Values, string Mro, string Message)> ByLabel =
            Rows.ToDictionary(r => r.Label);

        /// <summary>numpy's builtin classes a NumSharp exception is matched to by name, most specific first.</summary>
        private static readonly string[] KnownClasses =
            { "DTypePromotionError", "AxisError", "UnicodeEncodeError", "UnicodeDecodeError", "UnicodeError", "IndexError", "TypeError", "ValueError" };

        /// <summary>The operand of a kind, as the generator builds it.</summary>
        /// <param name="kind"><c>U</c>, <c>S</c>, <c>T</c>, <c>TNone</c>, <c>Tnan</c> or <c>TNA</c>.</param>
        /// <returns>A fresh 4-element array.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An unknown kind.</exception>
        private static NDArray Make(string kind) => kind switch
        {
            "U" => np.array(Data).astype(np.dtype("U3")),
            "S" => np.array(Data, np.dtype("S3")),
            "T" => np.array(Data, np.dtypes.StringDType()),
            "TNone" => np.array(new object[] { "ab", null, "", "xyz" }, np.dtypes.StringDType((object)null)),
            "Tnan" => np.array(new object[] { "ab", double.NaN, "", "xyz" }, np.dtypes.StringDType(double.NaN)),
            "TNA" => np.array(new object[] { "ab", "NA", "", "xyz" }, np.dtypes.StringDType("NA")),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        /// <summary>
        ///     Holds NumSharp's outcome of <paramref name="compute"/> to numpy's recorded row: the dtype and the elements, or
        ///     the exception class (by name or a house convention) and text.
        /// </summary>
        /// <param name="label">The row's label.</param>
        /// <param name="compute">NumSharp's computation of the row's operation.</param>
        /// <returns>A description of the disagreement, or null when NumSharp gives numpy's answer.</returns>
        private static string Check(string label, Func<NDArray> compute)
        {
            var row = ByLabel[label];
            NDArray result = null;
            Exception error = null;
            try
            {
                result = compute();
            }
            catch (Exception e)
            {
                error = e;
            }

            if (row.DType != null)
            {
                if (error != null)
                    return $"{label}: NumSharp raises {error.GetType().Name}: {error.Message} -- numpy returns {row.DType}";
                string dtype = row.DType.StartsWith("StringDType", StringComparison.Ordinal) ? result.dtype.ToString() : result.dtype.str;
                if (dtype != row.DType)
                    return $"{label}: dtype {dtype} -- numpy {row.DType}";
                if (result.size != row.Values.Length)
                    return $"{label}: {result.size} elements -- numpy {row.Values.Length}";
                using NDArray flat = result.ndim == 0 ? result.reshape(1) : result.flatten();
                for (int i = 0; i < row.Values.Length; i++)
                {
                    object actual = Normalize(flat.GetValue(i));
                    if (!Equals(actual, row.Values[i]) && !(actual is double a && row.Values[i] is double b && double.IsNaN(a) && double.IsNaN(b)))
                        return $"{label}: element {i} is {Show(actual)} -- numpy {Show(row.Values[i])}";
                }

                return null;
            }

            if (error == null)
                return $"{label}: NumSharp returns {result.dtype} -- numpy raises {row.Mro.Split(',')[0]}: {row.Message}";
            string[] chain = row.Mro.Split(',');
            string pyClass = chain.FirstOrDefault(KnownClasses.Contains) ?? chain[0];
            var names = new HashSet<string>();
            for (Type t = error.GetType(); t != null; t = t.BaseType)
                names.Add(t.Name);
            bool classMatches = names.Contains(pyClass) ||
                                (chain.Contains("UFuncTypeError") && error is ArgumentException) ||
                                (chain.Contains("ValueError") && error is IncorrectShapeException);
            if (!classMatches)
                return $"{label}: NumSharp raises {error.GetType().Name} -- numpy {pyClass}";
            string text = error.Message;
            if (error is ArgumentException argument && !string.IsNullOrEmpty(argument.ParamName))
            {
                int at = text.IndexOf($" (Parameter '{argument.ParamName}')", StringComparison.Ordinal);
                if (at >= 0)
                    text = text.Substring(0, at);
            }

            return text == row.Message ? null : $"{label}: text \"{text}\" -- numpy \"{row.Message}\"";
        }

        /// <summary>
        ///     One element as the rows spell it: an <c>S</c> element's bytes as Latin-1 text, an integer as a <see cref="long"/>,
        ///     a float as a <see cref="double"/>, a bool, a text, or an NA (<c>null</c>, a NaN, the NA string).
        /// </summary>
        /// <param name="value">The element NumSharp read.</param>
        /// <returns>The comparable value.</returns>
        private static object Normalize(object value) => value switch
        {
            byte[] bytes => Encoding.Latin1.GetString(bytes),
            sbyte or byte or short or ushort or int or uint or long => Convert.ToInt64(value),
            float f => (double)f,
            Half h => (double)h,
            _ => value,
        };

        /// <summary>A value for a failure message (a string quoted, a null spelled).</summary>
        /// <param name="value">The value.</param>
        /// <returns>Its display text.</returns>
        private static string Show(object value) => value switch { null => "null", string s => $"'{s}'", _ => value.ToString() };

        /// <summary>Fails with every disagreeing row, one per line.</summary>
        /// <param name="failures">The disagreements (nulls are agreements and are dropped).</param>
        /// <param name="family">The family's label.</param>
        /// <exception cref="AssertFailedException">At least one row disagreed.</exception>
        private static void AssertAll(IEnumerable<string> failures, string family)
        {
            string[] list = failures.Where(f => f != null).ToArray();
            if (list.Length > 0)
                Assert.Fail($"{list.Length} {family} rows disagree with numpy 2.4.2:\n" + string.Join("\n", list));
        }

        /// <summary>
        ///     Operands of two text classes or NA objects (every ordered pair of the six kinds) through <c>add</c>,
        ///     <c>equal</c> and <c>less</c>: numpy promotes <c>U</c> / <c>StringDType</c> mixes to <c>StringDType</c> with
        ///     the NA of the side that has one, refuses <c>U</c> with <c>S</c> ("cannot use operands" / "did not contain a
        ///     loop"), <c>S</c> with <c>StringDType</c>, two different NA objects, and a None NA in <c>add</c>.
        /// </summary>
        [TestMethod]
        public void CrossClassOperands_AreNumpys()
        {
            string[] kinds = { "U", "S", "T", "TNone", "Tnan", "TNA" };
            var failures = new List<string>();
            foreach (string a in kinds)
            foreach (string b in kinds)
            {
                failures.Add(Check($"{a} add {b}", () => np.strings.add(Make(a), Make(b)["::-1"])));
                failures.Add(Check($"{a} equal {b}", () => np.equal(Make(a), Make(b)["::-1"])));
                failures.Add(Check($"{a} less {b}", () => np.less(Make(a), Make(b)["::-1"])));
            }

            AssertAll(failures, "cross-class");
        }

        /// <summary>
        ///     The three NA kinds through the functions whose answer depends on the kind: <c>str_len</c>, <c>upper</c>,
        ///     <c>isalpha</c>, <c>isnan</c>, <c>find</c>, <c>startswith</c>, <c>sort</c>, a stable <c>argsort</c>, and the
        ///     casts to <c>U4</c> and <c>float64</c> — a None NA raises with each function's own text, a NaN-like NA
        ///     propagates and sorts last, a string NA reads as its text.
        /// </summary>
        [TestMethod]
        public void StringDTypeNAKinds_AreNumpys()
        {
            var ops = new (string Name, Func<NDArray, NDArray> Op)[]
            {
                ("str_len", x => np.strings.str_len(x)),
                ("upper", x => np.strings.upper(x)),
                ("isalpha", x => np.strings.isalpha(x)),
                ("isnan", x => np.isnan(x)),
                ("find l", x => np.strings.find(x, "b")),
                ("startswith a", x => np.strings.startswith(x, "a")),
                ("sort", x => np.sort(x)),
                ("argsort", x => np.argsort(x, kind: "stable")),
                ("astype U4", x => x.astype(np.dtype("U4"))),
                ("astype float64", x => x.astype(np.dtype("float64"))),
            };
            var failures = from kind in new[] { "TNone", "Tnan", "TNA" }
                           from op in ops
                           select Check($"{kind} {op.Name}", () => op.Op(Make(kind)));
            AssertAll(failures, "NA kinds");
        }

        /// <summary>
        ///     Calls a ufunc-shaped function with <c>out=</c> and returns the out array, after checking that the function
        ///     returned the out OBJECT itself (numpy's ufuncs do).
        /// </summary>
        /// <param name="call">The call, given the out array.</param>
        /// <param name="out">The out array.</param>
        /// <returns><paramref name="out"/>.</returns>
        /// <exception cref="InvalidOperationException">The function returned another object.</exception>
        private static NDArray Out(Func<NDArray, NDArray> call, NDArray @out)
        {
            if (!ReferenceEquals(call(@out), @out))
                throw new InvalidOperationException("the ufunc returned another object than out=");
            return @out;
        }

        /// <summary>
        ///     The ufunc keywords over text, with numpy's answers: <c>str_len</c> into an int64 (with a <c>where=</c> mask:
        ///     the masked elements keep -1), an int8 and a float64 <c>out=</c>; <c>dtype=int32</c> refused (no such loop);
        ///     <c>equal</c> into an int64 <c>out=</c>; <c>add</c> into a wide <c>U10</c> with a mask, a NARROW <c>U2</c> (each
        ///     result cut), a <c>StringDType</c> out from <c>U</c> operands (refused: no cast back) and from
        ///     <c>StringDType</c> ones, and an <c>S10</c> out from <c>U</c> operands (the same-kind cast refusal) — and
        ///     every computed out is the object passed in.
        /// </summary>
        [TestMethod]
        public void UfuncKeywords_AreNumpys()
        {
            NDArray U() => Make("U");
            NDArray Mask() => np.array(new[] { true, false, true, false });
            AssertAll(new[]
            {
                Check("str_len out int64 where", () => Out(o => np.strings.str_len(U(), @out: o, where: Mask()), np.full(new Shape(4), -1, np.dtype("int64")))),
                Check("str_len out int8", () => Out(o => np.strings.str_len(U(), @out: o), np.full(new Shape(4), -1, np.dtype("int8")))),
                Check("str_len out float64", () => Out(o => np.strings.str_len(U(), @out: o), np.full(new Shape(4), -1.0, np.dtype("float64")))),
                Check("str_len dtype=int32", () => np.strings.str_len(U(), dtype: np.dtype("int32"))),
                Check("equal out int64", () => Out(o => np.equal(U(), U()["::-1"], @out: o), np.full(new Shape(4), 7, np.dtype("int64")))),
                Check("add out U10 where", () => Out(o => np.strings.add(U(), U(), @out: o, where: Mask()), np.full(new Shape(4), "Q", np.dtype("U10")))),
                Check("add out U2", () => Out(o => np.strings.add(U(), U(), @out: o), np.full(new Shape(4), "Q", np.dtype("U2")))),
                Check("add U out T", () => Out(o => np.strings.add(U(), U(), @out: o), np.full(new Shape(4), "Q", np.dtypes.StringDType()))),
                Check("add T out T", () => Out(o => np.strings.add(Make("T"), Make("T"), @out: o), np.full(new Shape(4), "Q", np.dtypes.StringDType()))),
                Check("add out S10 from U", () => Out(o => np.strings.add(U(), U(), @out: o), np.full(new Shape(4), new[] { (byte)'Q' }, np.dtype("S10")))),
            }, "keywords");
        }

        /// <summary>
        ///     <c>numpy.char</c>'s comparisons strip trailing whitespace first (<c>'ab ' == 'ab'</c> and <c>'' == ' '</c> are
        ///     True there), <c>np.equal</c> does not, a chararray's <c>==</c> strips, and <c>compare_chararrays</c> strips
        ///     only when told to.
        /// </summary>
        [TestMethod]
        public void CharComparisons_StripTrailingWhitespace_AsNumpy()
        {
            NDArray C1() => np.array(new[] { "ab ", "c", "x  ", "" });
            NDArray C2() => np.array(new[] { "ab", "c  ", "y", " " });
            AssertAll(new[]
            {
                Check("char.equal", () => np.@char.equal(C1(), C2())),
                Check("char.less", () => np.@char.less(C1(), C2())),
                Check("np.equal", () => np.equal(C1(), C2())),
                Check("chararray ==", () => np.@char.array(C1()) == np.@char.array(C2())),
                Check("compare_chararrays == True", () => np.@char.compare_chararrays(C1(), C2(), "==", true)),
                Check("compare_chararrays == False", () => np.@char.compare_chararrays(C1(), C2(), "==", false)),
                Check("compare_chararrays >= True", () => np.@char.compare_chararrays(C1(), C2(), ">=", true)),
            }, "numpy.char");
        }
    }
}
