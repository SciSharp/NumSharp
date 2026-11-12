using System;

namespace NumSharp.Tests.APIs;

/// <summary>
/// Battle tests for np.can_cast - comprehensive coverage of type casting rules.
/// </summary>
[TestClass]
public class NpCanCastBattleTests
{
    #region Safe Casting - Type to Type

    [TestMethod]
    public void CanCast_SameType_AlwaysTrue()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int32).Should().BeTrue();
        np.can_cast(NPTypeCode.Double, NPTypeCode.Double).Should().BeTrue();
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Boolean).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_BoolToIntegers_Safe()
    {
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Byte).Should().BeTrue();
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Int32).Should().BeTrue();
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Int64).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_BoolToFloats_Safe()
    {
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Single).Should().BeTrue();
        np.can_cast(NPTypeCode.Boolean, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_ByteUpcast_Safe()
    {
        np.can_cast(NPTypeCode.Byte, NPTypeCode.Int16).Should().BeTrue();
        np.can_cast(NPTypeCode.Byte, NPTypeCode.Int32).Should().BeTrue();
        np.can_cast(NPTypeCode.Byte, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_Int32ToInt64_Safe()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_Int64ToInt32_NotSafe()
    {
        np.can_cast(NPTypeCode.Int64, NPTypeCode.Int32).Should().BeFalse();
    }

    [TestMethod]
    public void CanCast_Float32ToFloat64_Safe()
    {
        np.can_cast(NPTypeCode.Single, NPTypeCode.Double).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_Float64ToFloat32_NotSafe()
    {
        np.can_cast(NPTypeCode.Double, NPTypeCode.Single).Should().BeFalse();
    }

    [TestMethod]
    public void CanCast_FloatToInt_NotSafe()
    {
        np.can_cast(NPTypeCode.Double, NPTypeCode.Int32).Should().BeFalse();
    }

    #endregion

    #region Casting Modes

    [TestMethod]
    public void CanCast_NoMode_OnlySameType()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int32, "no").Should().BeTrue();
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64, "no").Should().BeFalse();
    }

    [TestMethod]
    public void CanCast_SameKindMode_AllowsDowncastWithinKind()
    {
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Int16, "same_kind").Should().BeTrue();
        np.can_cast(NPTypeCode.Double, NPTypeCode.Single, "same_kind").Should().BeTrue();
        // int8 (SByte) participates: down/across signed and float16 both allowed.
        np.can_cast(NPTypeCode.Int32, NPTypeCode.SByte, "same_kind").Should().BeTrue();
        np.can_cast(NPTypeCode.Single, NPTypeCode.Half, "same_kind").Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_SameKindMode_DirectionalKindOrdering()
    {
        // NumPy same_kind is a DIRECTIONAL kind ordering (bool < unsigned < signed < float < complex):
        // moving to the same-or-higher kind is allowed; moving DOWN is not.
        // int -> float is allowed (NOT rejected — the old test asserted the opposite).
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Single, "same_kind").Should().BeTrue();
        np.can_cast(NPTypeCode.Int16, NPTypeCode.Half, "same_kind").Should().BeTrue();
        // unsigned -> signed allowed; signed -> unsigned NOT (asymmetric).
        np.can_cast(NPTypeCode.Byte, NPTypeCode.SByte, "same_kind").Should().BeTrue();
        np.can_cast(NPTypeCode.SByte, NPTypeCode.Byte, "same_kind").Should().BeFalse();
        // float -> int and complex -> float move DOWN a kind: rejected.
        np.can_cast(NPTypeCode.Single, NPTypeCode.Int32, "same_kind").Should().BeFalse();
        np.can_cast(NPTypeCode.Complex, NPTypeCode.Double, "same_kind").Should().BeFalse();
        // anything (non-bool) -> bool moves down: rejected.
        np.can_cast(NPTypeCode.Int32, NPTypeCode.Boolean, "same_kind").Should().BeFalse();
    }

    [TestMethod]
    public void CanCast_UnsafeMode_AllowsAnything()
    {
        np.can_cast(NPTypeCode.Double, NPTypeCode.Int32, "unsafe").Should().BeTrue();
        np.can_cast(NPTypeCode.Int64, NPTypeCode.Byte, "unsafe").Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_InvalidMode_Throws()
    {
        new Action(() => np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64, "invalid")).Should().Throw<Exception>();
    }

    #endregion

    #region Python numbers (NEP 50): NumPy 2 refuses them

    // NumPy 2.4.2: np.can_cast(1, np.int8), (300, np.int8), (1.5, np.float32), (True, np.int8), (0, np.uint8) and (1j,
    // np.complex64) all raise this TypeError (array_can_cast_safely's PyArray_IsPythonNumber branch). NumSharp answered
    // by value until API-parity batch 9, with holes (can_cast(1, SByte) was False).
    private const string PythonNumberRefusal =
        "can_cast() does not support Python ints, floats, and complex because the result used to depend on the value.\n"
        + "This change was part of adopting NEP 50, we may explicitly allow them again in the future.";

    /// <summary>
    ///     A C# int literal is NumPy's Python int: <c>np.can_cast(255, np.uint8)</c> and every other value raise NEP
    ///     50's TypeError, where NumSharp answered by value (with holes: 1 into SByte was False) until API-parity batch
    ///     9.
    /// </summary>
    [TestMethod]
    public void CanCast_IntLiteral_IsNumPysTypeError()
    {
        // `np.can_cast(0, NPTypeCode.Byte)` does not compile (CS0121): the literal 0 also converts to NPTypeCode.
        new Action(() => np.can_cast(255, NPTypeCode.Byte)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(256, NPTypeCode.Byte)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(-1, NPTypeCode.Byte)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(1, NPTypeCode.SByte)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(300, NPTypeCode.SByte)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
    }

    /// <summary>
    ///     The value table's bool rows (0 / 1 fit, 2 does not) are gone with it: NumPy refuses the Python int whatever
    ///     the target.
    /// </summary>
    [TestMethod]
    public void CanCast_IntLiteral_ToBoolean_IsNumPysTypeError()
    {
        new Action(() => np.can_cast(1, NPTypeCode.Boolean)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(2, NPTypeCode.Boolean)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
    }

    /// <summary>
    ///     A C# long is a Python int too (it binds the NDArray form through the literal-marked implicit conversion):
    ///     the int32 range checks of the removed long overload become NumPy's TypeError.
    /// </summary>
    [TestMethod]
    public void CanCast_LongLiteral_IsNumPysTypeError()
    {
        new Action(() => np.can_cast((long)int.MaxValue, NPTypeCode.Int32)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast((long)int.MaxValue + 1, NPTypeCode.Int32)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
    }

    /// <summary>
    ///     The DType spelling of the target binds <c>can_cast(NDArray, DType)</c>, which refuses the literal the same
    ///     way (int into uint8, float into float32, complex into complex128).
    /// </summary>
    [TestMethod]
    public void CanCast_PythonNumber_DTypeTarget_IsNumPysTypeError()
    {
        // The DType spelling of the target binds can_cast(NDArray, DType): the same refusal.
        new Action(() => np.can_cast(0, np.uint8)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(1.5, np.float32)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(new System.Numerics.Complex(0, 1), DType.Complex)).Should().Throw<TypeError>()
            .WithMessage(PythonNumberRefusal);
    }

    /// <summary>
    ///     A boxed C# number reaches the object form, which ports <c>array_can_cast_safely</c>'s Python-number branch:
    ///     the same TypeError for an int, a bool and a double.
    /// </summary>
    [TestMethod]
    public void CanCast_BoxedPythonNumber_IsNumPysTypeError()
    {
        new Action(() => np.can_cast((object)1, NPTypeCode.SByte)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast((object)0, NPTypeCode.Byte)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast((object)true, NPTypeCode.SByte)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast((object)1.5, NPTypeCode.Single)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
    }

    /// <summary>
    ///     NumPy parses <c>to</c> and <c>casting</c> before it looks at <c>from_</c>: a None target is the 'None'
    ///     TypeError and an unknown casting name the ValueError, even for a Python number.
    /// </summary>
    [TestMethod]
    public void CanCast_PythonNumber_ArgumentErrorsComeFirst()
    {
        // NumPy parses `to` and `casting` before it looks at from_: np.can_cast(1, None) is the 'None' TypeError and an
        // unknown casting name is the ValueError, not the Python-number refusal.
        new Action(() => np.can_cast(1, (DType)null)).Should().Throw<TypeError>()
            .WithMessage("did not understand one of the types; 'None' not accepted");
        new Action(() => np.can_cast(1, NPTypeCode.SByte, "bogus")).Should().Throw<ValueError>();
        new Action(() => np.can_cast((object)1, NPTypeCode.SByte, "bogus")).Should().Throw<ValueError>();
    }

    /// <summary>
    ///     Only Python numbers are refused: a 0-d array, an <c>NDArray.Scalar</c> (NumPy's scalar), a bool array and a
    ///     <see cref="Half"/> (a NumPy float16 scalar in NumSharp's model) are judged by their dtype, never by value.
    /// </summary>
    [TestMethod]
    public void CanCast_NumPyScalarsAndArrays_KeepTheDtypeRule()
    {
        // np.can_cast(np.int16(1), np.int8) and np.can_cast(np.array(1, np.int16), np.int8) are False; np.can_cast(
        // np.array(True), np.int8) and np.can_cast(np.float16(1), np.float32) are True: a NumPy scalar or 0-d array is
        // judged by its dtype, never by value.
        np.can_cast(np.array((short)1), NPTypeCode.SByte).Should().BeFalse();
        np.can_cast(NDArray.Scalar((short)1), NPTypeCode.SByte).Should().BeFalse();
        np.can_cast(np.array(true), NPTypeCode.SByte).Should().BeTrue();
        np.can_cast((Half)1, NPTypeCode.Single).Should().BeTrue();
        np.can_cast((object)(Half)1, NPTypeCode.Single).Should().BeTrue();
    }

    /// <summary>
    ///     The object form converts any other <c>from_</c> as a dtype specifier (<c>PyArray_DescrConverter2</c>): a
    ///     boxed DType, NPTypeCode, Type or dtype string, and a boxed array by its descriptor.
    /// </summary>
    [TestMethod]
    public void CanCast_BoxedDtypeSpellings_UseTheDtypeRule()
    {
        // array_can_cast_safely converts any other from_ as a dtype specifier (PyArray_DescrConverter2).
        np.can_cast((object)DType.Int16, NPTypeCode.SByte).Should().BeFalse();
        np.can_cast((object)NPTypeCode.Byte, NPTypeCode.Int16).Should().BeTrue();
        np.can_cast((object)typeof(short), NPTypeCode.Int32).Should().BeTrue();
        np.can_cast((object)"i2", NPTypeCode.SByte, "same_kind").Should().BeTrue();
        np.can_cast((object)np.array(new short[] { 1, 2 }), NPTypeCode.SByte).Should().BeFalse();
    }

    /// <summary>
    ///     An object that names no dtype (a C# array, NumPy's list) and a null source are NumPy's <c>did not understand
    ///     one of the types; 'None' not accepted</c>.
    /// </summary>
    [TestMethod]
    public void CanCast_NonDtypeObjects_AreNumPysTypeError()
    {
        // NumPy: np.can_cast([1, 2], np.int8) and np.can_cast(None, np.int8) raise
        // TypeError: did not understand one of the types; 'None' not accepted
        new Action(() => np.can_cast((object)new[] { 1, 2 }, NPTypeCode.SByte)).Should().Throw<TypeError>()
            .WithMessage("did not understand one of the types; 'None' not accepted");
        new Action(() => np.can_cast((object)null, NPTypeCode.SByte)).Should().Throw<TypeError>()
            .WithMessage("did not understand one of the types; 'None' not accepted");
    }

    /// <summary>
    ///     A C# <c>decimal</c> has no NumPy analog, so its 0-d array stays unmarked and keeps the dtype rule (NumSharp's
    ///     decimal dtype), where every C# number NumPy knows raises the NEP 50 TypeError.
    /// </summary>
    [TestMethod]
    [Misaligned]
    public void CanCast_DecimalLiteral_UsesTheDecimalDtypeRule()
    {
        np.can_cast(1m, NPTypeCode.Decimal).Should().BeTrue();
        np.can_cast((object)1m, NPTypeCode.Decimal).Should().BeTrue();
    }

    #endregion

    #region C# type spellings (the generic can_cast<TFrom, TTo> went in API-parity batch 9, OD7)

    // NumPy asks with dtypes: np.can_cast(np.int32, np.int64) is True, (np.int64, np.int32) False,
    // (np.float32, np.float64) True, (np.int32, np.int16, 'same_kind') True. A C# type converts to the DType.

    /// <summary>
    ///     The generic <c>can_cast&lt;int, long&gt;()</c> went (OD7): its C# type spelling <c>can_cast(typeof(int),
    ///     typeof(long))</c> and the np dtypes give NumPy's <c>np.can_cast(np.int32, np.int64)</c>, True.
    /// </summary>
    [TestMethod]
    public void CanCast_TypeofIntToLong()
    {
        np.can_cast(typeof(int), typeof(long)).Should().BeTrue();
        np.can_cast(np.int32, np.int64).Should().BeTrue();
    }

    /// <summary>
    ///     The narrowing direction, NumPy's <c>np.can_cast(np.int64, np.int32)</c>: False under 'safe'.
    /// </summary>
    [TestMethod]
    public void CanCast_TypeofLongToInt()
    {
        np.can_cast(typeof(long), typeof(int)).Should().BeFalse();
        np.can_cast(np.int64, np.int32).Should().BeFalse();
    }

    /// <summary>
    ///     float32 widens safely to float64 (NumPy's <c>np.can_cast(np.float32, np.float64)</c>, True).
    /// </summary>
    [TestMethod]
    public void CanCast_TypeofFloatToDouble()
    {
        np.can_cast(typeof(float), typeof(double)).Should().BeTrue();
    }

    /// <summary>
    ///     'same_kind' allows int32 to int16 (NumPy's <c>np.can_cast(np.int32, np.int16, 'same_kind')</c>, True).
    /// </summary>
    [TestMethod]
    public void CanCast_TypeofSameKind()
    {
        np.can_cast(typeof(int), typeof(short), "same_kind").Should().BeTrue();
    }

    #endregion

    #region Every C# number type is a Python number (the typed value overloads went in API-parity batch 9)

    /// <summary>
    ///     Every C# number type of the removed typed overloads (byte, sbyte, short, ushort, uint, ulong, float, bool)
    ///     is a Python number to NumPy's can_cast: each raises the NEP 50 TypeError whatever its value.
    /// </summary>
    [TestMethod]
    public void CanCast_EveryCSharpNumberLiteral_IsNumPysTypeError()
    {
        // Each binds can_cast(NDArray, NPTypeCode) through NDArray's implicit conversion, which marks the 0-d array as a
        // Python scalar (WasPythonLiteral / WasPythonBool): NumPy's refusal, whatever the value.
        new Action(() => np.can_cast((byte)100, NPTypeCode.Int32)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast((sbyte)-1, NPTypeCode.Int32)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast((short)-100, NPTypeCode.Byte)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast((ushort)1, NPTypeCode.Byte)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(1u, NPTypeCode.Int64)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(ulong.MaxValue, NPTypeCode.Int64)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(1.0f, NPTypeCode.Double)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
        new Action(() => np.can_cast(true, NPTypeCode.Int32)).Should().Throw<TypeError>().WithMessage(PythonNumberRefusal);
    }

    #endregion

    #region NDArray Tests

    [TestMethod]
    public void CanCast_NDArray_ToLargerType()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.can_cast(arr, NPTypeCode.Int64).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_NDArray_ToSmallerType()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        np.can_cast(arr, NPTypeCode.Int16).Should().BeFalse();
    }

    #endregion

    #region Type Overload Tests

    [TestMethod]
    public void CanCast_Type_IntToLong()
    {
        np.can_cast(typeof(int), typeof(long)).Should().BeTrue();
    }

    [TestMethod]
    public void CanCast_Type_LongToInt()
    {
        np.can_cast(typeof(long), typeof(int)).Should().BeFalse();
    }

    #endregion
}
