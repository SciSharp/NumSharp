using System;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.eye raises where NumPy's does, in NumPy's order and with NumPy's texts: its body is
    ///     <c>m = zeros((N, M), dtype=dtype, order=order, device=device)</c>, so the order's letter is checked first (an
    ///     argument conversion), then the device, then the order's C-or-F rule, then the dimensions (the allocation's
    ///     "negative dimensions are not allowed"). Its one is the Python int 1 stored by the class's setitem. Every expected
    ///     value and text was probed against NumPy 2.4.2 (the np-layer dtype plan's batch 5 replay found the divergences).
    /// </summary>
    [TestClass]
    public class np_eye_NumPyOrder_Test
    {
        /// <summary>A negative N or M raises the allocation's text, without NumSharp's old "(N=-1)" / "(M=-1)" suffix.</summary>
        [TestMethod]
        public void NegativeDimension_RaisesTheAllocationsText()
        {
            Action n = () => np.eye(-1);
            n.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
            Action m = () => np.eye(2, -1);
            m.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
            Action identity = () => np.identity(-3);
            identity.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
        }

        /// <summary>np.zeros refuses A and K before it allocates: <c>np.eye(2, -1, order='A')</c> is the order's error.</summary>
        [TestMethod]
        public void OrderRule_PrecedesTheDimensions()
        {
            Action a = () => np.eye(2, -1, order: 'A');
            a.Should().Throw<ValueError>().WithMessage("only 'C' or 'F' order is permitted");
            Action k = () => np.eye(2, -1, order: 'K');
            k.Should().Throw<ValueError>().WithMessage("only 'C' or 'F' order is permitted");
        }

        /// <summary>
        ///     The order's letter is converted before the device and the C-or-F rule is applied after it:
        ///     <c>np.eye(2, order='Z', device='cuda')</c> is the order's error, <c>np.eye(2, order='A', device='cuda')</c> the
        ///     device's.
        /// </summary>
        [TestMethod]
        public void OrderLetter_PrecedesTheDevice_WhichPrecedesTheCOrFRule()
        {
            Action z = () => np.eye(2, order: 'Z', device: "cuda");
            z.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Z')");
            Action a = () => np.eye(2, order: 'A', device: "cuda");
            a.Should().Throw<ArgumentException>().WithMessage("Device not understood. Only \"cpu\" is allowed, but received: cuda*");
        }

        /// <summary>
        ///     The one is the class's setitem of the Python int 1, so a <c>StringDType(coerce=False)</c> refuses it as NumPy's
        ///     flat assignment does (np.ones of that dtype casts an int64 1 and stores '1').
        /// </summary>
        [TestMethod]
        public void StringDTypeWithoutCoercion_RefusesTheOne()
        {
            Action e = () => np.eye(2, dtype: np.dtypes.StringDType(coerce: false));
            e.Should().Throw<ValueError>().WithMessage("StringDType only allows string data when string coercion is disabled.");
            using var ones = np.ones(new Shape(2), np.dtypes.StringDType(coerce: false));
            ones.ToString(true).Should().Be("array(['1', '1'], dtype=StringDType(coerce=False))");
        }

        /// <summary>A text dtype's one is the text '1' (an unsized request is one character, as np.zeros allocates it).</summary>
        [TestMethod]
        public void TextDtype_HoldsTheTextOne()
        {
            using var u = np.eye(2, dtype: "U");
            u.ToString(true).Should().Be("array([['1', ''],\n       ['', '1']], dtype='<U1')");
        }
    }
}
