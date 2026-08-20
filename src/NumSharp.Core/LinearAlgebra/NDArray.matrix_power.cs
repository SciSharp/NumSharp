namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Raises this square matrix to the (integer) power <paramref name="power"/>.
        /// </summary>
        /// <remarks>
        ///     The method form of <see cref="np.linalg.matrix_power"/>; see it for the full contract.
        ///     <para>
        ///     This used to reject a NEGATIVE power outright ("matrix_power just work with int >= 0"),
        ///     which was never NumPy's rule — <c>a**-n</c> is <c>inv(a)**n</c>. It now takes that
        ///     route, so a negative power computes wherever a matrix backend is installed and raises
        ///     <see cref="OpenBlasMissingBackendException"/> where none is. Three other behaviours came
        ///     with the delegation: a non-square operand now raises <see cref="LinAlgError"/> rather
        ///     than failing inside the product, <c>power == 0</c> returns the identity in THIS array's
        ///     dtype instead of always float64, and the chain is evaluated by binary exponentiation
        ///     rather than one multiply per step.
        ///     </para>
        /// </remarks>
        public NDArray matrix_power(int power) => np.linalg.matrix_power(this, power);
    }
}
