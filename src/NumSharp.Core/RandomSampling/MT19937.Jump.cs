namespace NumSharp
{
    public sealed partial class MT19937
    {
        /// <summary>MEXP: the Mersenne exponent — the recurrence's degree (19937).</summary>
        private const int MEXP = 19937;

        /// <summary>
        ///     NumPy's <c>poly_coef</c> (<c>src/mt19937/mt19937-jump.h</c>): the jump polynomial for 2**128 steps
        ///     over the MT19937 recurrence, 624 32-bit words (coefficient <c>d</c> is bit <c>d &amp; 31</c> of word
        ///     <c>d &gt;&gt; 5</c>). Produced upstream by randomgen's <c>mt19937-generate-jump-poly.c</c> (a modified
        ///     <c>minipoly_mt19937.c</c> from Matsumoto's jump-ahead package); transcribed verbatim.
        /// </summary>
        private static readonly uint[] JumpPoly =
        {
            0x72DE3963u, 0xB5709EC4u, 0x88279BB6u, 0xA823F8E5u, 0x26D83E59u, 0x041F2259u,
            0xE7FDBB15u, 0x8B521777u, 0x48B5E756u, 0xBF2812D5u, 0xE4B0ADB9u, 0x0B4849AAu,
            0x3E928B83u, 0xE96D39CEu, 0xAF6131D3u, 0x09EAF2E8u, 0x33548456u, 0xC1814C7Bu,
            0x893A7C83u, 0xFEBD07BCu, 0x01BD8267u, 0x5147DCBFu, 0xE2A67DE6u, 0x9AFEF574u,
            0xB8334D09u, 0xF0D3DECAu, 0x5561FD58u, 0xD884703Bu, 0xEF5C803Bu, 0xB39B8F42u,
            0x20DFB761u, 0xD61CFED3u, 0xCF5F3E5Bu, 0x47416177u, 0x8E8442E9u, 0x8EA9CFABu,
            0x585D0EC0u, 0x60DDF78Du, 0x2C9B8528u, 0xF0F7D60Eu, 0xB2BB3BFCu, 0xCA3EE37Du,
            0x81C9E659u, 0x870ED969u, 0x9573A0DEu, 0xCE524851u, 0x77683B94u, 0x73CDA5EDu,
            0x56BCFCBCu, 0xF43B956Cu, 0x1F91DE14u, 0xBF04B400u, 0x9438C481u, 0x1D859831u,
            0xCA6AE0A2u, 0x9D97AED5u, 0x9E464218u, 0xE75C9519u, 0x253C5486u, 0xCD43455Cu,
            0x73B5CCD8u, 0x7F8282D4u, 0xC8CACD44u, 0x192DDF99u, 0xD6BE8546u, 0x5288B589u,
            0xB4F26CA7u, 0x9819557Fu, 0x200570EBu, 0x03E73D28u, 0x264ACC04u, 0x78A114C9u,
            0x95F0FB7Bu, 0x42EEE897u, 0xABCC80C2u, 0x67E751E8u, 0x1330CC85u, 0x140E87EFu,
            0x913B9A96u, 0xD3F8525Eu, 0x3EE3D205u, 0x1BA1158Fu, 0x2C4CDB89u, 0x1F6AA87Du,
            0x9B5E9A3Au, 0x878B3223u, 0xA498C3EDu, 0xA48C7778u, 0x974AC066u, 0x1D08F055u,
            0xC8A08242u, 0xD6DE80E9u, 0xA1CF0B40u, 0x2892CE4Cu, 0x842731C7u, 0x604168AEu,
            0xDD23EE6Du, 0xBECFF8B2u, 0xDFAC7287u, 0xA4369751u, 0xBA8BC89Du, 0x4A5840D9u,
            0xA7A58582u, 0xF53BDBEDu, 0xCFBA4997u, 0xA4149D1Cu, 0xD5C66FC3u, 0xF2C72905u,
            0xCE68AD39u, 0xAE4D8E96u, 0xF213A9B5u, 0xC588F396u, 0x9D6116BBu, 0x2C618D4Eu,
            0xB34420D1u, 0xEBFB61F3u, 0x3B702ED7u, 0xCBDCA6F2u, 0x7CB78166u, 0xBE283395u,
            0x03A2436Au, 0x20C0D096u, 0xE190AA6Fu, 0xBF49B815u, 0x49D78DC3u, 0x9B45B903u,
            0x0AA4C4C8u, 0x67EB90E3u, 0xF32B13F0u, 0x7F5CEAB1u, 0xCCC48294u, 0x641EAEDBu,
            0x6D6AAFB6u, 0x80B55358u, 0x72B55832u, 0xF1FA779Au, 0x3B60AF74u, 0x8992AEFDu,
            0x4FA609F2u, 0x28359472u, 0x61E7AAF1u, 0x527DC1A9u, 0x834E8087u, 0xBCAD693Fu,
            0xC9CA3BF6u, 0x95171796u, 0x9F41164Au, 0xB7D36775u, 0xCF20CF3Bu, 0x5C77677Bu,
            0xF4765B01u, 0x47DFD69Fu, 0xD90D6E15u, 0xD708247Fu, 0x5FE95113u, 0xAD799628u,
            0xC627F9F2u, 0xFCFB0CE2u, 0x0F2441CEu, 0x4B003380u, 0x72161100u, 0x50FA780Bu,
            0x1F72B11Au, 0xB71CA8B7u, 0xFFAB42FDu, 0x5475BACEu, 0x91C28B39u, 0x356EEF78u,
            0x1441C9C3u, 0xDC80086Du, 0x96C47491u, 0xB5C30EC9u, 0xA254E42Du, 0xA9321ADDu,
            0x963A3612u, 0xC30BEE5Bu, 0x635C75C7u, 0xDF141323u, 0x38308F58u, 0x8926E38Fu,
            0x71B69592u, 0x897754D8u, 0x3CDDDE5Eu, 0x5BC06174u, 0xAD520904u, 0xBEBB80A7u,
            0x5CC284D4u, 0xD91D5D33u, 0x8C6BA748u, 0x11090E41u, 0x33BB9929u, 0x462CFFBCu,
            0xC42A508Eu, 0xEFC68605u, 0x602A3A14u, 0x230E6CD9u, 0x26C6F9F4u, 0x49B8EB31u,
            0x51BD358Fu, 0x7C49E7A4u, 0x47B592CBu, 0x1910BB39u, 0x3CED6A5Bu, 0xAD0CA518u,
            0x93461DCBu, 0xD98CA579u, 0x9526948Eu, 0xECC5CB65u, 0xFD1A431Bu, 0x0BDDC87Du,
            0x5D694024u, 0x7D9820ACu, 0xFFEB5538u, 0x716C1AE1u, 0x13CFFB2Fu, 0x04F8ED86u,
            0xD777F039u, 0x1B32EB97u, 0x87C1A95Fu, 0x893DA4EEu, 0xC235F16Cu, 0x965118D4u,
            0xE87994BAu, 0xF99023E2u, 0xBB8C4545u, 0x891268A5u, 0xE7CF46B4u, 0x4D163861u,
            0x0B2C5681u, 0xCA688C0Eu, 0x36702E5Fu, 0xB86346B5u, 0x55E311BBu, 0x72A60137u,
            0x142FDC5Cu, 0x47D10E13u, 0xA34CE0CBu, 0xAC088C30u, 0x8F9503FEu, 0x4D79A2E8u,
            0x937670C7u, 0x02B4C095u, 0x20F8F5E0u, 0x080533C0u, 0x81FE8F32u, 0xAB1D0C25u,
            0x048F776Du, 0xB601BB28u, 0x96004A47u, 0xF8B8E16Eu, 0x6862AF7Bu, 0x4A9FA042u,
            0xB0B6F662u, 0x54384AD4u, 0xA350C0EEu, 0x81670A57u, 0x26061DC1u, 0x3A2C2820u,
            0xB575F899u, 0xB9749667u, 0x738DFC2Au, 0xAA853838u, 0x00CCC442u, 0xA53A92A4u,
            0xCFAF5A3Eu, 0xBDC8CFA2u, 0x09884265u, 0x529FEE9Du, 0xA4D7F84Fu, 0x966C709Eu,
            0x4C80BC42u, 0xD14265D4u, 0xF5EBE7F3u, 0xB23C2AEDu, 0x804523F1u, 0xB7D47C42u,
            0xA7CB0AA9u, 0x73370568u, 0x06D90AC5u, 0x66158A1Eu, 0x9805C7ADu, 0xC4A3898Cu,
            0x7890ADDEu, 0x7FC53690u, 0x85C39B20u, 0xC5427E08u, 0xC0C864F8u, 0x2FBA05EDu,
            0xC365017Au, 0x210AD2BFu, 0x8FFB95EAu, 0x609CA003u, 0x8E6C4F72u, 0x84E663C4u,
            0x3C110562u, 0x753C1CA8u, 0x8700B723u, 0x48642AFCu, 0x14AC952Cu, 0xCEF1123Eu,
            0xED84973Cu, 0xF075B8B8u, 0x0CEAC5C9u, 0xF00A255Au, 0xDFCD487Cu, 0x7E77E0DAu,
            0x8BE5750Cu, 0x0071CB97u, 0x560827FEu, 0x28C4386Fu, 0xAF4049F0u, 0xBF6B3AD6u,
            0xA911AADDu, 0x2E3006D1u, 0x5EB5BB74u, 0x2E8489F9u, 0xC36FB83Du, 0x84278164u,
            0x82302B47u, 0x61E0E6BEu, 0x0422260Eu, 0x11B59C56u, 0xE4F20C9Cu, 0x9CD5ECAAu,
            0xF866E2DAu, 0x9BC72523u, 0x52C41667u, 0x816F533Cu, 0x47A3235Eu, 0xA0DBFF9Eu,
            0x0C62A756u, 0xEA9CA5A3u, 0xDE0761A6u, 0xC51267E9u, 0x3EED2AF6u, 0xF28B8866u,
            0x695ED01Fu, 0xFD769663u, 0x9065AF4Eu, 0xBC47FCDFu, 0xDFCA6259u, 0x424E389Cu,
            0x166C2C1Bu, 0xBB03335Eu, 0x2A73A1A1u, 0xC4BE33DDu, 0xE690D058u, 0x45746BC2u,
            0x94B43407u, 0x07D38D7Fu, 0x60854FB3u, 0x74B851E4u, 0xDB3D2AC2u, 0xD99DF507u,
            0x86D3323Bu, 0x5D6C254Cu, 0x82BFAC22u, 0xB4DD3032u, 0xB27E023Bu, 0xB7261A5Fu,
            0x34FE8179u, 0x40F361BFu, 0x6C9E7858u, 0xE716500Eu, 0x65873B06u, 0x35C6EE0Bu,
            0xFB2864E7u, 0xE4C5D4FCu, 0x281901C6u, 0x858EE284u, 0xE5FCA3CDu, 0x44803A65u,
            0xF850F7F6u, 0xF9F41E41u, 0x65EB5539u, 0x87CBF3C9u, 0xBE2F8074u, 0xAE056412u,
            0x3C5CB955u, 0xD8FE916Fu, 0xAEC289DFu, 0xD18CCB5Eu, 0x0EEF81BFu, 0x446157F2u,
            0x4690364Au, 0xDE982175u, 0xC1597EA0u, 0xD094591Bu, 0xB1ED3E17u, 0x79676E7Au,
            0xC495EBC1u, 0xA283BDF6u, 0x648C3570u, 0x6A06B25Cu, 0x398B0580u, 0x0DEB138Cu,
            0xE51108EDu, 0x4E3D096Au, 0x1DDA7416u, 0xAFDE012Bu, 0x722F0317u, 0xCB001892u,
            0x23875CF7u, 0x82D756D2u, 0xC99114DEu, 0x2091CE44u, 0xD24757B4u, 0x8A944EF9u,
            0x8594145Au, 0xEDF8F12Bu, 0x998C4AFFu, 0xF30C0CE9u, 0x9CE601A0u, 0xBA657A58u,
            0x36A851DDu, 0x94E6EC8Du, 0xED46B938u, 0x86ADA470u, 0x409B507Du, 0x46C714B9u,
            0x05C862A8u, 0xB628043Eu, 0x7AC4A188u, 0x8D763A8Cu, 0x0ADC18B6u, 0x7F5BA797u,
            0x69073599u, 0x5DB4BC6Bu, 0x444D59D3u, 0x3D087E22u, 0xE9C04E89u, 0x61466F51u,
            0x548AA4E6u, 0x151FD405u, 0x91555389u, 0x60905661u, 0x5E8D5619u, 0x3E3C8561u,
            0x39C6B81Cu, 0x2491156Cu, 0xFC2FD4A6u, 0x17B4D42Cu, 0x82C9BCF9u, 0x2BD704CFu,
            0x7B2568ECu, 0x05403240u, 0x5D2268D9u, 0x7E037B6Bu, 0xD86BEC7Au, 0x231F10E7u,
            0xBA016830u, 0x964F8501u, 0xA3B7321Fu, 0x9873C321u, 0x350AC2DDu, 0xA5A250E1u,
            0x26578385u, 0xC738D247u, 0x012541CAu, 0xCD33873Cu, 0xC5907F19u, 0xD0CDC82Cu,
            0x5C2B540Au, 0x5656CCA4u, 0x1F887DD1u, 0xA3D987B8u, 0x83E7FE48u, 0x06A28478u,
            0x945682DBu, 0x465F2DF8u, 0x9B494CE1u, 0xFAC8FFBCu, 0x598F39CDu, 0xB12AC825u,
            0xFA99231Bu, 0x3E5C217Eu, 0x3B2D8BA2u, 0xE550FDBAu, 0x8E510006u, 0x846A6733u,
            0x3E573194u, 0xEE48A926u, 0x5CCD36BDu, 0x41C394C8u, 0x10A79620u, 0xA19B67F2u,
            0x8B3FD2A6u, 0x8A285C06u, 0x3A1797D9u, 0x3637050Au, 0x63DFCA07u, 0x7295647Eu,
            0x7A7B3BBAu, 0xBE8E7601u, 0xEA660549u, 0x3C1E511Au, 0xC7A1931Au, 0x06C40C25u,
            0x3796CF70u, 0x7D188664u, 0xCCD9FA38u, 0xB9F70031u, 0x601E2C75u, 0x87FE9735u,
            0xF8CD68B0u, 0xEF645DD6u, 0x7D05B323u, 0x535D7138u, 0x5C02F47Fu, 0x90327A26u,
            0x63ECD3B2u, 0xABD5EA25u, 0x01624325u, 0x302C1641u, 0xDBFBEB93u, 0x1CDFA6BCu,
            0x866519A2u, 0xB15987EDu, 0x113296F1u, 0x0C31EC84u, 0x232A35B2u, 0xB4132090u,
            0x92D0C3C5u, 0x535172E3u, 0x095FFCCBu, 0xFC24A0A9u, 0x932C038Eu, 0x2546326Eu,
            0xCCC15E47u, 0x1BBAFC54u, 0x3CF2A838u, 0xA8486630u, 0x1057E025u, 0x8405B4AEu,
            0xDA36738Du, 0x1EEC4C73u, 0x88B30F90u, 0x4F9FF104u, 0x85EEA780u, 0x6EAB7DA8u,
            0x40D9FDBEu, 0x6FE9593Du, 0x3C850D3Cu, 0x65606C0Cu, 0xB078A231u, 0x70308A34u,
            0x635AF9BDu, 0x6D9A7CBEu, 0xED73EE32u, 0x63660519u, 0x1701DD8Du, 0x0E62955Fu,
            0x180DB0E9u, 0x9CB66A13u, 0xD3C2CD3Eu, 0x78FB88AAu, 0x85FDBE48u, 0xA2859C52u,
            0x9579F8F8u, 0x902FFD41u, 0x4B7C6A7Bu, 0x1F5E048Au, 0x8E262D89u, 0x706D2495u,
            0xEBBBD878u, 0x816D7F42u, 0x88CDFBF1u, 0x3E6CC58Au, 0x754A64ABu, 0xAA7DFAFDu,
            0xE98D0A02u, 0xB63CD2F7u, 0x38C8C85Cu, 0x72C5B57Fu, 0xB97F2B0Au, 0xE479DA34u,
            0x553E33F7u, 0x7C86232Au, 0xB35CC8F8u, 0xEDC6266Du, 0xCA67E7FEu, 0x14B7F688u,
            0x072D997Bu, 0xB3D3D66Fu, 0x528C6A42u, 0x121005B9u, 0x0DF2B622u, 0x87D31F39u,
            0x12CE5FD4u, 0xEDAEDB37u, 0x49DEC2F4u, 0x8E53FF25u, 0xE79E435Au, 0x764041AAu,
            0x29A3EE70u, 0xB359BD5Eu, 0x5AA2B047u, 0x303ACD04u, 0xB82A2D07u, 0x165795C2u,
            0xA64AB733u, 0x950FAAC1u, 0xDFA2861Fu, 0xFF195E03u, 0x8CD6E865u, 0x5EB360ECu,
            0x639CB063u, 0x19E1A74Du, 0x7EC12528u, 0x775C20D6u, 0xA44C4DDFu, 0x08722D7Fu,
            0xB0C92D32u, 0x83D145BCu, 0x3B2207E8u, 0x73DA60E4u, 0xA13D0929u, 0x962813B9u,
            0x738F420Bu, 0xEB6572D6u, 0x151A52CAu, 0x80A4A0EFu, 0x23EEE457u, 0x00000000u,
        };

        /// <summary>
        ///     Advances this generator's state by 2**128 draws in place (NumPy's <c>mt19937_jump</c> →
        ///     <c>mt19937_jump_state</c>).
        /// </summary>
        /// <remarks>
        ///     A position of 624 (key exhausted, twist pending) is first normalized to 0 WITHOUT twisting — NumPy's
        ///     <c>if (state-&gt;pos &gt;= N) state-&gt;pos = 0;</c> — because the jump works on the circular key as a
        ///     linear-recurrence state, where that position is equivalent. The caller must own the instance
        ///     exclusively (it is a freshly built <see cref="jumped"/> copy).
        /// </remarks>
        private void JumpInPlace()
        {
            if (_pos >= N)
                _pos = 0;
            Horner1();
        }

        /// <summary>NumPy's <c>get_coef</c>: coefficient <paramref name="deg"/> of the jump polynomial.</summary>
        /// <param name="deg">The coefficient index.</param>
        /// <returns>True when the coefficient is 1.</returns>
        private static bool GetCoef(int deg) => (JumpPoly[deg >> 5] & (1U << (deg & 0x1f))) != 0;

        /// <summary>
        ///     NumPy's <c>horner1</c>: evaluates the jump polynomial at the recurrence using Horner's method —
        ///     <c>temp = state</c>; then for each lower coefficient step <c>temp</c> once and XOR <c>state</c> in where
        ///     the coefficient is 1 — and replaces this state with the result.
        /// </summary>
        private void Horner1()
        {
            int i = MEXP - 1;
            var tempKey = new uint[N]; // calloc: a zeroed temporary state
            int tempPos = 0;

            while (!GetCoef(i))
                i--;

            if (i > 0)
            {
                System.Array.Copy(_key, tempKey, N);
                tempPos = _pos;
                GenNext(tempKey, ref tempPos);
                i--;
                for (; i > 0; i--)
                {
                    if (GetCoef(i))
                        AddState(tempKey, tempPos, _key, _pos);
                    GenNext(tempKey, ref tempPos);
                }
                if (GetCoef(0))
                    AddState(tempKey, tempPos, _key, _pos);
            }
            else if (i == 0)
            {
                System.Array.Copy(_key, tempKey, N);
                tempPos = _pos;
            }

            System.Array.Copy(tempKey, _key, N);
            _pos = tempPos;
        }

        /// <summary>
        ///     NumPy's jump-module <c>gen_next</c>: regenerates ONE word of a circular MT state (the incremental form
        ///     of the twist) and advances its position.
        /// </summary>
        /// <param name="key">The state's key.</param>
        /// <param name="pos">The state's position (advanced; wraps to 0 after word 623).</param>
        private static void GenNext(uint[] key, ref int pos)
        {
            int num = pos;
            uint y;
            if (num < N - M)
            {
                y = (key[num] & UPPER_MASK) | (key[num + 1] & LOWER_MASK);
                key[num] = key[num + M] ^ (y >> 1) ^ ((y & 1) != 0 ? MATRIX_A : 0U);
                pos++;
            }
            else if (num < N - 1)
            {
                y = (key[num] & UPPER_MASK) | (key[num + 1] & LOWER_MASK);
                key[num] = key[num + (M - N)] ^ (y >> 1) ^ ((y & 1) != 0 ? MATRIX_A : 0U);
                pos++;
            }
            else if (num == N - 1)
            {
                y = (key[N - 1] & UPPER_MASK) | (key[0] & LOWER_MASK);
                key[N - 1] = key[M - 1] ^ (y >> 1) ^ ((y & 1) != 0 ? MATRIX_A : 0U);
                pos = 0;
            }
        }

        /// <summary>
        ///     NumPy's <c>add_state</c>: XORs state 2 into state 1 with both circular keys aligned at their own
        ///     positions (addition in the F2 vector space the jump polynomial acts on).
        /// </summary>
        /// <param name="k1">State 1's key (modified).</param>
        /// <param name="pt1">State 1's position.</param>
        /// <param name="k2">State 2's key (read).</param>
        /// <param name="pt2">State 2's position.</param>
        private static void AddState(uint[] k1, int pt1, uint[] k2, int pt2)
        {
            int i;
            if (pt2 - pt1 >= 0)
            {
                for (i = 0; i < N - pt2; i++)
                    k1[i + pt1] ^= k2[i + pt2];
                for (; i < N - pt1; i++)
                    k1[i + pt1] ^= k2[i + (pt2 - N)];
                for (; i < N; i++)
                    k1[i + (pt1 - N)] ^= k2[i + (pt2 - N)];
            }
            else
            {
                for (i = 0; i < N - pt1; i++)
                    k1[i + pt1] ^= k2[i + pt2];
                for (; i < N - pt2; i++)
                    k1[i + (pt1 - N)] ^= k2[i + pt2];
                for (; i < N; i++)
                    k1[i + (pt1 - N)] ^= k2[i + (pt2 - N)];
            }
        }
    }
}
