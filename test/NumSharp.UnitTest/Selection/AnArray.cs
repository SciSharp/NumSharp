using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;

namespace NumSharp.UnitTest.Selection
{
    public partial class Array
    {
        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public Array(List<Array> values)
        {
            Values = values;
        }

        public Array(string val)
        {
            IsValue = true;
            Value = val ?? throw new ArgumentNullException(nameof(val));
        }

        public void Add(Array child)
        {
            Values.Add(child);
        }

        public bool IsValue;
        public string Value;
        public List<Array> Values { get; set; }


        [DebuggerDisplay("{Depth} - {Match}")]
        private class ExpressionTrack
        {
            public Guid Id { get; } = Guid.NewGuid();
            private Match _match;
            public string AssignedVariable { get; set; }

            public Match Match
            {
                get => _match;
                set
                {
                    _match = value;
                    Index = value.Index;
                    Length = value.Length;
                }
            }

            public int Depth { get; set; }
            public int Index { get; set; }
            public int Length { get; set; }
        }

        public static Array Parse(string @string)
        {
            const string arrayElementsSeperationRegex = @"\s{1,}";

            const string BracketDepthPattern2 = @"\[([^\[\]]*)\]";
            var arrayStr = @string;

            //here we perform an algorithm similar to shunting-yard 
            //var parsed = new Dictionary<Guid, Data>();
            //var parsedMap = new Dictionary<string, string>();
            var tracks = new List<ExpressionTrack>();
            string @in = arrayStr;
            int depth = 0;
            int i = 0;
            var parent = new Array(new List<Array>());
            Array _last = null;
            var uniqueid = new string(Guid.NewGuid().ToString("N").SkipWhile(char.IsDigit).ToArray());
            while (true)
            {
                var matches = Regex.Matches(@in, BracketDepthPattern2, RegexOptions.Multiline | RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant);
                if (matches.Count == 0)
                    break;

                foreach (Match expressionMatch in matches.Cast<Match>().Reverse())
                {
                    //iterate matches, last to first
                    //expressionMatch.DisplayMatchResults();
                    var expression = StripBrackets(expressionMatch.Value);
                    var expressionTrack = new ExpressionTrack() {Depth = depth, Match = expressionMatch};
                    tracks.Add(expressionTrack);

                    //the following might be useless, left here in-case problems rise up.
                    //for (int j = 0; j < tracks.Count - 1; j++) { 
                    //    expression = expression.Replace(
                    //        tracks[j].Match.Value,
                    //        tracks[j].AssignedVariable);
                    //}

                    var input = StripBrackets(expression);

                    var _parsed = Regex.Split(input, arrayElementsSeperationRegex, RegexOptions.Multiline | RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant)
                        .ToList();

                    //_last = new Array(_parsed);
                    parent.Add(_last);

                    var key = $"__{uniqueid}{i++}";
                    //variables.Add(key, arr);
                    expressionTrack.AssignedVariable = key; //parsedMap[expressionTrack.Match.Value] = 
                    @in = @in.Replace(expressionMatch.Value, expressionTrack.AssignedVariable);
                }
            }

            //remove used variables
            //foreach (var k in variables.Keys.ToArray().Where(k => k.StartsWith($"__{uniqueid}")))
            //{
            //    variables.Remove(k);
            //}

            return _last;

            string StripBrackets(string input)
            {
                if (input.StartsWith("[") && input.EndsWith("]"))
                    input = input.Substring(1, input.Length - 2);
                return input;
            }
        }
    }
}
