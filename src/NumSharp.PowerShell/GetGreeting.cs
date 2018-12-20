using System.Management.Automation;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.PowerShell
{
    [Cmdlet(VerbsCommon.New,"NDArray")]
    [OutputType(typeof(NDArray))]
    public class NewNDArray : Cmdlet
    {
        protected dynamic _NDArray;
        [ValidateNotNull]
        [Parameter(Position = 0, HelpMessage = "A collection of numbers")]
        public IList<object> Array {get;set;}
        [ValidateSet("float","double","int32","int64","Complex")]
        [Parameter(Position = 1, HelpMessage = "The data type of NDArray")]
        public string DataType {get;set;} = "double"; 
        protected override void ProcessRecord()
        {
            switch (DataType)
            {
                case "float" : 
                {
                    float[] array = Array.Select(x => (float) x).ToArray();
                    _NDArray = new NumSharp.Core.NDArray();
                    _NDArray.Data = array;
                    _NDArray.Shape = new Shape(array.Length);
                    break;
                }
                case "double" : 
                {
                    double[] array = Array.Select(x => (double) x).ToArray();
                    _NDArray = new NumSharp.Core.NDArray();
                    _NDArray.Data = array;
                    _NDArray.Shape = new Shape(array.Length);
                    break;
                }
                default : 
                {
                    break;
                }

            }
            WriteObject(_NDArray);
        }
    }
    /// <summary>
    ///     A simple Cmdlet that outputs a greeting to the pipeline.
    /// </summary>
    [Cmdlet(VerbsCommon.Get, "Greeting")]
    public class GetGreeting
        : Cmdlet
    {
        /// <summary>
        ///     The name of the person to greet.
        /// </summary>
        [ValidateNotNullOrEmpty]
        [Parameter(Position = 0, HelpMessage = "The name of the person to greet")]
        public string Name { get; set; } = "Stranger";

        /// <summary>
        ///     Perform Cmdlet processing.
        /// </summary>
        protected override void ProcessRecord()
        {
            WriteObject($"Hello, {Name}!");
        }
    }
}
