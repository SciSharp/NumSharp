using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Interop;

[TestClass]
public class GistSignalSourceDemonstrationsLiveTests : InteropTestBase
{
    private const string VertexReference = """
        def vertex(y, i):
            x = .5*(y[i-1]-y[i+1])/(y[i-1]-2*y[i]+y[i+1])+i
            return x, y[i]-.25*(y[i-1]-y[i+1])*(x-i)
        """;

    [DataTestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void CrossingReductionStages_ExplicitAxisMatchesNumpyPairwiseMean(bool harmonicDemo)
    {
        using var scope = NDScope.Open();
        double sampleRate = harmonicDemo ? 8192 : 44100;
        NDArray signal;
        if (harmonicDemo)
        {
            var time = np.arange(1024).astype(NPTypeCode.Double) / sampleRate;
            signal = np.zeros(new Shape(1024), NPTypeCode.Double);
            for (int h = 1; h <= 7; h++) signal = signal + np.sin(2 * Math.PI * 384 * h * time) / h;
        }
        else
        {
            var samples = new double[2048];
            for (int i = 0; i < samples.Length; i++) samples[i] = Math.Sin(2 * Math.PI * 1000 * i / sampleRate + .4);
            signal = np.array(samples);
        }
        var indices = np.nonzero((signal["1:"] >= 0.0) & (signal[":-1"] < 0.0))[0];
        var crossings = np.empty(new Shape(indices.size), NPTypeCode.Double);
        for (long k = 0; k < indices.size; k++)
        {
            long i = indices.item<long>(k);
            double left = signal.item<double>(i), right = signal.item<double>(i + 1);
            crossings[k] = i - left / (right - left);
        }
        var periods = np.diff(crossings);
        double pairwiseSum = np.sum(periods, axis: 0).item<double>();
        double pairwiseMean = np.mean(periods, axis: 0).item<double>();
        double flatMean = np.mean(periods).item<double>();
        var stages = np.array(new[] { pairwiseSum, pairwiseMean, FrequencyEstimation.FromCrossings(signal, sampleRate) });
        Console.WriteLine($"CROSSING_REDUCTION harmonic={harmonicDemo}; n={periods.size}; flatMean={flatMean:R}/0x{BitConverter.DoubleToUInt64Bits(flatMean):X16}; pairwiseMean={pairwiseMean:R}/0x{BitConverter.DoubleToUInt64Bits(pairwiseMean):X16}");
        ExportTo("s", signal);
        PyExec($"""
            fs={sampleRate:R}
            i=np.flatnonzero((s[1:]>=0)&(s[:-1]<0))
            crossings=np.array([j-s[j]/(s[j+1]-s[j]) for j in i])
            periods=np.diff(crossings)
            stages=np.array([np.sum(periods),np.mean(periods),fs/np.mean(periods)])
            """);
        using (Gil())
        {
            using var expectedCrossings = Scope.Eval("crossings");
            using var expectedPeriods = Scope.Eval("periods");
            using var expectedStages = Scope.Eval("stages");
            GistParity.AssertExact(crossings, expectedCrossings, "intersample crossings before reduction");
            GistParity.AssertExact(periods, expectedPeriods, "period differences before reduction");
            GistParity.AssertExact(stages, expectedStages, "NumPy pairwise sum, divide-by-count and final frequency");
        }
    }

    [TestMethod]
    public void ParabolicMain_PlotMarkers_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        var values = np.array(new double[] { 2, 1, 4, 8, 11, 10, 7, 3, 1, 1 });
        long i = np.argmax(values);
        var (x, y) = FrequencyEstimation.Parabolic(values, i);
        var result = np.array(new double[,] { { i, values.item<double>(i) }, { x, y } });
        ExportTo("s", values);
        PyExec(VertexReference + "\ni=int(np.argmax(s))\nmarkers=np.array([[i,s[i]],vertex(s,i)],dtype=np.float64)");
        using (Gil())
        {
            using var expected = Scope.Eval("markers");
            GistParity.AssertExact(result, expected, "parabolic.py main: silver and blue plot markers");
        }
    }

    [TestMethod]
    public void ParabolicReadme_ThreePointPolyfit_ExactLiveNumpy()
    {
        Assert.IsTrue(OpenBlasEngine.LapackAvailable, "Pinned three-point polyfit requires LAPACK.");
        using var scope = NDScope.Open();
        var values = np.array(new double[] { 2, 3, 1, 6, 4, 2, 3, 1 });
        var (x, y) = FrequencyEstimation.ParabolicPolyfit(values, 3, 3);
        var result = np.array(new[] { x, y });
        ExportTo("s", values);
        PyExec("a,b,c=np.polyfit(np.array([2.,3.,4.]),s[2:5],2)\nx=-.5*b/a\nresult=np.array([x,a*x**2+b*x+c])");
        using (Gil())
        {
            using var expected = Scope.Eval("result");
            GistParity.AssertExact(result, expected, "parabolic.md three-point fit (current NumPy, not historical printed rounding)");
        }
    }

    [TestMethod]
    public void PeakMain_PlotScatterColumns_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        // Same numeric source fixture, normalized to the port's declared float64
        // contract; the original Python integer list would otherwise infer int64.
        var values = np.array(new double[] { 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, -2, 0 });
        var (maxima, minima) = PeakDetection.Detect(values, .3);
        ExportTo("s", values);
        PyExec("""
            hi,lo=-np.inf,np.inf
            hx,lx=np.nan,np.nan
            waiting=True
            high,low=[],[]
            for i,value in enumerate(s):
                if value>hi: hi,hx=value,i
                if value<lo: lo,lx=value,i
                if waiting:
                    if value<hi-.3:
                        high.append((hx,hi))
                        lo,lx,waiting=value,i,False
                elif value>lo+.3:
                    low.append((lx,lo))
                    hi,hx,waiting=value,i,True
            high=np.asarray(high,dtype=np.float64)
            low=np.asarray(low,dtype=np.float64)
            """);
        using (Gil())
        {
            using var high = Scope.Eval("high");
            using var low = Scope.Eval("low");
            GistParity.AssertExact(maxima, high, "peakdetect.py main: blue scatter coordinates");
            GistParity.AssertExact(minima, low, "peakdetect.py main: red scatter coordinates");
        }
    }

    /// <summary>
    /// The synthetic-sine crossing and FFT measurements vs live NumPy, exact. The reference imports
    /// <c>scipy.signal.windows</c>, so this is inconclusive where SciPy is not installed.
    /// </summary>
    [TestMethod]
    [PythonEcosystem]
    public void SyntheticSine_MeasuredCrossingAndFftOutputs_ExactLiveNumpy()
    {
        // Checked BEFORE any export, so a missing SciPy cannot leave pinned exports behind (the leak
        // would cascade into every interop test that asserts an absolute LiveExports count).
        SkipUnless("scipy");
        using var scope = NDScope.Open();
        var samples = new double[2048];
        for (int i = 0; i < samples.Length; i++) samples[i] = Math.Sin(2 * Math.PI * 1000 * i / 44100.0 + .4);
        var signal = np.array(samples);
        var result = np.array(new[] { FrequencyEstimation.FromCrossings(signal, 44100), FrequencyEstimation.FromFft(signal, 44100) });
        ExportTo("s", signal);
        PyExec(VertexReference + """

            from scipy.signal.windows import blackmanharris
            i=np.flatnonzero((s[1:]>=0)&(s[:-1]<0))
            crossings=np.array([j-s[j]/(s[j+1]-s[j]) for j in i])
            m=abs(np.fft.rfft(s*blackmanharris(len(s))))
            expected=np.array([44100/np.mean(np.diff(crossings)),44100*vertex(np.log(m),np.argmax(m))[0]/len(s)])
            """);
        using (Gil())
        {
            using var expected = Scope.Eval("expected");
            GistParity.AssertExact(result, expected, "synthetic 1000Hz signal, NOT the unidentified historical recording");
        }
    }

    /// <summary>
    /// All four measurements of the synthetic demo harmonics vs live NumPy, exact. The reference imports
    /// <c>scipy.signal</c>, so this is inconclusive where SciPy is not installed.
    /// </summary>
    [TestMethod]
    [PythonEcosystem]
    public void SyntheticDemoHarmonics_AllFourMeasurements_ExactLiveNumpy()
    {
        SkipUnless("scipy");   // before any export — see SyntheticSine_MeasuredCrossingAndFftOutputs_ExactLiveNumpy
        using var scope = NDScope.Open();
        var time = np.arange(1024).astype(NPTypeCode.Double) / 8192.0;
        var signal = np.zeros(new Shape(1024), NPTypeCode.Double);
        for (int h = 1; h <= 7; h++) signal = signal + np.sin(2 * Math.PI * 384 * h * time) / h;
        var result = np.array(new[] {
            FrequencyEstimation.FromCrossings(signal, 8192), FrequencyEstimation.FromFft(signal, 8192),
            FrequencyEstimation.FromAutocorrelation(signal, 8192), FrequencyEstimation.FromHarmonicProductSpectrum(signal, 8192)
        });
        ExportTo("s", signal);
        PyExec(VertexReference + """

            from scipy.signal.windows import blackmanharris
            from scipy.signal import correlate
            i=np.flatnonzero((s[1:]>=0)&(s[:-1]<0))
            crossings=np.array([j-s[j]/(s[j+1]-s[j]) for j in i])
            m=abs(np.fft.rfft(s*blackmanharris(len(s))))
            c=correlate(s,s,mode='full'); c=c[len(c)//2:]
            start=np.flatnonzero(np.diff(c)>0)[0]
            peak=start+np.argmax(c[start:])
            product=m[:len(m)//7].copy()
            for h in range(2,8): product*=m[::h][:len(product)]
            hp=1+np.argmax(product[1:])
            expected=np.array([8192/np.mean(np.diff(crossings)),8192*vertex(np.log(m),np.argmax(m))[0]/len(s),
                               8192/vertex(c,peak)[0],8192*vertex(np.log(product),hp)[0]/len(s)])
            """);
        using (Gil())
        {
            using var expected = Scope.Eval("expected");
            GistParity.AssertExact(result, expected, "synthetic demo estimates; matching bytes does not remove autocorrelation's 0.504Hz estimation bias");
        }
    }
}
