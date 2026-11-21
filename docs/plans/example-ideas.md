# Example ideas for NumSharp

The owner asked on 2026-11-21 for ideas, of any category, for examples built on NumSharp, its NuGet packages and its
existing examples. The numbers continue after the 0-20 table of the home page's two ladders (0-10 the live ladder,
11-20 journey5's real-world ladder), so one number picks one idea from either list.

- ★ a strong candidate for the home page or a gallery: short, visual or surprising.
- [OpenBLAS] needs NumSharp.Interop.OpenBLAS (SVD, eigenvalues, lstsq / polyfit, polynomial roots).
- [pythonnet], [ORT] (ONNX Runtime), [ML.NET], [TorchSharp], [Tensors] (System.Numerics.Tensors), [Bitmap]: the released
  companion packages.
- [Drawing] NumSharp.Drawing (Pillow's Image on NDArray), on feat/drawing, not released; [Parquet]
  NumSharp.Interop.ParquetNet, not released.
- "gap": needs a NumPy function NumSharp lacks.
- Pictures in and out without [Drawing]: [Bitmap] on Windows, or PGM / PPM files written with `tofile`.

## Already built in examples/ (on master; the site links none of them yet)

- 21 ★ **MaterialLab** — real-time liquids, sand, snow, jelly and lava (MLS-MPM), every material a list of fused `np.evaluate` stages; 72-208 FPS, a 27-check physics gate.
- 22 ★ **Gravity sandbox** — the figure-eight three-body orbit, the solar system, relativistic precession; every force from one broadcast subtraction and two sums; energy kept to 1e-9.
- 23 ★ **Falling sand** — sand, water, oil, fire, lava → obsidian; a mass-conserving cellular automaton; its Player renders 1000×500 PNGs without Unity.
- 24 **Life Arcade** — an Avalonia game: a Pong ball shatters a living Conway colony.
- 25 **Keras-style neural networks** — an MNIST MLP with callbacks, weights saved as `.npz` that NumPy opens.
- 26 ★ **Karpathy ports** — microGPT, a character RNN, a batched LSTM, Pong from pixels by policy gradient, natural evolution strategies, the Stable Diffusion latent walk.
- 27 **The ten most-starred NumPy gists** — forecasting and ranking metrics, frequency estimation, peak detection, TensorBoard histograms, Catch Q-learning, time-series windows, GoogLeNet preprocessing; 88 managed + 55 live NumPy tests.
- 28 **ImageNet classification** — an ONNX model fed from an NDArray's own memory [ORT].
- 29 **ML.NET on arrays** — blobs trained from `AsDataView`; house prices trained in ML.NET, exported to ONNX, run in ONNX Runtime [ML.NET] [ORT].
- 30 **OpenBLAS** — products and linear algebra at native speed, backend choice, reproducible bits [OpenBLAS].
- 31 **Python from C#** — a 12-step pythonnet tour (zero-copy, GIL, PyTorch, pandas), a CI suite that tests NumPy, SciPy, pandas, scikit-learn, Matplotlib and PyTorch from C#, and the `Python.*` facade driving scikit-learn [pythonnet].
- 32 **NDScoping** — one computation five ways, from a leak to `[NDScoped]` (NumSharp.Build).
- 33 **Pillow's Image on NDArray** — the SkiaSharp and System.Drawing examples [Drawing].

## Ports of examples NumPy users already know

- 34 ★ **The 100 NumPy exercises** (rougier/numpy-100, MIT) — every one side by side, each output checked against NumPy.
- 35 **The official NumPy tutorials** — Moore's law from real data, masked arrays on real data, X-ray image processing, fractals, static equilibrium; SVD image compression [OpenBLAS].
- 36 **Project Euler 1-10** — multiples of 3 or 5, even Fibonacci numbers, primes below two million by a sieve, each a line or two.
- 37 **Advent of Code grid puzzles** — seating automata, flood fills, beam paths, solved with arrays (own inputs only).
- 38 **More ranked gists** — run-length encoding (#14), Holt-Winters (#18), RGB ↔ HSV (#20), image pyramids (#30), 3-D finite elements in 100 lines (#36), Euler-angle rotations (#37), reservoir sampling (#44), RANSAC plane fitting (#59).
- 39 **Classic datasets from scratch** — Iris by k-NN, Titanic by logistic regression (`np.loadtxt(usecols:)`), digits by PCA [OpenBLAS].

## Finance and business

- 40 **Max drawdown** — an equity curve's running peak and worst fall; gap: `np.maximum.accumulate` (in ~42,000 GitHub files).
- 41 ★ **Efficient frontier** — 20,000 random portfolios from `np.random.dirichlet`, risk and return by `einsum`, the best Sharpe ratio by `argmax`.
- 42 **Minimum-variance portfolio** — one `np.linalg.solve` on `np.cov` (managed, no backend).
- 43 **Value at Risk** — historical (`np.percentile`), parametric and Monte Carlo (`multivariate_normal`) side by side.
- 44 ★ **Black-Scholes for a whole options chain** — one fused `np.evaluate` expression; NumPy has no `erf`, so a polynomial normal CDF sits inside it.
- 45 **Loan amortization** — the payment formula, principal and interest columns by `cumsum`, printed as NumPy prints.
- 46 **Retirement fan chart** — 10,000 `cumprod` paths of random returns, the 5th / 50th / 95th percentile each year.
- 47 **Bollinger bands and RSI** — from `np.lib.stride_tricks.sliding_window_view`, no loop.
- 48 ★ **Benford's law fraud check** — first digits by `log10` and `floor`, `bincount` against log₁₀(1 + 1/d).
- 49 **ABC inventory** — `argsort` by value, `cumsum` shares, `searchsorted` for the 80 % cut.
- 50 **Cohort retention** — `ravel_multi_index` + `bincount` into a signup-month × months-since grid.
- 51 **Currency arbitrage** — min-plus products of a −log rates matrix by broadcasting.
- 52 **Leontief economy** — the production a demand needs: `solve(I − A, d)`.

## Data analysis without pandas

- 53 **Pivot table** — region × month from `ravel_multi_index` + `bincount(weights:)`.
- 54 **Missing values** — `np.ma` means and spreads that skip gaps, then `filled`.
- 55 **Outliers by IQR** — `np.percentile` 25 / 75, flags by `np.where`.
- 56 **One-hot encoding** — `np.eye(k)[labels]`, and back by `argmax`.
- 57 **Correlation heatmap** — `np.corrcoef` of a dataset as a colored grid.
- 58 **A/B test without tables** — a bootstrap interval of the lift (`choice(replace: true)`) and a permutation test.
- 59 **Kernel density estimate** — Gaussians broadcast over a grid, `(g, 1) − (1, n)`.
- 60 **Records without structured dtypes** — parallel column arrays sorted together by one `argsort` (NumSharp has no structured dtype; the docs page shows this pattern).
- 61 **CSV to a summary** — `np.loadtxt`, then count, mean, std, min, quartiles and max per column, like pandas' `describe()`.
- 62 **Weather map from stations** — inverse-distance weighting by broadcast distances.

## Machine learning from scratch

- 63 ★ **k-nearest neighbours** — `argpartition` finds the top k without sorting, `bincount` votes.
- 64 **Linear regression twice** — normal equations (`solve`) and gradient descent reach the same coefficients.
- 65 **Logistic regression** — two blobs, the decision boundary drawn as a character map.
- 66 ★ **Confusion matrix in one line** — `bincount(y * k + p).reshape(k, k)`, then precision and recall per class.
- 67 **ROC curve and AUC** — `argsort` the scores, `cumsum` the hits, `np.trapezoid`.
- 68 **Naive Bayes spam filter** — word counts per class (`np.strings.lower`, `np.unique(return_inverse:)`, `bincount`), log-probabilities, `argmax`.
- 69 **Gaussian mixture by EM** — responsibilities by broadcasting, a stable log-sum-exp.
- 70 **PCA** — the top component by power iteration (managed), the full decomposition by `np.linalg.svd` [OpenBLAS].
- 71 **Anomalies by Mahalanobis distance** — `inv(cov)` and `einsum("ij,jk,ik->i")`.
- 72 **SMOTE** — oversample a rare class by interpolating between neighbours (imbalanced-learn's idea).
- 73 **k-fold cross-validation** — `permutation` + `array_split`, accuracy mean ± std.
- 74 **ALS recommender** — matrix factorization by alternating `solve` per user and per item.
- 75 **Multi-armed bandits** — ε-greedy vs UCB vs Thompson sampling (`np.random.beta`), regret curves.
- 76 **Q-learning in a grid world** — the learned policy printed as arrows.
- 77 **Optimizers race on the Rosenbrock valley** — gradient descent, momentum and Adam paths over a contour map.
- 78 **Travelling salesman** — a genetic algorithm on random-key `argsort` permutations, simulated annealing beside it.
- 79 **CartPole by the cross-entropy method** — a Gym.NET environment, a NumSharp policy.
- 80 **OptunaSharp tunes a NumSharp model** — ridge α or the k of k-NN.

## Deep learning and AI

- 81 ★ **Attention from scratch** — `einsum` for Q·Kᵀ, a causal mask, softmax, the weighted V: one transformer block.
- 82 **A convolution layer as im2col** — `sliding_window_view` + `tensordot`, checked against a loop.
- 83 **micrograd on NDArray** — a 100-line reverse-mode autograd that trains a small MLP.
- 84 **Diffusion on 2-D points** — a DDPM noise schedule, forward noising, reverse sampling: the spiral comes back.
- 85 **Byte-pair-encoding tokenizer** — adjacent pairs counted by `np.unique(axis: 0, return_counts:)`, the top pair merged, repeat.
- 86 **Word analogies** — king − man + woman by cosine similarity over embeddings loaded from `.npy`.
- 87 ★ **YOLO object detection** — letterboxing, IoU by broadcasting and non-max suppression in NumSharp, the model in ONNX Runtime [ORT].
- 88 **Sentiment from a BERT model** — `Postprocess.Softmax` and `TopK` on the logits [ORT].
- 89 **Whisper's front end** — a log-mel spectrogram in NumSharp (windowed `rfft`, a mel filterbank `matmul`) feeding the ONNX encoder [ORT].
- 90 **Text to speech** — an ONNX TTS model (KokoroSharp's route) to a WAV file [ORT].
- 91 **Inference on an Android phone** — ONNX Runtime's NNAPI and XNNPACK providers, proven for 0.81.0 [ORT].
- 92 **Train in TorchSharp, prepare data in NumSharp** — zero-copy both ways [TorchSharp].
- 93 **Semantic search for a RAG bot** — cosine top-k over 100,000 embeddings with SIMD `TensorPrimitives` on `AsTensorSpan` [Tensors].

## Signals and audio

- 94 ★ **Make music** — a chord progression from `sin` and an ADSR envelope, saved as a WAV (a 44-byte header, then int16 samples).
- 95 ★ **Spectrogram of a chirp** — `sliding_window_view` × `np.hanning`, `rfft`, decibels, drawn as an image.
- 96 **Low-pass filter** — a windowed sinc (`np.sinc`, `np.hamming`) applied by `np.convolve`, spectra before and after.
- 97 **DTMF decoder** — which phone keys were pressed, from FFT peaks at the row and column tones.
- 98 **Heart rate from an ECG** — band-pass, peaks, beats per minute.
- 99 **Tempo detection** — an onset envelope, `np.correlate`, beats per minute.
- 100 **Noise reduction** — spectral subtraction with `rfft` / `irfft`.
- 101 **Wavelet denoising** — Haar averages and differences by `reshape(-1, 2)`, a soft threshold, the inverse (PyWavelets' idea).
- 102 **Morse code** — text to a tone WAV and back by envelope run lengths.
- 103 **Radar range-Doppler map** — a 2-D FFT finds a target's distance and speed.

## Images and vision

- 104 ★ **Histogram equalization** — `histogram`, a `cumsum` lookup table, every pixel fancy-indexed through it.
- 105 **Otsu threshold** — the between-class variance of all 256 thresholds at once, `argmax`.
- 106 ★ **Seam carving** — content-aware resizing: gradient energy, then a row-by-row DP with `np.minimum` of shifted rows.
- 107 ★ **JPEG in 30 lines** — the 8×8 DCT as matrix products, a quantization table, reconstruction and PSNR.
- 108 **Blur by FFT** — `fft2` × the kernel's spectrum, `ifft2`, against direct convolution.
- 109 **1-bit dithering** — a `np.tile`d Bayer matrix, one comparison, `packbits`.
- 110 **ASCII-art photo** — block means by `reshape(h, b, w, b).mean((1, 3))`, a character ramp by fancy indexing.
- 111 **Green screen** — a chroma mask by `np.where`, feathered edges.
- 112 **Hue rotation** — RGB → HSV → RGB, vectorized.
- 113 **Erosion and dilation** — `sliding_window_view(...)` then `min` / `max`.
- 114 **Template matching** — a normalized cross-correlation map, `unravel_index(argmax)`.
- 115 **Document scanner** — a homography from four corners (`solve` on 8×8), `meshgrid`, bilinear sampling.
- 116 **Pixel sorting** — glitch art: `argsort` each row by brightness, `take_along_axis`.
- 117 **Laplacian-pyramid blending** — the apple-orange picture.
- 118 **Color quantization** — 16 colors by k-means in NumSharp vs Pillow's median cut and octree [Drawing].
- 119 ★ **Animated GIF from arrays** — the Game of Life or any simulation through Pillow's ported GIF encoder [Drawing].
- 120 **Color science** — sRGB → Lab through the LittleCMS port, ΔE between swatches [Drawing].
- 121 **Edit a Bitmap in place** — a zero-copy NDArray view over System.Drawing's pixels [Bitmap].
- 122 **OpenCV draws on NumSharp memory** — through Python and `np.frombuffer` [pythonnet].

## Physics and simulation

- 123 ★ **Reaction-diffusion** — Gray-Scott spots and stripes from a 5-point Laplacian on slices.
- 124 ★ **Wind tunnel** — a lattice Boltzmann vortex street behind a cylinder (D2Q9, about 80 lines).
- 125 **Heat in a plate** — explicit finite differences, checked against the steady state.
- 126 **Ripple tank** — the 2-D wave equation, two sources interfering.
- 127 ★ **1,000 double pendulums at once** — vectorized RK4, chaos fanning out from nearly equal starts.
- 128 **Lorenz attractor** — 10,000 trajectories, a density image by `histogram2d`.
- 129 **Ising model** — Metropolis checkerboard updates, magnetization against temperature.
- 130 **Epidemics** — SIR / SEIR curves, and a grid version with neighbour infection.
- 131 **Phantom traffic jams** — the Nagel-Schreckenberg automaton's space-time diagram.
- 132 **Forest fires and sandpiles** — two automata, one critical state.
- 133 **Boids** — flocking from pairwise distances by broadcasting.
- 134 **Brownian motion** — 10,000 walkers, the mean squared displacement grows linearly.
- 135 **Rope and cloth** — Verlet integration with distance constraints.
- 136 **A Lennard-Jones gas** — molecular dynamics, temperature, the radial distribution by `histogram`.
- 137 **Quantum tunnelling** — the split-step Fourier Schrödinger solver in complex128.
- 138 **Quantum circuits** — a state-vector simulator: H and CNOT by `np.kron`, a Bell state's statistics.
- 139 **Planet positions** — Kepler's equation for 10,000 dates at once by vectorized Newton steps.
- 140 **Drug dosing** — concentration from repeated doses by convolving the schedule with one dose's curve.
- 141 **Balance a chemical equation** — the element matrix with one coefficient fixed, then `solve`.

## Math, puzzles and teaching

- 142 ★ **Sieve of Eratosthenes** — boolean slicing finds the primes below 10⁷, then an Ulam spiral image.
- 143 ★ **Collatz** — stopping times of 1 to 10⁶ in a `while (active.any())` loop over arrays.
- 144 **Sierpinski triangle** — Pascal's triangle mod 2.
- 145 ★ **Hamming(7,4)** — encode by `G @ bits % 2`, flip a bit, the syndrome finds and fixes it.
- 146 **Sudoku** — checked through reshaped 3×3 boxes, solved by constraint propagation on candidate masks.
- 147 **Monty Hall and the birthday paradox** — a million trials each, a line each.
- 148 ★ **Newton fractal** — z³ − 1 in complex128, basins colored by root.
- 149 **Barnsley fern** — an iterated function system, `np.random.choice(p:)` picking each point's map.
- 150 ★ **Strange attractors** — Clifford and de Jong: 10 million points, `histogram2d`, log scale, art.
- 151 **Fibonacci by `np.linalg.matrix_power`** — and int64 overflowing at F₉₃.
- 152 **Fourier epicycles** — the FFT of a closed drawing redraws it with rotating circles.
- 153 **Chebyshev approximation** — `np.polynomial`, error against degree (fitting needs [OpenBLAS]).
- 154 **Central limit theorem** — sums of uniforms turn into a bell.
- 155 **Numerical calculus** — `np.gradient` and `np.trapezoid` against exact answers.
- 156 **Shapes under matrices** — rotation, shear and scale applied to a point cloud.
- 157 **Voronoi diagram** — `argmin` over distances to the seeds.
- 158 **Floyd-Warshall** — all shortest paths, one broadcast `np.minimum` per pivot.
- 159 **Graph facts from matrices** — triangles as `trace(A³) / 6`, breadth-first search by boolean products.
- 160 **Knapsack** — the DP vectorized over capacities, one `np.maximum` per item.
- 161 **Spectral clustering** — a graph Laplacian's eigenvectors (`eigh`) [OpenBLAS].

## Games and toys

- 162 ★ **Minesweeper** — random mines, neighbour counts from eight shifted slices.
- 163 **2048** — a move compresses each row by a stable sort on the zero mask, then merges pairs.
- 164 **Connect Four** — win checks by `sliding_window_view` in four directions.
- 165 **Tetris** — full lines by `all(axis: 1)`, cleared by `np.delete` + `vstack`.
- 166 ★ **Wordle helper** — the best first guess by expected information: feedback patterns as base-3 ints, `bincount`, entropy.
- 167 **Procedural terrain** — fractal noise, then water, sand, grass and snow by `np.digitize`.
- 168 **Cave generator** — a smoothing cellular automaton.
- 169 **Flow-field pathfinding** — a breadth-first wavefront by repeated `np.minimum` of shifted grids; 1,000 agents follow it.
- 170 **Raycaster** — Wolfenstein-style, one ray per screen column, vectorized.
- 171 **Demoscene effects** — Doom fire and plasma in a palette.
- 172 **Elo ratings** — a season of games, win probabilities.
- 173 **Election night** — polls with error, Monte Carlo seat counts.

## Earth, life and engineering

- 174 **Nearest airport** — haversine distances for 1,000 GPS fixes, broadcast `(n, 1)` × `(1, m)`.
- 175 **A bike ride's GPS** — speed, climbing and the steepest stretch (`diff`, `gradient`, `cumsum`).
- 176 **Climate stripes** — yearly temperature anomalies as a striped picture, missing years masked.
- 177 **Earthquakes** — the Gutenberg-Richter b-value from magnitudes.
- 178 **Exoplanet hunting** — fold a light curve by trial periods, find the transit dip.
- 179 **Sunrise and sunset** — every day of a year from the solar declination.
- 180 **DNA** — GC content, k-mer counts (`S1` bytes, `np.unique(return_counts:)`), the reverse complement by a lookup table.
- 181 **Protein structures** — RMSD after a Kabsch alignment (`svd`) [OpenBLAS].
- 182 **Genetic drift** — Wright-Fisher generations by `np.random.binomial`.
- 183 **Robot arm** — forward kinematics from a `cumsum` of joint angles, the reachable workspace.
- 184 **Kalman filter** — tracking a noisy target in 2-D.
- 185 **PID controller** — step responses over a grid of gains, overshoot and settling time.
- 186 **MRI** — an image from k-space by `ifft2` and `fftshift`, and the artifacts of undersampling.

## Text and strings

- 187 ★ **Caesar and Vigenère** — `(b − 65 + k) % 26` on bytes, then letter frequencies crack it.
- 188 **Hill cipher** — encryption as a matrix product mod 26.
- 189 **Anagram groups** — letters as uint8, `sort(axis: 1)`, `np.unique(axis: 0, return_inverse:)`.
- 190 **A table printer** — `np.strings.ljust`, `rjust`, `center` and `zfill`.
- 191 **Log analysis** — `np.strings.partition`, `find` and `slice` pull the status codes, `np.unique` counts them.
- 192 **Spelling suggestions** — Levenshtein distances computed along anti-diagonals.
- 193 **Run-length encoding** — `diff`, `flatnonzero`, `repeat`.
- 194 **Base64 by hand** — `unpackbits`, regroup by 6 bits, `packbits`, a lookup table.

## Interop and hosts

- 195 ★ **Edit in Python, see it in C#** — one buffer shared by an NDArray and a numpy array [pythonnet].
- 196 **Matplotlib in a desktop window** — a figure drawn from C#, shown through its pixel buffer [pythonnet].
- 197 **SciPy optimizes a C# function** — `optimize.minimize` on a C# objective [pythonnet].
- 198 **pandas, polars and PyArrow** — DataFrame and Series columns zero-copy (`np.frombuffer`) [pythonnet].
- 199 **Parquet** — columns straight into NDArrays [Parquet].
- 200 **Numpy.NET** — a PyObject handed over without a copy.
- 201 **`.npy` / `.npz` both ways** — saved in C#, loaded in Python, the same SHA-256.
- 202 **SIMD on NDArray memory** — `TensorPrimitives` cosine similarity and softmax [Tensors].
- 203 **OpenBLAS** — a 2,000 × 2,000 product managed vs native, SVD and eigenvalues unlocked, NumPy 2.5.4's own bits [OpenBLAS].
- 204 **F#** — a NumSharp script with `#r "nuget: NumSharp"`.
- 205 **A one-file .NET 10 script** — `dotnet run mandel.cs` with `#:package NumSharp@0.81.0`, run like a Python script.
- 206 **A web API** — a Mandelbrot tile server returning PNG tiles computed by NumSharp (ASP.NET minimal API).
- 207 **Notebooks** — a Jupyter gallery with a .NET kernel (check which kernel is still maintained).
- 208 **Hosts nobody tried yet** — PowerShell 7, Excel-DNA user functions, Godot's .NET games.

## The engine and NumPy parity

- 209 ★ **Fused vs chained** — one `np.evaluate` pass with no temporaries against six ufunc passes: time and memory side by side.
- 210 **A million windows, no copy** — `sliding_window_view` / `as_strided` allocate nothing.
- 211 **Your own kernel** — NDIter over a transposed, strided view.
- 212 **Threads on demand** — `np.multithreading(true, 8)` before and after a large reduction.
- 213 ★ **Byte for byte** — the same script in Python and C# prints the same text; `np.sum` of a million 0.1s gives NumPy's exact bits.
- 214 **The same errors** — a broadcasting failure raises NumPy's exception with NumPy's message.
- 215 **The same printing** — `np.set_printoptions(precision: 3, suppress: true)` reproduces NumPy's repr.

## With the other SciSharp packages

- 216 **SharpCV** — OpenCV's `Mat` and NDArray (its NumSharp upgrade is on a fork; no PR as of 11-11).
- 217 **PlotNET** — plots of NumSharp arrays (its 4 OpenBugs first).
- 218 **TensorFlow.NET** — after its TF 2.21 migration plan.
- 219 **KokoroSharp and EDMTranslator** — real apps on ONNX Runtime + NumSharp (speech, translation); upgrades on forks, no PRs as of 11-11.

## Notes from the survey

- 21-33 already exist on master, yet the site links none of them: an Examples gallery page with their pictures
  (LifeAndPong's previews, the falling sand's 1000×500 PNG) would cost little.
- Gaps the ideas expose: `np.maximum.accumulate` (40); structured dtypes are deliberately absent (60); SVD,
  eigenvalues, lstsq / polyfit and polynomial roots need [OpenBLAS]; datetime arithmetic is absent, so dates stay int64
  offsets.
- Sources: `examples/` and `examples/RUNDOWN.md`, the 0.80.0 and 0.81.0 release notes, the integrations diagram's 18
  items, NuGet (9 NumSharp packages at 0.81.0, 114 SciSharp packages), and the saved gist and GitHub rankings by stars ×
  NumPy calls (`K:\source\paper-gateway\articles\{gist,github}\rankings\stars-x-usage-2026-09-09`).
