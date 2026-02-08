# #507: np.maximum error

- **URL:** https://github.com/SciSharp/NumSharp/issues/507
- **State:** OPEN
- **Author:** @Thanatos0173
- **Created:** 2024-02-16T18:29:17Z
- **Updated:** 2024-02-17T14:43:40Z

## Description

Hello, 
I'm working on a Neural Network for the video game Celeste, and I'm using Numsharp to convert a code that a friend made, but he used python.
Currently, I'm calling a train function a huge number of time, and randomly, this error pops out :
```
One or more errors occurred.
-> at System.Threading.Tasks.Task.ThrowIfExceptional()
-> at System.Threading.Tasks.Task.Wait()
-> at System.Threading.Tasks.Task.Wait()
-> at System.Threading.Tasks.Parallel.ForWorker[TLocal]()
-> at System. Threading.Tasks. Parallel.For()
-> at NumSharp. Backends. DefaultEngine.ClipNDArray()
-> at NumSharp.np.maximum()
```
I've googled it but found nothing relevant.
I don't really know if I can do a proper reproductible exemple, but if you need more informations about this error, i would be pleased to send them.

Here is some of the code :

```csharp
        public static void TrainCommand()
        {
            NeuralNetwork.NeuralNetwork.Open();

            for (int i = 0; i < Directory.GetFiles("Mia/Saves","*",SearchOption.AllDirectories).Length/2; i++)
            {
                var tdimarray = np.load($"Mia/Saves/ArraySaved_{i}.npy"); // 20x20 array
                var tdiminput = np.load($"Mia/Saves/InputSaved_{i}.npy"); // 56x1 array 
                for(int j = 0; j < 10_000; j++)
                {
                    double[] input = (double[])(Array)tdiminput[j];
                    NeuralNetwork.NeuralNetwork.Train(lr, tdimarray[j], Utils.AllArrayFromOld(input ));
                }
            }
        }
```

```csharp
        public class FirstLayers
        {
            public NDArray weights;
            public NDArray biases;
            public NDArray inputs;
            public NDArray outputNotActivated;
            public NDArray output;
            public NDArray outputGradient;
            public FirstLayers(NDArray weights, NDArray biases)
            {
                this.weights = weights;
                this.biases = biases;
            }

            public void Forward(NDArray inputs)
            {
                this.inputs = inputs;
                this.outputNotActivated = np.dot(inputs, this.weights) + this.biases; //np.dot
                this.output = np.maximum(0, this.outputNotActivated);
            }

            public void FirstLayerBackward(NDArray inputGradient, double learningRate)
            {
                bool one = DerivRelu(this.outputNotActivated) == null;
                inputGradient = inputGradient * DerivRelu(this.outputNotActivated);
                this.outputGradient = np.dot(inputGradient, this.weights.T);
                this.weights -= np.dot(this.inputs.T, inputGradient) * learningRate / inputGradient.shape[0];
                this.biases -= np.mean(inputGradient, axis: 0) * learningRate;
            }

            public NDArray DerivRelu(NDArray x)
            {
                NDArray result = np.zeros(x.Shape);
                for (int i = 0; i < x.Shape[0]; i++)
                {
                    for (int j = 0; j < x.Shape[1]; j++)
                    {
                        result[i, j] = (x[i, j].Data<double>()[0] > 0 ? 1 : 0);
                    }
                }
                return result;
            }

        }

        public class LastLayer
        {
            public NDArray weights;
            public NDArray biases;
            public NDArray inputs;
            public NDArray outputGradient;
            public NDArray output;

            public LastLayer(NDArray weights, NDArray biases)
            {
                this.weights = weights;
                this.biases = biases;
            }

            public void Forward(NDArray inputs)
            {
                this.inputs = inputs;
                NDArray outputNotActivated = np.dot(inputs, this.weights) + this.biases;
                NDArray expValues = np.exp(outputNotActivated - np.max(outputNotActivated, axis: 1, keepdims: true));
                this.output = expValues / np.sum(expValues.astype(NPTypeCode.Float), axis: 1, keepdims: true);
            }

            public void LastLayerBackward(NDArray yPred, NDArray yTrue, double learningRate)
            {
                NDArray inputGradient = yPred - yTrue;
                this.outputGradient = np.dot(inputGradient, this.weights.T);
                this.weights -= np.dot(this.inputs.T, inputGradient) * learningRate  / inputGradient.shape[0];
                this.biases -= np.mean(inputGradient, axis: 0) * learningRate;
            }
        }

        private static Tuple<List<FirstLayers>, LastLayer> nn;
        
        public static void Open()
        {
            var weights = new List<NDArray> ();
            var biases = new List<NDArray> ();
            foreach(string file in Directory.GetFiles("Mia/weights"))
            {
                weights.Add(np.load (file));
            }
            foreach (string file in Directory.GetFiles("Mia/biases"))
            {
                biases.Add(np.load(file));
            }

            int n = weights.Count;
        
            nn = new Tuple<List<FirstLayers>, LastLayer>(
                new List<FirstLayers>(), 
                new LastLayer(weights[n - 1], biases[n - 1])); // I gave it a new value... We'll see.
            for (int j = 0; j < n - 1; j++)
            {
                nn.Item1.Add(new FirstLayers(weights[j], biases[j]));
            }
        }

        public static void Train(double lr, NDArray allTiles, int[] keypress)
        {
            NDArray trueInputs = allTiles.reshape(1, 400);
            NDArray labels = new NDArray(keypress);

            NDArray output = ForPropagation(trueInputs);


            BackPropagation(output, labels, lr);


        }
        public static NDArray ForPropagation(NDArray input)
        {
            nn.Item1[0].Forward(input);
            for (int i = 1; i < nn.Item1.Count; i++) // Adding all the values inside of FirstLayer
            {
                nn.Item1[i].Forward(nn.Item1[i - 1].output);
            }

            nn.Item2.Forward(nn.Item1[nn.Item1.Count - 1].output);

            return nn.Item2.output;
        }

        public static void BackPropagation(NDArray yPred, NDArray yTrue, double lr)
        {
            // Your implementation for the BackPropagation function
            nn.Item2.LastLayerBackward(yPred, yTrue, lr); 
            nn.Item1[nn.Item1.Count - 1].FirstLayerBackward(nn.Item2.outputGradient, lr);
            for (int i = nn.Item1.Count - 3; i >= 0; i--)
            {
                nn.Item1[i].FirstLayerBackward(nn.Item1[i + 1].outputGradient, lr);
            }
        }
    }
}

```
