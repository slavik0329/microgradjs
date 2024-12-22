export class Value {
  public data: number;
  public grad: number;
  public label: string;
  public _prev: Value[];
  public _backward: () => void;
  private _op: string;

  constructor(data: number, _children: Value[] = [], _op = "", label = "") {
    this.data = data;
    this._prev = _children;
    this._op = _op;
    this.label = label;
    this.grad = 0;
    this._backward = () => {};
  }

  add(other: Value): Value {
    const out = new Value(this.data + other.data, [this, other], "+");

    out._backward = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };

    return out;
  }

  mul(other: Value): Value {
    const out = new Value(this.data * other.data, [this, other], "*");

    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };

    return out;
  }

  tanh(): Value {
    // Performs tanh on this.data
    const t = (Math.exp(2 * this.data) - 1) / (Math.exp(2 * this.data) + 1);

    const out = new Value(t, [this], "tanh");

    out._backward = () => {
      this.grad += (1 - t ** 2) * out.grad;
    };

    return out;
  }

  log(): Value {
    // Instead of Math.log(this.data),
    // clamp to a small positive value
    const eps = 1e-9;
    const valClamped = this.data < eps ? eps : this.data;
    const t = Math.log(valClamped);

    const out = new Value(t, [this], "log");
    out._backward = () => {
      this.grad += (1 / valClamped) * out.grad;
    };

    return out;
  }

  sigmoid(): Value {
    const s = 1 / (1 + Math.exp(-this.data));

    const out = new Value(s, [this], "sig");

    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };

    return out;
  }

  relu(): Value {
    const out = new Value(this.data < 0 ? 0 : this.data, [this], "ReLu");

    out._backward = () => {
      this.grad += out.data > 0 ? out.grad : 0;
    };

    return out;
  }

  exp(): Value {
    const x = this.data;
    const out = new Value(Math.exp(x), [this], "exp");

    out._backward = () => {
      this.grad += out.data * out.grad;
    };

    return out;
  }

  pow(other: number): Value {
    const out = new Value(this.data ** other, [this], `**${other}`);

    out._backward = () => {
      this.grad += other * this.data ** (other - 1) * out.grad;
    };

    return out;
  }

  backward() {
    let topo: Value[] = [];
    let visited = new Set<Value>();

    /** Builds backward graph of nodes to iterate through */
    function buildTopo(v: Value) {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._prev) {
          buildTopo(child);
        }
        topo.push(v);
      }
    }

    buildTopo(this);
    topo.reverse();

    // Initialize gradients
    this.grad = 1;
    for (const node of topo) {
      if (node !== this) {
        node.grad = 0;
      }
    }

    // Backward pass
    for (const node of topo) {
      node._backward();
    }
  }

  sub(other: Value): Value {
    return this.add(other.neg());
  }

  neg(): Value {
    return this.mul(new Value(-1));
  }

  div(other: Value): Value {
    return this.mul(other.pow(-1));
  }
}

type ActivationFunctionType = "relu" | "sigmoid" | "tanh" | "softmax";
type NeuronRandomizer = "ZERO_TO_ONE" | "NEG_ZERO_TO_ONE";

export class Neuron {
  public w: Value[];
  public b: Value;

  constructor(nin: number, neuronRandomizer: NeuronRandomizer = "ZERO_TO_ONE") {
    let randomFunction = getRandomNeuronValueZeroAndOne;

    if (neuronRandomizer === "ZERO_TO_ONE") {
      randomFunction = getRandomNeuronValueZeroAndOne;
    } else {
      randomFunction = getRandomNeuronValueNegOneAndOne;
    }

    // Xavier/Glorot initialization
    const scale = Math.sqrt(2.0 / nin);
    this.w = new Array(nin)
      .fill(0)
      .map(() => new Value(randomFunction() * scale));
    this.b = new Value(0); // Initialize biases to zero
  }

  call(x: Value[], activationFunction: ActivationFunctionType) {
    const activation = x.reduce(
      (prev, cur, i) => prev.add(this.w[i].mul(cur)),
      this.b
    );

    if (activationFunction === "relu") {
      return activation.relu();
    } else if (activationFunction === "sigmoid") {
      return activation.sigmoid();
    } else if (activationFunction === "tanh") {
      return activation.tanh();
    } else if (activationFunction === "softmax") {
      return activation; // Softmax will be applied at the layer level
    } else {
      return activation.relu();
    }
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }
}

export class Layer {
  public neurons: Neuron[];
  public activationFunction: ActivationFunctionType;

  constructor(
    nin: number,
    nout: number,
    activationFunction: ActivationFunctionType = "relu",
    neuronRandomizer: NeuronRandomizer = "ZERO_TO_ONE"
  ) {
    this.neurons = new Array(nout)
      .fill(0)
      .map(() => new Neuron(nin, neuronRandomizer));
    this.activationFunction = activationFunction;
  }

  call(x: Value[]): Value[] {
    const outputs = this.neurons.map((neuron) =>
      neuron.call(x, this.activationFunction)
    );
    return outputs;
  }

  parameters(): Value[] {
    let params: Value[] = [];

    return this.neurons.reduce(
      (prev, cur) => [...prev, ...cur.parameters()],
      params
    );
  }
}

type MLPLayer = {
  numLayer: number;
  activationFunction: ActivationFunctionType;
};

export class MLP {
  public layers: Layer[];

  constructor(
    nin: number,
    nouts: MLPLayer[],
    neuronRandomizer: NeuronRandomizer = "ZERO_TO_ONE"
  ) {
    const layersArr = nouts.map((out) => out.numLayer);

    const sz = [nin, ...layersArr];
    this.layers = new Array(nouts.length)
      .fill(0)
      .map(
        (val, i) =>
          new Layer(
            sz[i],
            sz[i + 1],
            nouts[i].activationFunction,
            neuronRandomizer
          )
      );
  }

  call(x: Value[]): Value[] {
    for (const layer of this.layers) {
      x = layer.call(x);
    }
    return x;
  }

  parameters(): Value[] {
    let params: Value[] = [];

    return this.layers.reduce(
      (prev, cur) => [...prev, ...cur.parameters()],
      params
    );
  }
}

export interface TrainingItemUnnormalized {
  /** Numbers normalized from -1 – +1 */
  input: number[];
  /** Expected result */
  output: number[];
}

export interface TrainingItemNormalized {
  /** Values normalized from -1 – +1 */
  input: Value[];
  /** Expected result */
  output: Value[];
}

export interface NetworkOptions {
  /** Number of inputs into the network */
  nin: number;
  /** Array of neuron counts per layer
   *  eg: [4, 4, 1] In this case there is only 1 output neuron
   * */
  nouts: MLPLayer[];
  iterations?: number;
}

export interface TrainerOptions extends NetworkOptions {
  trainingSet: TrainingItemUnnormalized[];
  learningRate?: number;
  bs?: number;
  lossType?: LossType;
  neuronRandomizer?: NeuronRandomizer;
}

export type LossType = "MSE" | "CROSS_ENTROPY";

export class Trainer extends MLP {
  private trainingSet: TrainingItemUnnormalized[];
  private iterations: number;
  private learningRate: number;
  private neuronRandomizer: NeuronRandomizer;
  private readonly bs: number;
  private readonly lossType: LossType;

  constructor({
    nin,
    nouts,
    bs = 64,
    trainingSet,
    iterations = 100,
    learningRate = 0.01,
    lossType = "MSE",
    neuronRandomizer = "ZERO_TO_ONE",
  }: TrainerOptions) {
    super(nin, nouts, neuronRandomizer);

    this.bs = bs;
    this.learningRate = learningRate;
    this.iterations = iterations;
    this.trainingSet = trainingSet;
    this.lossType = lossType;
    this.neuronRandomizer = neuronRandomizer;
  }

  createBatch(
    /** Which part of the training set to get the batch from */
    batchIndex: number
  ) {
    const start = batchIndex * this.bs;
    const rawBatch = this.trainingSet.filter(
      (item, i) => i >= start && i < start + this.bs
    );

    const normalizedBatch: TrainingItemNormalized[] = rawBatch.map((item) => ({
      input: item.input.map((it) => v(it)),
      output: item.output.map((it) => v(it)),
    }));

    return normalizedBatch;
  }

  train() {
    for (
      let iterationIndex = 0;
      iterationIndex < this.iterations;
      iterationIndex++
    ) {
      for (
        let batchNum = 0;
        batchNum < Math.ceil(this.trainingSet.length / this.bs);
        batchNum++
      ) {
        const batch = this.createBatch(batchNum);

        const output = this.trainOnePassOnBatch(batch);
        console.log(
          `Iteration: ${iterationIndex} Batch #: ${batchNum} Loss: ${output.totalLoss.data} Accuracy: ${output.accuracy} LR: ${this.learningRate}`
        );
      }
    }

    const out = this;
  }

  trainOnePassOnBatch(normalizedBatch: TrainingItemNormalized[]): {
    totalLoss: Value;
    accuracy: number;
  } {
    // Forward pass
    const predictions = normalizedBatch.map((x) => this.call(x.input));

    let lossFunction = this.getMSELoss;
    let accuracyFunction = this.getSimpleAccuracy;

    if (this.lossType === "CROSS_ENTROPY") {
      lossFunction = this.getSoftmaxCrossEntropyLoss;
      accuracyFunction = this.getMultiClassAccuracy;
    }

    const totalLoss = lossFunction.call(this, predictions, normalizedBatch);

    // Backward pass
    // Initialize all gradients back to zero
    for (const p of this.parameters()) {
      p.grad = 0;
    }

    totalLoss.backward();

    // Update with gradient clipping
    const maxGradNorm = 1.0;
    let gradNorm = 0;
    const params = this.parameters();

    // Calculate gradient norm
    for (const p of params) {
      gradNorm += p.grad * p.grad;
    }
    gradNorm = Math.sqrt(gradNorm);

    // Scale factor for gradient clipping
    const scale = Math.min(1.0, maxGradNorm / (gradNorm + 1e-6));

    // Update with clipped gradients
    for (const p of params) {
      p.data += -this.learningRate * p.grad * scale;
    }

    const accuracy = accuracyFunction.call(this, predictions, normalizedBatch);

    return { totalLoss, accuracy };
  }

  private getMultiClassAccuracy(
    predictions: Value[][],
    normalizedBatch: TrainingItemNormalized[]
  ) {
    let correct = 0;
    const total = predictions.length;

    for (let i = 0; i < total; i++) {
      const pred = predictions[i];
      const truth = normalizedBatch[i].output;

      // Get predicted class (max index)
      let maxPredIdx = 0;
      let maxPredVal = pred[0].data;
      for (let j = 1; j < pred.length; j++) {
        if (pred[j].data > maxPredVal) {
          maxPredVal = pred[j].data;
          maxPredIdx = j;
        }
      }

      // Get true class (max index)
      let maxTruthIdx = 0;
      let maxTruthVal = truth[0].data;
      for (let j = 1; j < truth.length; j++) {
        if (truth[j].data > maxTruthVal) {
          maxTruthVal = truth[j].data;
          maxTruthIdx = j;
        }
      }

      if (maxPredIdx === maxTruthIdx) {
        correct++;
      }
    }

    return correct / total;
  }

  private getSimpleAccuracy(
    predictions: Value[][],
    normalizedBatch: TrainingItemNormalized[]
  ) {
    return (
      predictions.reduce((prev, curPred, predictionIndex) => {
        const result =
          curPred.reduce((prevItem, curItem, itemIndex) => {
            const trueThreshold =
              this.neuronRandomizer === "ZERO_TO_ONE" ? 0.5 : 0;
            const isGoodPrediction =
              curItem.data > trueThreshold ===
              normalizedBatch[predictionIndex].output[itemIndex].data >
                trueThreshold;
            return prevItem + (isGoodPrediction ? 1 : 0);
          }, 0) / curPred.length;
        return result + prev;
      }, 0) / predictions.length
    );
  }

  private getMSELoss(
    predictions: Value[][],
    normalizedBatch: TrainingItemNormalized[]
  ): Value {
    const lossPerExample = predictions.map((pred, predsI) => {
      const lossPerCorrespondingValue = pred.map((predictedVal, predI) => {
        return predictedVal.sub(normalizedBatch[predsI].output[predI]).pow(2);
      });

      return lossPerCorrespondingValue.reduce((prev, cur) => prev.add(cur));
    });

    return lossPerExample
      .reduce((prev, cur) => prev.add(cur))
      .div(v(normalizedBatch.length));
  }

  private getSoftmaxCrossEntropyLoss(
    predictions: Value[][],
    normalizedBatch: TrainingItemNormalized[]
  ): Value {
    let totalLoss = new Value(0);

    // Process one example at a time to save memory
    for (let i = 0; i < predictions.length; i++) {
      const logits = predictions[i];

      // Compute max for numerical stability
      let maxLogit = logits[0].data;
      for (let j = 1; j < logits.length; j++) {
        maxLogit = Math.max(maxLogit, logits[j].data);
      }

      // Compute exp(logits - max_logit) and sum
      let sumExp = new Value(0);
      const exps: Value[] = [];
      for (const logit of logits) {
        const exp = logit.sub(new Value(maxLogit)).exp();
        exps.push(exp);
        sumExp = sumExp.add(exp);
      }

      // Compute softmax probabilities and loss
      let exampleLoss = new Value(0);
      for (let j = 0; j < logits.length; j++) {
        const prob = exps[j].div(sumExp);
        const target = normalizedBatch[i].output[j];
        if (target.data > 0) {
          // Only compute log prob for the true class
          const eps = 1e-15;
          const safeProb = Math.max(eps, Math.min(1 - eps, prob.data));
          exampleLoss = exampleLoss.add(
            target.mul(new Value(Math.log(safeProb)))
          );
        }
      }

      totalLoss = totalLoss.add(exampleLoss.neg());
    }

    return totalLoss.div(v(predictions.length));
  }
}

function softMax(x: Value[]): Value[] {
  // Subtract max for numerical stability
  const maxVal = Math.max(...x.map((v) => v.data));
  const shiftedX = x.map((v) => new Value(v.data - maxVal));

  const exps = shiftedX.map((v) => v.exp());
  const sumExp = exps.reduce((a, b) => a.add(b));

  return exps.map((exp) => exp.div(sumExp));
}

function crossEntropyLoss(truth: Value[], prediction: Value[]): Value {
  // Add numerical stability by clamping predictions to avoid log(0)
  const eps = 1e-15;
  const clampedPreds = prediction.map((p) => {
    const val = p.data < eps ? eps : p.data > 1 - eps ? 1 - eps : p.data;
    return new Value(val);
  });

  // Cross entropy loss for multi-class classification
  const lossItems = truth.map((t, i) => {
    return t.mul(clampedPreds[i].log());
  });

  // Sum up and negate
  let loss = lossItems.reduce((prev, cur) => prev.add(cur)).neg();

  return loss;
}

export class Classifier extends Trainer {
  constructor(options: TrainerOptions) {
    super(options);
  }
}

/** Helper for creating easy Value objects */
export function v(num: number): Value {
  return new Value(Number(num));
}

/** Get random number between 0 and +1 */
function getRandomNeuronValueZeroAndOne(): number {
  return Math.random();
}

/** Get random number between -1 and +1 */
function getRandomNeuronValueNegOneAndOne(): number {
  return Math.random() * 2 - 1;
}
