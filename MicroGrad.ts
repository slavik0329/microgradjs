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
    // Performs tanh on this.data
    const t = Math.log(this.data);

    const out = new Value(t, [this], "log");

    out._backward = () => {
      this.grad += (1 / this.data) * out.grad;
    };

    return out;
  }

  sigmoid(): Value {
    // Performs tanh on this.data
    const s = 1 / (1 + Math.exp(-this.data));

    const out = new Value(s, [this], "sig");

    out._backward = () => {
      this.grad += s - (1 - s) * out.grad;
    };

    return out;
  }

  relu(): Value {
    // Performs tanh on this.data
    const out = new Value(this.data < 0 ? 0 : this.data, [this], "ReLu");

    out._backward = () => {
      this.grad += out.data > 0 ? out.data * out.grad : 0;
    };

    return out;
  }

  exp(): Value {
    const x = this.data;
    const out = new Value(Math.exp(x), [this], "exp");

    out._backward = () => {
      this.grad = out.data * out.grad;
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

    // Set output node grad to 1
    this.grad = 1;
    topo.reverse();

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

type ActivationFunctionType = "relu" | "sigmoid" | "tanh";
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

    this.w = new Array(nin).fill(0).map(() => new Value(randomFunction()));
    this.b = new Value(randomFunction());
  }

  call(x: Value[], activationFunction?: ActivationFunctionType) {
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
    return this.neurons.map((neuron) =>
      neuron.call(x, this.activationFunction)
    );
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
    for (let i = 0; i < this.iterations; i++) {
      for (
        let batchNum = 0;
        batchNum < Math.ceil(this.trainingSet.length / this.bs);
        batchNum++
      ) {
        const batch = this.createBatch(batchNum);

        const loss = this.trainOnePassOnBatch(batch);
        console.log(
          `Iteration: ${i} Batch #: ${batchNum} Loss: ${loss.data} LR: ${this.learningRate}`
        );
      }
    }

    const out = this;
  }

  trainOnePassOnBatch(normalizedBatch: TrainingItemNormalized[]): Value {
    // Forward pass
    const predictions = normalizedBatch.map((x) => this.call(x.input));

    let lossFunction = this.getMSELoss;

    if (this.lossType === "CROSS_ENTROPY") {
      lossFunction = this.getSoftmaxCrossEntropyLoss;
    } else if (this.lossType === "MSE") {
      lossFunction = this.getMSELoss;
    }

    const totalLoss = lossFunction(predictions, normalizedBatch);

    // Backward pass
    // Initialize all gradients back to zero
    for (const p of this.parameters()) {
      p.grad = 0;
    }

    totalLoss.backward();

    // Update
    for (const p of this.parameters()) {
      p.data += -this.learningRate * p.grad;
    }

    return totalLoss;
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

    return lossPerExample.reduce((prev, cur) => prev.add(cur));
  }

  private getSoftmaxCrossEntropyLoss(
    predictions: Value[][],
    normalizedBatch: TrainingItemNormalized[]
  ): Value {
    const lossPerExample = predictions.map((pred, predictionsI) => {
      const softMaxPrediction = softMax(pred);
      return crossEntropyLoss(
        normalizedBatch[predictionsI].output,
        softMaxPrediction
      );
    });

    return lossPerExample.reduce((prev, cur) => prev.add(cur));
  }
}

function softMax(x: Value[]): Value[] {
  const sumParts = x.map((x) => x.exp());
  const sum = sumParts.reduce((prev, cur) => prev.add(cur));

  return x.map((item) => item.exp().div(sum));
}

export function crossEntropyLoss(truth: Value[], prediction: Value[]) {
  const lossItems = truth.map((cur, index) =>
    cur
      .neg()
      .mul(prediction[index])
      .sub(v(1).sub(cur))
      .mul(v(1).sub(prediction[index]).log())
  );

  const loss = lossItems.reduce((prev, cur) => prev.add(cur));

  return loss.div(v(truth.length));
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
