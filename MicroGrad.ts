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

export class Neuron {
  public w: Value[];
  public b: Value;

  constructor(nin: number) {
    this.w = new Array(nin)
      .fill(0)
      .map(() => new Value(getRandomNeuronValue()));
    this.b = new Value(getRandomNeuronValue());
  }

  call(x: Value[]) {
    const activation = x.reduce(
      (prev, cur, i) => prev.add(this.w[i].mul(cur)),
      this.b
    );
    return activation.tanh();
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }
}

export class Layer {
  public neurons: Neuron[];

  constructor(nin: number, nout: number) {
    this.neurons = new Array(nout).fill(0).map(() => new Neuron(nin));
  }

  call(x: Value[]): Value[] {
    return this.neurons.map((neuron) => neuron.call(x));
  }

  parameters(): Value[] {
    let params: Value[] = [];

    return this.neurons.reduce(
      (prev, cur) => [...prev, ...cur.parameters()],
      params
    );
  }
}

export class MLP {
  public layers: Layer[];

  constructor(nin: number, nouts: number[]) {
    const sz = [nin, ...nouts];
    this.layers = new Array(nouts.length)
      .fill(0)
      .map((val, i) => new Layer(sz[i], sz[i + 1]));
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
  nouts: number[];
  iterations?: number;
}

export interface TrainerOptions extends NetworkOptions {
  trainingSet: TrainingItemUnnormalized[];
  learningRate?: number;
}

export class Trainer extends MLP {
  private trainingSet: TrainingItemUnnormalized[];
  private iterations: number;
  private learningRate: number;

  constructor({
    nin,
    nouts,
    trainingSet,
    iterations = 100,
    learningRate = 0.05,
  }: TrainerOptions) {
    super(nin, nouts);

    this.learningRate = learningRate;
    this.iterations = iterations;
    this.trainingSet = trainingSet;
    // this.trainingSet = trainingSet.map((trainingItem) => ({
    //   input: trainingItem.input.map((item) => v(item)),
    //   output: trainingItem.output.map((item) => v(item)),
    // }));
  }

  train() {
    // for (let i = 0; i < this.iterations; i++) {
    //   // Forward pass
    //   const ypred = this.trainingSet.map((x) => this.call(x.input));
    //   const lossPerExample = ypred.map((pred, i) => {
    //     const lossPerCorrespondingValue = pred.map((predictedVal, i) => {
    //       return predictedVal.sub(this.trainingSet[i].output[i]).pow(2);
    //     });
    //
    //     return lossPerCorrespondingValue.reduce((prev, cur) => prev.add(cur));
    //   });
    //
    //   const totalLoss = lossPerExample.reduce((prev, cur) => prev.add(cur));
    //
    //   // Backward pass
    //   // Initialize all gradients back to zero
    //   for (const p of this.parameters()) {
    //     p.grad = 0;
    //   }
    //
    //   totalLoss.backward();
    //
    //   // Update
    //   for (const p of this.parameters()) {
    //     p.data += -this.learningRate * p.grad;
    //   }
    //
    //   console.log(`Step: ${i} Loss: ${totalLoss.data}`);
    // }
  }
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

/** Get random number between -1 and +1 */
function getRandomNeuronValue(): number {
  return Math.random() * 2 - 1;
}
