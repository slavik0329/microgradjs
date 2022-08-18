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

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    if (typeof other === "number") {
      other = new Value(other);
    }

    return this.add(other.neg());
  }

  add(other: Value | number): Value {
    if (typeof other === "number") {
      other = new Value(other);
    }

    const out = new Value(this.data + other.data, [this, other], "+");

    out._backward = () => {
      if (typeof other === "number") {
        throw "Shouldn't happen!";
      }

      this.grad += out.grad;
      other.grad += out.grad;
    };

    return out;
  }

  mul(other: Value | number): Value {
    if (typeof other === "number") {
      other = new Value(other);
    }

    const out = new Value(this.data * other.data, [this, other], "*");

    out._backward = () => {
      if (typeof other === "number") {
        return;
      }

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

  div(other: Value): Value {
    return this.mul(other.pow(-1));
  }

  backward() {
    let topo: Value[] = [];
    let visited = new Set<Value>();

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
    this.grad = 1;
    topo.reverse();

    for (const node of topo) {
      node._backward();
    }
  }
}

function getRandomNeuronValue() {
  return Math.random() * 2 - 1;
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

  call(x: Value[]) {
    return this.neurons.map((neuron) => neuron.call(x));
  }

  parameters() {
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

  call(x: Value[]) {
    for (const layer of this.layers) {
      x = layer.call(x);
    }
    return x;
  }

  parameters() {
    let params: Value[] = [];

    return this.layers.reduce(
      (prev, cur) => [...prev, ...cur.parameters()],
      params
    );
  }
}
