import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-node";

export type ActivationFunctionType = "relu" | "sigmoid" | "tanh" | "softmax";
export type NeuronRandomizer = "ZERO_TO_ONE" | "NEG_ZERO_TO_ONE";

export interface TrainingItemUnnormalized {
  input: number[];
  output: number[];
}

export interface NetworkOptions {
  nin: number;
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

type MLPLayer = {
  numLayer: number;
  activationFunction: ActivationFunctionType;
};

export class GPUTrainer {
  private model: tf.Sequential;
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
  }: TrainerOptions) {
    // Replace the webgl backend setup with:
    require("@tensorflow/tfjs-node");
    // The Node backend will automatically use Metal/Core ML on M1/M2 Macs
    // for hardware acceleration

    // If you need to explicitly set the backend:
    tf.setBackend("tensorflow");

    this.bs = bs;
    this.learningRate = learningRate;
    this.iterations = iterations;
    this.trainingSet = trainingSet;
    this.lossType = lossType;

    // Create sequential model
    this.model = tf.sequential();

    // Add input layer
    this.model.add(
      tf.layers.dense({
        units: nouts[0].numLayer,
        activation: this.getActivationFunction(nouts[0].activationFunction),
        inputShape: [nin],
        kernelInitializer: "glorotNormal",
      })
    );

    // Add hidden layers
    for (let i = 1; i < nouts.length; i++) {
      this.model.add(
        tf.layers.dense({
          units: nouts[i].numLayer,
          activation: this.getActivationFunction(nouts[i].activationFunction),
          kernelInitializer: "glorotNormal",
        })
      );
    }

    // Compile model
    this.model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss:
        this.lossType === "CROSS_ENTROPY"
          ? "categoricalCrossentropy"
          : "meanSquaredError",
      metrics: ["accuracy"],
    });
  }

  private getActivationFunction(
    activation: ActivationFunctionType
  ): tf.layers.ActivationIdentifier {
    switch (activation) {
      case "relu":
        return "relu";
      case "sigmoid":
        return "sigmoid";
      case "tanh":
        return "tanh";
      case "softmax":
        return "softmax";
      default:
        return "relu";
    }
  }

  async train() {
    console.log("Converting data to tensors...");

    // Convert training data to tensors
    const inputs = tf.tensor2d(
      this.trainingSet.map((item) => item.input),
      [this.trainingSet.length, this.trainingSet[0].input.length]
    );

    const outputs = tf.tensor2d(
      this.trainingSet.map((item) => item.output),
      [this.trainingSet.length, this.trainingSet[0].output.length]
    );

    console.log("Starting training on GPU...");

    // Train the model
    await this.model.fit(inputs, outputs, {
      batchSize: this.bs,
      epochs: this.iterations,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(
            `Epoch ${epoch + 1}/${this.iterations} - ` +
              `loss: ${logs?.loss.toFixed(4)} - ` +
              `accuracy: ${(logs?.acc * 100).toFixed(1)}% - ` +
              `val_loss: ${logs?.val_loss.toFixed(4)} - ` +
              `val_accuracy: ${(logs?.val_acc * 100).toFixed(1)}%`
          );
        },
      },
    });

    // Clean up tensors
    inputs.dispose();
    outputs.dispose();
  }

  async predict(input: number[]): Promise<number[]> {
    const inputTensor = tf.tensor2d([input], [1, input.length]);
    const prediction = this.model.predict(inputTensor) as tf.Tensor;
    const result = (await prediction.array()) as number[][];

    // Clean up tensors
    inputTensor.dispose();
    prediction.dispose();

    return result[0];
  }

  dispose() {
    this.model.dispose();
  }
}
