/** Work in progress */

import { GPUTrainer } from "./MicroGradGPU";
import { getTrainingSet } from "./utils";
import { join } from "path";

async function go() {
  try {
    console.log("Loading and parsing MNIST training data...");
    const trainingSet = await getTrainingSet(
      join(__dirname, "../trainData/train.csv")
    );

    console.log("Initializing neural network on GPU...");
    const net = new GPUTrainer({
      trainingSet,
      lossType: "CROSS_ENTROPY",
      nin: 784,
      nouts: [
        { numLayer: 128, activationFunction: "relu" },
        { numLayer: 64, activationFunction: "relu" },
        { numLayer: 10, activationFunction: "softmax" },
      ],
      bs: 256,
      iterations: 100,
      learningRate: 0.001,
    });

    console.log("Starting GPU training...");
    console.log(`Total training examples: ${trainingSet.length}`);
    console.log(`Batch size: ${net["bs"]}`);
    console.log(`Network architecture: 784 -> 128 -> 64 -> 10`);
    console.time("Training duration");

    await net.train();

    console.timeEnd("Training duration");
    console.log("Training completed successfully");

    // Clean up GPU resources
    net.dispose();
  } catch (error) {
    console.error("Error during training:", error);
    process.exit(1);
  }
}

// Handle unhandled promise rejections
process.on("unhandledRejection", (error) => {
  console.error("Unhandled promise rejection:", error);
  process.exit(1);
});

go();
