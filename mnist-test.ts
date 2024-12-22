/** Work in progress */

import { Trainer } from "./MicroGrad";
import { getTrainingSet } from "./utils";

async function go() {
  console.log("Parsing CSV");

  const trainingSet = await getTrainingSet("./trainData/train.csv");

  const net = new Trainer({
    trainingSet,
    lossType: "CROSS_ENTROPY",
    neuronRandomizer: "NEG_ZERO_TO_ONE",
    nin: 784,
    nouts: [
      { numLayer: 128, activationFunction: "relu" },
      { numLayer: 64, activationFunction: "relu" },
      { numLayer: 10, activationFunction: "softmax" },
    ],
    bs: 32,
    iterations: 50,
    learningRate: 0.003,
  });
  console.log("Training");

  net.train();

  console.log("");
}

go();
