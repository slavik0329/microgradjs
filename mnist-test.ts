/** Work in progress */

import { Trainer } from "./MicroGrad";
import { getTrainingSet } from "./utils";

async function go() {
  console.log("Parsing CSV");

  const trainingSet = await getTrainingSet("./trainData/train.csv");

  const net = new Trainer({
    trainingSet,
    lossType: "CROSS_ENTROPY",
    neuronRandomizer: "ZERO_TO_ONE",
    nin: 784,
    nouts: [
      { numLayer: 100, activationFunction: "relu" },
      { numLayer: 10, activationFunction: "relu" },
    ],
    bs: 20,
    learningRate: 0.03,
  });
  console.log("Training");

  net.train();

  console.log("");
}

go();
