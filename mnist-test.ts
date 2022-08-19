import { Trainer } from "./MicroGrad";
import { getTrainingSet } from "./utils";

async function go() {
  console.log("Parsing CSV");

  const trainingSet = await getTrainingSet("./trainData/train.csv");

  const net = new Trainer({
    trainingSet,
    lossType: "CROSS_ENTROPY",
    nin: 784,
    nouts: [
      { numLayer: 100, activationFunction: "relu" },
      { numLayer: 64, activationFunction: "relu" },
      { numLayer: 10, activationFunction: "sigmoid" },
    ],
    bs: 20,
    learningRate: 0.03,
  });
  console.log("Training");

  // for (let i = 0; i < 10; i++) {
  net.train();
  // }

  console.log("");
}

go();
