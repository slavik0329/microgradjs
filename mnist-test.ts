import { Classifier, v } from "./MicroGrad";
import { getTrainingSet } from "./utils";

async function go() {
  console.log("Parsing CSV");

  const trainingSet = await getTrainingSet("./trainData/train.csv");

  const net = new Classifier({
    trainingSet: trainingSet,
    nin: 784,
    nouts: [
      { numLayer: 100, activationFunction: "sigmoid" },
      { numLayer: 10, activationFunction: "sigmoid" },
    ],
    bs: 20,
    learningRate: 0.05,
  });
  console.log("Training");

  // for (let i = 0; i < 10; i++) {
  net.train();
  // }

  console.log("");
}

go();
