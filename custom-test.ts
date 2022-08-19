import { Classifier, TrainingItemUnnormalized, v } from "./MicroGrad";
import { getTrainingSet } from "./utils";

async function go() {
  const trainingSet: TrainingItemUnnormalized[] = [
    {
      input: [0, 0],
      output: [0],
    },
    {
      input: [1, 0],
      output: [0],
    },
    {
      input: [1, 1],
      output: [1],
    },
    {
      input: [0, 1],
      output: [0],
    },
  ];

  const net = new Classifier({
    trainingSet: trainingSet,
    neuronRandomizer: "NEG_ZERO_TO_ONE",
    iterations: 1000,
    lossType: "MSE",
    nin: 2,
    nouts: [
      { numLayer: 4, activationFunction: "tanh" },
      { numLayer: 4, activationFunction: "tanh" },
      { numLayer: 1, activationFunction: "tanh" },
    ],
    bs: 100,
    learningRate: 0.05,
  });
  console.log("Training");

  net.train();

  const res = net.call(trainingSet[2].input.map((it) => v(it)));

  console.log(res);
}

go();
