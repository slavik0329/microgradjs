import { Classifier, TrainingItemUnnormalized, v } from "./MicroGrad";

async function go() {
  const trainingSet: TrainingItemUnnormalized[] = [{
    input: 
      [ 0.03,
       0.7,
       0.5]
    ,
    output: 
      [1,0]
    
  },
  {
    input: [
      0.16,
      0.09,
       0.2
    ],
    output: [0,1]
     
  },
  {
    input: [
      0.5,
      0.5,
      1.0
    ],
    output: [0,1]
  }
];

  const net = new Classifier({
    trainingSet: trainingSet,
    neuronRandomizer: "ZERO_TO_ONE",
    iterations: 1000,
    lossType: "MSE",
    nin: 3,
    nouts: [
      { numLayer: 4, activationFunction: "tanh" },
      { numLayer: 4, activationFunction: "tanh" },
      { numLayer: 2, activationFunction: "tanh" },
    ],
    bs: 100,
    learningRate: 0.05,
  });
  console.log("Training");

  net.train();

  const res = net.call([1,.4,0].map((it) => v(it)));

  console.log(res);
}

go();
