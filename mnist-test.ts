import { parse } from "csv-parse";
import { readFile } from "node:fs/promises";
import { Classifier, TrainingItemUnnormalized } from "./MicroGrad";

const normalize = (n) => n / (254 / 2) - 1;

function makeOutputNumber(num: number): number[] {
  let arr: number[] = [];

  for (let i = 0; i < 10; i++) {
    arr.push(i === num ? 1 : 0);
  }

  return arr;
}

async function go() {
  console.log("Parsing CSV");

  const trainingSet = await getTrainingSet();

  const net = new Classifier({ trainingSet, nin: 784, nouts: [20, 10], bs: 2 });

  net.train();
}

function getTrainingSet(): Promise<TrainingItemUnnormalized[]> {
  return new Promise(async (resolve) => {
    const trainingSet: TrainingItemUnnormalized[] = [];

    let row = 0;
    const rawFile = await readFile("./trainData/train.csv", {
      encoding: "ascii",
    });

    const parser = parse(rawFile);

    parser.on("readable", () => {
      let item;
      while ((item = parser.read()) !== null) {
        if (row !== 0) {
          const output = makeOutputNumber(Number(item.shift()));

          trainingSet.push({
            input: item.map((item) => normalize(Number(item))),
            output,
          });
        }
        console.log("Row: " + row);
        row++;
      }
    });

    parser.on("end", () => {
      resolve(trainingSet);
    });
  });
}

go();
