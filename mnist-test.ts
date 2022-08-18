import { parse } from "csv-parse";
import { readFile } from "node:fs/promises";
import {
  TrainingItemNormalized,
  TrainingItemUnnormalized,
  v,
  Value,
} from "./MicroGrad";

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

  const rawTrainingSet: { [key: string]: string }[] = [];

  // .on("data", (data) => {
  //   // let input: number[] = [];
  //   //
  //   // for (let i = 0; i < 784; i++) {
  //   //   input.push(Number(data[`pixel${i}`]));
  //   // }
  //   //
  //   // trainingSet.push({
  //   //   input,
  //   //   output: makeOutputNumber(Number(data.label)),
  //   // });
  //   rawTrainingSet.push(data);
  //
  //   console.log("Row: " + row);
  //   row++;
  // })
  // .on("end", () => {
  //   // [
  //   //   { NAME: 'Daffy Duck', AGE: '24' },
  //   //   { NAME: 'Bugs Bunny', AGE: '22' }
  //   // ]
  // });
}

function getTrainingSet() {
  return new Promise(async (resolve) => {
    const trainingSet: TrainingItemUnnormalized[] = [];

    let row = 0;
    const rawFile = await readFile("./trainData/train.csv", {
      encoding: "ascii",
    });

    console.log("parsing");
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
