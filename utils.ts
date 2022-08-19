import { TrainingItemUnnormalized } from "./MicroGrad";
import { readFile } from "node:fs/promises";
import { parse } from "csv-parse";

const normalize: (n) => number = (n) => n / (254 / 2) - 1;

function makeOutputNumber(num: number): number[] {
  let arr: number[] = [];

  for (let i = 0; i < 10; i++) {
    arr.push(i === num ? 1 : 0);
  }

  return arr;
}

export function getTrainingSet(
  fileName: string
): Promise<TrainingItemUnnormalized[]> {
  return new Promise(async (resolve) => {
    const trainingSet: TrainingItemUnnormalized[] = [];

    let row = 0;
    const rawFile = await readFile(fileName, {
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
