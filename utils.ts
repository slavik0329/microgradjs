import { TrainingItemUnnormalized } from "./MicroGrad";
import { readFile } from "node:fs/promises";
import { parse } from "csv-parse";

export function getTrainingSet(): Promise<TrainingItemUnnormalized[]> {
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
