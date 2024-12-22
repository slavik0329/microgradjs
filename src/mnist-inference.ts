import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-node";
import { readFile } from "node:fs/promises";
import { parse } from "csv-parse";
import { join } from "path";
import { createInterface } from "readline";

const PIXEL_COUNT = 784; // 28x28 image

async function loadModel(path: string) {
  return await tf.loadLayersModel(`file://${path}/model.json`);
}

function displayDigit(pixels: number[]) {
  // Convert normalized values back to 0-255 range and create ASCII art
  const art = [];
  for (let i = 0; i < 28; i++) {
    let row = "";
    for (let j = 0; j < 28; j++) {
      const pixel = Math.round((pixels[i * 28 + j] + 1) * 127.5);
      // Use different characters for different intensity levels
      if (pixel < 50) row += " ";
      else if (pixel < 100) row += "·";
      else if (pixel < 150) row += "▪";
      else if (pixel < 200) row += "▮";
      else row += "█";
    }
    art.push(row);
  }
  return art.join("\n");
}

interface DigitData {
  label?: number;
  pixels: number[];
}

async function getRowFromCSV(
  fileName: string,
  rowIndex: number
): Promise<DigitData> {
  const rawFile = await readFile(fileName, { encoding: "utf-8" });
  const parser = parse(rawFile, {
    skip_empty_lines: true,
    from_line: 2, // Skip header
  });

  let currentRow = 0;
  for await (const item of parser) {
    if (currentRow === rowIndex) {
      const pixels = item.map((p: string) => parseInt(p, 10) / 127.5 - 1);
      return { pixels };
    }
    currentRow++;
  }
  throw new Error(`Row ${rowIndex} not found in CSV`);
}

async function main() {
  try {
    console.log("Loading model...");
    const model = await loadModel(join(__dirname, "../model"));

    const rl = createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    while (true) {
      const answer = await new Promise<string>((resolve) => {
        rl.question("Enter row number to test (or 'q' to quit): ", resolve);
      });

      if (answer.toLowerCase() === "q") {
        break;
      }

      const rowIndex = parseInt(answer, 10);
      if (isNaN(rowIndex)) {
        console.log("Please enter a valid number");
        continue;
      }

      try {
        const { label, pixels } = await getRowFromCSV(
          join(__dirname, "../trainData/test.csv"),
          rowIndex
        );

        // Display the digit
        console.log("\nDigit visualization:");
        console.log(displayDigit(pixels));

        if (label !== undefined) {
          console.log(`\nActual digit: ${label}`);
        }

        // Make prediction
        const inputTensor = tf.tensor2d([pixels], [1, PIXEL_COUNT]);
        const prediction = model.predict(inputTensor) as tf.Tensor;
        const probabilities = (await prediction.array()) as number[][];

        // Get the predicted digit (index of max probability)
        const predictedDigit = probabilities[0].indexOf(
          Math.max(...probabilities[0])
        );

        // Show prediction results
        console.log(`Predicted digit: ${predictedDigit}`);

        if (label !== undefined) {
          console.log(
            "Prediction correct:",
            label === predictedDigit ? "✅" : "❌"
          );
        }

        console.log("\nProbabilities:");
        probabilities[0].forEach((prob, digit) => {
          const percentage = (prob * 100).toFixed(2);
          const isCorrect = label !== undefined && digit === label;
          const isPredicted = digit === predictedDigit;
          const marker = isCorrect ? "✓" : isPredicted ? "!" : " ";
          console.log(`${digit}: ${percentage}% ${marker}`);
        });

        // Cleanup
        inputTensor.dispose();
        prediction.dispose();
      } catch (error) {
        console.error("Error processing row:", error);
      }
    }

    rl.close();
    model.dispose();
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
