import * as use from "@tensorflow-models/universal-sentence-encoder";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs";
import { getEmbedding } from "./utilities";

async function testGetEmbedding() {
  console.log("Loading model...");
  const startTime = performance.now();
  await tf.ready();
  tf.engine().startScope(); // Create a scope to better manage memory

  await tf.setBackend("cpu");
  console.log("tf done");
  const model = await use.load();
  const model2 = await model.loadModel();
  model2.save("model.json");

  const endTime = performance.now();
  console.log(`Model loaded in ${(endTime - startTime) / 1000} seconds`);

  const embedding = await getEmbedding("Hello, world!", model);
  console.log(embedding);
}

testGetEmbedding();
