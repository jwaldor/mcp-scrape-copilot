import * as use from "@tensorflow-models/universal-sentence-encoder";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs";

export async function getEmbedding(
  text: string,
  model: use.UniversalSentenceEncoder
): Promise<number[]> {
  // Generate embedding
  // Generate embedding
  const embeddings = await model.embed([text]);

  // Convert the tensor to array and return the first (and only) embedding
  const embeddingArray = await embeddings.array();
  return embeddingArray[0];
}

async function testGetEmbedding() {
  console.log("Loading model...");
  const startTime = performance.now();

  await tf.setBackend("cpu");
  await tf.ready();
  console.log("tf done");
  const model = await use.load();

  const endTime = performance.now();
  console.log(`Model loaded in ${(endTime - startTime) / 1000} seconds`);

  const embedding = await getEmbedding("Hello, world!", model);
  console.log(embedding);
}

testGetEmbedding();
