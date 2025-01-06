import * as use from "@tensorflow-models/universal-sentence-encoder";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs";

export async function makeRequest(
  url: string,
  type: string,
  headers: Record<string, string>,
  body: any
) {
  try {
    const response = await fetch(url, {
      method: type,
      headers,
      body:
        body && (type === "POST" || type === "PUT")
          ? JSON.stringify(body)
          : undefined,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return {
      status: response.status,
      data: await response.text(),
      headers: Object.fromEntries(response.headers),
    };
  } catch (error) {
    console.error("Error making request:", error);
    throw error;
  }
}

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
  await tf.setBackend("cpu");
  await tf.ready();
  const model = await use.load();
  console.log("Model loaded");
  const embedding = await getEmbedding("Hello, world!", model);
  console.log(embedding);
}

testGetEmbedding();
