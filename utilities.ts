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

export async function semanticSearchRequests(
  query: string,
  requests: Array<{
    url: string;
    method: string;
    headers: Record<string, string>;
    resourceType: string;
    postData: any;
    embedding: number[];
  }>,
  model: use.UniversalSentenceEncoder
): Promise<Array<any>> {
  // Get embedding for the query
  const queryEmbedding = await getEmbedding(query, model);

  // Calculate cosine similarity scores for all requests
  const scoredRequests = requests.map((request) => {
    // Compute cosine similarity between query and request embeddings
    const similarity = cosineSimilarity(queryEmbedding, request.embedding);
    return { ...request, similarity };
  });

  // Sort by similarity score (highest first) and take top 10
  return scoredRequests
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 10);
}

// Helper function to compute cosine similarity between two vectors
function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}
