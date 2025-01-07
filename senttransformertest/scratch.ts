// interface Request {
//   url: string;
//   headers: Record<string>;
//   body?: any;
// }

// Combine request parts into meaningful text
function requestToText(req: Request): string {
  return `${req.url} ${JSON.stringify(req.headers)} ${JSON.stringify(
    req.body
  )}`;
}

// Get embeddings using sentence-transformers
async function getEmbedding(text: string) {
  const model = await import("@xenova/transformers");
  const pipeline = await model.pipeline("feature-extraction");
  return pipeline(text);
}

getEmbedding("Hello, world!").then((embedding) => {
  console.log(embedding);
});
