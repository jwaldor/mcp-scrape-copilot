#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  CallToolResult,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import puppeteer, { Browser, Page } from "puppeteer";
import {
  getEmbedding,
  getEmbeddingSentTransformer,
  initializeModelSentTransformer,
  makeRequest,
  semanticSearchRequests,
  semanticSearchRequestsSentTransformer,
} from "./utilities.js";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs";
import { RequestRecord } from "./types.js";
import { FeatureExtractionPipeline } from "@xenova/transformers";
// Define the tools once to avoid repetition
const TOOLS: Tool[] = [
  {
    name: "puppeteer_navigate",
    description: "Navigate to a URL",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string" },
      },
      required: ["url"],
    },
  },
  {
    name: "puppeteer_page_history",
    description: "Get the history of visited URLs, most recent urls first",
    inputSchema: {
      type: "object",
      properties: {},
      required: [],
    },
  },
  // {
  //   name: "get_page_requests",
  //   description:
  //     "Get all of the HTTP requests made for a specific URL, the most recent requests first",
  //   inputSchema: {
  //     type: "object",
  //     properties: {
  //       url: {
  //         type: "string",
  //         description: "The URL to get requests for",
  //       },
  //     },
  //     required: ["url"],
  //   },
  // },
  {
    name: "make_http_request",
    description: "Make an HTTP request with curl",
    inputSchema: {
      type: "object",
      properties: {
        type: {
          type: "string",
          description: "Type of the request. GET, POST, PUT, DELETE",
        },
        url: {
          type: "string",
          description: "Url to make the request to",
        },
        headers: {
          type: "object",
          description: "Headers to include in the request",
        },
        body: {
          type: "object",
          description: "Body to include in the request",
        },
      },
      required: ["type", "url", "headers", "body"],
    },
  },
  {
    name: "semantic_search_requests",
    description:
      "Semantically search for requests that occurred within a page URL. Returns the top 10 results.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description:
            "Your search request. Make this specific and detailed to get the best results",
        },
        page_url: {
          type: "string",
          description: "The page within which to search for requests",
        },
      },
      required: ["query", "page_url"],
    },
  },
];

// Global state
let browser: Browser | undefined;
let page: Page | undefined;
const consoleLogs: string[] = [];
const requests: Map<string, RequestRecord[]> = new Map(); // collects all results
const urlHistory: Array<string> = [];
let model: use.UniversalSentenceEncoder | undefined;
let pipeline: FeatureExtractionPipeline | undefined;
console.error("log model", model);
initializeModelSentTransformer().then((sent_pipeline) => {
  console.error("model loaded");
  console.error("model", sent_pipeline);
  pipeline = sent_pipeline;
});
// ensureModel();
async function doStuff() {
  console.error("doStuff haha");
  await use.load();
  return;
}
// doStuff();
async function ensureModel() {
  console.error("ensureModel", model);
  if (!model) {
    await tf.setBackend("cpu");
    await tf.ready();
    console.error("loading model now");
    model = await use.load();
    console.error("model loaded");
  }
  return;
}

async function ensureBrowser() {
  if (!browser) {
    const npx_args = { headless: false };
    const docker_args = {
      headless: true,
      args: ["--no-sandbox", "--single-process", "--no-zygote"],
    };
    browser = await puppeteer.launch(
      process.env.DOCKER_CONTAINER ? docker_args : npx_args
    );
    const pages = await browser.pages();
    page = pages[0];
    page.setRequestInterception(true);

    page.on("console", (msg) => {
      const logEntry = `[${msg.type()}] ${msg.text()}`;
      consoleLogs.push(logEntry);
      server.notification({
        method: "notifications/resources/updated",
        params: { uri: "console://logs" },
      });
    });

    page.on("request", async (request) => {
      if (!pipeline) {
        console.error(
          "Request made before model was loaded.",
          request.url(),
          page.url()
        );
        request.continue();
        return;
      }
      if (requests.has(page.url())) {
        requests.get(page.url()).unshift({
          url: request.url(),
          resourceType: request.resourceType(),
          method: request.method(),
          headers: request.headers(),
          postData: request.postData(),
          embedding: await getEmbeddingSentTransformer(
            request.url() +
              request.method() +
              JSON.stringify(request.headers()) +
              JSON.stringify(request.postData()),
            pipeline
          ),
        });
        // server.sendLoggingMessage({
        //   level: "info",
        //   data: JSON.stringify({
        //     resourceType: request.resourceType(),
        //     method: request.method(),
        //     headers: request.headers(),
        //     postData: request.postData(),
        //   }),
        // });
        console.error("request", {
          url: request.url(),
          resourceType: request.resourceType(),
          method: request.method(),
          headers: request.headers(),
          postData: request.postData(),
        });
      } else {
        requests.set(page.url(), [
          {
            url: request.url(),
            resourceType: request.resourceType(),
            method: request.method(),
            headers: request.headers(),
            postData: request.postData(),
            embedding: await getEmbeddingSentTransformer(
              request.url() +
                request.method() +
                JSON.stringify(request.headers()) +
                JSON.stringify(request.postData()),
              pipeline
            ),
          },
        ]);
      }
      request.continue();
    });
    // page.on("load", () => {
    //   urlHistory.push(page.url());
    // });
  }
  return page!;
}

declare global {
  interface Window {
    mcpHelper: {
      logs: string[];
      originalConsole: Partial<typeof console>;
    };
  }
}

async function handleToolCall(
  name: string,
  args: any
): Promise<CallToolResult> {
  const page = await ensureBrowser();
  // const model = await ensureModel();
  switch (name) {
    case "puppeteer_navigate":
      await page.goto(args.url);
      return {
        content: [
          {
            type: "text",
            text: `Navigated to ${args.url}`,
          },
        ],
        isError: false,
      };

    case "page_history":
      return {
        content: [
          {
            type: "text",
            text: urlHistory.reverse().join("\n"),
          },
        ],
        isError: false,
      };

    // case "get_page_requests": {
    //   console.error(
    //     "get_page_requests",
    //     "arg url",
    //     args.url,
    //     "all_requests",
    //     requests
    //   );
    //   const requestData = requests.get(args.url);
    //   if (!requestData) {
    //     return {
    //       content: [
    //         {
    //           type: "text",
    //           text: `No requests found for URL: ${args.url}`,
    //         },
    //       ],
    //       isError: false,
    //     };
    //   }
    //   return {
    //     content: [
    //       {
    //         type: "text",
    //         text: JSON.stringify(requestData, null, 2),
    //       },
    //     ],
    //     isError: false,
    //   };
    // }

    case "make_http_request": {
      const response = await makeRequest(
        args.url,
        args.type,
        args.headers,
        args.body
      );
      return {
        content: [{ type: "text", text: JSON.stringify(response, null, 2) }],
        isError: false,
      };
    }

    case "semantic_search_requests": {
      if (!pipeline) {
        return {
          content: [{ type: "text", text: "Model not defined" }],
          isError: true,
        };
      }
      const searchResults = await semanticSearchRequestsSentTransformer(
        args.query,
        requests.get(args.page_url),
        pipeline
      );
      const withoutEmbedding = searchResults.map(
        ({ embedding, similarity, ...rest }) => rest
      );
      return {
        content: [
          { type: "text", text: JSON.stringify(withoutEmbedding, null, 2) },
        ],
        isError: false,
      };
    }

    default:
      return {
        content: [
          {
            type: "text",
            text: `Unknown tool: ${name}`,
          },
        ],
        isError: true,
      };
  }
}

const server = new Server(
  {
    name: "mcp-scrape-copilot",
    version: "0.1.0",
  },
  {
    capabilities: {
      resources: {},
      tools: {},
    },
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS,
}));

server.setRequestHandler(CallToolRequestSchema, async (request) =>
  handleToolCall(request.params.name, request.params.arguments ?? {})
);

async function runServer() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

runServer().catch(console.error);

process.stdin.on("close", () => {
  console.error("Puppeteer MCP Server closed");
  server.close();
});
