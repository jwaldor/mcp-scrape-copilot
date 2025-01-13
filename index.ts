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
  getEmbeddingSentTransformer,
  initializeModelSentTransformer,
  makeRequest,
  semanticSearchRequestsSentTransformer,
} from "./utilities.js";

import { RequestRecord } from "./types.js";
import { FeatureExtractionPipeline } from "@xenova/transformers";
import { Command } from "commander";
import { getConfigCommand } from "./config.js";
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
            "Your search request, which should consist of a bunch of specific keywords for best results.",
        },
        page_url: {
          type: "string",
          description: "The page within which to search for requests",
        },
        exact_substrings: {
          type: "array",
          items: {
            type: "string",
          },
          description:
            "An array of strings that must be exactly present in the request URL, body, or headers to be included in the results (not case sensitive). The array can be empty, in which case no requests are filtered out in this way.",
        },
      },
      required: ["query", "page_url", "exact_substrings"],
    },
  },
];

// Global state
let browser: Browser | undefined;
let page: Page | undefined;
const consoleLogs: string[] = [];
const requests: Map<string, RequestRecord[]> = new Map(); // collects all results
const urlHistory: Array<string> = [];

let pipeline: FeatureExtractionPipeline | undefined;

initializeModelSentTransformer().then((sent_pipeline) => {
  console.error("model loaded");
  console.error("model", sent_pipeline);
  pipeline = sent_pipeline;
});

async function ensureBrowser() {
  if (!browser) {
    const npx_args = { headless: false };
    browser = await puppeteer.launch(npx_args);
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
      const add_request = async () => {
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
      };
      add_request();
      request.continue();
    });
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
      const filtered_requests: Array<RequestRecord> = requests
        .get(args.page_url)
        .filter((request) => {
          return args.exact_substrings.every(
            (substring) =>
              request.url.toLowerCase().includes(substring.toLowerCase()) ||
              (request.postData &&
                JSON.stringify(request.postData)
                  .toLowerCase()
                  .includes(substring.toLowerCase())) ||
              JSON.stringify(request.headers)
                .toLowerCase()
                .includes(substring.toLowerCase())
          );
        });
      const searchResults = await semanticSearchRequestsSentTransformer(
        args.query,
        filtered_requests,
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

process.stdin.on("close", () => {
  console.error("Puppeteer MCP Server closed");
  server.close();
});

async function main() {
  runServer().catch((error) => {
    console.error("Server error:", error);
    process.exit(1);
  });
}

export function createServer() {
  return new Command("serve").action(() => {
    main().catch((error) => {
      console.error("Server error:", error);
      process.exit(1);
    });
  });
}

const program = new Command();
program
  .name("mcp-scrape-copilot")
  .version("0.0.1")
  .description("MCP server for supercharging API reverse engineering");
program.addCommand(createServer());
program.addCommand(getConfigCommand());
program.parse(process.argv);
