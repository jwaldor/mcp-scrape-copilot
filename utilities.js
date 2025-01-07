"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
exports.__esModule = true;
exports.semanticSearchRequests = exports.getEmbedding = exports.makeRequest = void 0;
require("@tensorflow/tfjs-backend-webgl");
function makeRequest(url, type, headers, body) {
    return __awaiter(this, void 0, void 0, function () {
        var response, error_1;
        var _a;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    _b.trys.push([0, 3, , 4]);
                    return [4 /*yield*/, fetch(url, {
                            method: type,
                            headers: headers,
                            body: body && (type === "POST" || type === "PUT")
                                ? JSON.stringify(body)
                                : undefined
                        })];
                case 1:
                    response = _b.sent();
                    if (!response.ok) {
                        throw new Error("HTTP error! status: ".concat(response.status));
                    }
                    _a = {
                        status: response.status
                    };
                    return [4 /*yield*/, response.text()];
                case 2: return [2 /*return*/, (_a.data = _b.sent(),
                        _a.headers = Object.fromEntries(response.headers),
                        _a)];
                case 3:
                    error_1 = _b.sent();
                    console.error("Error making request:", error_1);
                    throw error_1;
                case 4: return [2 /*return*/];
            }
        });
    });
}
exports.makeRequest = makeRequest;
function getEmbedding(text, model) {
    return __awaiter(this, void 0, void 0, function () {
        var embeddings, embeddingArray;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, model.embed([text])];
                case 1:
                    embeddings = _a.sent();
                    return [4 /*yield*/, embeddings.array()];
                case 2:
                    embeddingArray = _a.sent();
                    return [2 /*return*/, embeddingArray[0]];
            }
        });
    });
}
exports.getEmbedding = getEmbedding;
// async function testGetEmbedding() {
//   console.log("Loading model...");
//   await tf.setBackend("cpu");
//   await tf.ready();
//   const model = await use.load();
//   console.log("Model loaded");
//   const embedding = await getEmbedding("Hello, world!", model);
//   console.log(embedding);
// }
// testGetEmbedding();
function semanticSearchRequests(query, requests, model) {
    return __awaiter(this, void 0, void 0, function () {
        var queryEmbedding, scoredRequests;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, getEmbedding(query, model)];
                case 1:
                    queryEmbedding = _a.sent();
                    scoredRequests = requests.map(function (request) {
                        // Compute cosine similarity between query and request embeddings
                        var similarity = cosineSimilarity(queryEmbedding, request.embedding);
                        return __assign(__assign({}, request), { similarity: similarity });
                    });
                    // Sort by similarity score (highest first) and take top 10
                    return [2 /*return*/, scoredRequests
                            .sort(function (a, b) { return b.similarity - a.similarity; })
                            .slice(0, 10)];
            }
        });
    });
}
exports.semanticSearchRequests = semanticSearchRequests;
// Helper function to compute cosine similarity between two vectors
function cosineSimilarity(a, b) {
    var dotProduct = a.reduce(function (sum, val, i) { return sum + val * b[i]; }, 0);
    var magnitudeA = Math.sqrt(a.reduce(function (sum, val) { return sum + val * val; }, 0));
    var magnitudeB = Math.sqrt(b.reduce(function (sum, val) { return sum + val * val; }, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}
