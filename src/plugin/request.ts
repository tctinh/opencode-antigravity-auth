import crypto from "node:crypto";
import {
  ANTIGRAVITY_HEADERS,
  ANTIGRAVITY_ENDPOINT,
} from "../constants";
import { logAntigravityDebugResponse, type AntigravityDebugContext } from "./debug";
import {
  // Types
  type AntigravityApiBody,
  type ToolDebugInfo,
  // API response helpers
  extractUsageFromSsePayload,
  extractUsageMetadata,
  parseAntigravityApiBody,
  rewriteAntigravityPreviewAccessError,
  // Model detection
  parseModelFromUrl,
  hasAssistantHistory,
  // Thinking config
  extractThinkingSource,
  resolveThinkingConfig,
  applyThinkingConfig,
  // Cached content
  extractCachedContent,
  cleanupCachedContentFields,
  // Tool normalization
  normalizeToolsForClaude,
  normalizeToolsForGemini,
  // Claude transformations
  filterUnsignedThinkingBlocks,
  assignFunctionCallIds,
} from "./request-helpers";

// ============================================================================
// Constants
// ============================================================================

const SYNTHETIC_PROJECT_ADJECTIVES = ["useful", "bright", "swift", "calm", "bold"];
const SYNTHETIC_PROJECT_NOUNS = ["fuze", "wave", "spark", "flow", "core"];

// ============================================================================
// Utility Functions
// ============================================================================

function generateSyntheticProjectId(): string {
  const adj = SYNTHETIC_PROJECT_ADJECTIVES[Math.floor(Math.random() * SYNTHETIC_PROJECT_ADJECTIVES.length)];
  const noun = SYNTHETIC_PROJECT_NOUNS[Math.floor(Math.random() * SYNTHETIC_PROJECT_NOUNS.length)];
  const randomPart = crypto.randomUUID().slice(0, 5).toLowerCase();
  return `${adj}-${noun}-${randomPart}`;
}

function generateRequestId(): string {
  return `agent-${crypto.randomUUID()}`;
}

function generateSessionId(): string {
  return `-${Math.floor(Math.random() * 1e15)}`;
}

/**
 * Detects requests headed to the Google Generative Language API so we can intercept them.
 */
export function isGenerativeLanguageRequest(input: RequestInfo): input is string {
  return typeof input === "string" && input.includes("generativelanguage.googleapis.com");
}

// ============================================================================
// Response Transformation
// ============================================================================

/**
 * Transforms thinking/reasoning content in response parts to OpenCode's expected format.
 * Handles both Gemini-style (thought: true) and Anthropic-style (type: "thinking") formats.
 */
function transformThinkingParts(response: unknown): unknown {
  if (!response || typeof response !== "object") {
    return response;
  }

  const resp = response as Record<string, unknown>;

  // Handle Anthropic-style content array with type: "thinking"
  if (Array.isArray(resp.content)) {
    resp.content = resp.content.map((block: any) => {
      if (block && typeof block === "object" && block.type === "thinking") {
        return {
          type: "reasoning",
          text: block.thinking || "",
          ...(block.signature ? { signature: block.signature } : {}),
        };
      }
      return block;
    });
  }

  // Handle Gemini-style candidates with parts containing thought: true
  if (Array.isArray(resp.candidates)) {
    resp.candidates = resp.candidates.map((candidate: any) => {
      if (!candidate || typeof candidate !== "object") {
        return candidate;
      }
      const content = candidate.content;
      if (!content || typeof content !== "object" || !Array.isArray(content.parts)) {
        return candidate;
      }

      const thinkingTexts: string[] = [];
      content.parts = content.parts.map((part: any) => {
        if (part && typeof part === "object" && part.thought === true) {
          thinkingTexts.push(part.text || "");
          return { ...part, type: "reasoning" };
        }
        return part;
      });

      if (thinkingTexts.length > 0) {
        candidate.reasoning_content = thinkingTexts.join("\n\n");
      }

      return candidate;
    });
  }

  return resp;
}

/**
 * Rewrites SSE payloads so downstream consumers see only the inner `response` objects,
 * with thinking/reasoning blocks transformed to OpenCode's expected format.
 */
function transformStreamingPayload(payload: string): string {
  return payload
    .split("\n")
    .map((line) => {
      if (!line.startsWith("data:")) {
        return line;
      }
      const json = line.slice(5).trim();
      if (!json) {
        return line;
      }
      try {
        const parsed = JSON.parse(json) as { response?: unknown };
        if (parsed.response !== undefined) {
          const transformed = transformThinkingParts(parsed.response);
          return `data: ${JSON.stringify(transformed)}`;
        }
      } catch (_) { }
      return line;
    })
    .join("\n");
}

// ============================================================================
// Request Preparation Types
// ============================================================================

export interface PrepareRequestResult {
  request: RequestInfo;
  init: RequestInit;
  streaming: boolean;
  requestedModel?: string;
  effectiveModel?: string;
  projectId?: string;
  endpoint?: string;
  toolDebugMissing?: number;
  toolDebugSummary?: string;
  toolDebugPayload?: string;
}

// ============================================================================
// Request Body Transformation
// ============================================================================

/**
 * Transforms the request payload for Antigravity API.
 */
function transformRequestPayload(
  parsedBody: Record<string, unknown>,
  modelInfo: { rawModel: string; isClaude: boolean; isThinkingModel: boolean },
  toolDebugInfo: ToolDebugInfo
): Record<string, unknown> {
  const payload: Record<string, unknown> = { ...parsedBody };

  // 1. Handle thinking config
  const sourceThinking = extractThinkingSource(payload);
  const hasHistory = hasAssistantHistory(payload.contents);
  const thinkingConfig = resolveThinkingConfig(
    sourceThinking,
    modelInfo.isThinkingModel,
    modelInfo.isClaude,
    hasHistory
  );
  applyThinkingConfig(payload, thinkingConfig);

  // 2. Normalize system instruction field name
  if ("system_instruction" in payload) {
    payload.systemInstruction = payload.system_instruction;
    delete payload.system_instruction;
  }

  // 3. Handle cached content
  const cachedContent = extractCachedContent(payload);
  if (cachedContent) {
    payload.cachedContent = cachedContent;
  }
  cleanupCachedContentFields(payload);

  // 4. Normalize tools
  if (Array.isArray(payload.tools)) {
    payload.tools = modelInfo.isClaude
      ? normalizeToolsForClaude(payload.tools, toolDebugInfo)
      : normalizeToolsForGemini(payload.tools, toolDebugInfo);

    try {
      toolDebugInfo.payload = JSON.stringify(payload.tools);
    } catch {
      // Ignore serialization errors
    }
  }

  // 5. Claude-specific content transformations
  if (modelInfo.isClaude && Array.isArray(payload.contents)) {
    payload.contents = filterUnsignedThinkingBlocks(payload.contents as any[]);
    payload.contents = assignFunctionCallIds(payload.contents as any[]);
  }

  // 6. Remove model field (will be set at wrapper level)
  delete payload.model;

  return payload;
}

/**
 * Wraps the request payload in Antigravity envelope.
 */
function wrapForAntigravity(
  requestPayload: Record<string, unknown>,
  model: string,
  projectId: string
): Record<string, unknown> {
  const effectiveProjectId = projectId.trim() || generateSyntheticProjectId();

  return {
    project: effectiveProjectId,
    model,
    request: {
      ...requestPayload,
      sessionId: generateSessionId(),
    },
    userAgent: "antigravity",
    requestId: generateRequestId(),
  };
}

/**
 * Sets up headers for Antigravity requests.
 */
function setupHeaders(
  baseHeaders: Headers,
  accessToken: string,
  streaming: boolean,
  toolDebugMissing: number
): void {
  baseHeaders.set("Authorization", `Bearer ${accessToken}`);
  baseHeaders.delete("x-api-key");

  if (streaming) {
    baseHeaders.set("Accept", "text/event-stream");
  }

  baseHeaders.set("User-Agent", ANTIGRAVITY_HEADERS["User-Agent"]);
  baseHeaders.set("X-Goog-Api-Client", ANTIGRAVITY_HEADERS["X-Goog-Api-Client"]);
  baseHeaders.set("Client-Metadata", ANTIGRAVITY_HEADERS["Client-Metadata"]);

  if (toolDebugMissing > 0) {
    baseHeaders.set("X-Opencode-Tools-Debug", String(toolDebugMissing));
  }
}

// ============================================================================
// Main Request Preparation Function
// ============================================================================

/**
 * Rewrites OpenAI-style requests into Antigravity shape, normalizing model, headers,
 * optional cached_content, and thinking config. Also toggles streaming mode for SSE actions.
 */
export function prepareAntigravityRequest(
  input: RequestInfo,
  init: RequestInit | undefined,
  accessToken: string,
  projectId: string,
  endpointOverride?: string,
): PrepareRequestResult {
  const baseInit: RequestInit = { ...init };
  const headers = new Headers(init?.headers ?? {});

  // Early return for non-generative language requests
  if (!isGenerativeLanguageRequest(input)) {
    return {
      request: input,
      init: { ...baseInit, headers },
      streaming: false,
    };
  }

  // Parse model info from URL
  const modelInfo = parseModelFromUrl(input);
  if (!modelInfo) {
    headers.set("Authorization", `Bearer ${accessToken}`);
    headers.delete("x-api-key");
    return {
      request: input,
      init: { ...baseInit, headers },
      streaming: false,
    };
  }

  const { rawModel, action, isStreaming, isClaude, isThinkingModel } = modelInfo;
  const baseEndpoint = endpointOverride ?? ANTIGRAVITY_ENDPOINT;
  const transformedUrl = `${baseEndpoint}/v1internal:${action}${isStreaming ? "?alt=sse" : ""}`;

  // Initialize debug info
  const toolDebugInfo: ToolDebugInfo = { missing: 0, summaries: [] };
  let resolvedProjectId = projectId?.trim() || "";
  let body = baseInit.body;

  // Transform request body if present
  if (typeof baseInit.body === "string" && baseInit.body) {
    const parsedBody = JSON.parse(baseInit.body) as Record<string, unknown>;
    const isAlreadyWrapped = typeof parsedBody.project === "string" && "request" in parsedBody;

    if (isAlreadyWrapped) {
      // Already wrapped - just update model
      body = JSON.stringify({ ...parsedBody, model: rawModel });
    } else {
      // Transform and wrap the payload
      const transformedPayload = transformRequestPayload(
        parsedBody,
        { rawModel, isClaude, isThinkingModel },
        toolDebugInfo
      );

      const wrappedBody = wrapForAntigravity(transformedPayload, rawModel, projectId);
      resolvedProjectId = wrappedBody.project as string;
      body = JSON.stringify(wrappedBody);
    }
  }

  // Setup headers
  setupHeaders(headers, accessToken, isStreaming, toolDebugInfo.missing);

  return {
    request: transformedUrl,
    init: { ...baseInit, headers, body },
    streaming: isStreaming,
    requestedModel: rawModel,
    effectiveModel: rawModel,
    projectId: resolvedProjectId,
    endpoint: transformedUrl,
    toolDebugMissing: toolDebugInfo.missing,
    toolDebugSummary: toolDebugInfo.summaries.slice(0, 20).join(" | "),
    toolDebugPayload: toolDebugInfo.payload,
  };
}

// ============================================================================
// Response Transformation
// ============================================================================

/**
 * Normalizes Antigravity responses: applies retry headers, extracts cache usage into headers,
 * rewrites preview errors, flattens streaming payloads, and logs debug metadata.
 */
export async function transformAntigravityResponse(
  response: Response,
  streaming: boolean,
  debugContext?: AntigravityDebugContext | null,
  requestedModel?: string,
  projectId?: string,
  endpoint?: string,
  effectiveModel?: string,
  toolDebugMissing?: number,
  toolDebugSummary?: string,
  toolDebugPayload?: string,
): Promise<Response> {
  const contentType = response.headers.get("content-type") ?? "";
  const isJsonResponse = contentType.includes("application/json");
  const isEventStreamResponse = contentType.includes("text/event-stream");

  if (!isJsonResponse && !isEventStreamResponse) {
    logAntigravityDebugResponse(debugContext, response, {
      note: "Non-JSON response (body omitted)",
    });
    return response;
  }

  try {
    const text = await response.text();
    const headers = new Headers(response.headers);

    if (!response.ok) {
      let errorBody;
      try {
        errorBody = JSON.parse(text);
      } catch {
        errorBody = { error: { message: text } };
      }

      // Inject Debug Info
      if (errorBody?.error) {
        const debugInfo = `\n\n[Debug Info]\nRequested Model: ${requestedModel || "Unknown"}\nEffective Model: ${effectiveModel || "Unknown"}\nProject: ${projectId || "Unknown"}\nEndpoint: ${endpoint || "Unknown"}\nStatus: ${response.status}\nRequest ID: ${headers.get('x-request-id') || "N/A"}${toolDebugMissing !== undefined ? `\nTool Debug Missing: ${toolDebugMissing}` : ""}${toolDebugSummary ? `\nTool Debug Summary: ${toolDebugSummary}` : ""}${toolDebugPayload ? `\nTool Debug Payload: ${toolDebugPayload}` : ""}`;
        errorBody.error.message = (errorBody.error.message || "Unknown error") + debugInfo;

        return new Response(JSON.stringify(errorBody), {
          status: response.status,
          statusText: response.statusText,
          headers
        });
      }

      if (errorBody?.error?.details && Array.isArray(errorBody.error.details)) {
        const retryInfo = errorBody.error.details.find(
          (detail: any) => detail['@type'] === 'type.googleapis.com/google.rpc.RetryInfo'
        );

        if (retryInfo?.retryDelay) {
          const match = retryInfo.retryDelay.match(/^([\d.]+)s$/);
          if (match && match[1]) {
            const retrySeconds = parseFloat(match[1]);
            if (!isNaN(retrySeconds) && retrySeconds > 0) {
              const retryAfterSec = Math.ceil(retrySeconds).toString();
              const retryAfterMs = Math.ceil(retrySeconds * 1000).toString();
              headers.set('Retry-After', retryAfterSec);
              headers.set('retry-after-ms', retryAfterMs);
            }
          }
        }
      }
    }

    const init = {
      status: response.status,
      statusText: response.statusText,
      headers,
    };

    const usageFromSse = streaming && isEventStreamResponse ? extractUsageFromSsePayload(text) : null;
    const parsed: AntigravityApiBody | null = !streaming || !isEventStreamResponse ? parseAntigravityApiBody(text) : null;
    const patched = parsed ? rewriteAntigravityPreviewAccessError(parsed, response.status, requestedModel) : null;
    const effectiveBody = patched ?? parsed ?? undefined;

    const usage = usageFromSse ?? (effectiveBody ? extractUsageMetadata(effectiveBody) : null);
    if (usage?.cachedContentTokenCount !== undefined) {
      headers.set("x-antigravity-cached-content-token-count", String(usage.cachedContentTokenCount));
      if (usage.totalTokenCount !== undefined) {
        headers.set("x-antigravity-total-token-count", String(usage.totalTokenCount));
      }
      if (usage.promptTokenCount !== undefined) {
        headers.set("x-antigravity-prompt-token-count", String(usage.promptTokenCount));
      }
      if (usage.candidatesTokenCount !== undefined) {
        headers.set("x-antigravity-candidates-token-count", String(usage.candidatesTokenCount));
      }
    }

    logAntigravityDebugResponse(debugContext, response, {
      body: text,
      note: streaming ? "Streaming SSE payload" : undefined,
      headersOverride: headers,
    });

    if (streaming && response.ok && isEventStreamResponse) {
      return new Response(transformStreamingPayload(text), init);
    }

    if (!parsed) {
      return new Response(text, init);
    }

    if (effectiveBody?.response !== undefined) {
      const transformed = transformThinkingParts(effectiveBody.response);
      return new Response(JSON.stringify(transformed), init);
    }

    if (patched) {
      return new Response(JSON.stringify(patched), init);
    }

    return new Response(text, init);
  } catch (error) {
    logAntigravityDebugResponse(debugContext, response, {
      error,
      note: "Failed to transform Antigravity response",
    });
    return response;
  }
}
