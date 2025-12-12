const ANTIGRAVITY_PREVIEW_LINK = "https://goo.gle/enable-preview-features"; // TODO: Update to Antigravity link if available

// ============================================================================
// Constants
// ============================================================================

export const DEFAULT_THINKING_BUDGET = 16000;
export const THINKING_MODEL_PATTERNS = ["thinking", "gemini-3", "opus"] as const;

// ============================================================================
// Types
// ============================================================================

export interface ToolDebugInfo {
  missing: number;
  summaries: string[];
  payload?: string;
}

export interface ParsedModelInfo {
  rawModel: string;
  action: string;
  isStreaming: boolean;
  isClaude: boolean;
  isThinkingModel: boolean;
}

export interface FunctionDeclaration {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface AntigravityApiError {
  code?: number;
  message?: string;
  status?: string;
  [key: string]: unknown;
}

/**
 * Minimal representation of Antigravity API responses we touch.
 */
export interface AntigravityApiBody {
  response?: unknown;
  error?: AntigravityApiError;
  [key: string]: unknown;
}

/**
 * Usage metadata exposed by Antigravity responses. Fields are optional to reflect partial payloads.
 */
export interface AntigravityUsageMetadata {
  totalTokenCount?: number;
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  cachedContentTokenCount?: number;
}

/**
 * Normalized thinking configuration accepted by Antigravity.
 */
export interface ThinkingConfig {
  thinkingBudget?: number;
  includeThoughts?: boolean;
}

/**
 * Ensures thinkingConfig is valid: includeThoughts only allowed when budget > 0.
 */
export function normalizeThinkingConfig(config: unknown): ThinkingConfig | undefined {
  if (!config || typeof config !== "object") {
    return undefined;
  }

  const record = config as Record<string, unknown>;
  const budgetRaw = record.thinkingBudget ?? record.thinking_budget;
  const includeRaw = record.includeThoughts ?? record.include_thoughts;

  const thinkingBudget = typeof budgetRaw === "number" && Number.isFinite(budgetRaw) ? budgetRaw : undefined;
  const includeThoughts = typeof includeRaw === "boolean" ? includeRaw : undefined;

  const enableThinking = thinkingBudget !== undefined && thinkingBudget > 0;
  const finalInclude = enableThinking ? includeThoughts ?? false : false;

  if (!enableThinking && finalInclude === false && thinkingBudget === undefined && includeThoughts === undefined) {
    return undefined;
  }

  const normalized: ThinkingConfig = {};
  if (thinkingBudget !== undefined) {
    normalized.thinkingBudget = thinkingBudget;
  }
  if (finalInclude !== undefined) {
    normalized.includeThoughts = finalInclude;
  }
  return normalized;
}

/**
 * Parses an Antigravity API body; handles array-wrapped responses the API sometimes returns.
 */
export function parseAntigravityApiBody(rawText: string): AntigravityApiBody | null {
  try {
    const parsed = JSON.parse(rawText);
    if (Array.isArray(parsed)) {
      const firstObject = parsed.find((item: unknown) => typeof item === "object" && item !== null);
      if (firstObject && typeof firstObject === "object") {
        return firstObject as AntigravityApiBody;
      }
      return null;
    }

    if (parsed && typeof parsed === "object") {
      return parsed as AntigravityApiBody;
    }

    return null;
  } catch {
    return null;
  }
}

/**
 * Extracts usageMetadata from a response object, guarding types.
 */
export function extractUsageMetadata(body: AntigravityApiBody): AntigravityUsageMetadata | null {
  const usage = (body.response && typeof body.response === "object"
    ? (body.response as { usageMetadata?: unknown }).usageMetadata
    : undefined) as AntigravityUsageMetadata | undefined;

  if (!usage || typeof usage !== "object") {
    return null;
  }

  const asRecord = usage as Record<string, unknown>;
  const toNumber = (value: unknown): number | undefined =>
    typeof value === "number" && Number.isFinite(value) ? value : undefined;

  return {
    totalTokenCount: toNumber(asRecord.totalTokenCount),
    promptTokenCount: toNumber(asRecord.promptTokenCount),
    candidatesTokenCount: toNumber(asRecord.candidatesTokenCount),
    cachedContentTokenCount: toNumber(asRecord.cachedContentTokenCount),
  };
}

/**
 * Walks SSE lines to find a usage-bearing response chunk.
 */
export function extractUsageFromSsePayload(payload: string): AntigravityUsageMetadata | null {
  const lines = payload.split("\n");
  for (const line of lines) {
    if (!line.startsWith("data:")) {
      continue;
    }
    const jsonText = line.slice(5).trim();
    if (!jsonText) {
      continue;
    }
    try {
      const parsed = JSON.parse(jsonText);
      if (parsed && typeof parsed === "object") {
        const usage = extractUsageMetadata({ response: (parsed as Record<string, unknown>).response });
        if (usage) {
          return usage;
        }
      }
    } catch {
      continue;
    }
  }
  return null;
}

/**
 * Enhances 404 errors for Antigravity models with a direct preview-access message.
 */
export function rewriteAntigravityPreviewAccessError(
  body: AntigravityApiBody,
  status: number,
  requestedModel?: string,
): AntigravityApiBody | null {
  if (!needsPreviewAccessOverride(status, body, requestedModel)) {
    return null;
  }

  const error: AntigravityApiError = body.error ?? {};
  const trimmedMessage = typeof error.message === "string" ? error.message.trim() : "";
  const messagePrefix = trimmedMessage.length > 0
    ? trimmedMessage
    : "Antigravity preview features are not enabled for this account.";
  const enhancedMessage = `${messagePrefix} Request preview access at ${ANTIGRAVITY_PREVIEW_LINK} before using this model.`;

  return {
    ...body,
    error: {
      ...error,
      message: enhancedMessage,
    },
  };
}

function needsPreviewAccessOverride(
  status: number,
  body: AntigravityApiBody,
  requestedModel?: string,
): boolean {
  if (status !== 404) {
    return false;
  }

  if (isAntigravityModel(requestedModel)) {
    return true;
  }

  const errorMessage = typeof body.error?.message === "string" ? body.error.message : "";
  return isAntigravityModel(errorMessage);
}

function isAntigravityModel(target?: string): boolean {
  if (!target) {
    return false;
  }

  // Check for Antigravity models instead of Gemini 3
  return /antigravity/i.test(target) || /opus/i.test(target) || /claude/i.test(target);
}

// ============================================================================
// Model Detection Helpers
// ============================================================================

const STREAM_ACTION = "streamGenerateContent";

/**
 * Parses model information from a generativelanguage API URL.
 */
export function parseModelFromUrl(url: string): ParsedModelInfo | null {
  const match = url.match(/\/models\/([^:]+):(\w+)/);
  if (!match) return null;

  const [, rawModel = "", action = ""] = match;
  const modelLower = rawModel.toLowerCase();

  return {
    rawModel,
    action,
    isStreaming: action === STREAM_ACTION,
    isClaude: modelLower.includes("claude"),
    isThinkingModel: THINKING_MODEL_PATTERNS.some((p) => modelLower.includes(p)),
  };
}

/**
 * Checks if contents array has assistant/model history (multi-turn conversation).
 */
export function hasAssistantHistory(contents: unknown): boolean {
  if (!Array.isArray(contents)) return false;
  return contents.some(
    (c: any) => c?.role === "model" || c?.role === "assistant"
  );
}

// ============================================================================
// Thinking Config Helpers
// ============================================================================

/**
 * Extracts thinking config from multiple possible source locations.
 */
export function extractThinkingSource(payload: Record<string, unknown>): unknown {
  const generationConfig = payload.generationConfig as Record<string, unknown> | undefined;
  const extraBody = payload.extra_body as Record<string, unknown> | undefined;

  // Check for thinkingConfig in generationConfig, extra_body, or top-level
  let config = generationConfig?.thinkingConfig ?? extraBody?.thinkingConfig ?? payload.thinkingConfig;

  // Convert Anthropic-style "thinking" option
  if (!config) {
    const anthropicThinking = extraBody?.thinking ?? payload.thinking;
    if (anthropicThinking && typeof anthropicThinking === "object") {
      const thinking = anthropicThinking as Record<string, unknown>;
      if (thinking.type === "enabled" || thinking.budgetTokens) {
        config = {
          includeThoughts: true,
          thinkingBudget: thinking.budgetTokens ?? DEFAULT_THINKING_BUDGET,
        };
      }
    }
  }

  return config;
}

/**
 * Resolves final thinking config based on model type and conversation history.
 */
export function resolveThinkingConfig(
  sourceConfig: unknown,
  isThinkingModel: boolean,
  isClaude: boolean,
  hasHistory: boolean
): ThinkingConfig | undefined {
  // For Claude models with history, disable thinking to avoid signature issues
  if (isClaude && hasHistory) {
    return { includeThoughts: false, thinkingBudget: 0 };
  }

  // Force thinking for thinking-capable models (unless Claude with history)
  if (isThinkingModel) {
    const existingBudget = (sourceConfig as any)?.thinkingBudget;
    return {
      includeThoughts: true,
      thinkingBudget: existingBudget > 0 ? existingBudget : DEFAULT_THINKING_BUDGET,
    };
  }

  return normalizeThinkingConfig(sourceConfig);
}

/**
 * Applies thinking config to the payload, cleaning up source fields.
 */
export function applyThinkingConfig(
  payload: Record<string, unknown>,
  config: ThinkingConfig | undefined
): void {
  const generationConfig = payload.generationConfig as Record<string, unknown> | undefined;
  const extraBody = payload.extra_body as Record<string, unknown> | undefined;

  if (config) {
    if (generationConfig) {
      generationConfig.thinkingConfig = config;
    } else {
      payload.generationConfig = { thinkingConfig: config };
    }
  } else if (generationConfig?.thinkingConfig) {
    delete generationConfig.thinkingConfig;
  }

  // Clean up source fields
  if (extraBody) {
    delete extraBody.thinkingConfig;
    delete extraBody.thinking;
  }
  delete payload.thinkingConfig;
  delete payload.thinking;
}

// ============================================================================
// Cached Content Helpers
// ============================================================================

/**
 * Extracts and normalizes cached content from various locations.
 */
export function extractCachedContent(payload: Record<string, unknown>): string | undefined {
  const extraBody = payload.extra_body as Record<string, unknown> | undefined;
  const fromExtra = extraBody?.cached_content ?? extraBody?.cachedContent;

  return (
    (payload.cached_content as string | undefined) ??
    (payload.cachedContent as string | undefined) ??
    (fromExtra as string | undefined)
  );
}

/**
 * Cleans up cached content fields from payload and extra_body.
 */
export function cleanupCachedContentFields(payload: Record<string, unknown>): void {
  delete payload.cached_content;
  delete payload.cachedContent;

  const extraBody = payload.extra_body as Record<string, unknown> | undefined;
  if (extraBody) {
    delete extraBody.cached_content;
    delete extraBody.cachedContent;
    if (Object.keys(extraBody).length === 0) {
      delete payload.extra_body;
    }
  }
}

// ============================================================================
// Tool Normalization Helpers
// ============================================================================

const EMPTY_SCHEMA = { type: "object", properties: {} };

/**
 * Extracts schema from various possible locations in a tool definition.
 */
function extractToolSchema(tool: any, decl?: any): any {
  return (
    decl?.parameters ||
    decl?.input_schema ||
    decl?.inputSchema ||
    tool.parameters ||
    tool.input_schema ||
    tool.inputSchema ||
    tool.function?.parameters ||
    tool.function?.input_schema ||
    tool.function?.inputSchema ||
    tool.custom?.parameters ||
    tool.custom?.input_schema
  );
}

/**
 * Extracts name from various possible locations in a tool definition.
 */
function extractToolName(tool: any, decl?: any, fallback?: string): string {
  return (
    decl?.name ||
    tool.name ||
    tool.function?.name ||
    tool.custom?.name ||
    fallback ||
    "unknown-tool"
  );
}

/**
 * Extracts description from various possible locations in a tool definition.
 */
function extractToolDescription(tool: any, decl?: any): string {
  return (
    decl?.description ||
    tool.description ||
    tool.function?.description ||
    tool.custom?.description ||
    ""
  );
}

/**
 * Normalizes tools for Claude models into functionDeclarations format.
 */
export function normalizeToolsForClaude(
  tools: any[],
  debugInfo: ToolDebugInfo
): any[] {
  const functionDeclarations: FunctionDeclaration[] = [];
  const passthroughTools: any[] = [];

  for (const tool of tools) {
    // Handle existing functionDeclarations array
    if (Array.isArray(tool.functionDeclarations) && tool.functionDeclarations.length > 0) {
      for (const decl of tool.functionDeclarations) {
        const schema = extractToolSchema(tool, decl);
        const name = extractToolName(tool, decl, `tool-${functionDeclarations.length}`);

        if (!schema) debugInfo.missing++;

        functionDeclarations.push({
          name,
          description: extractToolDescription(tool, decl),
          parameters: schema ?? EMPTY_SCHEMA,
        });

        debugInfo.summaries.push(`decl=${name},src=functionDeclarations,hasSchema=${schema ? "y" : "n"}`);
      }
      continue;
    }

    // Handle function/custom style definitions
    if (tool.function || tool.custom || tool.parameters || tool.input_schema || tool.inputSchema) {
      const source = tool.function ?? tool.custom ?? tool;
      const schema = extractToolSchema(tool, source);
      const name = extractToolName(tool, source, `tool-${functionDeclarations.length}`);

      if (!schema) debugInfo.missing++;

      functionDeclarations.push({
        name,
        description: extractToolDescription(tool, source),
        parameters: schema ?? EMPTY_SCHEMA,
      });

      debugInfo.summaries.push(`decl=${name},src=function/custom,hasSchema=${schema ? "y" : "n"}`);
      continue;
    }

    // Preserve non-function tools (e.g., codeExecution)
    passthroughTools.push(tool);
  }

  const result: any[] = [];
  if (functionDeclarations.length > 0) {
    result.push({ functionDeclarations });
  }
  return result.concat(passthroughTools);
}

/**
 * Normalizes tools for Gemini/non-Claude models.
 */
export function normalizeToolsForGemini(
  tools: any[],
  debugInfo: ToolDebugInfo
): any[] {
  return tools.map((tool, index) => {
    const newTool = { ...tool };

    const schema = extractToolSchema(newTool);

    // Ensure function has input_schema
    if (newTool.function && !newTool.function.input_schema && schema) {
      newTool.function.input_schema = schema;
    }

    // Create custom wrapper if needed (for internal use)
    if (newTool.custom && !newTool.custom.input_schema) {
      newTool.custom.input_schema = schema ?? EMPTY_SCHEMA;
      if (!schema) debugInfo.missing++;
    }

    debugInfo.summaries.push(
      `idx=${index},hasCustom=${!!newTool.custom},hasFunction=${!!newTool.function}`
    );

    // Strip custom wrappers for Gemini - only function-style is accepted
    delete newTool.custom;

    return newTool;
  });
}

// ============================================================================
// Claude Content Transformations
// ============================================================================

/**
 * Checks if a part is a thinking/reasoning block.
 */
function isThinkingPart(part: any): boolean {
  if (!part || typeof part !== "object") return false;
  return (
    part.thinking !== undefined ||
    part.type === "thinking" ||
    part.type === "reasoning" ||
    part.thought === true
  );
}

/**
 * Checks if a thinking part has a valid signature.
 */
function hasValidSignature(part: any): boolean {
  return !!(part.signature || part.thinking?.signature || part.thoughtSignature);
}

/**
 * Filters out unsigned thinking blocks from Claude conversation history.
 */
export function filterUnsignedThinkingBlocks(contents: any[]): any[] {
  return contents.map((content) => {
    if (!content || !Array.isArray(content.parts)) {
      return content;
    }

    const filteredParts = content.parts.filter((part: any) => {
      if (!isThinkingPart(part)) return true;
      return hasValidSignature(part);
    });

    return { ...content, parts: filteredParts };
  });
}

/**
 * Assigns IDs to function calls and matches responses using FIFO per function name.
 */
export function assignFunctionCallIds(contents: any[]): any[] {
  let callCounter = 0;
  const pendingIds = new Map<string, string[]>();

  // First pass: assign IDs to function calls
  const withCallIds = contents.map((content) => {
    if (!content || !Array.isArray(content.parts)) return content;

    const newParts = content.parts.map((part: any) => {
      if (!part?.functionCall) return part;

      const call = { ...part.functionCall };
      if (!call.id) {
        call.id = `tool-call-${++callCounter}`;
      }

      const nameKey = typeof call.name === "string" ? call.name : `tool-${callCounter}`;
      const queue = pendingIds.get(nameKey) || [];
      queue.push(call.id);
      pendingIds.set(nameKey, queue);

      return { ...part, functionCall: call };
    });

    return { ...content, parts: newParts };
  });

  // Second pass: match function responses to calls (FIFO)
  return withCallIds.map((content) => {
    if (!content || !Array.isArray(content.parts)) return content;

    const newParts = content.parts.map((part: any) => {
      if (!part?.functionResponse) return part;

      const resp = { ...part.functionResponse };
      if (!resp.id && typeof resp.name === "string") {
        const queue = pendingIds.get(resp.name);
        if (queue?.length) {
          resp.id = queue.shift();
          pendingIds.set(resp.name, queue);
        }
      }

      return { ...part, functionResponse: resp };
    });

    return { ...content, parts: newParts };
  });
}
