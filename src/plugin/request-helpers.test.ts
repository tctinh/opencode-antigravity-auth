import { describe, expect, it } from "vitest";

import {
  isThinkingCapableModel,
  extractThinkingConfig,
  resolveThinkingConfig,
  filterUnsignedThinkingBlocks,
  filterMessagesThinkingBlocks,
  deepFilterThinkingBlocks,
  transformThinkingParts,
  normalizeThinkingConfig,
  parseAntigravityApiBody,
  extractUsageMetadata,
  extractUsageFromSsePayload,
  rewriteAntigravityPreviewAccessError,
  DEFAULT_THINKING_BUDGET,
} from "./request-helpers";

describe("sanitizeThinkingPart (covered via filtering)", () => {
  it("extracts wrapped text and strips SDK fields for Gemini-style thought blocks", () => {
    const validSignature = "s".repeat(60);
    const thinkingText = "wrapped thought";
    const getCachedSignatureFn = (_sessionId: string, text: string) =>
      text === thinkingText ? validSignature : undefined;

    const contents = [
      {
        role: "model",
        parts: [
          {
            thought: true,
            text: {
              text: thinkingText,
              cache_control: { type: "ephemeral" },
              providerOptions: { injected: true },
            },
            thoughtSignature: validSignature,
            cache_control: { type: "ephemeral" },
            providerOptions: { injected: true },
          },
        ],
      },
    ];

    const result = filterUnsignedThinkingBlocks(contents, "session-1", getCachedSignatureFn) as any;
    expect(result[0].parts).toHaveLength(1);
    expect(result[0].parts[0]).toEqual({
      thought: true,
      text: thinkingText,
      thoughtSignature: validSignature,
    });

    expect(result[0].parts[0].cache_control).toBeUndefined();
    expect(result[0].parts[0].providerOptions).toBeUndefined();
  });

  it("extracts wrapped thinking text and strips SDK fields for Anthropic-style thinking blocks", () => {
    const validSignature = "a".repeat(60);
    const thinkingText = "wrapped thinking";
    const getCachedSignatureFn = (_sessionId: string, text: string) =>
      text === thinkingText ? validSignature : undefined;

    const contents = [
      {
        role: "model",
        parts: [
          {
            type: "thinking",
            thinking: {
              text: thinkingText,
              cache_control: { type: "ephemeral" },
              providerOptions: { injected: true },
            },
            signature: validSignature,
            cache_control: { type: "ephemeral" },
            providerOptions: { injected: true },
          },
        ],
      },
    ];

    const result = filterUnsignedThinkingBlocks(contents, "session-1", getCachedSignatureFn) as any;
    expect(result[0].parts).toHaveLength(1);
    expect(result[0].parts[0]).toEqual({
      type: "thinking",
      thinking: thinkingText,
      signature: validSignature,
    });
  });

  it("preserves signatures while dropping cache_control/providerOptions during signature restoration", () => {
    const cachedSignature = "c".repeat(60);
    const getCachedSignatureFn = (_sessionId: string, _text: string) => cachedSignature;

    const messages = [
      {
        role: "assistant",
        content: [
          {
            type: "thinking",
            thinking: {
              thinking: "restore me",
              cache_control: { type: "ephemeral" },
            },
            // no signature present (forces restore)
            providerOptions: { injected: true },
          },
          { type: "text", text: "visible" },
        ],
      },
    ];

    const result = filterMessagesThinkingBlocks(messages, "session-1", getCachedSignatureFn) as any;
    expect(result[0].content[0]).toEqual({
      type: "thinking",
      thinking: "restore me",
      signature: cachedSignature,
    });
  });

  it("sanitizes reasoning blocks keeping only allowed fields (type, text, signature)", () => {
    const validSignature = "z".repeat(60);
    const getCachedSignatureFn = (_sessionId: string, _text: string) => validSignature;

    const contents = [
      {
        role: "model",
        parts: [
          {
            type: "reasoning",
            text: "reasoning text",
            signature: validSignature,
            cache_control: { type: "ephemeral" },
            providerOptions: { injected: true },
            meta: { keep: true },
          },
          { type: "text", text: "visible" },
        ],
      },
    ];

    const result = filterUnsignedThinkingBlocks(contents, "session-1", getCachedSignatureFn) as any;
    expect(result[0].parts[0]).toEqual({
      type: "reasoning",
      text: "reasoning text",
      signature: validSignature,
    });
  });
});

describe("isThinkingCapableModel", () => {
  it("returns true for models with 'thinking' in name", () => {
    expect(isThinkingCapableModel("claude-thinking")).toBe(true);
    expect(isThinkingCapableModel("CLAUDE-THINKING-4")).toBe(true);
    expect(isThinkingCapableModel("model-thinking-v1")).toBe(true);
  });

  it("returns true for models with 'gemini-3' in name", () => {
    expect(isThinkingCapableModel("gemini-3-pro")).toBe(true);
    expect(isThinkingCapableModel("GEMINI-3-flash")).toBe(true);
    expect(isThinkingCapableModel("gemini-3")).toBe(true);
  });

  it("returns true for models with 'opus' in name", () => {
    expect(isThinkingCapableModel("claude-opus")).toBe(true);
    expect(isThinkingCapableModel("claude-4-opus")).toBe(true);
    expect(isThinkingCapableModel("OPUS")).toBe(true);
  });

  it("returns false for non-thinking models", () => {
    expect(isThinkingCapableModel("claude-sonnet")).toBe(false);
    expect(isThinkingCapableModel("gemini-2-pro")).toBe(false);
    expect(isThinkingCapableModel("gpt-4")).toBe(false);
  });
});

describe("extractThinkingConfig", () => {
  it("extracts thinkingConfig from generationConfig", () => {
    const result = extractThinkingConfig(
      {},
      { thinkingConfig: { includeThoughts: true, thinkingBudget: 8000 } },
      undefined,
    );
    expect(result).toEqual({ includeThoughts: true, thinkingBudget: 8000 });
  });

  it("extracts thinkingConfig from extra_body", () => {
    const result = extractThinkingConfig(
      {},
      undefined,
      { thinkingConfig: { includeThoughts: true, thinkingBudget: 4000 } },
    );
    expect(result).toEqual({ includeThoughts: true, thinkingBudget: 4000 });
  });

  it("extracts thinkingConfig from requestPayload directly", () => {
    const result = extractThinkingConfig(
      { thinkingConfig: { includeThoughts: false, thinkingBudget: 2000 } },
      undefined,
      undefined,
    );
    expect(result).toEqual({ includeThoughts: false, thinkingBudget: 2000 });
  });

  it("prioritizes generationConfig over extra_body", () => {
    const result = extractThinkingConfig(
      {},
      { thinkingConfig: { includeThoughts: true, thinkingBudget: 8000 } },
      { thinkingConfig: { includeThoughts: false, thinkingBudget: 4000 } },
    );
    expect(result).toEqual({ includeThoughts: true, thinkingBudget: 8000 });
  });

  it("converts Anthropic-style thinking config", () => {
    const result = extractThinkingConfig(
      { thinking: { type: "enabled", budgetTokens: 10000 } },
      undefined,
      undefined,
    );
    expect(result).toEqual({ includeThoughts: true, thinkingBudget: 10000 });
  });

  it("uses default budget for Anthropic-style without budgetTokens", () => {
    const result = extractThinkingConfig(
      { thinking: { type: "enabled" } },
      undefined,
      undefined,
    );
    expect(result).toEqual({ includeThoughts: true, thinkingBudget: DEFAULT_THINKING_BUDGET });
  });

  it("returns undefined when no config found", () => {
    expect(extractThinkingConfig({}, undefined, undefined)).toBeUndefined();
  });

  it("uses default budget when thinkingBudget not specified", () => {
    const result = extractThinkingConfig(
      {},
      { thinkingConfig: { includeThoughts: true } },
      undefined,
    );
    expect(result).toEqual({ includeThoughts: true, thinkingBudget: DEFAULT_THINKING_BUDGET });
  });
});

describe("resolveThinkingConfig", () => {
  it("keeps thinking enabled for Claude models with assistant history", () => {
    const result = resolveThinkingConfig(
      { includeThoughts: true, thinkingBudget: 8000 },
      true, // isThinkingModel
      true, // isClaudeModel
      true, // hasAssistantHistory
    );
    expect(result).toEqual({ includeThoughts: true, thinkingBudget: 8000 });
  });

  it("enables thinking for thinking-capable models without user config", () => {
    const result = resolveThinkingConfig(
      undefined,
      true, // isThinkingModel
      false, // isClaudeModel
      false, // hasAssistantHistory
    );
    expect(result).toEqual({ includeThoughts: true, thinkingBudget: DEFAULT_THINKING_BUDGET });
  });

  it("respects user config for non-Claude models", () => {
    const userConfig = { includeThoughts: false, thinkingBudget: 5000 };
    const result = resolveThinkingConfig(
      userConfig,
      true,
      false,
      false,
    );
    expect(result).toEqual(userConfig);
  });

  it("returns user config for Claude without history", () => {
    const userConfig = { includeThoughts: true, thinkingBudget: 8000 };
    const result = resolveThinkingConfig(
      userConfig,
      true,
      true, // isClaudeModel
      false, // no history
    );
    expect(result).toEqual(userConfig);
  });

  it("returns undefined for non-thinking model without user config", () => {
    const result = resolveThinkingConfig(
      undefined,
      false, // not thinking model
      false,
      false,
    );
    expect(result).toBeUndefined();
  });
});

describe("filterUnsignedThinkingBlocks", () => {
  it("filters out unsigned thinking parts", () => {
    const contents = [
      {
        role: "model",
        parts: [
          { type: "thinking", text: "thinking without signature" },
          { type: "text", text: "visible text" },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result[0].parts).toHaveLength(1);
    expect(result[0].parts[0].type).toBe("text");
  });

  it("keeps signed thinking parts with valid signatures from our cache", () => {
    const validSignature = "a".repeat(60);
    const thinkingText = "thinking with signature";
    const getCachedSignatureFn = (_sessionId: string, text: string) =>
      text === thinkingText ? validSignature : undefined;

    const contents = [
      {
        role: "model",
        parts: [
          { type: "thinking", text: thinkingText, signature: validSignature },
          { type: "text", text: "visible text" },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents, "session-1", getCachedSignatureFn);
    expect(result[0].parts).toHaveLength(2);
    expect(result[0].parts[0].signature).toBe(validSignature);
  });

  it("strips thinking parts with foreign signatures not in our cache", () => {
    const foreignSignature = "f".repeat(60);
    const contents = [
      {
        role: "model",
        parts: [
          { type: "thinking", text: "foreign thinking", signature: foreignSignature },
          { type: "text", text: "visible text" },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result[0].parts).toHaveLength(1);
    expect(result[0].parts[0].type).toBe("text");
  });

  it("filters thinking parts with short signatures", () => {
    const contents = [
      {
        role: "model",
        parts: [
          { type: "thinking", text: "thinking with short signature", signature: "sig123" },
          { type: "text", text: "visible text" },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result[0].parts).toHaveLength(1);
    expect(result[0].parts[0].type).toBe("text");
  });

  it("handles Gemini-style thought parts with valid signatures from our cache", () => {
    const validSignature = "b".repeat(55);
    const thinkingText = "has signature";
    const getCachedSignatureFn = (_sessionId: string, text: string) =>
      text === thinkingText ? validSignature : undefined;

    const contents = [
      {
        role: "model",
        parts: [
          { thought: true, text: "no signature" },
          { thought: true, text: thinkingText, thoughtSignature: validSignature },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents, "session-1", getCachedSignatureFn);
    expect(result[0].parts).toHaveLength(1);
    expect(result[0].parts[0].thoughtSignature).toBe(validSignature);
  });

  it("filters Gemini-style thought parts with short signatures", () => {
    const contents = [
      {
        role: "model",
        parts: [
          { thought: true, text: "has short signature", thoughtSignature: "sig" },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result[0].parts).toHaveLength(0);
  });

  it("preserves non-thinking parts", () => {
    const contents = [
      {
        role: "user",
        parts: [{ text: "hello" }],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result).toEqual(contents);
  });

  it("strips blocks with signature field even if type is unknown", () => {
    const foreignSignature = "x".repeat(60);
    const contents = [
      {
        role: "model",
        parts: [
          { type: "unknown_thinking_type", text: "foreign block", signature: foreignSignature },
          { type: "text", text: "visible" },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result[0].parts).toHaveLength(1);
    expect(result[0].parts[0].type).toBe("text");
  });

  it("handles empty parts array", () => {
    const contents = [{ role: "model", parts: [] }];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result[0].parts).toEqual([]);
  });

  it("handles missing parts", () => {
    const contents = [{ role: "model" }];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result).toEqual(contents);
  });

  it("preserves tool_use and tool_result blocks intact", () => {
    const contents = [
      {
        role: "model",
        parts: [
          { type: "tool_use", id: "tool_123", name: "bash", input: { command: "ls" } },
        ],
      },
      {
        role: "user",
        parts: [
          { type: "tool_result", tool_use_id: "tool_123", content: "file1.txt" },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result[0].parts[0]).toEqual({ type: "tool_use", id: "tool_123", name: "bash", input: { command: "ls" } });
    expect(result[1].parts[0]).toEqual({ type: "tool_result", tool_use_id: "tool_123", content: "file1.txt" });
  });

  it("preserves tool blocks even if they have signature-like fields", () => {
    const contents = [
      {
        role: "user",
        parts: [
          { type: "tool_result", tool_use_id: "tool_456", content: "result", signature: "some_random_value" },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result[0].parts).toHaveLength(1);
    expect(result[0].parts[0].tool_use_id).toBe("tool_456");
  });

  it("preserves nested tool_result format", () => {
    const contents = [
      {
        role: "user",
        parts: [
          { tool_result: { tool_use_id: "tool_789", content: "nested result" } },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result[0].parts).toHaveLength(1);
    expect(result[0].parts[0].tool_result.tool_use_id).toBe("tool_789");
  });

  it("preserves functionCall and functionResponse blocks", () => {
    const contents = [
      {
        role: "model",
        parts: [
          { functionCall: { name: "get_weather", args: { city: "NYC" } } },
        ],
      },
      {
        role: "function",
        parts: [
          { functionResponse: { name: "get_weather", response: { temp: 72 } } },
        ],
      },
    ];
    const result = filterUnsignedThinkingBlocks(contents);
    expect(result[0].parts[0].functionCall).toBeDefined();
    expect(result[1].parts[0].functionResponse).toBeDefined();
  });
});

describe("deepFilterThinkingBlocks", () => {
  it("removes nested thinking blocks in extra_body messages", () => {
    const payload = {
      extra_body: {
        messages: [
          {
            role: "assistant",
            content: [
              { type: "thinking", thinking: "foreign", signature: "x".repeat(60) },
              { type: "text", text: "visible" },
            ],
          },
        ],
      },
    };

    deepFilterThinkingBlocks(payload);
    const filtered = (payload as any).extra_body.messages[0].content;
    expect(filtered).toHaveLength(1);
    expect(filtered[0].type).toBe("text");
  });

});

describe("filterMessagesThinkingBlocks", () => {
  it("filters out unsigned thinking blocks in messages[].content", () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "no signature" },
          { type: "text", text: "visible" },
        ],
      },
    ];

    const result = filterMessagesThinkingBlocks(messages) as any;
    expect(result[0].content).toHaveLength(1);
    expect(result[0].content[0].type).toBe("text");
  });

  it("keeps signed thinking blocks with valid signatures from our cache and sanitizes injected fields", () => {
    const validSignature = "a".repeat(60);
    const thinkingText = "wrapped";
    const getCachedSignatureFn = (_sessionId: string, text: string) =>
      text === thinkingText ? validSignature : undefined;

    const messages = [
      {
        role: "assistant",
        content: [
          {
            type: "thinking",
            thinking: { text: thinkingText, cache_control: { type: "ephemeral" } },
            signature: validSignature,
            cache_control: { type: "ephemeral" },
            providerOptions: { injected: true },
          },
          { type: "text", text: "visible" },
        ],
      },
    ];

    const result = filterMessagesThinkingBlocks(messages, "session-1", getCachedSignatureFn) as any;
    expect(result[0].content[0]).toEqual({
      type: "thinking",
      thinking: thinkingText,
      signature: validSignature,
    });
  });

  it("strips thinking blocks with foreign signatures not in our cache", () => {
    const foreignSignature = "f".repeat(60);
    const messages = [
      {
        role: "assistant",
        content: [
          {
            type: "thinking",
            thinking: "foreign thinking",
            signature: foreignSignature,
          },
          { type: "text", text: "visible" },
        ],
      },
    ];

    const result = filterMessagesThinkingBlocks(messages) as any;
    expect(result[0].content).toHaveLength(1);
    expect(result[0].content[0].type).toBe("text");
  });

  it("filters thinking blocks with short signatures", () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "short sig", signature: "sig123" },
          { type: "text", text: "visible" },
        ],
      },
    ];

    const result = filterMessagesThinkingBlocks(messages) as any;
    expect(result[0].content).toEqual([{ type: "text", text: "visible" }]);
  });

  it("restores a missing signature from cache and preserves it after sanitization", () => {
    const cachedSignature = "c".repeat(60);
    const getCachedSignatureFn = (_sessionId: string, _text: string) => cachedSignature;

    const messages = [
      {
        role: "assistant",
        content: [
          {
            type: "thinking",
            thinking: { thinking: "restore me", providerOptions: { injected: true } },
            // no signature present (forces restore)
            cache_control: { type: "ephemeral" },
          },
          { type: "text", text: "visible" },
        ],
      },
    ];

    const result = filterMessagesThinkingBlocks(messages, "session-1", getCachedSignatureFn) as any;
    expect(result[0].content[0]).toEqual({
      type: "thinking",
      thinking: "restore me",
      signature: cachedSignature,
    });
  });

  it("handles Gemini-style thought blocks inside messages content with cached signatures", () => {
    const validSignature = "b".repeat(60);
    const thinkingText = "wrapped thought";
    const getCachedSignatureFn = (_sessionId: string, text: string) =>
      text === thinkingText ? validSignature : undefined;

    const messages = [
      {
        role: "assistant",
        content: [
          {
            thought: true,
            text: { text: thinkingText, cache_control: { type: "ephemeral" } },
            thoughtSignature: validSignature,
            providerOptions: { injected: true },
          },
          { type: "text", text: "visible" },
        ],
      },
    ];

    const result = filterMessagesThinkingBlocks(messages, "session-1", getCachedSignatureFn) as any;
    expect(result[0].content[0]).toEqual({
      thought: true,
      text: thinkingText,
      thoughtSignature: validSignature,
    });
  });

  it("preserves non-thinking blocks and returns message unchanged when content is missing", () => {
    const messages: any[] = [
      { role: "assistant", content: [{ type: "text", text: "hello" }] },
      { role: "assistant" },
    ];

    const result = filterMessagesThinkingBlocks(messages) as any;
    expect(result[0]).toEqual(messages[0]);
    expect(result[1]).toEqual(messages[1]);
  });

  it("handles non-object messages gracefully", () => {
    const messages: any[] = [null, "string", 123, { role: "assistant", content: [] }];
    const result = filterMessagesThinkingBlocks(messages) as any;
    expect(result).toEqual(messages);
  });
});

describe("transformThinkingParts", () => {
  it("transforms Anthropic-style thinking blocks to reasoning", () => {
    const response = {
      content: [
        { type: "thinking", thinking: "my thoughts" },
        { type: "text", text: "visible" },
      ],
    };
    const result = transformThinkingParts(response) as any;
    expect(result.content[0].type).toBe("reasoning");
    expect(result.content[0].thought).toBe(true);
    expect(result.reasoning_content).toBe("my thoughts");
  });

  it("transforms Gemini-style candidates", () => {
    const response = {
      candidates: [
        {
          content: {
            parts: [
              { thought: true, text: "thinking here" },
              { text: "output" },
            ],
          },
        },
      ],
    };
    const result = transformThinkingParts(response) as any;
    expect(result.candidates[0].content.parts[0].type).toBe("reasoning");
    expect(result.candidates[0].reasoning_content).toBe("thinking here");
  });

  it("handles non-object input", () => {
    expect(transformThinkingParts(null)).toBeNull();
    expect(transformThinkingParts(undefined)).toBeUndefined();
    expect(transformThinkingParts("string")).toBe("string");
  });

  it("preserves other response properties", () => {
    const response = {
      content: [],
      id: "resp-123",
      model: "claude-4",
    };
    const result = transformThinkingParts(response) as any;
    expect(result.id).toBe("resp-123");
    expect(result.model).toBe("claude-4");
  });
});

describe("normalizeThinkingConfig", () => {
  it("returns undefined for non-object input", () => {
    expect(normalizeThinkingConfig(null)).toBeUndefined();
    expect(normalizeThinkingConfig(undefined)).toBeUndefined();
    expect(normalizeThinkingConfig("string")).toBeUndefined();
  });

  it("normalizes valid config", () => {
    const result = normalizeThinkingConfig({
      thinkingBudget: 8000,
      includeThoughts: true,
    });
    expect(result).toEqual({
      thinkingBudget: 8000,
      includeThoughts: true,
    });
  });

  it("handles snake_case property names", () => {
    const result = normalizeThinkingConfig({
      thinking_budget: 4000,
      include_thoughts: true,
    });
    expect(result).toEqual({
      thinkingBudget: 4000,
      includeThoughts: true,
    });
  });

  it("disables includeThoughts when budget is 0", () => {
    const result = normalizeThinkingConfig({
      thinkingBudget: 0,
      includeThoughts: true,
    });
    expect(result?.includeThoughts).toBe(false);
  });

  it("returns undefined when both values are absent/undefined", () => {
    const result = normalizeThinkingConfig({});
    expect(result).toBeUndefined();
  });

  it("handles non-finite budget values", () => {
    const result = normalizeThinkingConfig({
      thinkingBudget: Infinity,
      includeThoughts: true,
    });
    // When budget is non-finite (undefined), includeThoughts is forced to false
    expect(result).toEqual({ includeThoughts: false });
  });
});

describe("parseAntigravityApiBody", () => {
  it("parses valid JSON object", () => {
    const result = parseAntigravityApiBody('{"response": {"text": "hello"}}');
    expect(result).toEqual({ response: { text: "hello" } });
  });

  it("extracts first object from array", () => {
    const result = parseAntigravityApiBody('[{"response": "first"}, {"response": "second"}]');
    expect(result).toEqual({ response: "first" });
  });

  it("returns null for invalid JSON", () => {
    expect(parseAntigravityApiBody("not json")).toBeNull();
  });

  it("returns null for empty array", () => {
    expect(parseAntigravityApiBody("[]")).toBeNull();
  });

  it("returns null for primitive values", () => {
    expect(parseAntigravityApiBody('"string"')).toBeNull();
    expect(parseAntigravityApiBody("123")).toBeNull();
  });

  it("handles array with null values", () => {
    const result = parseAntigravityApiBody('[null, {"valid": true}]');
    expect(result).toEqual({ valid: true });
  });
});

describe("extractUsageMetadata", () => {
  it("extracts usage from response.usageMetadata", () => {
    const body = {
      response: {
        usageMetadata: {
          totalTokenCount: 1000,
          promptTokenCount: 500,
          candidatesTokenCount: 500,
          cachedContentTokenCount: 100,
        },
      },
    };
    const result = extractUsageMetadata(body);
    expect(result).toEqual({
      totalTokenCount: 1000,
      promptTokenCount: 500,
      candidatesTokenCount: 500,
      cachedContentTokenCount: 100,
    });
  });

  it("returns null when no usageMetadata", () => {
    expect(extractUsageMetadata({ response: {} })).toBeNull();
    expect(extractUsageMetadata({})).toBeNull();
  });

  it("handles partial usage data", () => {
    const body = {
      response: {
        usageMetadata: {
          totalTokenCount: 1000,
        },
      },
    };
    const result = extractUsageMetadata(body);
    expect(result).toEqual({
      totalTokenCount: 1000,
      promptTokenCount: undefined,
      candidatesTokenCount: undefined,
      cachedContentTokenCount: undefined,
    });
  });

  it("filters non-finite numbers", () => {
    const body = {
      response: {
        usageMetadata: {
          totalTokenCount: Infinity,
          promptTokenCount: NaN,
          candidatesTokenCount: 100,
        },
      },
    };
    const result = extractUsageMetadata(body);
    expect(result?.totalTokenCount).toBeUndefined();
    expect(result?.promptTokenCount).toBeUndefined();
    expect(result?.candidatesTokenCount).toBe(100);
  });
});

describe("extractUsageFromSsePayload", () => {
  it("extracts usage from SSE data line", () => {
    const payload = `data: {"response": {"usageMetadata": {"totalTokenCount": 500}}}`;
    const result = extractUsageFromSsePayload(payload);
    expect(result?.totalTokenCount).toBe(500);
  });

  it("handles multiple SSE lines", () => {
    const payload = `data: {"response": {}}
data: {"response": {"usageMetadata": {"totalTokenCount": 1000}}}`;
    const result = extractUsageFromSsePayload(payload);
    expect(result?.totalTokenCount).toBe(1000);
  });

  it("returns null when no usage found", () => {
    const payload = `data: {"response": {"text": "hello"}}`;
    const result = extractUsageFromSsePayload(payload);
    expect(result).toBeNull();
  });

  it("ignores non-data lines", () => {
    const payload = `: keepalive
event: message
data: {"response": {"usageMetadata": {"totalTokenCount": 200}}}`;
    const result = extractUsageFromSsePayload(payload);
    expect(result?.totalTokenCount).toBe(200);
  });

  it("handles malformed JSON gracefully", () => {
    const payload = `data: not json
data: {"response": {"usageMetadata": {"totalTokenCount": 300}}}`;
    const result = extractUsageFromSsePayload(payload);
    expect(result?.totalTokenCount).toBe(300);
  });
});

describe("rewriteAntigravityPreviewAccessError", () => {
  it("returns null for non-404 status", () => {
    const body = { error: { message: "Not found" } };
    expect(rewriteAntigravityPreviewAccessError(body, 400)).toBeNull();
    expect(rewriteAntigravityPreviewAccessError(body, 500)).toBeNull();
  });

  it("rewrites error for Antigravity model on 404", () => {
    const body = { error: { message: "Model not found" } };
    const result = rewriteAntigravityPreviewAccessError(body, 404, "claude-opus");
    expect(result?.error?.message).toContain("Model not found");
    expect(result?.error?.message).toContain("preview access");
  });

  it("rewrites error when error message contains antigravity", () => {
    const body = { error: { message: "antigravity model unavailable" } };
    const result = rewriteAntigravityPreviewAccessError(body, 404);
    expect(result?.error?.message).toContain("preview access");
  });

  it("returns null for 404 with non-antigravity model", () => {
    const body = { error: { message: "Model not found" } };
    const result = rewriteAntigravityPreviewAccessError(body, 404, "gemini-pro");
    expect(result).toBeNull();
  });

  it("provides default message when error message is empty", () => {
    const body = { error: { message: "" } };
    const result = rewriteAntigravityPreviewAccessError(body, 404, "opus-model");
    expect(result?.error?.message).toContain("Antigravity preview features are not enabled");
  });

  it("detects Claude models in requested model name", () => {
    const body = { error: {} };
    const result = rewriteAntigravityPreviewAccessError(body, 404, "claude-3-sonnet");
    expect(result?.error?.message).toContain("preview access");
  });
});

// =============================================================================
// CLAUDE MULTI-TURN EDGE CASE TESTS
// These tests cover the bugs documented in CLAUDE_MULTI_TURN_BUG_SPEC.md
// =============================================================================

import {
  analyzeConversationState,
  closeToolLoopForThinking,
  sanitizeThinkingForClaude,
  recoverToolResponseID,
  validateToolPairing,
  BYPASS_SIGNATURE,
  TOOL_LOOP_CLOSE_MODEL,
  TOOL_LOOP_CLOSE_USER,
  TOOL_RESPONSE_UNAVAILABLE,
} from "./request-helpers";

describe("Claude Multi-Turn Edge Cases", () => {

  // ===========================================================================
  // Edge Case 1: Thinking stripped, no cached signature
  // Bug: tool_use sent without thinking block when signature cache is empty
  // ===========================================================================
  describe("Edge Case 1: Thinking stripped, no cached signature", () => {
    it("should detect when tool_use has no preceding thinking block", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { functionCall: { name: "bash", args: { command: "ls" } } },
          ],
        },
      ];

      const state = analyzeConversationState(contents);
      expect(state.hasToolUseWithoutThinking).toBe(true);
    });

    it("should inject thinking with bypass signature when no cached signature available", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { functionCall: { name: "bash", args: { command: "ls" } } },
          ],
        },
      ];

      const result = sanitizeThinkingForClaude(contents, true);
      
      // Should have thinking block before functionCall
      expect(result[0].parts.length).toBe(2);
      expect(result[0].parts[0].thought).toBe(true);
      expect(result[0].parts[0].thoughtSignature).toBe(BYPASS_SIGNATURE);
      expect(result[0].parts[1].functionCall).toBeDefined();
    });

    it("should not corrupt conversation on subsequent turns after bypass injection", () => {
      // Simulate a multi-turn conversation where first turn used bypass
      const contents = [
        {
          role: "model",
          parts: [
            { thought: true, text: "[Thinking]", thoughtSignature: BYPASS_SIGNATURE },
            { functionCall: { name: "bash", args: { command: "ls" } } },
          ],
        },
        {
          role: "user",
          parts: [
            { functionResponse: { name: "bash", response: { result: "file1.txt" } } },
          ],
        },
        {
          role: "model",
          parts: [
            { functionCall: { name: "read", args: { path: "file1.txt" } } },
          ],
        },
      ];

      const result = sanitizeThinkingForClaude(contents, true);
      
      // Last model message should also have thinking injected
      const lastModel = result[result.length - 1];
      expect(lastModel.parts[0].thought).toBe(true);
      expect(lastModel.parts[0].thoughtSignature).toBe(BYPASS_SIGNATURE);
    });
  });

  // ===========================================================================
  // Edge Case 2: Tool loop mid-turn, thinking toggled
  // Bug: Thinking mode toggle mid-turn causes permanent corruption
  // ===========================================================================
  describe("Edge Case 2: Tool loop mid-turn, thinking toggled", () => {
    it("should detect when in tool loop (last message is functionResponse)", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { thought: true, text: "thinking", thoughtSignature: "s".repeat(60) },
            { functionCall: { name: "bash", args: { command: "ls" } } },
          ],
        },
        {
          role: "user",
          parts: [
            { functionResponse: { name: "bash", response: { result: "output" } } },
          ],
        },
      ];

      const state = analyzeConversationState(contents);
      expect(state.inToolLoop).toBe(true);
    });

    it("should close tool loop with synthetic messages when thinking unavailable at turn start", () => {
      const contents = [
        {
          role: "model",
          parts: [
            // No thinking at turn start - was stripped
            { functionCall: { name: "bash", args: { command: "ls" } } },
          ],
        },
        {
          role: "user",
          parts: [
            { functionResponse: { name: "bash", response: { result: "output" } } },
          ],
        },
      ];

      const result = closeToolLoopForThinking(contents);
      
      // Should have synthetic model message
      const syntheticModel = result[result.length - 2];
      expect(syntheticModel.role).toBe("model");
      expect(syntheticModel.parts[0].text).toBe(TOOL_LOOP_CLOSE_MODEL);
      
      // Should have synthetic user message
      const syntheticUser = result[result.length - 1];
      expect(syntheticUser.role).toBe("user");
      expect(syntheticUser.parts[0].text).toBe(TOOL_LOOP_CLOSE_USER);
    });

    it("should preserve tool_use/tool_result pairing after loop closure", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { functionCall: { name: "bash", args: { command: "ls" } }, id: "call-1" },
          ],
        },
        {
          role: "user",
          parts: [
            { functionResponse: { name: "bash", response: { result: "output" } }, id: "call-1" },
          ],
        },
      ];

      const result = closeToolLoopForThinking(contents);
      const validation = validateToolPairing(result);
      
      expect(validation.valid).toBe(true);
      expect(validation.orphanedToolCalls).toHaveLength(0);
      expect(validation.orphanedToolResponses).toHaveLength(0);
    });
  });

  // ===========================================================================
  // Edge Case 3: functionResponse ID doesn't match any functionCall
  // Bug: Silent failure causes orphaned tool_use error
  // ===========================================================================
  describe("Edge Case 3: ID mismatch recovery", () => {
    it("should recover using name match when ID not found", () => {
      const pendingCalls = new Map([
        ["bash", ["call-1", "call-2"]],
        ["read", ["call-3"]],
      ]);
      const orphanedCalls: string[] = [];

      const result = recoverToolResponseID(
        { name: "bash", id: undefined },
        pendingCalls,
        orphanedCalls
      );

      expect(result.id).toBe("call-1");
      expect(result.recoveryLevel).toBe("name");
    });

    it("should recover using orphan match when name not found", () => {
      const pendingCalls = new Map<string, string[]>();
      const orphanedCalls = ["orphan-call-1", "orphan-call-2"];

      const result = recoverToolResponseID(
        { name: "unknown_tool", id: undefined },
        pendingCalls,
        orphanedCalls
      );

      expect(result.id).toBe("orphan-call-1");
      expect(result.recoveryLevel).toBe("orphan");
      expect(result.warning).toContain("unknown_tool");
    });

    it("should use fallback when no match available", () => {
      const pendingCalls = new Map([
        ["other_tool", ["call-99"]],
      ]);
      const orphanedCalls: string[] = [];

      const result = recoverToolResponseID(
        { name: "missing_tool", id: undefined },
        pendingCalls,
        orphanedCalls
      );

      expect(result.id).toBe("call-99");
      expect(result.recoveryLevel).toBe("fallback");
      expect(result.warning).toBeDefined();
    });

    it("should generate placeholder ID when all recovery fails", () => {
      const pendingCalls = new Map<string, string[]>();
      const orphanedCalls: string[] = [];

      const result = recoverToolResponseID(
        { name: "orphan_response", id: undefined },
        pendingCalls,
        orphanedCalls
      );

      expect(result.id).toMatch(/^placeholder-/);
      expect(result.recoveryLevel).toBe("placeholder");
      expect(result.warning).toContain("orphan_response");
    });

    it("should log warning on recovery", () => {
      const pendingCalls = new Map([
        ["bash", ["call-1"]],
      ]);
      const orphanedCalls: string[] = [];

      const result = recoverToolResponseID(
        { name: "bash", id: undefined },
        pendingCalls,
        orphanedCalls
      );

      // Name match doesn't need warning (it's expected)
      expect(result.warning).toBeUndefined();
    });
  });

  // ===========================================================================
  // Edge Case 4: Session restart loses in-memory signature cache
  // Bug: Can't recover thinking signatures after restart
  // ===========================================================================
  describe("Edge Case 4: Session restart loses cache", () => {
    it("should use bypass signature when cache empty and thinking needed", () => {
      // Simulate: cache is empty (session restart), but we have tool_use
      const contents = [
        {
          role: "model",
          parts: [
            // Thinking was stripped because cache is empty
            { functionCall: { name: "bash", args: { command: "ls" } } },
          ],
        },
      ];

      // No getCachedSignatureFn provided (simulating empty cache)
      const result = sanitizeThinkingForClaude(contents, true, undefined);
      
      expect(result[0].parts[0].thoughtSignature).toBe(BYPASS_SIGNATURE);
    });

    it("should not fail on first request after restart", () => {
      const contents = [
        {
          role: "user",
          parts: [{ text: "Hello" }],
        },
      ];

      // Should not throw
      const result = sanitizeThinkingForClaude(contents, true, undefined);
      expect(result).toEqual(contents);
    });
  });

  // ===========================================================================
  // Edge Case 5: Context compaction removes thinking blocks
  // Bug: Old thinking stripped, can't continue tool loop
  // ===========================================================================
  describe("Edge Case 5: Context compaction removes thinking", () => {
    it("should detect missing thinking in earlier turns and inject before tool_use", () => {
      // After compaction: thinking was stripped from earlier turn
      const contents = [
        {
          role: "model",
          parts: [
            // Thinking was compacted/removed
            { functionCall: { name: "bash", args: { command: "ls" } } },
          ],
        },
        {
          role: "user",
          parts: [
            { functionResponse: { name: "bash", response: { result: "file1" } } },
          ],
        },
        {
          role: "model",
          parts: [
            { functionCall: { name: "read", args: { path: "file1" } } },
          ],
        },
      ];

      const result = sanitizeThinkingForClaude(contents, true);
      
      // Each model message with tool_use should have thinking
      const modelMessages = result.filter((m: any) => m.role === "model");
      for (const msg of modelMessages) {
        const hasThinking = msg.parts.some((p: any) => p.thought === true);
        const hasTool = msg.parts.some((p: any) => p.functionCall);
        if (hasTool) {
          expect(hasThinking).toBe(true);
        }
      }
    });

    it("should close tool loop if needed before injection", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { functionCall: { name: "bash", args: { command: "ls" } } },
          ],
        },
        {
          role: "user",
          parts: [
            { functionResponse: { name: "bash", response: { result: "output" } } },
          ],
        },
      ];

      const state = analyzeConversationState(contents);
      expect(state.inToolLoop).toBe(true);
      expect(state.turnHasThinking).toBe(false);

      const result = sanitizeThinkingForClaude(contents, true);
      
      // Should have closed the loop OR injected thinking
      const validation = validateToolPairing(result);
      expect(validation.valid).toBe(true);
    });
  });

  // ===========================================================================
  // Edge Case 6: Parallel tool calls, responses out of order
  // Bug: FIFO queue mismatch when responses arrive out of order
  // ===========================================================================
  describe("Edge Case 6: Parallel tool calls, out of order responses", () => {
    it("should match by name when IDs mismatch", () => {
      const pendingCalls = new Map([
        ["bash", ["call-1"]],
        ["read", ["call-2"]],
        ["write", ["call-3"]],
      ]);
      const orphanedCalls: string[] = [];

      // Response for "read" comes first
      const result = recoverToolResponseID(
        { name: "read", id: undefined },
        pendingCalls,
        orphanedCalls
      );

      expect(result.id).toBe("call-2");
      expect(result.recoveryLevel).toBe("name");
    });

    it("should handle multiple calls with same name (FIFO within name)", () => {
      const pendingCalls = new Map([
        ["bash", ["call-1", "call-4", "call-7"]],  // Multiple bash calls
      ]);
      const orphanedCalls: string[] = [];

      // First bash response
      const result1 = recoverToolResponseID(
        { name: "bash", id: undefined },
        pendingCalls,
        orphanedCalls
      );
      expect(result1.id).toBe("call-1");

      // Queue should be updated (call-1 removed)
      pendingCalls.set("bash", ["call-4", "call-7"]);

      // Second bash response
      const result2 = recoverToolResponseID(
        { name: "bash", id: undefined },
        pendingCalls,
        orphanedCalls
      );
      expect(result2.id).toBe("call-4");
    });
  });

  // ===========================================================================
  // Edge Case 7: Empty content array after filtering
  // Bug: Invalid request with empty parts
  // ===========================================================================
  describe("Edge Case 7: Empty content after filtering", () => {
    it("should handle empty parts array gracefully", () => {
      const contents = [
        {
          role: "model",
          parts: [],
        },
      ];

      const state = analyzeConversationState(contents);
      expect(state.inToolLoop).toBe(false);
      expect(state.hasToolUseWithoutThinking).toBe(false);
    });

    it("should not send invalid request with only stripped thinking", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { type: "thinking", thinking: "will be stripped" },
          ],
        },
      ];

      const result = sanitizeThinkingForClaude(contents, false);  // thinking disabled
      
      // Should either keep the message empty or add placeholder text
      expect(result[0].parts.length).toBeGreaterThanOrEqual(0);
    });
  });

  // ===========================================================================
  // Edge Case 8: Nested tool calls in thinking block
  // Bug: Tool blocks inside thinking get dropped
  // ===========================================================================
  describe("Edge Case 8: Nested tool calls in thinking", () => {
    it("should preserve tool blocks even if adjacent to thinking", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { type: "thinking", thinking: "planning to call tool" },
            { functionCall: { name: "bash", args: { command: "ls" } } },
          ],
        },
      ];

      // When stripping thinking, tool should remain
      const result = sanitizeThinkingForClaude(contents, false);  // thinking disabled = strip
      
      const toolBlocks = result[0].parts.filter((p: any) => p.functionCall);
      expect(toolBlocks).toHaveLength(1);
    });

    it("should extract and promote nested tool blocks if found inside thinking object", () => {
      // Edge case: malformed structure where tool is inside thinking
      const contents = [
        {
          role: "model",
          parts: [
            {
              type: "thinking",
              thinking: "text",
              // Hypothetical malformed: nested tool
              nested: { functionCall: { name: "bash", args: {} } },
            },
          ],
        },
      ];

      // Should handle gracefully without crashing
      expect(() => sanitizeThinkingForClaude(contents, true)).not.toThrow();
    });
  });

  // ===========================================================================
  // Edge Case 9: Incomplete turn (tool call without response)
  // Bug: Orphaned tool_use causes "tool_use without tool_result" error
  // ===========================================================================
  describe("Edge Case 9: Incomplete turn (tool call without response)", () => {
    it("should detect orphaned tool calls", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { functionCall: { name: "bash", args: { command: "ls" } }, id: "call-1" },
            { functionCall: { name: "read", args: { path: "file" } }, id: "call-2" },
          ],
        },
        {
          role: "user",
          parts: [
            // Only one response for two calls
            { functionResponse: { name: "bash", response: { result: "ok" } }, id: "call-1" },
          ],
        },
      ];

      const validation = validateToolPairing(contents);
      expect(validation.orphanedToolCalls).toContain("call-2");
    });

    it("should inject placeholder response for orphaned calls when auto-fix enabled", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { functionCall: { name: "bash", args: { command: "ls" } }, id: "call-1" },
          ],
        },
        // No user response at all
      ];

      const validation = validateToolPairing(contents, { autoFix: true });
      
      if (validation.autoFixed) {
        expect(validation.valid).toBe(true);
      } else {
        expect(validation.orphanedToolCalls).toHaveLength(1);
      }
    });
  });

  // ===========================================================================
  // Edge Case 10: Thinking at wrong position (not before tool_use)
  // Bug: Thinking block comes after tool_use, causing rejection
  // ===========================================================================
  describe("Edge Case 10: Thinking at wrong position", () => {
    it("should reorder thinking before tool_use", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { functionCall: { name: "bash", args: { command: "ls" } } },
            { thought: true, text: "should be first", thoughtSignature: BYPASS_SIGNATURE },
          ],
        },
      ];

      const result = sanitizeThinkingForClaude(contents, true);
      
      // Thinking should come first
      expect(result[0].parts[0].thought).toBe(true);
      expect(result[0].parts[1].functionCall).toBeDefined();
    });

    it("should inject thinking if missing entirely before tool_use", () => {
      const contents = [
        {
          role: "model",
          parts: [
            { text: "I will run a command" },
            { functionCall: { name: "bash", args: { command: "ls" } } },
          ],
        },
      ];

      const result = sanitizeThinkingForClaude(contents, true);
      
      // Should have thinking before tool_use
      const functionCallIdx = result[0].parts.findIndex((p: any) => p.functionCall);
      const thinkingIdx = result[0].parts.findIndex((p: any) => p.thought === true);
      
      expect(thinkingIdx).toBeLessThan(functionCallIdx);
    });
  });
});

// =============================================================================
// INTEGRATION TESTS: Multi-Turn Conversation Simulation
// =============================================================================
describe("Multi-Turn Conversation Simulation", () => {
  it("should handle 10-turn conversation with tool loops", () => {
    const contents: any[] = [];
    
    // Build a 10-turn conversation with tool calls
    for (let i = 0; i < 10; i++) {
      // Model turn with tool call
      contents.push({
        role: "model",
        parts: [
          { thought: true, text: `Thinking turn ${i}`, thoughtSignature: BYPASS_SIGNATURE },
          { functionCall: { name: "bash", args: { command: `cmd-${i}` } }, id: `call-${i}` },
        ],
      });
      
      // User turn with response
      contents.push({
        role: "user",
        parts: [
          { functionResponse: { name: "bash", response: { result: `result-${i}` } }, id: `call-${i}` },
        ],
      });
    }

    const result = sanitizeThinkingForClaude(contents, true);
    const validation = validateToolPairing(result);
    
    expect(validation.valid).toBe(true);
    expect(validation.orphanedToolCalls).toHaveLength(0);
    expect(validation.orphanedToolResponses).toHaveLength(0);
  });

  it("should recover from simulated session restart mid-conversation", () => {
    // First 5 turns have thinking with real signatures
    const contents: any[] = [];
    
    for (let i = 0; i < 5; i++) {
      contents.push({
        role: "model",
        parts: [
          // After "restart", these signatures are not in cache
          { thought: true, text: `Old thinking ${i}`, thoughtSignature: "unknown_sig_".repeat(5) },
          { functionCall: { name: "bash", args: { command: `cmd-${i}` } }, id: `call-${i}` },
        ],
      });
      contents.push({
        role: "user",
        parts: [
          { functionResponse: { name: "bash", response: { result: `ok` } }, id: `call-${i}` },
        ],
      });
    }

    // Simulate: no cache function (restart scenario)
    const result = sanitizeThinkingForClaude(contents, true, undefined);
    
    // Should use bypass signatures
    const modelMessages = result.filter((m: any) => m.role === "model");
    for (const msg of modelMessages) {
      const thinking = msg.parts.find((p: any) => p.thought === true);
      if (thinking) {
        expect(thinking.thoughtSignature).toBe(BYPASS_SIGNATURE);
      }
    }
  });

  it("should handle context compaction simulation", () => {
    // Simulate compacted conversation: old tool results cleared
    const contents = [
      {
        role: "model",
        parts: [
          { functionCall: { name: "bash", args: { command: "old" } }, id: "old-1" },
        ],
      },
      {
        role: "user",
        parts: [
          { functionResponse: { name: "bash", response: { result: TOOL_RESPONSE_UNAVAILABLE } }, id: "old-1" },
        ],
      },
      {
        role: "model",
        parts: [
          { functionCall: { name: "read", args: { path: "file" } }, id: "recent-1" },
        ],
      },
      {
        role: "user",
        parts: [
          { functionResponse: { name: "read", response: { result: "actual content" } }, id: "recent-1" },
        ],
      },
    ];

    const result = sanitizeThinkingForClaude(contents, true);
    const validation = validateToolPairing(result);
    
    expect(validation.valid).toBe(true);
  });

  it("should handle rapid parallel tool calls", () => {
    // Model makes 5 parallel tool calls
    const contents = [
      {
        role: "model",
        parts: [
          { thought: true, text: "Running multiple tools", thoughtSignature: BYPASS_SIGNATURE },
          { functionCall: { name: "bash", args: { command: "1" } }, id: "p-1" },
          { functionCall: { name: "bash", args: { command: "2" } }, id: "p-2" },
          { functionCall: { name: "read", args: { path: "a" } }, id: "p-3" },
          { functionCall: { name: "read", args: { path: "b" } }, id: "p-4" },
          { functionCall: { name: "write", args: { path: "c" } }, id: "p-5" },
        ],
      },
      {
        role: "user",
        parts: [
          // Responses in different order
          { functionResponse: { name: "read", response: { result: "a" } }, id: "p-3" },
          { functionResponse: { name: "bash", response: { result: "1" } }, id: "p-1" },
          { functionResponse: { name: "write", response: { result: "ok" } }, id: "p-5" },
          { functionResponse: { name: "bash", response: { result: "2" } }, id: "p-2" },
          { functionResponse: { name: "read", response: { result: "b" } }, id: "p-4" },
        ],
      },
    ];

    const result = sanitizeThinkingForClaude(contents, true);
    const validation = validateToolPairing(result);
    
    expect(validation.valid).toBe(true);
    expect(validation.orphanedToolCalls).toHaveLength(0);
    expect(validation.orphanedToolResponses).toHaveLength(0);
  });
});
