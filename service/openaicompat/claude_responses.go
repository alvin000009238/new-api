package openaicompat

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
)

var ErrUnsupportedResponsesTool = errors.New("unsupported responses tool for Claude conversion")

type responsesInputItem struct {
	Type      string          `json:"type,omitempty"`
	ID        string          `json:"id,omitempty"`
	Role      string          `json:"role,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	Output    any             `json:"output,omitempty"`
}

func ClaudeMessagesRequestToResponsesRequest(req *dto.ClaudeRequest) (*dto.OpenAIResponsesRequest, error) {
	if req == nil {
		return nil, errors.New("request is nil")
	}
	if strings.TrimSpace(req.Model) == "" {
		return nil, errors.New("model is required")
	}

	inputItems := make([]map[string]any, 0, len(req.Messages))
	for _, msg := range req.Messages {
		items, err := claudeMessageToResponsesInputItems(msg)
		if err != nil {
			return nil, err
		}
		inputItems = append(inputItems, items...)
	}
	inputRaw, err := common.Marshal(inputItems)
	if err != nil {
		return nil, err
	}

	out := &dto.OpenAIResponsesRequest{
		Model:       req.Model,
		Input:       inputRaw,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		ServiceTier: req.ServiceTier,
	}
	if req.MaxTokens != nil {
		out.MaxOutputTokens = common.GetPointer(*req.MaxTokens)
	}
	if instructions, err := claudeSystemToResponsesInstructions(req.System); err != nil {
		return nil, err
	} else if len(instructions) > 0 {
		out.Instructions = instructions
	}
	if len(req.Metadata) > 0 {
		out.Metadata = req.Metadata
	}
	if len(req.OutputFormat) > 0 {
		out.Text = req.OutputFormat
	}
	if req.Tools != nil {
		toolsRaw, err := claudeToolsToResponsesTools(req.Tools)
		if err != nil {
			return nil, err
		}
		out.Tools = toolsRaw
	}
	if req.ToolChoice != nil {
		toolChoiceRaw, parallelToolCallsRaw, err := claudeToolChoiceToResponsesToolChoice(req.ToolChoice)
		if err != nil {
			return nil, err
		}
		out.ToolChoice = toolChoiceRaw
		out.ParallelToolCalls = parallelToolCallsRaw
	}
	if reasoning := claudeThinkingToResponsesReasoning(req); reasoning != nil {
		out.Reasoning = reasoning
	}

	return out, nil
}

func OpenAIResponsesRequestToClaudeMessagesRequest(req *dto.OpenAIResponsesRequest) (*dto.ClaudeRequest, error) {
	if req == nil {
		return nil, errors.New("request is nil")
	}
	if strings.TrimSpace(req.Model) == "" {
		return nil, errors.New("model is required")
	}

	out := &dto.ClaudeRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxOutputTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		ServiceTier: req.ServiceTier,
	}
	if len(req.Instructions) > 0 {
		system, err := responsesInstructionsToClaudeSystem(req.Instructions)
		if err != nil {
			return nil, err
		}
		out.System = system
	}
	if len(req.Tools) > 0 {
		tools, err := responsesToolsToClaudeTools(req.Tools)
		if err != nil {
			return nil, err
		}
		out.Tools = tools
	}
	if len(req.ToolChoice) > 0 || len(req.ParallelToolCalls) > 0 {
		toolChoice, err := responsesToolChoiceToClaudeToolChoice(req.ToolChoice, req.ParallelToolCalls)
		if err != nil {
			return nil, err
		}
		out.ToolChoice = toolChoice
	}
	if req.Reasoning != nil && strings.TrimSpace(req.Reasoning.Effort) != "" {
		out.Thinking = reasoningEffortToClaudeThinking(req.Reasoning.Effort)
	}
	if len(req.Metadata) > 0 {
		out.Metadata = req.Metadata
	}

	messages, err := responsesInputToClaudeMessages(req.Input)
	if err != nil {
		return nil, err
	}
	out.Messages = messages
	return out, nil
}

func ClaudeResponseToOpenAIResponsesResponse(resp *dto.ClaudeResponse, fallbackID string, createdAt int64) (*dto.OpenAIResponsesResponse, *dto.Usage, error) {
	if resp == nil {
		return nil, nil, errors.New("response is nil")
	}

	responseID := strings.TrimSpace(resp.Id)
	if responseID == "" {
		responseID = fallbackID
	}
	outputs, err := claudeContentToResponsesOutputs(resp.Content, responseID)
	if err != nil {
		return nil, nil, err
	}
	usage := OpenAIUsageFromClaudeUsage(resp.Usage)
	out := &dto.OpenAIResponsesResponse{
		ID:        responseID,
		Object:    "response",
		CreatedAt: int(createdAt),
		Status:    rawJSONString("completed"),
		Model:     resp.Model,
		Output:    outputs,
		Usage:     usage,
	}
	return out, usage, nil
}

func OpenAIUsageFromClaudeUsage(claudeUsage *dto.ClaudeUsage) *dto.Usage {
	if claudeUsage == nil {
		return &dto.Usage{
			UsageSemantic: "openai",
			UsageSource:   "anthropic",
		}
	}

	cacheCreation5m, cacheCreation1h := normalizeClaudeCacheCreationSplit(
		claudeUsage.CacheCreationInputTokens,
		claudeUsage.GetCacheCreation5mTokens(),
		claudeUsage.GetCacheCreation1hTokens(),
	)
	cacheCreationTotal := claudeUsage.CacheCreationInputTokens
	if cacheCreationTotal == 0 {
		cacheCreationTotal = cacheCreation5m + cacheCreation1h
	}
	inputTokens := claudeUsage.InputTokens + claudeUsage.CacheReadInputTokens + cacheCreationTotal
	outputTokens := claudeUsage.OutputTokens
	usage := &dto.Usage{
		PromptTokens:     inputTokens,
		CompletionTokens: outputTokens,
		TotalTokens:      inputTokens + outputTokens,
		InputTokens:      inputTokens,
		OutputTokens:     outputTokens,
		PromptTokensDetails: dto.InputTokenDetails{
			CachedTokens:         claudeUsage.CacheReadInputTokens,
			CachedCreationTokens: cacheCreationTotal,
		},
		InputTokensDetails: &dto.InputTokenDetails{
			CachedTokens:         claudeUsage.CacheReadInputTokens,
			CachedCreationTokens: cacheCreationTotal,
		},
		ClaudeCacheCreation5mTokens: cacheCreation5m,
		ClaudeCacheCreation1hTokens: cacheCreation1h,
		UsageSemantic:               "openai",
		UsageSource:                 "anthropic",
	}
	return usage
}

type ClaudeToResponsesStreamConverter struct {
	ResponseID string
	CreatedAt  int
	Model      string

	Usage     *dto.Usage
	Completed bool

	outputs []dto.ResponsesOutput
	blocks  map[int]*responsesStreamBlock
}

type responsesStreamBlock struct {
	Index       int
	OutputIndex int
	Type        string
	ItemID      string
	ToolName    string
	ToolArgs    strings.Builder
	Text        strings.Builder
}

func NewClaudeToResponsesStreamConverter(responseID string, createdAt int64, model string) *ClaudeToResponsesStreamConverter {
	return &ClaudeToResponsesStreamConverter{
		ResponseID: responseID,
		CreatedAt:  int(createdAt),
		Model:      model,
		Usage:      &dto.Usage{UsageSemantic: "openai", UsageSource: "anthropic"},
		blocks:     make(map[int]*responsesStreamBlock),
	}
}

func (c *ClaudeToResponsesStreamConverter) Convert(event *dto.ClaudeResponse) ([]dto.ResponsesStreamResponse, *dto.Usage, error) {
	if c == nil {
		return nil, nil, errors.New("converter is nil")
	}
	if event == nil {
		return nil, c.Usage, nil
	}

	var events []dto.ResponsesStreamResponse
	switch event.Type {
	case "message_start":
		if event.Message != nil {
			if strings.TrimSpace(event.Message.Id) != "" {
				c.ResponseID = event.Message.Id
			}
			if strings.TrimSpace(event.Message.Model) != "" {
				c.Model = event.Message.Model
			}
			if event.Message.Usage != nil {
				c.Usage = OpenAIUsageFromClaudeUsage(event.Message.Usage)
			}
		}
		events = append(events, dto.ResponsesStreamResponse{
			Type:     "response.created",
			Response: c.response("in_progress"),
		})
	case "content_block_start":
		block := c.startBlock(event)
		if block == nil {
			return events, c.Usage, nil
		}
		events = append(events, dto.ResponsesStreamResponse{
			Type:        dto.ResponsesOutputTypeItemAdded,
			OutputIndex: common.GetPointer(block.OutputIndex),
			Item:        c.outputForBlock(block, false),
		})
	case "content_block_delta":
		block := c.blockForEvent(event)
		if block == nil || event.Delta == nil {
			return events, c.Usage, nil
		}
		switch event.Delta.Type {
		case "text_delta":
			text := event.Delta.GetText()
			if text != "" {
				block.Text.WriteString(text)
				events = append(events, dto.ResponsesStreamResponse{
					Type:         "response.output_text.delta",
					Delta:        text,
					OutputIndex:  common.GetPointer(block.OutputIndex),
					ContentIndex: common.GetPointer(0),
					ItemID:       block.ItemID,
				})
			}
		case "thinking_delta":
			if event.Delta.Thinking != nil && *event.Delta.Thinking != "" {
				events = append(events, dto.ResponsesStreamResponse{
					Type:         "response.reasoning_summary_text.delta",
					Delta:        *event.Delta.Thinking,
					OutputIndex:  common.GetPointer(block.OutputIndex),
					ContentIndex: common.GetPointer(0),
					ItemID:       block.ItemID,
				})
			}
		case "input_json_delta":
			if event.Delta.PartialJson != nil && *event.Delta.PartialJson != "" {
				block.ToolArgs.WriteString(*event.Delta.PartialJson)
				events = append(events, dto.ResponsesStreamResponse{
					Type:        "response.function_call_arguments.delta",
					Delta:       *event.Delta.PartialJson,
					OutputIndex: common.GetPointer(block.OutputIndex),
					ItemID:      block.ItemID,
				})
			}
		}
	case "content_block_stop":
		block := c.blockForEvent(event)
		if block == nil {
			return events, c.Usage, nil
		}
		item := c.outputForBlock(block, true)
		c.outputs = append(c.outputs, *item)
		events = append(events, dto.ResponsesStreamResponse{
			Type:        dto.ResponsesOutputTypeItemDone,
			OutputIndex: common.GetPointer(block.OutputIndex),
			Item:        item,
			ItemID:      block.ItemID,
		})
		delete(c.blocks, block.Index)
	case "message_delta":
		if event.Usage != nil {
			c.Usage = OpenAIUsageFromClaudeUsage(event.Usage)
		}
	case "message_stop":
		events = append(events, c.completeEvent())
		c.Completed = true
	}
	return events, c.Usage, nil
}

func (c *ClaudeToResponsesStreamConverter) Complete() dto.ResponsesStreamResponse {
	if c.Completed {
		return dto.ResponsesStreamResponse{}
	}
	c.Completed = true
	return c.completeEvent()
}

func claudeMessageToResponsesInputItems(msg dto.ClaudeMessage) ([]map[string]any, error) {
	role := strings.TrimSpace(msg.Role)
	if role == "" {
		role = "user"
	}
	if msg.IsStringContent() {
		return []map[string]any{{
			"role":    role,
			"content": msg.GetStringContent(),
		}}, nil
	}

	contents, err := msg.ParseContent()
	if err != nil {
		return nil, err
	}
	contentParts := make([]map[string]any, 0, len(contents))
	items := make([]map[string]any, 0, 1)
	for _, content := range contents {
		switch content.Type {
		case "text", "input_text":
			partType := "input_text"
			if role == "assistant" {
				partType = "output_text"
			}
			contentParts = append(contentParts, map[string]any{
				"type": partType,
				"text": content.GetText(),
			})
		case "image":
			contentParts = append(contentParts, map[string]any{
				"type":      "input_image",
				"image_url": claudeSourceToDataOrURL(content.Source),
			})
		case "document":
			filePart := map[string]any{"type": "input_file"}
			if source := claudeSourceToDataOrURL(content.Source); source != "" {
				if strings.HasPrefix(source, "data:") {
					filePart["file_data"] = source
				} else {
					filePart["file_url"] = source
				}
			}
			contentParts = append(contentParts, filePart)
		case "tool_use":
			argsRaw, err := common.Marshal(content.Input)
			if err != nil {
				return nil, err
			}
			items = append(items, map[string]any{
				"type":      "function_call",
				"call_id":   content.Id,
				"name":      content.Name,
				"arguments": json.RawMessage(argsRaw),
			})
		case "tool_result":
			items = append(items, map[string]any{
				"type":    "function_call_output",
				"call_id": content.ToolUseId,
				"output":  claudeToolResultOutput(content),
			})
		}
	}
	if len(contentParts) > 0 {
		items = append([]map[string]any{{
			"role":    role,
			"content": contentParts,
		}}, items...)
	}
	return items, nil
}

func claudeSystemToResponsesInstructions(system any) (json.RawMessage, error) {
	if system == nil {
		return nil, nil
	}
	if str, ok := system.(string); ok {
		if strings.TrimSpace(str) == "" {
			return nil, nil
		}
		return common.Marshal(str)
	}
	parts, err := common.Any2Type[[]dto.ClaudeMediaMessage](system)
	if err != nil {
		return nil, err
	}
	var texts []string
	for _, part := range parts {
		if part.Type == "text" && strings.TrimSpace(part.GetText()) != "" {
			texts = append(texts, part.GetText())
		}
	}
	if len(texts) == 0 {
		return nil, nil
	}
	return common.Marshal(strings.Join(texts, "\n"))
}

func claudeToolsToResponsesTools(toolsAny any) (json.RawMessage, error) {
	toolsMap, err := common.Any2Type[[]map[string]any](toolsAny)
	if err != nil {
		return nil, err
	}
	out := make([]map[string]any, 0, len(toolsMap))
	for _, tool := range toolsMap {
		toolType := common.Interface2String(tool["type"])
		name := common.Interface2String(tool["name"])
		if strings.HasPrefix(toolType, "web_search") || name == "web_search" {
			out = append(out, map[string]any{
				"type":                dto.BuildInToolWebSearchPreview,
				"search_context_size": maxUsesToSearchContextSize(common.Interface2String(tool["max_uses"])),
			})
			continue
		}
		inputSchema := tool["input_schema"]
		if inputSchema == nil {
			inputSchema = map[string]any{"type": "object"}
		}
		out = append(out, map[string]any{
			"type":        "function",
			"name":        name,
			"description": tool["description"],
			"parameters":  inputSchema,
		})
	}
	return common.Marshal(out)
}

func claudeToolChoiceToResponsesToolChoice(toolChoice any) (json.RawMessage, json.RawMessage, error) {
	toolChoiceMap, err := common.Any2Type[map[string]any](toolChoice)
	if err != nil {
		if str, ok := toolChoice.(string); ok {
			raw, marshalErr := common.Marshal(str)
			return raw, nil, marshalErr
		}
		return nil, nil, err
	}
	choiceType := common.Interface2String(toolChoiceMap["type"])
	var choice any
	switch choiceType {
	case "auto":
		choice = "auto"
	case "any":
		choice = "required"
	case "none":
		choice = "none"
	case "tool":
		choice = map[string]any{
			"type": "function",
			"name": common.Interface2String(toolChoiceMap["name"]),
		}
	default:
		choice = toolChoice
	}
	choiceRaw, err := common.Marshal(choice)
	if err != nil {
		return nil, nil, err
	}
	if disableParallel, ok := toolChoiceMap["disable_parallel_tool_use"].(bool); ok {
		parallelRaw, err := common.Marshal(!disableParallel)
		return choiceRaw, parallelRaw, err
	}
	return choiceRaw, nil, nil
}

func claudeThinkingToResponsesReasoning(req *dto.ClaudeRequest) *dto.Reasoning {
	if req == nil {
		return nil
	}
	var outputConfig struct {
		Effort string `json:"effort,omitempty"`
	}
	if len(req.OutputConfig) > 0 {
		_ = common.Unmarshal(req.OutputConfig, &outputConfig)
	}
	if strings.TrimSpace(outputConfig.Effort) != "" {
		return &dto.Reasoning{Effort: outputConfig.Effort, Summary: "detailed"}
	}
	if req.Thinking == nil {
		return nil
	}
	switch {
	case req.Thinking.Type == "adaptive":
		return &dto.Reasoning{Effort: "high", Summary: "detailed"}
	case req.Thinking.GetBudgetTokens() >= 4096:
		return &dto.Reasoning{Effort: "high", Summary: "detailed"}
	case req.Thinking.GetBudgetTokens() >= 2048:
		return &dto.Reasoning{Effort: "medium", Summary: "detailed"}
	case req.Thinking.GetBudgetTokens() > 0:
		return &dto.Reasoning{Effort: "low", Summary: "detailed"}
	default:
		return nil
	}
}

func responsesInstructionsToClaudeSystem(raw json.RawMessage) (any, error) {
	switch common.GetJsonType(raw) {
	case "string":
		var system string
		if err := common.Unmarshal(raw, &system); err != nil {
			return nil, err
		}
		return system, nil
	case "array":
		var parts []dto.ClaudeMediaMessage
		if err := common.Unmarshal(raw, &parts); err == nil {
			return parts, nil
		}
	}
	return raw, nil
}

func responsesInputToClaudeMessages(raw json.RawMessage) ([]dto.ClaudeMessage, error) {
	if len(raw) == 0 {
		return nil, errors.New("input is required")
	}
	if common.GetJsonType(raw) == "string" {
		var text string
		if err := common.Unmarshal(raw, &text); err != nil {
			return nil, err
		}
		return []dto.ClaudeMessage{{Role: "user", Content: text}}, nil
	}

	var items []responsesInputItem
	if err := common.Unmarshal(raw, &items); err != nil {
		return nil, err
	}
	messages := make([]dto.ClaudeMessage, 0, len(items))
	for _, item := range items {
		switch item.Type {
		case "function_call":
			input, err := responsesArgumentsToClaudeInput(item.Arguments)
			if err != nil {
				return nil, err
			}
			messages = append(messages, dto.ClaudeMessage{
				Role: "assistant",
				Content: []dto.ClaudeMediaMessage{{
					Type:  "tool_use",
					Id:    firstNonEmpty(item.CallID, item.ID),
					Name:  item.Name,
					Input: input,
				}},
			})
		case "function_call_output":
			messages = append(messages, dto.ClaudeMessage{
				Role: "user",
				Content: []dto.ClaudeMediaMessage{{
					Type:      "tool_result",
					ToolUseId: item.CallID,
					Content:   item.Output,
				}},
			})
		default:
			role := strings.TrimSpace(item.Role)
			if role == "" {
				role = "user"
			}
			content, err := responsesContentToClaudeContent(item.Content)
			if err != nil {
				return nil, err
			}
			messages = append(messages, dto.ClaudeMessage{
				Role:    role,
				Content: content,
			})
		}
	}
	return messages, nil
}

func responsesContentToClaudeContent(raw json.RawMessage) (any, error) {
	if len(raw) == 0 || common.GetJsonType(raw) == "null" {
		return "", nil
	}
	if common.GetJsonType(raw) == "string" {
		var text string
		if err := common.Unmarshal(raw, &text); err != nil {
			return nil, err
		}
		return text, nil
	}

	var parts []map[string]any
	if err := common.Unmarshal(raw, &parts); err != nil {
		return nil, err
	}
	claudeParts := make([]dto.ClaudeMediaMessage, 0, len(parts))
	for _, part := range parts {
		partType := common.Interface2String(part["type"])
		switch partType {
		case "input_text", "output_text", "text":
			claudeParts = append(claudeParts, dto.ClaudeMediaMessage{
				Type: "text",
				Text: common.GetPointer(common.Interface2String(part["text"])),
			})
		case "input_image":
			imageURL := stringFromStringOrURLObject(part["image_url"])
			if imageURL == "" {
				continue
			}
			claudeParts = append(claudeParts, dto.ClaudeMediaMessage{
				Type:   "image",
				Source: sourceFromDataOrURL(imageURL, "image/png"),
			})
		case "input_file":
			fileURL := stringFromStringOrURLObject(part["file_url"])
			fileData := common.Interface2String(part["file_data"])
			fileID := common.Interface2String(part["file_id"])
			if fileData == "" && fileURL == "" && fileID != "" {
				return nil, fmt.Errorf("responses input_file file_id is not supported for Claude conversion")
			}
			sourceValue := firstNonEmpty(fileData, fileURL)
			if sourceValue == "" {
				continue
			}
			claudeParts = append(claudeParts, dto.ClaudeMediaMessage{
				Type:   "document",
				Source: sourceFromDataOrURL(sourceValue, mimeTypeFromFilePart(part)),
			})
		}
	}
	if len(claudeParts) == 1 && claudeParts[0].Type == "text" {
		return claudeParts[0].GetText(), nil
	}
	return claudeParts, nil
}

func responsesToolsToClaudeTools(raw json.RawMessage) ([]any, error) {
	var tools []map[string]any
	if err := common.Unmarshal(raw, &tools); err != nil {
		return nil, err
	}
	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		toolType := common.Interface2String(tool["type"])
		switch toolType {
		case "function":
			inputSchema, _ := common.Any2Type[map[string]interface{}](tool["parameters"])
			if inputSchema == nil {
				inputSchema = map[string]interface{}{"type": "object"}
			}
			out = append(out, dto.Tool{
				Name:        common.Interface2String(tool["name"]),
				Description: common.Interface2String(tool["description"]),
				InputSchema: inputSchema,
			})
		case dto.BuildInToolWebSearchPreview:
			out = append(out, dto.ClaudeWebSearchTool{
				Type:    "web_search_20250305",
				Name:    "web_search",
				MaxUses: searchContextSizeToMaxUses(common.Interface2String(tool["search_context_size"])),
			})
		default:
			return nil, fmt.Errorf("%w: %s", ErrUnsupportedResponsesTool, toolType)
		}
	}
	return out, nil
}

func responsesToolChoiceToClaudeToolChoice(rawChoice json.RawMessage, rawParallel json.RawMessage) (*dto.ClaudeToolChoice, error) {
	var choice *dto.ClaudeToolChoice
	if len(rawChoice) > 0 {
		if common.GetJsonType(rawChoice) == "string" {
			var value string
			if err := common.Unmarshal(rawChoice, &value); err != nil {
				return nil, err
			}
			switch value {
			case "auto":
				choice = &dto.ClaudeToolChoice{Type: "auto"}
			case "required":
				choice = &dto.ClaudeToolChoice{Type: "any"}
			case "none":
				choice = &dto.ClaudeToolChoice{Type: "none"}
			}
		} else {
			var value map[string]any
			if err := common.Unmarshal(rawChoice, &value); err != nil {
				return nil, err
			}
			if common.Interface2String(value["type"]) == "function" {
				choice = &dto.ClaudeToolChoice{
					Type: "tool",
					Name: common.Interface2String(value["name"]),
				}
			}
		}
	}
	if len(rawParallel) > 0 && common.GetJsonType(rawParallel) == "boolean" {
		var parallel bool
		if err := common.Unmarshal(rawParallel, &parallel); err != nil {
			return nil, err
		}
		if choice == nil {
			choice = &dto.ClaudeToolChoice{Type: "auto"}
		}
		if choice.Type != "none" {
			choice.DisableParallelToolUse = !parallel
		}
	}
	return choice, nil
}

func reasoningEffortToClaudeThinking(effort string) *dto.Thinking {
	switch effort {
	case "low":
		return &dto.Thinking{Type: "enabled", BudgetTokens: common.GetPointer(1280)}
	case "medium":
		return &dto.Thinking{Type: "enabled", BudgetTokens: common.GetPointer(2048)}
	case "high":
		return &dto.Thinking{Type: "enabled", BudgetTokens: common.GetPointer(4096)}
	default:
		return nil
	}
}

func responsesArgumentsToClaudeInput(raw json.RawMessage) (any, error) {
	if len(raw) == 0 || common.GetJsonType(raw) == "null" {
		return map[string]any{}, nil
	}
	if common.GetJsonType(raw) == "string" {
		argString := common.JsonRawMessageToString(raw)
		if strings.TrimSpace(argString) == "" {
			return map[string]any{}, nil
		}
		var obj map[string]any
		if err := common.UnmarshalJsonStr(argString, &obj); err == nil {
			return obj, nil
		}
		return argString, nil
	}
	var obj map[string]any
	if err := common.Unmarshal(raw, &obj); err == nil {
		return obj, nil
	}
	return common.JsonRawMessageToString(raw), nil
}

func claudeContentToResponsesOutputs(contents []dto.ClaudeMediaMessage, responseID string) ([]dto.ResponsesOutput, error) {
	outputs := make([]dto.ResponsesOutput, 0, len(contents))
	messageOutput := dto.ResponsesOutput{
		Type:   "message",
		ID:     responseID + "-message",
		Status: "completed",
		Role:   "assistant",
	}
	for _, content := range contents {
		switch content.Type {
		case "text":
			messageOutput.Content = append(messageOutput.Content, dto.ResponsesOutputContent{
				Type: "output_text",
				Text: content.GetText(),
			})
		case "thinking":
			if content.Thinking != nil && *content.Thinking != "" {
				messageOutput.Content = append(messageOutput.Content, dto.ResponsesOutputContent{
					Type: "output_text",
					Text: *content.Thinking,
				})
			}
		case "tool_use":
			argsRaw, err := common.Marshal(content.Input)
			if err != nil {
				return nil, err
			}
			outputs = append(outputs, dto.ResponsesOutput{
				Type:      "function_call",
				ID:        content.Id,
				CallId:    content.Id,
				Name:      content.Name,
				Arguments: argsRaw,
				Status:    "completed",
			})
		}
	}
	if len(messageOutput.Content) > 0 {
		outputs = append([]dto.ResponsesOutput{messageOutput}, outputs...)
	}
	return outputs, nil
}

func (c *ClaudeToResponsesStreamConverter) startBlock(event *dto.ClaudeResponse) *responsesStreamBlock {
	if event == nil || event.ContentBlock == nil {
		return nil
	}
	idx := event.GetIndex()
	blockType := event.ContentBlock.Type
	itemID := event.ContentBlock.Id
	if itemID == "" {
		itemID = fmt.Sprintf("%s-output-%d", c.ResponseID, idx)
	}
	block := &responsesStreamBlock{
		Index:       idx,
		OutputIndex: len(c.outputs) + len(c.blocks),
		Type:        blockType,
		ItemID:      itemID,
		ToolName:    event.ContentBlock.Name,
	}
	if blockType == "text" && event.ContentBlock.Text != nil {
		block.Text.WriteString(*event.ContentBlock.Text)
	}
	c.blocks[idx] = block
	return block
}

func (c *ClaudeToResponsesStreamConverter) blockForEvent(event *dto.ClaudeResponse) *responsesStreamBlock {
	if event == nil {
		return nil
	}
	idx := event.GetIndex()
	if block, ok := c.blocks[idx]; ok {
		return block
	}
	block := &responsesStreamBlock{
		Index:       idx,
		OutputIndex: len(c.outputs) + len(c.blocks),
		Type:        "text",
		ItemID:      fmt.Sprintf("%s-output-%d", c.ResponseID, idx),
	}
	c.blocks[idx] = block
	return block
}

func (c *ClaudeToResponsesStreamConverter) outputForBlock(block *responsesStreamBlock, done bool) *dto.ResponsesOutput {
	if block == nil {
		return nil
	}
	status := "in_progress"
	if done {
		status = "completed"
	}
	switch block.Type {
	case "tool_use":
		arguments := block.ToolArgs.String()
		if strings.TrimSpace(arguments) == "" {
			arguments = "{}"
		}
		return &dto.ResponsesOutput{
			Type:      "function_call",
			ID:        block.ItemID,
			CallId:    block.ItemID,
			Name:      block.ToolName,
			Arguments: json.RawMessage(arguments),
			Status:    status,
		}
	default:
		return &dto.ResponsesOutput{
			Type:   "message",
			ID:     block.ItemID,
			Status: status,
			Role:   "assistant",
			Content: []dto.ResponsesOutputContent{{
				Type: "output_text",
				Text: block.Text.String(),
			}},
		}
	}
}

func (c *ClaudeToResponsesStreamConverter) response(status string) *dto.OpenAIResponsesResponse {
	return &dto.OpenAIResponsesResponse{
		ID:        c.ResponseID,
		Object:    "response",
		CreatedAt: c.CreatedAt,
		Status:    rawJSONString(status),
		Model:     c.Model,
		Output:    append([]dto.ResponsesOutput(nil), c.outputs...),
		Usage:     c.Usage,
	}
}

func (c *ClaudeToResponsesStreamConverter) completeEvent() dto.ResponsesStreamResponse {
	for idx, block := range c.blocks {
		item := c.outputForBlock(block, true)
		if item != nil {
			c.outputs = append(c.outputs, *item)
		}
		delete(c.blocks, idx)
	}
	return dto.ResponsesStreamResponse{
		Type:     "response.completed",
		Response: c.response("completed"),
	}
}

func normalizeClaudeCacheCreationSplit(totalTokens int, tokens5m int, tokens1h int) (int, int) {
	remainder := totalTokens - tokens5m - tokens1h
	if remainder < 0 {
		remainder = 0
	}
	return tokens5m + remainder, tokens1h
}

func claudeSourceToDataOrURL(source *dto.ClaudeMessageSource) string {
	if source == nil {
		return ""
	}
	if strings.TrimSpace(source.Url) != "" {
		return source.Url
	}
	data := common.Interface2String(source.Data)
	if data == "" {
		return ""
	}
	mediaType := source.MediaType
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	return "data:" + mediaType + ";base64," + data
}

func claudeToolResultOutput(content dto.ClaudeMediaMessage) any {
	if content.IsStringContent() {
		return content.GetStringContent()
	}
	if content.Content != nil {
		return content.Content
	}
	return ""
}

func sourceFromDataOrURL(value string, fallbackMime string) *dto.ClaudeMessageSource {
	value = strings.TrimSpace(value)
	if strings.HasPrefix(value, "data:") {
		mediaType := fallbackMime
		data := value
		if header, payload, ok := strings.Cut(strings.TrimPrefix(value, "data:"), ","); ok {
			data = payload
			if media, _, found := strings.Cut(header, ";"); found && media != "" {
				mediaType = media
			} else if header != "" && !strings.Contains(header, ";") {
				mediaType = header
			}
		}
		return &dto.ClaudeMessageSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      data,
		}
	}
	if strings.HasPrefix(value, "http://") || strings.HasPrefix(value, "https://") {
		return &dto.ClaudeMessageSource{
			Type: "url",
			Url:  value,
		}
	}
	return &dto.ClaudeMessageSource{
		Type:      "base64",
		MediaType: fallbackMime,
		Data:      value,
	}
}

func stringFromStringOrURLObject(v any) string {
	switch typed := v.(type) {
	case string:
		return typed
	case map[string]any:
		return common.Interface2String(typed["url"])
	default:
		return common.Interface2String(v)
	}
}

func mimeTypeFromFilePart(part map[string]any) string {
	if mimeType := common.Interface2String(part["mime_type"]); mimeType != "" {
		return mimeType
	}
	filename := common.Interface2String(part["filename"])
	if filename == "" {
		filename = common.Interface2String(part["file_name"])
	}
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".pdf":
		return "application/pdf"
	case ".txt", ".md", ".json", ".csv":
		return "text/plain"
	default:
		return "application/octet-stream"
	}
}

func maxUsesToSearchContextSize(maxUses string) string {
	switch maxUses {
	case "1":
		return "low"
	case "10":
		return "high"
	default:
		return "medium"
	}
}

func searchContextSizeToMaxUses(size string) int {
	switch size {
	case "low":
		return 1
	case "high":
		return 10
	default:
		return 5
	}
}

func rawJSONString(s string) json.RawMessage {
	raw, _ := common.Marshal(s)
	return raw
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
