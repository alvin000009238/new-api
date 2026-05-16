package openaicompat

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestClaudeMessagesRequestToResponsesRequest(t *testing.T) {
	toolInput := map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{"city": map[string]interface{}{"type": "string"}},
	}
	req := &dto.ClaudeRequest{
		Model:       "claude-3-5-sonnet",
		System:      "be concise",
		MaxTokens:   common.GetPointer[uint](0),
		Stream:      common.GetPointer(false),
		Temperature: common.GetPointer(0.0),
		Tools: []any{dto.Tool{
			Name:        "weather",
			Description: "Get weather",
			InputSchema: toolInput,
		}},
		Messages: []dto.ClaudeMessage{
			{
				Role: "user",
				Content: []dto.ClaudeMediaMessage{
					{Type: "text", Text: common.GetPointer("hello")},
					{
						Type: "image",
						Source: &dto.ClaudeMessageSource{
							Type:      "base64",
							MediaType: "image/png",
							Data:      "aW1n",
						},
					},
				},
			},
			{
				Role: "assistant",
				Content: []dto.ClaudeMediaMessage{{
					Type:  "tool_use",
					Id:    "call_1",
					Name:  "weather",
					Input: map[string]any{"city": "Taipei"},
				}},
			},
			{
				Role: "user",
				Content: []dto.ClaudeMediaMessage{{
					Type:      "tool_result",
					ToolUseId: "call_1",
					Content:   "sunny",
				}},
			},
		},
	}

	converted, err := ClaudeMessagesRequestToResponsesRequest(req)
	require.NoError(t, err)
	require.Equal(t, "claude-3-5-sonnet", converted.Model)
	require.NotNil(t, converted.MaxOutputTokens)
	require.EqualValues(t, 0, *converted.MaxOutputTokens)
	require.NotNil(t, converted.Stream)
	require.False(t, *converted.Stream)
	require.NotNil(t, converted.Temperature)
	require.Equal(t, 0.0, *converted.Temperature)

	encoded, err := common.Marshal(converted)
	require.NoError(t, err)
	require.Equal(t, "be concise", gjson.GetBytes(encoded, "instructions").String())
	require.Equal(t, "input_image", gjson.GetBytes(encoded, "input.0.content.1.type").String())
	require.Equal(t, "function_call", gjson.GetBytes(encoded, "input.1.type").String())
	require.Equal(t, "call_1", gjson.GetBytes(encoded, "input.1.call_id").String())
	require.Equal(t, "function_call_output", gjson.GetBytes(encoded, "input.2.type").String())
	require.Equal(t, "function", gjson.GetBytes(encoded, "tools.0.type").String())
	require.True(t, gjson.GetBytes(encoded, "max_output_tokens").Exists())
	require.True(t, gjson.GetBytes(encoded, "stream").Exists())
	require.True(t, gjson.GetBytes(encoded, "temperature").Exists())
}

func TestOpenAIResponsesRequestToClaudeMessagesRequest(t *testing.T) {
	req := &dto.OpenAIResponsesRequest{
		Model:           "claude-3-5-sonnet",
		Input:           json.RawMessage(`[{"role":"user","content":[{"type":"input_text","text":"read this"},{"type":"input_file","filename":"spec.pdf","file_data":"JVBERi0xLjQK"}]},{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"},{"type":"function_call_output","call_id":"call_1","output":"ok"}]`),
		Instructions:    json.RawMessage(`"follow policy"`),
		MaxOutputTokens: common.GetPointer[uint](0),
		Stream:          common.GetPointer(false),
		TopP:            common.GetPointer(0.0),
		Tools:           json.RawMessage(`[{"type":"function","name":"lookup","parameters":{"type":"object"}},{"type":"web_search_preview","search_context_size":"low"}]`),
		ToolChoice:      json.RawMessage(`{"type":"function","name":"lookup"}`),
	}

	converted, err := OpenAIResponsesRequestToClaudeMessagesRequest(req)
	require.NoError(t, err)
	require.Equal(t, "follow policy", converted.System)
	require.NotNil(t, converted.MaxTokens)
	require.EqualValues(t, 0, *converted.MaxTokens)
	require.NotNil(t, converted.Stream)
	require.False(t, *converted.Stream)
	require.NotNil(t, converted.TopP)
	require.Equal(t, 0.0, *converted.TopP)
	require.Len(t, converted.Messages, 3)

	firstContent, ok := converted.Messages[0].Content.([]dto.ClaudeMediaMessage)
	require.True(t, ok)
	require.Len(t, firstContent, 2)
	require.Equal(t, "text", firstContent[0].Type)
	require.Equal(t, "document", firstContent[1].Type)
	require.NotNil(t, firstContent[1].Source)
	require.Equal(t, "application/pdf", firstContent[1].Source.MediaType)

	toolUse, ok := converted.Messages[1].Content.([]dto.ClaudeMediaMessage)
	require.True(t, ok)
	require.Equal(t, "tool_use", toolUse[0].Type)
	require.Equal(t, "lookup", toolUse[0].Name)

	tools := converted.GetTools()
	require.Len(t, tools, 2)
	require.IsType(t, dto.Tool{}, tools[0])
	require.IsType(t, dto.ClaudeWebSearchTool{}, tools[1])

	encoded, err := common.Marshal(converted)
	require.NoError(t, err)
	require.True(t, gjson.GetBytes(encoded, "max_tokens").Exists())
	require.True(t, gjson.GetBytes(encoded, "stream").Exists())
	require.True(t, gjson.GetBytes(encoded, "top_p").Exists())
	require.Equal(t, "tool", gjson.GetBytes(encoded, "tool_choice.type").String())
	require.Equal(t, "lookup", gjson.GetBytes(encoded, "tool_choice.name").String())
}

func TestOpenAIResponsesRequestToClaudeMessagesRequestRejectsUnsupportedTool(t *testing.T) {
	req := &dto.OpenAIResponsesRequest{
		Model: "claude-3-5-sonnet",
		Input: json.RawMessage(`"hi"`),
		Tools: json.RawMessage(`[{"type":"file_search"}]`),
	}

	_, err := OpenAIResponsesRequestToClaudeMessagesRequest(req)
	require.Error(t, err)
	require.True(t, errors.Is(err, ErrUnsupportedResponsesTool))
}

func TestClaudeResponseToOpenAIResponsesResponse(t *testing.T) {
	resp := &dto.ClaudeResponse{
		Id:    "msg_123",
		Type:  "message",
		Model: "claude-3-5-sonnet",
		Usage: &dto.ClaudeUsage{InputTokens: 10, CacheReadInputTokens: 2, CacheCreationInputTokens: 3, OutputTokens: 5},
		Content: []dto.ClaudeMediaMessage{
			{Type: "text", Text: common.GetPointer("hello")},
			{Type: "tool_use", Id: "call_1", Name: "lookup", Input: map[string]any{"q": "x"}},
		},
	}

	converted, usage, err := ClaudeResponseToOpenAIResponsesResponse(resp, "fallback", 123)
	require.NoError(t, err)
	require.Equal(t, "msg_123", converted.ID)
	require.Equal(t, "response", converted.Object)
	require.Equal(t, "message", converted.Output[0].Type)
	require.Equal(t, "hello", converted.Output[0].Content[0].Text)
	require.Equal(t, "function_call", converted.Output[1].Type)
	require.Equal(t, "lookup", converted.Output[1].Name)
	require.Equal(t, 15, usage.InputTokens)
	require.Equal(t, 5, usage.OutputTokens)
	require.Equal(t, "openai", usage.UsageSemantic)
	require.Equal(t, "anthropic", usage.UsageSource)
}

func TestClaudeToResponsesStreamConverter(t *testing.T) {
	converter := NewClaudeToResponsesStreamConverter("fallback", 123, "claude-3-5-sonnet")
	text := "he"
	more := "llo"
	idx := 0

	var eventTypes []string
	steps := []*dto.ClaudeResponse{
		{Type: "message_start", Message: &dto.ClaudeMediaMessage{Id: "msg_123", Model: "claude-3-5-sonnet"}},
		{Type: "content_block_start", Index: &idx, ContentBlock: &dto.ClaudeMediaMessage{Type: "text", Text: &text}},
		{Type: "content_block_delta", Index: &idx, Delta: &dto.ClaudeMediaMessage{Type: "text_delta", Text: &more}},
		{Type: "content_block_stop", Index: &idx},
		{Type: "message_delta", Usage: &dto.ClaudeUsage{InputTokens: 4, OutputTokens: 2}},
		{Type: "message_stop"},
	}
	for _, step := range steps {
		events, usage, err := converter.Convert(step)
		require.NoError(t, err)
		require.NotNil(t, usage)
		for _, event := range events {
			eventTypes = append(eventTypes, event.Type)
		}
	}

	require.Equal(t, []string{
		"response.created",
		dto.ResponsesOutputTypeItemAdded,
		"response.output_text.delta",
		dto.ResponsesOutputTypeItemDone,
		"response.completed",
	}, eventTypes)
	require.True(t, converter.Completed)
	require.Equal(t, 4, converter.Usage.InputTokens)
	require.Equal(t, 2, converter.Usage.OutputTokens)
}
