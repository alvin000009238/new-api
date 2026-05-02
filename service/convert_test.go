package service

import (
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
)

func TestStreamResponseOpenAI2ClaudeFinishUsesFallbackUsage(t *testing.T) {
	info := &relaycommon.RelayInfo{
		ClaudeConvertInfo: &relaycommon.ClaudeConvertInfo{
			LastMessagesType: relaycommon.LastMessageTypeText,
			Usage: &dto.Usage{
				PromptTokens:     7,
				CompletionTokens: 3,
				TotalTokens:      10,
			},
		},
	}
	finishReason := "stop"
	resp := &dto.ChatCompletionsStreamResponse{
		Id:    "chatcmpl-test",
		Model: "gpt-5.5",
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{FinishReason: common.GetPointer(finishReason)},
		},
	}

	claudeResponses := StreamResponseOpenAI2Claude(resp, info)
	if len(claudeResponses) != 3 {
		t.Fatalf("expected content_block_stop, message_delta, message_stop; got %d responses: %#v", len(claudeResponses), claudeResponses)
	}
	if claudeResponses[0].Type != "content_block_stop" {
		t.Fatalf("expected first response to close content block, got %q", claudeResponses[0].Type)
	}
	if claudeResponses[1].Type != "message_delta" {
		t.Fatalf("expected second response to be message_delta, got %q", claudeResponses[1].Type)
	}
	if claudeResponses[1].Usage == nil || claudeResponses[1].Usage.InputTokens != 7 || claudeResponses[1].Usage.OutputTokens != 3 {
		t.Fatalf("expected fallback usage on message_delta, got %#v", claudeResponses[1].Usage)
	}
	if claudeResponses[2].Type != "message_stop" {
		t.Fatalf("expected final response to be message_stop, got %q", claudeResponses[2].Type)
	}
	if !info.ClaudeConvertInfo.Done {
		t.Fatal("expected conversion to be marked done")
	}
}
