package service

import (
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/service/openaicompat"
)

type ClaudeToResponsesStreamConverter = openaicompat.ClaudeToResponsesStreamConverter

func ChatCompletionsRequestToResponsesRequest(req *dto.GeneralOpenAIRequest) (*dto.OpenAIResponsesRequest, error) {
	return openaicompat.ChatCompletionsRequestToResponsesRequest(req)
}

func ResponsesResponseToChatCompletionsResponse(resp *dto.OpenAIResponsesResponse, id string) (*dto.OpenAITextResponse, *dto.Usage, error) {
	return openaicompat.ResponsesResponseToChatCompletionsResponse(resp, id)
}

func ExtractOutputTextFromResponses(resp *dto.OpenAIResponsesResponse) string {
	return openaicompat.ExtractOutputTextFromResponses(resp)
}

func ClaudeMessagesRequestToResponsesRequest(req *dto.ClaudeRequest) (*dto.OpenAIResponsesRequest, error) {
	return openaicompat.ClaudeMessagesRequestToResponsesRequest(req)
}

func OpenAIResponsesRequestToClaudeMessagesRequest(req *dto.OpenAIResponsesRequest) (*dto.ClaudeRequest, error) {
	return openaicompat.OpenAIResponsesRequestToClaudeMessagesRequest(req)
}

func ClaudeResponseToOpenAIResponsesResponse(resp *dto.ClaudeResponse, fallbackID string, createdAt int64) (*dto.OpenAIResponsesResponse, *dto.Usage, error) {
	return openaicompat.ClaudeResponseToOpenAIResponsesResponse(resp, fallbackID, createdAt)
}

func OpenAIUsageFromClaudeUsage(usage *dto.ClaudeUsage) *dto.Usage {
	return openaicompat.OpenAIUsageFromClaudeUsage(usage)
}

func NewClaudeToResponsesStreamConverter(responseID string, createdAt int64, model string) *openaicompat.ClaudeToResponsesStreamConverter {
	return openaicompat.NewClaudeToResponsesStreamConverter(responseID, createdAt, model)
}
