import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { overlayConcurrentMessageChanges } from './utils'

describe('overlayConcurrentMessageChanges tool calls', () => {
  // A re-delivered open_requests marks the cached activation row "changed";
  // the raw concat used to carry the same clarify call twice, interleaved
  // with reasoning that ran between the copies.
  it('does not duplicate a tool call the re-fetched row already carries when merging the prefix', () => {
    const authoritative = [
      {
        id: 'assistant-stream-activation',
        role: 'assistant',
        parts: [
          { type: 'text', text: 'partial A' },
          { type: 'tool-call', toolCallId: 'call-clarify', toolName: 'clarify' }
        ],
        pending: true
      }
    ] as ChatMessage[]

    const current = [
      {
        id: 'assistant-stream-runtime',
        role: 'assistant',
        parts: [
          { type: 'tool-call', toolCallId: 'call-clarify', toolName: 'clarify' },
          { type: 'reasoning', text: 'mid reasoning' },
          { type: 'text', text: ' + delta B' }
        ],
        pending: true
      }
    ] as ChatMessage[]

    const overlaid = overlayConcurrentMessageChanges(authoritative, [], current)

    expect(overlaid.map(message => message.id)).toEqual(['assistant-stream-runtime'])
    const toolCallIds = overlaid[0].parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : []))
    expect(toolCallIds).toEqual(['call-clarify'])
    // The activation side still contributes what the fresh row lacks.
    expect(overlaid[0].parts[0]).toEqual({ type: 'text', text: 'partial A' })
  })
})
