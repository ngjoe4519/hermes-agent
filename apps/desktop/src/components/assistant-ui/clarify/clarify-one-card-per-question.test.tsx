import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { pendingClarifyToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-clarify'
import { I18nProvider } from '@/i18n'
import { type ChatMessage, restorePendingClarifyToolCall } from '@/lib/chat-messages'
import { $settledClarifyResults, type ClarifyRequest, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { answerParkedCard } from '@/store/parked-card-answers'
import { rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId } from '@/store/session'

import { ClarifyTool } from './index'

// A canonical gateway asks a multi-question clarify one card per question when the
// submitter did not declare `clarify_batch` (gateway/run_turn_runner.py::_clarify_callback_sync).
// Each card is a new request that re-arms the same tool row. These tests drive the seams
// production uses: the transcript re-arm `parkClarify` runs, the composer's
// `answerParkedCard`, and the card's own Skip.

const gatewayMocks = vi.hoisted(() => ({
  requestGatewayForAgent: vi.fn(async () => ({ ok: true }))
}))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent
}))

vi.mock('@assistant-ui/react', () => ({ useAuiState: () => true }))

beforeAll(() => {
  // jsdom has no Web Animations API; a demoted row's enter animation would throw.
  Element.prototype.animate ??= (() => ({ cancel: () => {}, finished: Promise.resolve() })) as never
})

afterEach(() => {
  cleanup()
  clearClarifyRequest()
  resetServerRequestsForTests()
  $activeSessionId.set(null)
  $gateway.set(null)
  $settledClarifyResults.set({})
  vi.clearAllMocks()
})

const TARGET = { choices: ['staging', 'production'], question: 'Which deployment target?' }
const REGION = { choices: ['us', 'eu'], question: 'Which region?' }

const card = (requestId: string, entry: typeof TARGET): ClarifyRequest => ({
  questions: [{ choices: entry.choices, multiSelect: false, qid: `${requestId}-q`, question: entry.question }],
  receivedAt: Date.now() / 1000,
  requestId,
  sessionId: 'session-1'
})

const tree = (ui: ReactNode) => (
  <I18nProvider configClient={null} initialLocale="en">
    {ui}
  </I18nProvider>
)

function rowProps(messages: ChatMessage[], toolCallId: string): ToolCallMessagePartProps {
  const part = messages[0].parts.find(entry => entry.type === 'tool-call' && entry.toolCallId === toolCallId)

  if (part?.type !== 'tool-call') {
    throw new Error(`no tool row ${toolCallId}`)
  }

  return {
    addResult: vi.fn(),
    args: part.args,
    argsText: JSON.stringify(part.args),
    isError: false,
    respondToApproval: vi.fn(),
    result: undefined,
    resume: vi.fn(),
    status: { type: 'running' },
    toolCallId,
    toolName: 'clarify',
    type: 'tool-call'
  } as ToolCallMessagePartProps
}

function clarifyCall(toolCallId: string, questions: (typeof TARGET)[]) {
  return { args: { questions }, toolCallId, toolName: 'clarify', type: 'tool-call' } as never
}

/** One assistant message whose clarify rows re-arm the way `parkClarify` does. */
function transcript(...calls: ReturnType<typeof clarifyCall>[]) {
  const request = vi.fn().mockResolvedValue({ ok: true })
  let messages: ChatMessage[] = [{ id: 'm1', parts: calls, pending: true, role: 'assistant' }]

  $activeSessionId.set('session-1')
  $gateway.set({ request } as never)

  return {
    add(call: ReturnType<typeof clarifyCall>) {
      messages = [{ ...messages[0], parts: [...messages[0].parts, call] }]
    },
    locks: () =>
      [...request.mock.calls, ...gatewayMocks.requestGatewayForAgent.mock.calls].filter(call =>
        (call as unknown[]).includes('clarify.lock')
      ),
    park(next: ClarifyRequest) {
      rememberServerRequest({ fail: vi.fn(), id: next.requestId, method: 'clarify', params: {}, respond: vi.fn() })
      setClarifyRequest(next)
      const projection = restorePendingClarifyToolCall(messages, pendingClarifyToolPayload(next))
      // Re-armed in place: a card never grows a second row.
      expect(projection.messages[0].parts).toHaveLength(messages[0].parts.length)
      messages = projection.messages
    },
    props: (toolCallId: string) => rowProps(messages, toolCallId)
  }
}

describe('a clarify asked one card per question', () => {
  it('shows the next card after the chat bar answered the first, and the chat bar answers that card', async () => {
    const chat = transcript(clarifyCall('call-1', [TARGET, REGION]))
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(<ClarifyTool {...chat.props('call-1')} />))

    act(() => {
      expect(answerParkedCard('session-1', 'staging')).toBe(true)
    })
    expect(view.container.querySelector('[data-clarify-settled]')).not.toBeNull()

    act(() => chat.park(card('prompt-2', REGION)))
    // The request can land before the row's re-armed args; the row stays a card meanwhile,
    // never a bare tool row.
    expect(view.container.querySelector('[data-clarify-settled], [data-clarify-batch]')).not.toBeNull()

    view.rerender(tree(<ClarifyTool {...chat.props('call-1')} />))
    expect(view.container.querySelector('[data-clarify-settled]')).toBeNull()
    expect(screen.getByText('Which region?')).toBeTruthy()
    expect(screen.getByRole('button', { name: /^.*eu/ }).hasAttribute('disabled')).toBe(false)

    await act(async () => {
      expect(answerParkedCard('session-1', 'eu')).toBe(true)
      await Promise.resolve()
    })
    expect(chat.locks().at(-1)).toEqual([
      'clarify.lock',
      { answer: 'eu', question_id: 'prompt-2-q', request_id: 'prompt-2' }
    ])
  })

  it('shows the next card after Skip on the first', () => {
    const chat = transcript(clarifyCall('call-1', [TARGET, REGION]))
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(<ClarifyTool {...chat.props('call-1')} />))

    fireEvent.click(screen.getByRole('button', { name: 'Skip' }))
    expect(view.container.querySelector('[data-clarify-settled]')).not.toBeNull()

    act(() => chat.park(card('prompt-2', REGION)))
    view.rerender(tree(<ClarifyTool {...chat.props('call-1')} />))

    expect(screen.getByText('Which region?')).toBeTruthy()
    expect(screen.getByRole('button', { name: /^.*eu/ }).hasAttribute('disabled')).toBe(false)
  })

  it("keeps an earlier call's answered row settled when a later call asks", () => {
    // The earlier row never got its tool.complete (a lost frame, or a duplicate row a re-delivered
    // activate left behind), so it is still a live clarify row when the later call's question lands.
    const chat = transcript(clarifyCall('call-1', [TARGET]))
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(<ClarifyTool {...chat.props('call-1')} />))

    act(() => {
      expect(answerParkedCard('session-1', 'staging')).toBe(true)
    })

    chat.add(clarifyCall('call-2', [REGION]))
    act(() => chat.park(card('prompt-2', REGION)))
    view.rerender(
      tree(
        <>
          <ClarifyTool {...chat.props('call-1')} />
          <ClarifyTool {...chat.props('call-2')} />
        </>
      )
    )

    expect(view.container.querySelectorAll('[data-clarify-settled]')).toHaveLength(1)
    expect(screen.getAllByText('Which region?')).toHaveLength(1)
    expect(screen.getByRole('button', { name: /^.*eu/ }).hasAttribute('disabled')).toBe(false)
  })
})
