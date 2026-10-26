import { expect, mock, test } from 'claude-code/testing'

test('replayed compaction callbacks keep one request identity as the clock advances', async ($, on) => {
  mock.env(on, { LICORICE_BEING_AGENT: 'michelle', LICORICE_BEING_STORE: '/private/native.sqlite', LICORICE_BEING_BIN: '/native/licorice-being' })
  on('session.id', () => ({ value: 'session-1' }))
  let clock = 1792595000000
  on('clock.now', () => ({ value: clock++ }))
  const requests: any[] = []
  on('process.run', ($, e) => {
    requests.push(JSON.parse(e.init.stdin))
    return { value: { exitCode: 0, stdout: JSON.stringify({ ok: true, result: { revision: 4, replayed: requests.length > 1 } }), stderr: '' } }
  })
  const messages = [{ role: 'user', text: 'Keep this PRIVATE_COMPACTION_WITNESS out of durable boundary metadata.', toolUses: [] }]
  on('session.compact', () => ({ messages }))
  await $.session.compact({ trigger: 'manual', instructions: 'Preserve the work.', messages })
  await $.session.compact({ trigger: 'manual', instructions: 'Preserve the work.', messages })
  expect(requests.length).toBe(2)
  expect(requests[0].request_id).toBe(requests[1].request_id)
  expect(JSON.stringify(requests)).not.toContain('PRIVATE_COMPACTION_WITNESS')
})

test('episode recall is explicitly enabled inside the native context request, without adoption', async ($, on) => {
  mock.env(on, { LICORICE_BEING_AGENT: 'michelle', LICORICE_BEING_STORE: '/private/native.sqlite', LICORICE_BEING_BIN: '/native/licorice-being', LICORICE_BEING_EPISODES: '1' })
  let request: any = null
  on('clock.now', () => ({ value: 1792595000000 }))
  on('process.run', ($, e) => {
    request = JSON.parse(e.init.stdin)
    return { value: { exitCode: 0, stdout: JSON.stringify({ ok: true, result: { text: 'Attributed unadopted observation.', usage: { tokenizer: 'cl100k_base:ordinary', tokens: 8, budget_tokens: 2000 } } }), stderr: '' } }
  })
  on('prompt.submit', ($, e) => ({ text: e.text, context: e.context }))
  const result = await $.prompt.submit({ text: 'What did we observe?' })
  expect(request.include_episodes).toBe(true)
  expect(request.query).toBe('What did we observe?')
  expect(result.context).toEqual(['Attributed unadopted observation.'])
})

test('the status command is handled by the actual native event matcher', async ($, on) => {
  mock.env(on, {})
  on('command.run', () => ({ text: 'host fallback' }))
  const result = await $.command.run({ command: 'being-memory', args: '' })
  expect(result.text).toContain('LICORICE_BEING_AGENT')
})

test('native prompt event preserves the prompt and attaches bounded attributed context', async ($, on) => {
  mock.env(on, { LICORICE_BEING_BACKEND: 'prototype', LICORICE_BEING_AGENT: 'michelle', LICORICE_BEING_STORE: '/private/memory.sqlite' })
  on('process.run', () => ({ value: { exitCode: 0, stdout: JSON.stringify({ ok: true, text: 'Attributed memory data.' }), stderr: '' } }))
  on('prompt.submit', ($, e) => ({ text: e.text, context: e.context }))
  const result = await $.prompt.submit({ text: 'Resume our work.' })
  expect(result.text).toBe('Resume our work.')
  expect(result.context).toEqual(['Attributed memory data.'])
})

test('unavailable memory produces a diagnostic without repeating or dropping the native prompt', async ($, on) => {
  mock.env(on, { LICORICE_BEING_BACKEND: 'prototype', LICORICE_BEING_AGENT: 'michelle', LICORICE_BEING_STORE: '/private/memory.sqlite' })
  let diagnostics = 0
  let forwards = 0
  on('process.run', () => ({ value: { exitCode: 1, stdout: '{}', stderr: 'private diagnostic' } }))
  on('ui.log', () => { diagnostics++; return { value: undefined } })
  on('prompt.submit', ($, e) => { forwards++; return { text: e.text } })
  const result = await $.prompt.submit({ text: 'Continue this task.' })
  expect(result.text).toBe('Continue this task.')
  expect(forwards).toBe(1)
  expect(diagnostics).toBe(1)
})

test('native backend uses the actual CLI envelope and token receipt rather than the prototype byte limit', async ($, on) => {
  mock.env(on, { LICORICE_BEING_AGENT: 'michelle', LICORICE_BEING_STORE: '/private/native.sqlite', LICORICE_BEING_BIN: '/native/licorice-being' })
  on('clock.now', () => ({ value: 1792595000000 }))
  let argv: readonly string[] = []
  let query = ''
  const text = 'ordinary '.repeat(400)
  on('process.run', ($, e) => {
    argv = e.argv
    query = JSON.parse(e.init.stdin).query
    return { value: { exitCode: 0, stdout: JSON.stringify({ ok: true, result: { text, usage: { tokenizer: 'cl100k_base:ordinary', tokens: 402, budget_tokens: 2000 } } }), stderr: '' } }
  })
  on('prompt.submit', ($, e) => ({ text: e.text, context: e.context }))
  const result = await $.prompt.submit({ text: 'Resume the work.' })
  expect(argv[0]).toBe('/native/licorice-being')
  expect(argv[1]).toBe('context')
  expect(query).toBe('Resume the work.')
  expect(result.context).toEqual([text])
})

test('native turn completion writes a harness boundary and keeps the completed response', async ($, on) => {
  mock.env(on, { LICORICE_BEING_AGENT: 'michelle', LICORICE_BEING_STORE: '/private/native.sqlite', LICORICE_BEING_BIN: '/native/licorice-being' })
  on('session.id', () => ({ value: 'session-1' }))
  let request: any = null
  let role = ''
  on('process.run', ($, e) => {
    request = JSON.parse(e.init.stdin)
    role = e.argv[e.argv.indexOf('--role') + 1]
    return { value: { exitCode: 0, stdout: JSON.stringify({ ok: true, result: { revision: 3, replayed: false } }), stderr: '' } }
  })
  on('turn.complete', () => ({ text: 'Completed response.' }))
  const result = await $.turn.complete({ turnId: 'turn-1', answer: 'Completed response.' })
  expect(result.text).toBe('Completed response.')
  expect(role).toBe('harness')
  expect(request.operation.kind).toBe('boundary')
  expect(request.operation.payload).toEqual({ turn: 'turn-1', phase: 'completed' })
})

test('native compaction records its boundary and preserves the compacted messages', async ($, on) => {
  mock.env(on, { LICORICE_BEING_AGENT: 'michelle', LICORICE_BEING_STORE: '/private/native.sqlite', LICORICE_BEING_BIN: '/native/licorice-being' })
  on('session.id', () => ({ value: 'session-1' }))
  on('clock.now', () => ({ value: 1792595000000 }))
  let request: any = null
  on('process.run', ($, e) => {
    request = JSON.parse(e.init.stdin)
    return { value: { exitCode: 0, stdout: JSON.stringify({ ok: true, result: { revision: 4, replayed: false } }), stderr: '' } }
  })
  const messages = [{ role: 'user', text: 'A compacted summary preserving our working agreement.', toolUses: [] }]
  on('session.compact', () => ({ messages }))
  const result = await $.session.compact({ trigger: 'manual', instructions: 'Preserve our commitments.', messages })
  expect(result.messages[0].text).toBe('A compacted summary preserving our working agreement.')
  expect(request.operation.kind).toBe('boundary')
  expect(request.operation.payload.phase).toBe('compacting')
})
