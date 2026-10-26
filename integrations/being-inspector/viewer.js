'use strict';
(() => {
  const bytes = Uint8Array.from(atob('__LICORICE_DATA_BASE64__'), c => c.charCodeAt(0));
  const data = JSON.parse(new TextDecoder().decode(bytes));
  const $ = id => document.getElementById(id);
  const pretty = value => JSON.stringify(value, null, 2);
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  };
  const s = data.snapshot, c = data.context.continuity;
  $('identity').textContent = data.agent_id + ' · revision ' + data.revision + ' · ' +
    (data.relationship_id ? 'global + ' + data.relationship_id : 'global scope');
  const selected = new Set(c.source_ids || []);
  const stats = [[s.claims.length, 'Standing judgments'], [data.observations.length, 'Attributed observations'],
    [s.commitments.filter(x => x.status === 'active').length, 'Active commitments'],
    [c.omitted_unit_ids.length, 'Continuity units omitted']];
  for (const [count, label] of stats) {
    const box = node('div', undefined, 'stat');
    box.append(node('strong', count), node('span', label)); $('overview').append(box);
  }
  $('tokens').textContent = data.context.usage.tokens + ' / ' + data.context.usage.budget_tokens + ' tokens';
  $('query').textContent = 'Query: ' + data.query;
  $('packet').textContent = data.context.text;
  $('selection').textContent = pretty({selected_claim_ids: c.selected_claim_ids,
    omitted_claim_ids: c.omitted_claim_ids, selected_unit_ids: c.selected_unit_ids,
    omitted_unit_ids: c.omitted_unit_ids, source_ids: c.source_ids,
    episode_search: c.episode_search, usage: data.context.usage});
  for (const [key, value] of Object.entries(data.limits)) $('limits').append(node('p', key + ': ' + value));
  const sections = {
    Judgments: s.claims, Sources: data.observations, Commitments: s.commitments,
    History: data.history, Relationships: s.relationships,
    Ontology: [...s.concepts, ...s.hypotheses, ...s.links]
  };
  let current = 'Judgments', sourceFilter = null;
  const buttons = new Map();
  function showSource(id) {
    current = 'Sources'; sourceFilter = id; $('search').value = ''; render();
    $('records').scrollIntoView({block: 'start', behavior: 'auto'});
  }
  function sources(card, ids) {
    if (!ids || !ids.length) return;
    const list = node('div', undefined, 'sources'); list.append(node('span', 'Sources: ', 'muted'));
    for (const id of ids) {
      const button = node('button', id); button.type = 'button';
      button.addEventListener('click', () => showSource(id)); list.append(button);
    }
    card.append(list);
  }
  function card(record) {
    const box = node('article', undefined, 'card');
    const meta = node('div', undefined, 'meta'); let title, body, reason, ids;
    if (current === 'Judgments') {
      title = record.proposal.statement; body = record.proposal.subject;
      reason = record.reason; ids = record.proposal.source_ids;
      for (const label of [record.stance, record.proposal.category, 'revision ' + record.revision]) meta.append(node('span', label, 'badge'));
      meta.append(node('span', 'credence ' + (record.credence === null ? 'unknown' : record.credence + ' (uncalibrated)'), 'badge'));
    } else if (current === 'Sources') {
      title = record.source_id; body = record.observation.text;
      for (const label of [record.observation.origin, record.assurance,
        selected.has(record.source_id) ? 'included in continuity' : 'outside continuity packet']) meta.append(node('span', label, 'badge'));
      reason = 'Subject: ' + record.observation.subject + ' · Source: ' + pretty(record.observation.source);
    } else if (current === 'Commitments') {
      title = record.definition.next_action; body = 'Counterparty: ' + record.definition.counterparty;
      reason = record.reason; ids = record.source_ids;
      meta.append(node('span', record.status, 'badge'));
    } else if (current === 'History') {
      const op = record.request.operation;
      title = 'Revision ' + record.revision + ' · ' + op.kind;
      body = op.payload.reason || op.payload.text || op.payload.statement || pretty(op.payload);
      reason = record.actor.principal + ' · ' + record.actor.role + ' · ' + record.actor.session;
      ids = op.payload.source_ids;
    } else {
      const d = record.definition;
      title = d.label || d.relationship_id || d.link_id;
      body = d.description || d.definition || pretty(d);
      reason = d.reason; ids = d.source_ids;
    }
    box.append(meta, node('h3', title), node('p', body, 'body'));
    if (reason) box.append(node('p', reason, 'reason body'));
    sources(box, ids);
    const raw = node('details'); raw.append(node('summary', 'Full attributed record'), node('pre', pretty(record))); box.append(raw);
    return box;
  }
  function render() {
    for (const [name, button] of buttons) button.setAttribute('aria-pressed', String(name === current));
    const query = $('search').value.toLocaleLowerCase();
    const items = sections[current].filter(record => (!sourceFilter || record.source_id === sourceFilter) &&
      pretty(record).toLocaleLowerCase().includes(query));
    $('records').replaceChildren();
    $('result-count').textContent = items.length + ' of ' + sections[current].length + ' ' + current.toLowerCase() +
      (sourceFilter ? ' · selected source' : '');
    if (!items.length) $('records').append(node('p', sourceFilter ? 'Source unavailable in this scoped snapshot.' : 'No matching records in this snapshot.', 'muted'));
    for (const item of (current === 'History' ? [...items].reverse() : items)) $('records').append(card(item));
  }
  for (const name of Object.keys(sections)) {
    const button = node('button', name + ' (' + sections[name].length + ')'); button.type = 'button';
    button.addEventListener('click', () => {current = name; sourceFilter = null; render();});
    buttons.set(name, button); $('tabs').append(button);
  }
  $('search').addEventListener('input', () => {sourceFilter = null; render();});
  render();
})();
