/* Same-origin memory API client. No model, credentials or sample data are embedded. */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  let state=null, busy=false, retry=null, apiKey=null, example='add', supersedes=null;
  class ServiceError extends Error {
    constructor(message,status,code) { super(message);this.status=status;this.code=code; }
  }
  function requestId() { return crypto.randomUUID(); }
  async function api(path,method='GET',body) {
    const controller=new AbortController(), timer=setTimeout(() => controller.abort(),45000);
    try {
      const response=await fetch(`/api/v1/${path}`,{method,credentials:'same-origin',headers:body === undefined ? {'Accept':'application/json'} : {'Accept':'application/json','Content-Type':'application/json'},...(body === undefined ? {} : {body:JSON.stringify(body)}),signal:controller.signal,cache:'no-store'});
      let data;
      try { data=await response.json(); }
      catch { throw new ServiceError('The service returned an unreadable response.',response.status,'invalid_response'); }
      if(!response.ok) throw new ServiceError(data.error?.message || 'The request could not be completed.',response.status,data.error?.code);
      return data;
    } finally { clearTimeout(timer); }
  }
  function node(tag,className,text) {
    const result=document.createElement(tag);if(className) result.className=className;
    if(text !== undefined) result.textContent=String(text);return result;
  }
  function status(message,{error=false,retryAction=null,dismiss=true}={}) {
    $('status-panel').hidden=!message;$('status-panel').dataset.error=String(error);
    $('status-message').textContent=message;retry=retryAction;
    $('retry-action').hidden=!retryAction;$('dismiss-action').hidden=!dismiss;controls();
  }
  function controls() {
    const locked=busy || Boolean(retry);
    for(const id of ['start-trial','remember-button','recall-button','new-conversation','delete-trial','rotate-api-key']) $(id).disabled=locked || (id !== 'start-trial' && !state);
    for(const id of ['remember-input','recall-query']) $(id).disabled=locked;
    for(const id of ['retry-action','dismiss-action','confirm-delete']) $(id).disabled=busy;
    document.querySelectorAll('[data-example],.update-memory').forEach(button => button.disabled=locked);
    $('workspace').setAttribute('aria-busy',String(busy));
    $('start-trial').textContent=busy && !state ? 'Opening your workspace…' : 'Open a memory workspace →';
  }
  function showKey(key) {
    apiKey=key || null;$('api-key-section').hidden=!apiKey;$('key-missing').hidden=Boolean(apiKey);
    $('api-key').value=apiKey || '';$('api-key').type='password';$('show-api-key').textContent='Show';
  }
  function renderExample() {
    const endpoint=`${location.origin}/api/v1/${example === 'add' ? 'observations' : 'recall'}`;
    const payload=example === 'add' ? '{"request_id":"replace-with-a-unique-request-id","text":"My project is called Night Orchard."}' : '{"query":"What is my current project?"}';
    $('api-example').textContent=[`curl '${endpoint}' \\`, '  -H "Authorization: Bearer $LICORICE_API_KEY" \\', "  -H 'Content-Type: application/json' \\", `  --data '${payload}'`].join('\n');
    $('example-add').setAttribute('aria-pressed',String(example === 'add'));
    $('example-search').setAttribute('aria-pressed',String(example === 'search'));
  }
  function renderContext(context) {
    const usage=context.usage || {}, ids=Array.isArray(context.source_ids) ? context.source_ids : [];
    const budget=Number(usage.budget_tokens ?? state?.limits?.context_tokens ?? 2000);
    $('recall-empty').hidden=true;$('context-result').hidden=false;
    $('context-usage').textContent=usage.tokens === undefined ? `Budget: ${budget.toLocaleString()} tokens` : `${Number(usage.tokens).toLocaleString()} / ${budget.toLocaleString()} tokens`;
    $('context-sources').textContent=`${ids.length} ${ids.length === 1 ? 'source' : 'sources'} selected`;
    $('context-text').textContent=context.text || 'No matching evidence was selected.';
    $('context-text').dataset.sourceIds=JSON.stringify(ids);
  }
  function renderState(next) {
    state=next;$('welcome').hidden=true;$('workspace').hidden=false;
    const expiry=new Date(state.expires_at);
    $('workspace-expiry').textContent=Number.isNaN(expiry.getTime()) ? 'Temporary workspace. Export what you want to keep.' : `Expires ${expiry.toLocaleString(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})}. Export what you want to keep.`;
    $('retrieval-note').textContent=state.retrieval?.mode === 'hybrid_semantic' ? 'Combines meaning-based retrieval with word matching using a local embedding model. Each report is represented by its first 256 model tokens; selected source text stays whole. Results are candidate reports, not proof of an answer; automatic extraction is off.' : 'This preview matches words in stored reports. Automatic fact extraction and semantic retrieval are not enabled.';
    const observations=state.memory?.observations || [], replaced=new Map(state.supersessions || []);
    $('memory-count').textContent=String(observations.length);
    $('memory-revision').textContent=`Revision ${state.memory?.revision ?? 0}`;
    $('memory-list').replaceChildren();$('memory-empty').hidden=observations.length>0;
    for(const stored of [...observations].reverse()) {
      const observation=stored.observation || stored, source=String(stored.source_id || 'source');
      const article=node('article',`memory-entry${replaced.has(source) ? ' superseded' : ''}`);
      const row=node('div','source-row');
      row.append(node('span','source-kind',replaced.has(source) ? 'Earlier version' : 'Your report'),node('span','source-id',source.length>21 ? `${source.slice(0,18)}…` : source));
      row.lastChild.title=source;article.append(row,node('p','',observation.text || ''));
      const details=node('details','source-details'), summary=node('summary','','Source & history');
      details.append(summary,node('p','tiny quiet',`Source: ${source}`),node('p','tiny quiet',`Origin: ${observation.origin || 'reported'} · Assurance: ${stored.assurance || 'reported'}${stored.recorded_revision === undefined ? '' : ` · Revision: ${stored.recorded_revision}`}`));
      if(replaced.has(source)) details.append(node('p','tiny quiet',`Superseded by ${replaced.get(source)}. Retained for history; excluded from current recall.`));
      article.append(details);
      if(!replaced.has(source)) {
        const update=node('button','update-memory','Update this memory');update.type='button';
        update.addEventListener('click',() => {supersedes=source;$('remember-input').value=observation.text || '';$('add-title').textContent='Update a memory. Keep the history.';$('remember-button').textContent='Save update';$('remember-input').focus();$('remember-input').scrollIntoView({behavior:'smooth',block:'center'});status('Editing a replacement. The earlier source stays in history and is excluded from current recall after this update.');});
        article.append(update);
      }
      $('memory-list').append(article);
    }
    const session=String(state.conversation_id || '');$('session-label').textContent=`Current session: ${session}`;
    $('remember-input').maxLength=state.limits?.message_bytes || 4000;
    renderExample();controls();
  }
  async function refresh() { const next=await api('workspace');renderState(next);return next; }
  function errorText(error) {
    if(error.name === 'AbortError') return 'The request timed out and may still have reached the service.';
    if(error instanceof ServiceError) return error.message;
    return 'The connection was interrupted. The request may still have reached the service.';
  }
  async function runAction({path,method='POST',body,label,onSuccess,uncertain=true,retryable=true}) {
    if(busy) return;
    const attempt=() => runAction({path,method,body,label,onSuccess,uncertain,retryable});
    busy=true;status(label,{dismiss:false});
    try {const result=await api(path,method,body);await onSuccess(result);status('');}
    catch(error) {
      if(error.status === 410) {resetView();status('This temporary workspace has expired. Its memory is no longer available through this workspace. You can open a new one.',{error:true});}
      else if(error.status === 401 || error.status === 404 && path === 'workspace') {resetView();status('This temporary workspace is no longer available. You can open a new one.',{error:true});}
      else {
        const canRetry=retryable && !(error.status >= 400 && error.status < 500 && ![408,429].includes(error.status));
        const note=canRetry && uncertain ? ' Retry sends the identical request ID and payload. Nothing is resent automatically.' : '';
        if(path === 'keys') showKey(null);
        const keyNote=path === 'keys' ? ' The displayed key has been cleared. Use Replace API key again to deliberately create a new key and revoke any previous one. Your browser still has workspace access.' : '';
        status(`${errorText(error)}${note}${keyNote}`,{error:true,retryAction:canRetry ? attempt : null});
      }
    } finally {busy=false;controls();}
  }
  function resetView() {
    state=null;supersedes=null;showKey(null);$('workspace').hidden=true;$('welcome').hidden=false;
    $('remember-input').value='';$('recall-query').value='';$('context-result').hidden=true;$('recall-empty').hidden=false;
    $('context-text').textContent='';$('memory-list').replaceChildren();resetUpdate();controls();
  }
  function resetUpdate() {supersedes=null;$('add-title').textContent='Leave something worth remembering.';$('remember-button').textContent='Add memory +';}
  function validateText(input) {
    const text=input.value.trim();if(!text) {input.focus();return null;}
    const max=state?.limits?.message_bytes || 4000;
    if(new TextEncoder().encode(text).length>max) {status(`Please keep this under ${max.toLocaleString()} UTF-8 bytes. Some characters use more than one byte.`,{error:true});input.focus();return null;}
    return text;
  }
  $('start-trial').addEventListener('click',() => runAction({path:'workspaces',body:{},label:'Opening your temporary workspace…',uncertain:false,onSuccess:async result => {showKey(result.api_key);await refresh();$('remember-input').focus();}}));
  $('remember-form').addEventListener('submit',event => {
    event.preventDefault();if(!state || busy || retry) return;const text=validateText($('remember-input'));if(!text) return;
    const body={request_id:requestId(),text,...(supersedes ? {supersedes} : {})};
    runAction({path:'observations',body,label:supersedes ? 'Saving your update and its source history…' : 'Storing your attributed observation…',onSuccess:async () => {await refresh();$('remember-input').value='';resetUpdate();$('remember-input').focus();}});
  });
  $('remember-input').addEventListener('keydown',event => {if(event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {event.preventDefault();$('remember-form').requestSubmit();}});
  document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click',() => {resetUpdate();$('remember-input').value=button.dataset.example;$('remember-input').focus();}));
  $('recall-form').addEventListener('submit',event => {
    event.preventDefault();if(!state || busy || retry) return;const query=validateText($('recall-query'));if(!query) return;
    runAction({path:'recall',body:{query},label:'Selecting evidence from your memory…',uncertain:false,onSuccess:async result => renderContext(result)});
  });
  $('new-conversation').addEventListener('click',() => runAction({path:'sessions',body:{request_id:requestId()},label:'Starting a fresh session. Your memory stays…',onSuccess:async () => {await refresh();$('context-result').hidden=true;$('recall-empty').hidden=false;$('context-text').textContent='';$('recall-query').focus();}}));
  $('delete-trial').addEventListener('click',() => $('delete-dialog').showModal());
  $('cancel-delete').addEventListener('click',() => $('delete-dialog').close());
  $('confirm-delete').addEventListener('click',() => {$('delete-dialog').close();runAction({path:'workspace',method:'DELETE',label:'Deleting your workspace memory and history…',onSuccess:async () => {resetView();$('start-trial').focus();}});});
  $('retry-action').addEventListener('click',() => {const action=retry;if(action && !busy) {retry=null;action();}});
  $('dismiss-action').addEventListener('click',() => {const hadRetry=Boolean(retry);status('');if(hadRetry) status('Retry dismissed. A submitted request may still have completed; reload before sending the same content again.');});
  $('show-api-key').addEventListener('click',() => {const show=$('api-key').type === 'password';$('api-key').type=show ? 'text' : 'password';$('show-api-key').textContent=show ? 'Hide' : 'Show';});
  async function copy(text) {try {await navigator.clipboard.writeText(text);status('Copied to clipboard.');}catch {status('Clipboard access is unavailable. Select and copy the text directly.',{error:true});}}
  $('copy-api-key').addEventListener('click',() => copy(apiKey || ''));
  $('copy-api-example').addEventListener('click',() => copy($('api-example').textContent));
  $('example-add').addEventListener('click',() => {example='add';renderExample();});
  $('example-search').addEventListener('click',() => {example='search';renderExample();});
  $('rotate-api-key').addEventListener('click',() => runAction({path:'keys',body:{},retryable:false,label:'Replacing this workspace’s API key…',onSuccess:async result => {showKey(result.api_key);}}));
  busy=true;controls();
  api('workspace').then(renderState).catch(error => {
    if(error.status === 410) {resetView();status('This temporary workspace has expired. Its memory is no longer available through this workspace. You can open a new one.',{error:true});}
    else if(![401,404].includes(error.status)) status(`Could not check for an existing workspace. ${errorText(error)}`,{error:true});
  }).finally(() => {busy=false;controls();});
})();
