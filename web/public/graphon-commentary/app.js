const $ = id => document.getElementById(id);
const video = $('video');
let run, inspect = false;

function words(text, source) {
  const fragment = document.createDocumentFragment();
  const tokens = [...text.matchAll(/[\p{L}\p{N}]+(?:[’']s)?/gu)];
  const normalize = word => word.toLowerCase().replace(/[’']s$/, '');
  const evidence = (source.match(/[\p{L}\p{N}]+(?:[’']s)?/gu) || []).map(normalize);
  const matched = new Set();
  for (let i = 0; i < tokens.length; i++) for (let j = 0; j < evidence.length; j++) {
    let n = 0;
    while (i + n < tokens.length && normalize(tokens[i + n][0]) === evidence[j + n]) n++;
    if (n >= 2) for (let k = i; k < i + n; k++) matched.add(k);
  }
  let end = 0;
  tokens.forEach((token, i) => {
    fragment.append(text.slice(end, token.index));
    if (matched.has(i)) { const mark = document.createElement('mark'); mark.textContent = token[0]; fragment.append(mark); }
    else fragment.append(token[0]);
    end = token.index + token[0].length;
  });
  fragment.append(text.slice(end));
  return fragment;
}

function render() {
  if (!run) return;
  const time = video.currentTime;
  const event = inspect || time >= 4;
  const retrieved = inspect || time >= run.playback.retrieved_at_s;
  const handed = inspect || time >= run.playback.handoff_at_s;
  const output = inspect || time >= run.playback.output_at_s;
  $('event').textContent = event ? run.event : 'Play to the goal.';
  $('event').parentElement.classList.toggle('active', event);
  $('query').textContent = event ? run.query : 'Waiting for the goal.';
  $('query').parentElement.classList.toggle('active', event);
  $('timing').textContent = retrieved ? `${(run.retrieval.search_elapsed_ms / 1000).toFixed(2)} s search` : '';
  $('source').hidden = !retrieved;
  $('retrieval-empty').hidden = retrieved;
  $('retrieval-empty').textContent = event ? 'Searching the saved research…' : 'The relevant passage will appear here.';
  $('input-event').textContent = handed ? run.writer_input.event : 'Waiting for evidence.';
  $('input-event').classList.toggle('active', handed);
  $('input-passage').hidden = !handed;
  $('output').classList.toggle('ready', output);
  $('output').replaceChildren(output ? words(run.commentary.text, run.selected.text) : 'The background becomes part of the call.');
  const current = run.delivery.find(line => time >= line.at_s && time < line.end_s);
  $('spoken').textContent = inspect ? '' : current ? `${current.voice === 'caller' ? 'Caller' : 'Analyst'}: ${current.text}` : '';
  $('voice-state').textContent = !inspect && !video.paused && current?.voice === 'analyst' ? 'Speaking' : '';
  $('status').textContent = inspect ? 'Full recorded run' : video.ended ? 'Replay complete' : !video.paused ? 'Playing the saved run' : time > 0 ? 'Paused' : 'Ready to play';
  $('replay').textContent = time > 0 || inspect ? 'Replay the demo' : 'Play the demo';
}

async function init() {
  const response = await fetch('/graphon-commentary/run.json');
  if (!response.ok) throw Error('The saved run could not be loaded. Reload to try again.');
  run = await response.json();
  if (!run.selected.text || !run.retrieval.read.results.some(r => r.text?.includes(run.selected.text))) throw Error('The recorded source passage could not be verified.');
  $('source-title').textContent = new URL(run.selected.url).hostname === 'observer.co.uk' ? 'Dan Ballard interview · The Observer' : run.selected.title;
  $('source-title').href = run.selected.url;
  $('source-date').textContent = `Published ${run.selected.published_at.slice(0, 10)} · archived excerpt`;
  $('passage').textContent = run.selected.text;
  $('input-passage').textContent = run.writer_input.fact.text;
  $('full-passage').textContent = run.selected.retrieved_text;
  for (const hit of run.retrieval.search.results) {
    const doc = run.sources.documents.find(item => item.key === hit.key);
    const p = document.createElement('p');
    const a = document.createElement('a');
    a.href = doc.url; a.target = '_blank'; a.rel = 'noreferrer';
    a.textContent = `${hit.rank}. ${doc.title}${hit.key === run.selected.span.key ? ' (selected)' : ''}`;
    p.append(a); $('results').append(p);
  }
  for (const doc of run.sources.documents) {
    const link = document.createElement('a'); link.textContent = doc.title; link.href = doc.url; link.target = '_blank'; link.rel = 'noreferrer';
    const date = document.createElement('span'); date.textContent = `${doc.published_at.slice(0, 10)} · ${doc.key}`;
    $('archive').append(link, date);
  }
  const note = document.createElement('p');
  note.textContent = 'Graphon really searched two archived pre-match excerpts. A selector chose a verbatim passage; a writer received only that passage and the reviewed goal event. This page replays the captured result and prepared speech timing. Playback makes no API calls and does not measure live latency.';
  const dl = document.createElement('dl');
  for (const [key, value] of Object.entries({ 'Workspace': 'commentart', 'Search': `${run.retrieval.search_elapsed_ms} ms`, 'Passage read': `${run.retrieval.read_elapsed_ms} ms`, 'Source version': run.selected.span.versionId, 'Models': 'GPT-5.4 Mini; ElevenLabs Flash v2.5', 'Model token cost': `$${run.budget.reported_usd.toFixed(6)}`, 'Speech': 'Existing caller audio; newly generated analyst line. Broadcast audio removed.' })) {
    const dt = document.createElement('dt'); dt.textContent = key; const dd = document.createElement('dd'); dd.textContent = value; dl.append(dt, dd);
  }
  $('record').append(note, dl);
  $('explanation').textContent = 'Blue follows the goal into the search. Green follows the exact retrieved passage into the analyst’s words.';
  $('replay').disabled = $('show').disabled = false;
  render();
}

$('replay').onclick = async () => { inspect = false; video.currentTime = 0; try { await video.play(); } catch { $('status').textContent = 'Use the video’s play control to start.'; } render(); };
$('show').onclick = () => { video.pause(); inspect = true; render(); };
for (const type of ['timeupdate', 'play', 'pause', 'seeked', 'ended']) video.addEventListener(type, render);
video.addEventListener('error', () => { $('error').hidden = false; $('error').textContent = 'The match video could not be loaded. You can still select Show full run to inspect the source and commentary.'; });
init().catch(error => { $('status').textContent = 'Run unavailable'; $('error').textContent = error.message; $('error').hidden = false; });
