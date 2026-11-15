// Offline only. Rebuild ignored frame assets and validate saved audio/trace alignment.
// Run from anywhere: node scripts/prepare_replay.mjs
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve, dirname, join } from 'node:path';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const trace = JSON.parse(readFileSync(join(root, 'web/public/soccer-commentary/prepared-penalty.trace.json')));
const source = join(root, trace.source.path);
const media = join(root, 'web/public/soccer-commentary/prepared-penalty-agents.mp4');
const reportPath = 'examples/recorded-agent-demo/render-report.md';
const report = readFileSync(join(root, reportPath), 'utf8');
const segments = [...report.matchAll(/^\| \d+ \| (caller|analyst) \| [\d.]+ \| ([\d.]+) \| [\d.]+ \| ([\d.]+) \| on cue \| (.+) \|$/gm)].map(m => ({ voice: m[1], start: +m[2], end: +m[3], text: m[4] }));
assert.equal(segments.length, trace.dialogue.lines.length);
segments.forEach((s, i) => {
  const l = trace.dialogue.lines[i];
  assert.equal(s.text, l.text); assert.equal(s.voice, l.voice);
  assert(Math.abs(s.start - l.at_s) < .01); assert(s.end > s.start);
  assert(s.end <= 20); if(i) assert(s.start >= segments[i-1].end);
});
for(const turn of trace.dialogue.turns) for(const id of turn.result.line?.evidence_ids ?? []) {
  assert(turn.visible_evidence.includes(id));
  const observation = trace.observations.find(o => o.id === id); assert(observation);
  assert(Math.max(...observation.action.support_at_s) <= turn.source_at_s);
}
assert(existsSync(source), `Missing original source: ${source}`);
assert.equal(createHash('sha256').update(readFileSync(source)).digest('hex'), trace.source.sha256, 'Source does not match saved trace');
assert(existsSync(media), `Missing voiced render: ${media}`);
const probe = JSON.parse(execFileSync('ffprobe', ['-v','error','-show_entries','stream=codec_type,duration','-of','json',media]));
assert(probe.streams.some(s => s.codec_type === 'audio' && Math.abs(+s.duration-20)<.1));
assert(probe.streams.some(s => s.codec_type === 'video' && Math.abs(+s.duration-20)<.1));
const frames = join(root, 'web/public/soccer-commentary/replay-frames'); mkdirSync(frames, {recursive:true});
const times = new Set(trace.observations.flatMap(o => o.action.support_at_s));
for(const t of times) execFileSync('ffmpeg', ['-hide_banner','-loglevel','error','-ss',String(trace.source.start_s+t),'-i',source,'-frames:v','1','-vf','scale=480:-1','-y',join(frames,`${t}.jpg`)]);
writeFileSync(join(root,'web/data/replay-delivery.json'), JSON.stringify({source:reportPath,segments},null,2)+'\n');
console.log(`Verified ${segments.length} original lines, evidence cutoffs and 20s audio/video. Prepared ${times.size} supporting frames. No API calls.`);
