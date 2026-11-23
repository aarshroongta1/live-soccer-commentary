"use strict";
const $ = id => document.getElementById(id);
const scenes = [
  { label: "Invented touch", kind: "review", duration: 12, videoAt: 0 },
  { label: "Invented lead", kind: "review", duration: 10, videoAt: 0 },
  { label: "Repeated finish", kind: "review", duration: 12, videoAt: 0 },
  { label: "Call the goal", kind: "later", duration: 8, videoAt: 0 },
  { label: "Add analysis", kind: "later", duration: 9, videoAt: 1.5 },
  { label: "Leave space", kind: "later", duration: 7, videoAt: 3.5 }
];
const total = scenes.reduce((sum, scene) => sum + scene.duration, 0);
let data, clock = 0, playing = false, previousTime = 0, activeScene = -1, frame, viewState = "";

function node(tag, value, className) {
  const element = document.createElement(tag);
  element.textContent = value;
  if (className) element.className = className;
  return element;
}
function highlight(element, value, phrase) {
  const index = value.indexOf(phrase);
  if (index < 0) { element.textContent = value; return; }
  element.replaceChildren(document.createTextNode(value.slice(0, index)),
    node("mark", phrase), document.createTextNode(value.slice(index + phrase.length)));
}
function current() {
  let start = 0;
  for (let i = 0; i < scenes.length; i++) {
    if (clock < start + scenes[i].duration || i === scenes.length - 1) return { index: i, elapsed: clock - start, start };
    start += scenes[i].duration;
  }
}
function pause() {
  playing = false;
  cancelAnimationFrame(frame);
  $("match").pause();
  $("play").textContent = clock >= total ? "Replay history" : "Play history";
}
function play() {
  if (clock >= total) { clock = 0; activeScene = -1; }
  playing = true;
  previousTime = performance.now();
  $("play").textContent = "Pause";
  render();
  if (!$("match").ended) void $("match").play().catch(pause);
  frame = requestAnimationFrame(tick);
}
function tick(now) {
  if (!playing) return;
  clock = Math.min(total, clock + (now - previousTime) / 1000);
  previousTime = now;
  render();
  if (clock >= total) { pause(); return; }
  frame = requestAnimationFrame(tick);
}
function seek(index) {
  pause();
  clock = scenes.slice(0, index).reduce((sum, scene) => sum + scene.duration, 0);
  activeScene = -1;
  render();
}
function render() {
  const { index, elapsed } = current();
  const reviewing = index < 3;
  const corrected = reviewing && elapsed >= 4;
  const saved = reviewing && elapsed >= scenes[index].duration - 3;
  const memoryCount = reviewing ? index + Number(saved) : 3;
  const later = reviewing ? null : data.later_calls[index - 3];
  const output = later?.output;
  const referenced = new Set(output?.lesson_ids || []);
  const state = `${index}-${corrected}-${saved}`;
  $("clock").value = String(clock);
  if (viewState === state && activeScene === index) return;
  viewState = state;
  const changed = index !== activeScene;
  if (changed) {
    activeScene = index;
    $("match").currentTime = scenes[index].videoAt;
    if (playing) void $("match").play().catch(pause);
    for (const [i, button] of [...$("history").children].entries()) {
      button.setAttribute("aria-current", i === index ? "step" : "false");
      button.classList.toggle("done", i < index);
    }
  }
  $("clock").value = String(clock);
  $("position").textContent = `${index + 1} / 6`;
  $("next").disabled = index === scenes.length - 1;
  $("call-heading").textContent = reviewing ? `Earlier call ${index + 1}` : `Later pass · ${index - 2} / 3`;
  $("video-label").textContent = reviewing ? "Same penalty · earlier take" : "Same penalty · later pass";
  $("speaker").textContent = reviewing ? data.history[index].call.voice : "Recent dialogue";
  if (reviewing) {
    const phrases = ["only getting a touch as it went in", "England are ahead", "with the ball clearly crossing the line"];
    highlight($("past-call"), data.history[index].call.text, phrases[index]);
  } else {
    const recent = later.input.recent_dialogue;
    $("past-call").textContent = recent.length ? recent.map(line => line.text).join(" ") : "A fresh pass, with three lessons now in memory.";
  }
  $("correction").hidden = !corrected;
  if (corrected) {
    $("correction-label").textContent = saved ? "Lesson saved" : "Correction";
    $("correction-text").textContent = [
      "The audit tag did not establish a keeper touch. Don’t add contact that isn’t supported.",
      "The call says England are ahead, but the goal levelled the match. A goal alone doesn’t establish the score.",
      "The caller already announced the goal. Explain a distinct detail—or leave space."
    ][index];
  }
  $("memory-count").textContent = `${memoryCount} ${memoryCount === 1 ? "lesson" : "lessons"}`;
  // Update only when memory state changes, so active highlights stay readable.
  const memoryState = `${memoryCount}-${index}-${saved}`;
  if ($("memory-list").dataset.state !== memoryState) {
    $("memory-list").dataset.state = memoryState;
    $("memory-list").replaceChildren(...data.memory.map((entry, i) => {
      if (i >= memoryCount) return node("div", `Lesson ${i + 1} · not saved yet`, "memory-empty");
      const item = node("article", "", "memory-entry");
      item.classList.toggle("fresh", reviewing && i === index && saved);
      item.classList.toggle("used", !reviewing && referenced.has(entry.id));
      const header = document.createElement("header");
      header.append(node("span", `Lesson ${i + 1}`, "memory-id"), node("span", `from call ${entry.from_call}`, "source"));
      item.append(header, node("p", entry.lesson.principle));
      if (!reviewing && referenced.has(entry.id)) item.append(node("span", "Referenced by this call", "used-label"));
      return item;
    }));
  }
  $("memory-status").textContent = reviewing
    ? saved ? "Saved. This lesson stays available for the later pass." : corrected ? "Saving a reusable lesson from this correction…" : "Listen to the call. Its correction arrives next."
    : "All three lessons were supplied. Green shows the IDs this call referenced.";
  $("waiting").hidden = !reviewing;
  $("later-content").hidden = reviewing;
  if (!reviewing) {
    $("moment").textContent = [
      data.history[0].observations.find(o => o.id === "v1-o2").description,
      "The caller has announced the goal. The analyst gets the same observations and the recent dialogue.",
      "The camera still shows the same celebration. No new action or information has arrived."
    ][index - 3];
    $("used-lessons").replaceChildren(...output.lesson_ids.map(id => node("span", `Lesson ${id.slice(1)}`)));
    $("later-speaker").textContent = output.abstain ? "Decision: stay silent" : output.voice;
    $("later-call").textContent = output.abstain ? "No new call." : output.text;
    $("call-reason").textContent = output.reason;
  }
}
async function init() {
  const response = await fetch("/learning-calls/session.json");
  if (!response.ok) throw new Error("Saved history unavailable");
  data = await response.json();
  if (data.status !== "generated_pending_review" || data.later_calls.some(c => c.status !== "generated")) throw new Error("Incomplete run");
  for (const [index, scene] of scenes.entries()) {
    const button = node("button", "");
    button.type = "button";
    button.append(node("small", index < 3 ? `Earlier call ${index + 1}` : `Later call ${index + 1}`), node("span", scene.label));
    button.addEventListener("click", () => seek(index));
    $("history").append(button);
  }
  $("clock").max = String(total);
  $("clock").addEventListener("input", event => {
    pause(); clock = Number(event.target.value); activeScene = -1; render();
    const { index, elapsed } = current();
    $("match").currentTime = Math.min(scenes[index].videoAt + elapsed, $("match").duration || 6.9);
  });
  $("play").disabled = false;
  $("next").disabled = false;
  $("jump").disabled = false;
  $("play").addEventListener("click", () => playing ? pause() : play());
  $("next").addEventListener("click", () => { seek(Math.min(5, activeScene + 1)); play(); });
  $("jump").addEventListener("click", () => { seek(3); play(); });
  document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
  $("cost").textContent = `This run used six GPT-5.4 Mini calls and $${data.budget.reported_usd.toFixed(5)} in reported tokens, under a $0.10 cap. No retries. All generated words are unchanged. The analyst still partly repeats the outcome; lesson references don’t guarantee perfect behavior.`;
  render();
}
init().catch(() => { pause(); $("error").hidden = false; });
