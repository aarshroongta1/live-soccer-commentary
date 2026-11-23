"use strict";

const $ = id => document.getElementById(id);
const titles = { save: "A real save", goal: "A crowded finish", dribble: "A dribble and goal" };
const descriptions = {
  save: "A low delivery, a close-range save, then an unresolved loose ball. This control checks that caution still allows a real save call.",
  goal: "The ball ends up in the net after a crowded scramble. The saved observations leave the exact contact uncertain.",
  dribble: "The saved observations describe a carry, a left-footed shot beyond the diving keeper, and a confirmed goal."
};
let data;
let active = "save";
let revealed = false;

function text(tag, value, className) {
  const node = document.createElement(tag);
  node.textContent = value;
  if (className) node.className = className;
  return node;
}

function reveal() {
  revealed = true;
  $("memory").classList.add("saved");
  $("memory-state").textContent = "Saved";
  $("save-lesson").textContent = "Lesson saved";
  $("save-status").textContent = "The same frozen lesson was supplied to all three memory calls. You’re viewing the recorded result.";
  try {
    localStorage.setItem("commentary-lesson-v1", JSON.stringify({
      lesson: data.lesson, sha256: data.lesson_sha256, source: data.source_sha256
    }));
  } catch { /* The saved experiment remains usable when browser storage is blocked. */ }
  render();
}

function render() {
  const item = data.cases.find(c => c.id === active);
  $("case-title").textContent = titles[active];
  $("case-description").textContent = descriptions[active];
  $("case-tabs").replaceChildren(...["save", "goal", "dribble"].map(id => {
    const button = text("button", titles[id]);
    button.type = "button";
    button.setAttribute("aria-pressed", String(active === id));
    button.addEventListener("click", () => { active = id; render(); });
    return button;
  }));
  $("observations").replaceChildren(...item.payload.observations.map(o => {
    const container = document.createElement("div");
    container.className = "observation";
    container.append(text("p", `${o.id} · ${o.at_s}s · ${o.kind}`, "small"));
    container.append(text("p", o.description));
    container.append(text("p", `Action: ${o.action.type}; ${o.action.certainty}. Tags: ${o.action.cues.join(", ")}.`, "small"));
    return container;
  }));
  for (const arm of ["baseline", "memory"]) {
    const outcome = item.arms.find(a => a.arm === arm);
    $(arm + "-lines").replaceChildren(...outcome.lines.map(line => {
      const node = document.createElement("div");
      node.append(text("span", line.voice === "caller" ? "Caller" : "Analyst", "voice"));
      const spoken = text("p", line.text, "spoken");
      if (active === "save" && arm === "baseline") {
        const phrase = "at the near post";
        const index = line.text.indexOf(phrase);
        if (index !== -1) {
          spoken.replaceChildren(document.createTextNode(line.text.slice(0, index)),
            text("mark", phrase), document.createTextNode(line.text.slice(index + phrase.length)));
        }
      }
      node.append(spoken);
      return node;
    }));
    $(arm + "-count").textContent = `${outcome.word_count} words · ${outcome.within_word_target ? "within" : "over"} the 35-word target`;
    $(arm + "-count").className = "word-count" + (outcome.within_word_target ? "" : " over");
  }
  $("cover").hidden = revealed;
  $("review").hidden = !revealed;
  $("memory-lines").inert = !revealed;
  $("verdict").textContent = item.review.verdict;
  $("verdict").className = item.review.verdict.toLowerCase();
  $("review-summary").textContent = item.review.summary;
  $("review-baseline").textContent = "Without memory: " + item.review.baseline;
  $("review-memory").textContent = "With memory: " + item.review.memory;
}

async function init() {
  const response = await fetch("/learning-calls/data.json");
  if (!response.ok) throw new Error("Data could not load");
  data = await response.json();
  const call = data.failure.call.text;
  const phrase = "only getting a touch as it went in";
  const index = call.indexOf(phrase);
  if (index < 0) $("past-call").textContent = call;
  else $("past-call").replaceChildren(document.createTextNode(call.slice(0, index)),
    text("mark", phrase), document.createTextNode(call.slice(index + phrase.length)));
  const observation = data.failure.observations.find(o => o.id === "v1-o2");
  $("past-observation").textContent = observation.description;
  $("past-cues").textContent = "Observer audit tags: " + observation.action.cues.join(", ");
  $("principle").textContent = data.lesson.principle;
  $("apply-when").textContent = data.lesson.apply_when;
  $("preserve").textContent = data.lesson.preserve;
  $("model-label").textContent = "GPT-5.4 Mini · one sample per arm";
  $("cost").textContent = `Total reported model-token cost: $${data.total_reported_usd.toFixed(5)} across ${data.total_calls} calls, including lesson extraction and the failed pilot. The complete experiment stayed below a $0.20 ceiling. No search or voice charges.`;
  $("save-lesson").disabled = false;
  $("save-lesson").addEventListener("click", reveal);
  $("reveal").addEventListener("click", reveal);
  render();
}

init().catch(() => { $("error").hidden = false; });
