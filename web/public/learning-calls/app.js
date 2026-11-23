"use strict";
const $ = id => document.getElementById(id);

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
    node("mark", phrase, "mistake"), document.createTextNode(value.slice(index + phrase.length)));
}

async function init() {
  const response = await fetch("/learning-calls/data.json");
  if (!response.ok) throw new Error("Saved run unavailable");
  const data = await response.json();
  const past = data.failure.observations.find(o => o.id === "v1-o2");
  const next = data.cases.find(c => c.id === "save");
  $("past-observation").textContent = past.description;
  highlight($("past-call"), data.failure.call.text, "only getting a touch as it went in");
  $("principle").textContent = data.lesson.principle;
  $("next-observation").textContent = next.payload.observations.find(o => o.action.type === "save").description;
  const caller = arm => next.arms.find(a => a.arm === arm).lines.find(l => l.voice === "caller").text;
  highlight($("baseline-call"), caller("baseline"), "at the near post");
  $("memory-call").textContent = caller("memory");
  const titles = { save: "Save · favorable example", goal: "Crowded finish · mixed", dribble: "Dribble and goal · regression" };
  for (const id of ["save", "goal", "dribble"]) {
    const item = data.cases.find(c => c.id === id);
    const section = node("section", "", "result");
    section.append(node("h3", titles[id]));
    const pair = node("div", "", "result-pair");
    for (const arm of ["baseline", "memory"]) {
      const outcome = item.arms.find(a => a.arm === arm);
      const column = document.createElement("div");
      column.append(node("h4", arm === "baseline" ? "Without lesson" : "With lesson"));
      for (const line of outcome.lines) column.append(node("p", `${line.voice}: ${line.text}`));
      column.append(node("small", `${outcome.word_count} words; ${outcome.within_word_target ? "within" : "over"} the 35-word target`));
      pair.append(column);
    }
    section.append(pair, node("p", item.review.summary + " " + item.review.memory, "review"));
    $("all-results").append(section);
  }
  $("cost").textContent = `Total: $${data.total_reported_usd.toFixed(5)} across ${data.total_calls} GPT-5.4 Mini calls, including an initial pilot with conflicting length instructions. All outputs and failures are retained.`;
}
init().catch(() => { $("error").hidden = false; });
