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
    node("mark", phrase), document.createTextNode(value.slice(index + phrase.length)));
}
async function init() {
  const response = await fetch("/learning-calls/session.json");
  if (!response.ok) throw new Error("Saved calls unavailable");
  const data = await response.json();
  if (data.later_calls.some(c => c.status !== "generated")) throw new Error("Incomplete run");
  const used = new Set(data.later_calls.flatMap(c => c.output.lesson_ids));
  const phrases = ["only getting a touch as it went in", "England are ahead", "with the ball clearly crossing the line"];
  $("pairs").replaceChildren(...data.history.map((item, index) => {
    const memory = data.memory[index];
    const pair = node("article", "", "pair");
    pair.append(node("h3", `Bad call ${index + 1} · ${item.call.voice}`, "label"));
    const quote = node("p", "", "bad-call");
    highlight(quote, item.call.text, phrases[index]);
    const lesson = node("div", "", "lesson" + (used.has(memory.id) ? " referenced" : ""));
    lesson.append(node("h3", `Saved lesson ${index + 1}`, "label"), node("p", memory.lesson.principle));
    pair.append(quote, lesson);
    return pair;
  }));
  $("later-calls").replaceChildren(...data.later_calls.map(call => {
    const output = call.output;
    const article = node("article", "", "new-call" + (output.abstain ? " abstention" : ""));
    article.append(node("h3", output.abstain ? "When nothing new happens" : output.voice, "label"));
    article.append(node("p", output.abstain ? "No new call." : output.text, "spoken"));
    const references = node("div", "", "references");
    for (const id of output.lesson_ids) references.append(node("span", `Lesson ${id.slice(1)}`));
    article.append(references);
    if (output.abstain) article.append(node("p", output.reason, "reason"));
    return article;
  }));
  $("cost").textContent = `Six GPT-5.4 Mini calls, $${data.budget.reported_usd.toFixed(5)} in reported tokens, under a $0.10 cap. No retries.`;
}
init().catch(() => { $("error").hidden = false; });
