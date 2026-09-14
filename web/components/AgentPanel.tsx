import { Chip, Empty, Field, Meter, Panel } from "@/components/ui";
import { label, percent, usd, videoClock } from "@/lib/format";
import type {
  BoardEvent,
  CallerEvent,
  CorrectionEvent,
  CostEvent,
  GateEvent,
  PreemptedEvent,
  Sighting,
  SightingEvent,
  TriggerEvent,
} from "@/lib/events";
import type { Counts, StreamStore } from "@/lib/store";

/**
 * Everything the viewer would otherwise have to read a log file to see.
 *
 * A commentary system is judged on what it did not say, and nothing about a
 * line that was rejected or a sentence that was cut off shows up in the audio.
 * So the rejections, the cuts, the trigger that fired, the form the caller
 * filled in and the money it cost all live here, next to the picture.
 */
export function AgentPanel({ store }: { store: StreamStore }) {
  return (
    <div className="flex flex-col gap-3">
      <SpeakDecision trigger={store.trigger} />
      <CallerForm caller={store.caller} />
      <Sightings sighting={store.sighting} counts={store.counts} />
      <FactGate rejected={store.rejected} counts={store.counts} />
      <Preempted cuts={store.cuts} counts={store.counts} />
      <BoardReader board={store.board} />
      {store.corrections.length > 0 ? <Corrections corrections={store.corrections} /> : null}
      <Ledger cost={store.cost} counts={store.counts} lagS={store.lastLagS} />
    </div>
  );
}

/** Why the system considered speaking — so silence reads as a choice, not a hang. */
function SpeakDecision({ trigger }: { trigger: TriggerEvent | null }) {
  return (
    <Panel
      label="speak predictor"
      meta={trigger ? videoClock(trigger.ts) : null}
    >
      {trigger ? (
        <div className="flex flex-col gap-3">
          <div>
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">urgency</span>
              <span className="tnum font-mono text-[13px] text-text">{trigger.urgency.toFixed(2)}</span>
            </div>
            <Meter value={trigger.urgency} tone={trigger.urgency > 0.8 ? "live" : "neutral"} />
          </div>

          <div className="flex flex-wrap gap-1.5">
            {trigger.triggers.length > 0 ? (
              trigger.triggers.map((name) => (
                <Chip key={name} tone={name === "board_change" ? "live" : "neutral"}>
                  {label(name)}
                </Chip>
              ))
            ) : (
              <Chip tone="quiet">no triggers</Chip>
            )}
            <Chip tone={trigger.shouldCall ? "caller" : "quiet"}>
              {trigger.shouldCall ? "call" : "hold"}
            </Chip>
          </div>

          {trigger.reason ? (
            <p className="text-[13px] leading-relaxed text-dim">{trigger.reason}</p>
          ) : null}
        </div>
      ) : (
        <Empty>Waiting for the first tick of the speak predictor.</Empty>
      )}
    </Panel>
  );
}

/** The caller fills a form, not just a sentence. This is the form. */
function CallerForm({ caller }: { caller: CallerEvent | null }) {
  return (
    <Panel label="caller form" meta={caller ? videoClock(caller.ts) : null}>
      {caller ? (
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-1.5">
            <Chip>{label(caller.scene)}</Chip>
            <Chip tone={caller.event === "goal" ? "live" : "neutral"}>{label(caller.event)}</Chip>
            {caller.team ? <Chip tone="caller">{caller.team}</Chip> : null}
            <Chip tone={caller.speak ? "caller" : "quiet"}>{caller.speak ? "speak" : "stay quiet"}</Chip>
          </div>

          <div>
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">confidence</span>
              <span className="tnum font-mono text-[13px] text-text">{percent(caller.confidence)}</span>
            </div>
            <Meter value={caller.confidence} tone={caller.confidence < 0.6 ? "reject" : "caller"} />
          </div>

          <div>
            <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">read off the picture</span>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {caller.sightings.length > 0 ? (
                caller.sightings.map((seen, index) => (
                  <Chip key={`${caller.seq}-${index}`}>{describe(seen)}</Chip>
                ))
              ) : (
                <Chip tone="quiet">nothing legible</Chip>
              )}
            </div>
          </div>

          <p className="border-l-2 border-line pl-3 text-[14px] leading-snug text-text">
            {caller.line || <span className="text-faint">No line — the caller chose silence.</span>}
          </p>
        </div>
      ) : (
        <Empty>The caller has not filed a form yet.</Empty>
      )}
    </Panel>
  );
}

/** A shirt read, as the caller wrote it down: `#10`, `Musiala`, or both. */
function describe(seen: Sighting): string {
  if (seen.number !== null && seen.name) return `#${seen.number} ${seen.name}`;
  if (seen.number !== null) return `#${seen.number}`;
  return seen.name ?? "unreadable";
}

/**
 * What the caller read off the shirts, and whether the team sheet agreed.
 *
 * This is the naming pipeline in one box. A number the caller read is only a
 * claim about pixels until the roster turns it into a player, and the gap
 * between what was read and what bound is the single number that says
 * whether open-play naming is working at all.
 */
function Sightings({ sighting, counts }: { sighting: SightingEvent | null; counts: Counts }) {
  const seenTotal = counts.sightingsBound + counts.sightingsDropped;
  return (
    <Panel
      label="sightings · roster bind"
      meta={
        <span className="tnum">
          {seenTotal === 0 ? "none yet" : `${counts.sightingsBound} of ${seenTotal} bound`}
        </span>
      }
    >
      {sighting && sighting.sightings.length > 0 ? (
        <ul className="flex flex-col gap-2">
          {sighting.sightings.map((seen, index) => (
            <li key={`${sighting.seq}-${index}`} className="flex items-baseline gap-2">
              <Chip tone={seen.bound ? "caller" : "reject"}>{describe(seen)}</Chip>
              {seen.side !== "unknown" ? (
                <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
                  {seen.side}
                </span>
              ) : null}
              <span className="min-w-0 text-[13px] leading-snug text-dim">
                {seen.bound ? (
                  (seen.as ?? "on the team sheet")
                ) : (
                  <span className="text-faint">names nobody on either sheet</span>
                )}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <Empty>
          Nothing read off a shirt yet. A number in a wide shot is four pixels tall, so most
          lines have no sighting at all.
        </Empty>
      )}
    </Panel>
  );
}

/**
 * The statistician overruling the screen. Off in the default runtime — the
 * thesis is that the picture is enough — so this panel only appears on a run
 * started with `--wire`.
 */
function Corrections({ corrections }: { corrections: readonly CorrectionEvent[] }) {
  return (
    <Panel label="wire corrections" meta={<span className="tnum">{corrections.length}</span>}>
      <ol className="flex flex-col gap-2">
        {corrections.map((correction) => (
          <li key={correction.seq} className="flex items-baseline gap-2">
            <span className="tnum font-mono text-[10px] text-faint">{videoClock(correction.ts)}</span>
            <Chip tone="live">{label(correction.event)}</Chip>
            <span className="min-w-0 text-[13px] leading-snug text-dim">{correction.what}</span>
          </li>
        ))}
      </ol>
    </Panel>
  );
}

/** The lines that never made it to air, and exactly why. */
function FactGate({ rejected, counts }: { rejected: readonly GateEvent[]; counts: Counts }) {
  return (
    <Panel
      label="fact gate · rejected"
      meta={
        <span className="tnum">
          {counts.gateRejected} blocked · {counts.gatePassed} passed
        </span>
      }
      bodyClassName="max-h-72 overflow-y-auto px-4 py-3"
    >
      {rejected.length === 0 ? (
        <Empty>Nothing rejected yet. Every line the caller filed has matched the board and the roster.</Empty>
      ) : (
        <ol className="flex flex-col gap-3">
          {rejected.map((verdict) => (
            <li key={verdict.seq} className="border-l-2 border-reject/50 pl-3">
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-reject">blocked</span>
                <span className="tnum font-mono text-[10px] text-faint">{videoClock(verdict.ts)}</span>
              </div>
              <p className="mt-1 text-[14px] leading-snug text-dim line-through decoration-reject/60">
                {verdict.line}
              </p>
              <ul className="mt-1.5 flex flex-col gap-1">
                {verdict.reasons.map((reason, index) => (
                  <li key={`${verdict.seq}-${index}`} className="flex items-baseline gap-2">
                    <Chip tone="reject">{reason.tag}</Chip>
                    {reason.detail ? (
                      <span className="min-w-0 text-[12px] leading-snug text-faint">{reason.detail}</span>
                    ) : null}
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}

/** Lines the director killed: cut mid-sentence, or aged out before their turn. */
function Preempted({ cuts, counts }: { cuts: readonly PreemptedEvent[]; counts: Counts }) {
  return (
    <Panel
      label="preempted"
      meta={
        <span className="tnum">
          {counts.cut} cut · {counts.stale} stale
        </span>
      }
      bodyClassName="max-h-64 overflow-y-auto px-4 py-3"
    >
      {cuts.length === 0 ? (
        <Empty>No line has been interrupted. A goal will take the analyst&rsquo;s microphone mid-word.</Empty>
      ) : (
        <ol className="flex flex-col gap-3">
          {cuts.map((cut) => {
            const unsaid = cut.text.startsWith(cut.spoken) ? cut.text.slice(cut.spoken.length) : cut.text;
            return (
              <li key={cut.seq} className="border-l-2 border-line pl-3">
                <div className="flex items-baseline gap-2">
                  <span
                    className={`font-mono text-[10px] uppercase tracking-[0.2em] ${
                      cut.reason === "cut" ? "text-reject" : "text-faint"
                    }`}
                  >
                    {cut.reason === "cut" ? "cut off" : "aged out"}
                  </span>
                  <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-dim">{cut.voice}</span>
                  <span className="tnum font-mono text-[10px] text-faint">{videoClock(cut.videoTs)}</span>
                </div>
                <p className="mt-1 text-[14px] leading-snug text-text">
                  {cut.spoken}
                  {unsaid ? <span className="text-faint line-through">{unsaid}</span> : null}
                </p>
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}

/** One glance at the score bug, and how sure the reader was of it. */
function BoardReader({ board }: { board: BoardEvent | null }) {
  return (
    <Panel label="board reader" meta={board ? videoClock(board.ts) : null}>
      {board ? (
        <div className="flex flex-col gap-2">
          <Field label="score bug">
            {board.bugVisible ? (
              <span className="text-text">visible</span>
            ) : (
              <span className="text-live">hidden — replay</span>
            )}
          </Field>
          <Field label="read">
            <span className="tnum font-mono">
              {board.homeScore ?? "–"}–{board.awayScore ?? "–"}
            </span>
          </Field>
          <Field label="clock">
            <span className="tnum font-mono">{board.clock ?? "—"}</span>
          </Field>
          <div className="pt-1">
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">confidence</span>
              <span className="tnum font-mono text-[13px] text-text">{percent(board.confidence)}</span>
            </div>
            <Meter value={board.confidence} tone={board.confidence < 0.7 ? "reject" : "caller"} />
          </div>
        </div>
      ) : (
        <Empty>No read yet. The board reader glances at the score bug every two seconds.</Empty>
      )}
    </Panel>
  );
}

/** What the match has cost so far, how much of it reached air, and how late. */
function Ledger({
  cost,
  counts,
  lagS,
}: {
  cost: CostEvent | null;
  counts: Counts;
  lagS: number | null;
}) {
  return (
    <Panel label="ledger">
      <div className="flex flex-col gap-3">
        <div className="flex items-baseline justify-between">
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">spend</span>
          <span className="tnum font-mono text-[26px] leading-none text-text">
            {cost ? usd(cost.totalUsd) : "—"}
          </span>
        </div>

        {cost && cost.perAgent.length > 0 ? (
          <div className="flex flex-col gap-1 border-t border-line pt-2">
            {cost.perAgent.map((agent) => (
              <Field key={agent.agent} label={label(agent.agent)}>
                <span className="tnum font-mono">{usd(agent.usd, 3)}</span>
              </Field>
            ))}
          </div>
        ) : null}

        <div className="flex flex-col gap-1 border-t border-line pt-2">
          <Field label="beats queued">
            <span className="tnum font-mono">{counts.beats}</span>
          </Field>
          <Field label="lines spoken">
            <span className="tnum font-mono">{counts.spoken}</span>
          </Field>
          {/* Buffer depth plus generation time — how far behind the live edge
              the last line landed. */}
          <Field label="lag">
            <span className="tnum font-mono">{lagS === null ? "—" : `${lagS.toFixed(1)}s`}</span>
          </Field>
        </div>
      </div>
    </Panel>
  );
}
