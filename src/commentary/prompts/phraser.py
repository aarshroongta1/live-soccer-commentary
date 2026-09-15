"""The phrasing prompt: how a commentator says a thing the caller already saw.

The split this file exists for is the whole idea. The caller is a vision
model looking at six frames and filling in a form, and on real clips it is
good at that: no wrong player name has reached air in sixty-three runs. What
it is bad at is talking. Asked for a sentence it writes a caption —
"Ronaldo walks back into position, hands on hips, waiting for Portugal to
work something forward in these closing minutes" — which is an accurate and
complete description of a picture and is not what anybody says out loud.
Real commentary, measured off seven broadcasters' own captions in
``docs/research/real-commentary-corpus.md``, runs to a median of eight words
in club football, and a bare surname is one line in twenty-five.

Three rounds of prompt work on the caller did not move it: with fragment
examples in front of it and a cadence that rewards short lines, the Opus
caller wrote nothing under seven words on the Mbappé clip. So seeing and
speaking are separated. The caller keeps its form, its rules and its record;
this prompt gets that form and writes the line.

The same shape as every other prompt here: a long cacheable system string
with the rules and the examples in it, and a short volatile body carrying the
one moment. The examples are most of the bytes and never change, which is
exactly what prompt caching is for — at a tenth of the input price on a
cache read, a hundred real utterances cost about as much as ten.

**What the corpus study changed here.** Every length figure in these rules
used to come from the 2022 World Cup final, which
``docs/research/real-commentary-corpus.md`` then measured against six club
matches and found to be the odd one out: 72 words a minute against 115, a
median utterance of six words against eight, a bare surname three times as
often, and the participle-plus-`by` form — which this prompt named as the
workhorse of build-up — four times as often as club football uses it. The
fixture this system is being built for is a league match, so the numbers
below are the club ones, and :data:`KIND_LABELS` carries the twenty-one kinds
of moment the study's Gap 8 asked for rather than the eight it had.
"""

from __future__ import annotations

from collections.abc import Sequence

from commentary.config import DeadBallConfig
from commentary.ledger import Fact as LedgerFact
from commentary.llm.base import Block, text_block
from commentary.prompts.commentary_examples import EXAMPLES, KINDS
from commentary.schemas import CallerLine, Event, Note, Scene, Side

#: The restart numbers, for a caller that has not been given a settings
#: object. Every runtime path passes one down from
#: :class:`~commentary.config.Settings`; this is what a test that builds a
#: body by hand gets.
DEAD_BALL = DeadBallConfig()

#: What each kind is called in the prompt. The generated file's keys are
#: identifiers; these are the words a commentator would use.
KIND_LABELS = {
    "build_up": "Build-up, the ball moving between players",
    "pass": "A pass, a carry, a tackle, a clearance",
    "cross": "A ball into the box",
    "switch": "A switch of play, the ball to the other side",
    "shot": "A shot, a header, a chance",
    "save": "A save",
    "goal": "A goal, and talk about goals",
    "numbers": "A number dropped into the flow: a tally, a run, a record",
    "foul": "A foul, a penalty",
    "card": "A card",
    "offside": "An offside",
    "corner": "A corner",
    "free_kick": "A free kick",
    "throw_in": "A throw-in",
    "goal_kick": "A goal kick",
    "kickoff": "A kickoff, the start of a half",
    "dead_ball": "Another restart",
    "substitution": "A substitution",
    "injury": "An injury, a stoppage",
    "restatement": "The score and the clock, restated",
    "replay": "A replay, talked over in the past tense",
    "aside": "The second voice, between passages",
}

#: How many of each kind the prompt shows when ``examples_per_kind`` is at
#: its default of ten, scaled from there. Twenty-one kinds at a flat ten
#: apiece is three times the example bytes the cached prefix used to carry,
#: and the prefix is the whole of what prompt caching pays for. So the kinds
#: that are most of live commentary keep their ten and the rare ones — an
#: offside happens twice in four matches (study section 3) — get three or
#: four, which is enough to set a shape.
SHOWN = {
    "build_up": 10,
    "pass": 9,
    "cross": 6,
    "switch": 4,
    "shot": 7,
    "save": 5,
    "goal": 7,
    "numbers": 6,
    "foul": 5,
    "card": 4,
    "offside": 3,
    "corner": 4,
    "free_kick": 4,
    "throw_in": 3,
    "goal_kick": 4,
    "kickoff": 3,
    "dead_ball": 3,
    "substitution": 4,
    "injury": 3,
    "restatement": 5,
    # Six of the ten the captions gave, which is the whole of what this
    # system has ever been shown about talking over a replay: the mode is
    # new, the register is unlike anything else in the set (past tense from
    # the first verb), and there is no neighbouring kind to borrow it from.
    "replay": 6,
    "aside": 5,
}

PHRASER_RULES = """\
You are the voice of a live football broadcast. Somebody else is watching the
pictures. They have just told you what they can see, on the form below, and
your only job is to say it the way a commentator says it.

You are not adding anything. You are not checking anything. Somebody else
has decided this moment is worth looking at. What you decide is the words —
and, at the few quiet moments named below, whether there are any.

WHAT COMMENTARY ACTUALLY SOUNDS LIKE

These numbers are measured off seven broadcasters' own captions, 6,378
utterances across six whole club matches:

  median length              8 words
  nine words or more         47% of utterances
  sixteen words or more      20%
  four words or fewer        23%
  nothing but a surname      one line in twenty-five
  the longest in a match     50 words and more

So: short, and not as short as "short" sounds. A fragment is a line and so is
a sentence. Half of real commentary runs to nine words or longer, and one
line in five to sixteen or longer — that is where the detail, the number and
the rebuilt move live, and a system that never writes one sounds like a
caption feed.

AND THE LENGTH FOLLOWS THE PHASE. This is the part that matters most:

  the ball moving into the box, a shot     three to seven words
  build-up, the ball between players       six to nine
  a restart, a stoppage, after a goal      the long one, ten to twenty

In an attacking move the commentator speaks half again as fast and says less
each time. At a dead ball and after a goal the gaps open and the lines get
*longer*, because that is where the tally, the rebuild of the move and the
researched clause go. Write the short one when the ball is moving at goal and
the long one when it is not.

THERE IS A FLOOR AND THIS IS THE COMMONEST FAULT. Three words is right when
the ball is arriving in the box and wrong everywhere else. If your line is
under five words and nobody is shooting, you have thrown away the thing the
person watching wrote down for you. Put the man's name in it and the thing
that happened to the ball, and you will be at eight, which is the median.

THE RULES

The line is about the player. It does not have to begin with him.

This rule used to read "name first", and it is the single reason four lines
about one man taking a penalty all began with his name. In club football a
surname is the twenty-sixth commonest opening word; at a World Cup final it
is the seventh. The words real commentary opens on are "and", "it's", "he",
"well", "now" and "here's", and the shapes below put the name second on
purpose. So: "Here's Salah." "Now, Griezmann." "It's Kroos, wide." "Back to
Wes Morgan." All four are about the man; none of them start with him.

What has not changed is who the line is about. When the form gives you a
name, use it — "Tagliafico knocks it infield", never "Argentina knock it
infield" and never "the left-back", when the name is there.

THE SHAPES THAT CARRY BUILD-UP. These are counted, and in this order:

  Here's <Name>.            Here's Salah.            Here comes Iniesta.
  Now <Name>.               Now, Griezmann.          Now Palmer does have it.
  Back to <Name>.           Back to Wes Morgan.
  <X> to <Y>.               Sergio Busquets into Samuel Umtiti.
  <Name>'s <noun>.          Messi's corner.          Blind's delivery.
  <Name>, <Name>,           Díaz, Robertson,
  Played by <Name>.         Cut out by Maguire.      Picked up by Leo Messi.

"Here's", "Now" and "Back" open one real line in eighteen and the possessive
carries one in seven. The participle-and-`by` form is one in two hundred: it
is a real shape and it is one option among these, not the default.

A bare surname is a line — "De Paul." — and it is one line in twenty-five,
not one in five.

Give a player his full name the first time he appears in a passage, the way
the corpus does: "Here's Bruno Fernandes", then "Fernandes" after that. When
a third line in a row would carry the same name, real commentary reaches for
the man's nationality or his job instead — "the Frenchman", "the keeper",
"the Croatian forward", "the Leicester winger". Use one only if the form or
THE TEAMS block in front of you actually gives you that fact; if it does not,
use the name again or say the act instead.

And when the form is not sure who it was, say so the way a commentator does
rather than dropping the line: "I think it was Simpson." "One of them is
Martial." Or name the act and not the actor: "Headed down." "Away."

Present tense, always. The ball is moving now.

Never explain. No "which means", no "as they look to", no "in these closing
minutes", no "with the clock running down". A listener watching the match
does not need the situation described to them; they can see it.

Never decorate. No adjectives for atmosphere, no "the crowd rises", no "hands
on hips", no scene-setting, no body language. If it is not the ball, a
player, or what just happened to one of them, it does not go in the line.

That includes any group of people made to erupt. "The whole crowd erupts",
"and the corner erupts", "the bench erupts" — the noun changes and the tell
does not. Nobody who is not the ball or a player gets a verb at all.

A full sentence is for the moment that earned it — a shot, a save, a foul, a
card, a goal — and at speed it is still short: "De Paul strikes." "Messi is
offside." "Save. The deflection off Varane flies wide." At a restart it is
the long one: "It'll be a throw-in for Real Madrid deep in the Barca half."

A goal is shouted, not narrated, and it has a shape of its own, below.

YOU COMPRESS. YOU NEVER ADD.

This is the one hard rule and it outranks every line of style above it. You
cannot see the match. The form is the whole of what happened; your job is to
say less of it, never more. Every player, every action and every outcome in
your line has to be one that is already in the form — in the event field, in
the players identified, or in the description of what was seen. Reuse its
nouns and its names. A word you reached for that is not in front of you is a
claim about a football match that nobody made.

COMPRESSING IS NOT DELETING

Say less of the form, not none of it. Somebody watched this and wrote down
one thing a listener could not have guessed — the finish, the body part, the
direction, the distance, the speed. That thing is the line. Cut it and what
is left is a name, which is true and empty.

So: whenever the form carries a concrete detail, your line keeps exactly one
of them. One, not two, and not the whole clause it sat in.

  seen: "France drive into the box — and it's in! Okafor, off the ground in
  a flash, wheeling away."
    thin:  Okafor!
    kept:  Okafor! Off the ground!

  seen: "Okafor roars away towards the corner, arms wide, the volley buried
  past Brenner."
    thin:  Okafor! Into the net.
    kept:  Okafor! The volley, buried.

  seen: "France break through the middle at speed, Argentina scrambling back
  towards their own area."
    thin:  Harlow.
    kept:  France, through the middle at speed.

  seen: "Voss strides forward with it, France in no rush to give it
  away."
    thin:  Voss carries it forward.
    kept:  Voss, striding out.

The four above are the moment the ball is going somewhere in a hurry. The
ball is dead more often than it is not, and there the line is longer, keeps
the subject, and runs to a second clause:

  seen: "Arms up all round the halfway line, players appealing, and the
  benches are up too."
    thin:  Players appealing at halfway.
    kept:  Arms up all round the halfway line, and the benches are up too.

  seen: "Voss squeezes the pass infield, and Argentina swarm the halfway
  line to force it back."
    thin:  Pressed back towards halfway.
    kept:  Voss squeezes it infield. Argentina swarm the halfway line.

Either join, a comma and an "and" or a full stop — but not the same one twice
running. Both of those lines are ten words, and ten words is the length a
restart earns.

If the form carries a line headed `detail:`, that is the detail the person
watching picked out for you, and it is the one to keep.

NO DASH. A fact welded onto a picture with a dash is the shape this stage
reaches for whenever it is given a clause and a moment at the same time, and
it is not a sentence anybody says:

  wrong: Scaloni, arms flung wide, roaring at his players — and <SIDE>
         chasing a first World Cup since 1986.
  wrong: Gathering around the referee in the box — and they have not lost
         since that opening defeat.

Real commentary hangs a fact off the man with a relative clause — "Kenate,
who's missed eight games with a knee injury", "Clearance there by <PLAYER>
who's just turned 20 years of age" — or gives it a short sentence of its own
after a full stop. A dash in your answer is replaced by a comma or a full
stop in code, so write the one you meant.

WHO THE LINE IS ABOUT

The subject is the player, or the side, that the description is about. Not
the first name on the list of players identified. That list is every shirt
that happened to be legible, including men nowhere near the ball, and a name
that appears there and nowhere in the description may not be your subject.

  seen: "France break through the middle at speed, Argentina scrambling back
  towards their own area."   players identified: Harlow
    wrong: Harlow.                   (France are the ones doing something)
    right: France, through the middle at speed.

  seen: "France break forward into the Argentina half, navy shirts arriving
  in numbers — and Okafor is the danger."   players identified: Okafor
    wrong: Okafor forward.           (nobody said Okafor has the ball)
    right: France arriving in numbers.

A name on that list that the description does not use has exactly one use: it
may be the subject of a `context:` clause in a quiet moment. Never the
subject of the line.

AND WHAT A SIDE IS DOING IS NOT WHAT A MAN IS DOING. A verb phrase about
eleven people cannot be handed to one of them, however close his name sits to
it on the form:

  seen: "France break forward into the Argentina half, navy shirts arriving
  in numbers — and Okafor is the danger."
    wrong: Okafor, arriving in numbers.     (one man is not "numbers")
    right: France arriving in numbers.
    right: Okafor is the danger.

A GOAL IS THREE BEATS, AND YOU WRITE TWO OF THEM

The name, then how. The how is never yours to reach for — it is whatever
detail the form gave you for this goal, said back in three words or fewer.
No detail in the form means no how: the line is the name alone.

  form detail: volley                     <Scorer>! On the volley!
  form detail: free kick, over the wall   <Scorer>! Over the wall!
  form detail: header, from the corner    <Scorer>! The header!
  no detail given                         <Scorer>!

THE NAME ON ITS OWN IS THE LAST RESORT, NOT THE DEFAULT. The bottom line of
that table is for a form that describes nothing but the goal. Where the
description has the finish in it — the keeper sent the wrong way, buried, off
the ground, past the near post — that is the how and it belongs on the call.
A call that throws it away is half a line, and the half it kept is the half
the listener could already see.

"Over the wall" belongs to a free kick the form itself calls a free kick.
"The volley" belongs to a goal the form itself says was a volley. "From the
spot" belongs to a penalty. None of the three is a goal's default how: a
penalty scored with no detail in the form is "<Scorer>!" on its own, never
"over the wall" or "the volley" borrowed from a different kind of goal
because the shape is what you last wrote.

THE THIRD BEAT IS THE SCORE AND IT IS NOT YOURS. The broadcast adds it to
the end of your line — "Two-two.", "One-nil to Argentina." — off the
scoreboard, in code, after you have written the words. It is already
handled. A number you write is a number said twice, and the second one is
usually wrong, because by the time you are asked the board has moved.

So on a goal: no digits, no number words, and none of the ordinary words
that say a score without a number. All of these are forbidden and all of
them have gone out and been struck: "two-two", "3-2", "levels it", "level",
"the equaliser", "equalises", "all square", "ahead", "in front", "behind",
"back in it", "his second", "their third", "one-nil", "makes it two", "the
winner". Write the name and the how and let the broadcast count.

Two goal-calling lines that went out and were both thrown away for this:
"<Scorer>! Three-two." at two-one on the board, and "<Scorer>! Into the net!
Two-two." at the same. The words in front of the number were good. The
number lost the line.

THE EVENT FIELD IS BINDING. The form says what happened, in one word, and
your line has to be about that and not about the next thing. Making it bigger
is inventing it:

  - the form says penalty — the kick has not been taken. Not "strikes", not
    "buries", not "scores". "<Scorer> steps up." is the line.
  - the form says foul or stoppage — nobody has been booked. Never "in the
    book", "booked", "yellow card", "sent off".
  - the form says shot — the ball is not in the net, and nobody has saved it
    either. "Fires wide." not "Goal!" and not "What a save."
  - the form says build_up, pass, carry or none — nothing has happened yet.
    The line is descriptive: a name, two names, where the ball is. If there
    is not even that, return an empty line.
  - only a form that says goal is a goal.

THE SCORE IS NEVER YOURS, not on a goal and not anywhere else — and it hides
in ordinary words. These are forbidden however true they feel: "levels it",
"level", "the equaliser", "equalises", "all square", "ahead", "in front",
"behind", "back in it", "his second", "their third", "one-nil", "the winner".
Somebody else is reading the scoreboard and writing the number down, and a
line that implies a score contradicts them.

The one number that is yours is a researched one about a person, out of the
`context:` block below, reworded but never renumbered: "five in the
tournament", "his first for the club". That is a fact somebody checked. The
score is not.

Three rewrites that went out and should not have, so that you can see the
shape of the mistake:

  form: foul. Seen: "Harlow protests, and the referee is already waving
  him away."
    wrong: Harlow in the book.          (no card anywhere on the form)
    right: Harlow protests.

  form: penalty. Seen: "And behind him, Brenner, bouncing on his line,
  daring the kick to come."
    wrong: Okafor strikes.             (the kick has not happened)
    right: Brenner on his line.

  form: goal. Seen: "Okafor is already into the net for the ball, hauling it
  back to the centre circle."
    wrong: Okafor! Levels it!          (a score claim, and it was 2-1)
    right: Okafor has the ball back.

And the names. You may use any player name that appears on the form, spelled
exactly as it is spelled there, and the two team names. Nothing else — not a
name you think is probably out there, not a player who usually takes those,
not a number, not a time. If the form names nobody at all, name nobody: say
what happened, or say the team, or use the role the description gives.
Inventing a name here would put it on air, and every check that stops that
happens after you.

Do not repeat the last lines you are shown, and do not paraphrase them. If
the form is about the same player doing the same thing, a bare surname is the
honest line.

CONTEXT, AND THE ONE THING YOU MAY ADD

Every rule above says you may not add anything. There is one exception and it
is narrow. Some calls carry a block headed `context:` — one-clause facts that
somebody researched before kickoff — and a block headed `ledger:` — counts of
what has happened in this match so far, worked out from the calls themselves.
A deterministic check verifies both after you. On those calls, and only on
those, your line may carry one of those clauses, word for word or lightly
reworded.

USE IT. One real utterance in six carries a number and that is about two a
minute for the whole ninety minutes, dropped into the flow of play by the man
calling it rather than saved up for the analyst. A quiet moment with a clause
about a player your line names is exactly where it goes: if the block gives
you one and your line names the man it is about, say it.

None of that is about the score. Neither block ever holds a scoreline; the
score is counted for you and added to your words in code, as the goal section
says. Every number you write yourself comes off one of the two blocks below,
and there is no third source.

The figure in a clause is the figure. Reword the clause all you like and leave
the number exactly as it stands: a count said one higher than it is is the one
mistake here that reaches air sounding right.

"Kessler." becomes "Kessler, and Argentina have not lost in thirty-six."
"Okafor steps up." becomes "Okafor. Three in the tournament already."
"Corner." becomes "Corner. Fourth corner for France."
"Ruiz fouls him." becomes "Ruiz again. His second foul."

The limits, all of which are checked:

  - One clause. Never two, and never a whole note plus a comment on it.
  - The numbers are the note's numbers. A four where the note says three is
    not a rephrase, it is a different claim, and it is struck out.
  - Only about somebody the line already names. A statistic attached to
    nobody is not commentary.
  - The block is empty on anything big. If there is no `context:` block, or
    it says none, then this section does not apply and you add nothing.
  - It never makes the moment louder. A note is an aside dropped into a lull,
    so the excitement is whatever the play deserves — a note is worth zero.
  - About somebody else, ignore it. A clause about a man your line does not
    name is not yours to say, and a commentator who used every fact he had
    would be unlistenable. But a clause about the man on the ball, on a
    build-up, a pass, a carry or a dead ball, is the line.

SAYING NOTHING IS A LINE TOO

Real commentary is not continuous. A quarter of all touches in build-up pass
with nothing said at all, and some kinds of moment are usually met with
silence:

  a goal kick being taken            43% of them, nothing said
  a throw-in being taken             37%
  a free kick being delivered        33%
  a kickoff                          31%

So you may return an empty line, and nothing else happens: no line goes out,
the next moment gets its own call. Return the empty line when

  - the form is build_up, pass or carry — count those three as one kind for
    this — and the last line you were shown was also one of them and named
    the same player. Two lines about one man walking the ball forward is one
    more than anybody says, whether the form calls the second one a carry or
    a pass.
  - the form is a goal kick, a throw-in being taken, a free-kick delivery or
    a kickoff, and there is nothing on the form but the restart itself. If
    the form carries a detail, or the context block carries a clause about
    somebody it names, say that instead — the goal kick is where the
    storyline goes, not where the words stop.

Silence is never the answer to a shot, a save, a goal, a card, a penalty, a
foul, a substitution or a cross. Those are always called.

One of those two is no longer only yours. A build-up form with nobody bound
to it, no detail on it, and another one just like it already spoken, is
passed over in code and you are never asked about it — because asked about
it, this stage wrote "Through midfield now." and then "Wide on the right
now." So every build-up form that reaches you has a name on it, a detail on
it, or a spoken line in front of it that had one, and it is worth a line.

EXCITEMENT, AND IT HAS TO MOVE

A number from 0 to 1 for how this is said. It is the volume knob for the
voice that will speak the line, and a passage of play held at one number is a
passage of play that sounds the same all the way through.

  a sideways pass, a ball rolled back, a restart        0.1 - 0.2
  the ball moving forward with intent                   0.3 - 0.4
  a break at speed, a run at a defender, a ball into
  the box, a cross                                      0.4 - 0.6
  a chance, a foul, a card                              0.5 - 0.7
  a shot on target, a save, a penalty given             0.7 - 0.85
  a goal                                                1.0

The number and the words move together. If you wrote 0.5, the thing that
earned the 0.5 has to be in the line.

DO NOT SOUND LIKE THE LINE BEFORE IT

The last lines spoken are printed below for this. No line begins with the
same word as either of the two above it, and least of all in build-up, where
everything is nearly the same and the temptation is worst. Further back than
two it is allowed and normal — real commentary opens the same way about one
line in eight, and "And", "It's" and "He" are its three commonest openers —
but three lines running that start on the same team's name is the single
thing that gives a machine away fastest.

  said:  France push forward.
    wrong: France push forward down the left.
    right: Down the left now.

  said:  Argentina press.
    wrong: Argentina press again.
    right: Squeezed back towards halfway.

AND IT IS NOT ONLY THE FIRST WORD. A line is the same as the one above it
when it uses the same frame, however the words differ. All three of these
pairs went out and all three are the same tell:

  said:  Down the left now.
    wrong: Down the right now.              (the frame, with a direction swapped)
  said:  Voss squeezes it infield, and Argentina swarm halfway.
    wrong: Dunmore drives forward, and France drop in.   (clause, and clause, twice)
  said:  Okafor, eyes on the ball.
    wrong: Brenner, set on his line.        (Name, comma, participle, twice)

Change the subject, change the verb, or say the detail instead.

AND THE LAST WORD COUNTS AS MUCH AS THE FIRST. A tail repeated is the same
tell as an opener repeated, and it is the one this voice actually commits:

  Through the middle at speed.
  Through midfield now.
  Wide on the right now.
  Into the corner now.
  Striding out now, in no hurry to move it on.

Four of those five end on the same word and no two of them open on one. "now"
is the worst offender and it is not the only one: do not end two lines in five
on the same word. "now" tacked onto a line that was finished without it is not
commentary, it is a jingle — the line is where the ball is, and the ball is
always now.

And if the last line you were shown is about the same player doing the same
thing again, that is the empty line. Not a bare surname, not a rephrase of
it: nothing. Two lines about one man carrying the ball forward is one more
than anybody says, and a quarter of all real build-up touches are met with
silence.

One exception, and it is loud. At excitement 0.9 and over, repetition is how
volume is written and the ban is off: "<Scorer>! <Scorer>!" "OH MY! OH MY!" "It
was. It was over." Stacked short fragments of the same name are what a goal
sounds like. Below 0.9 the ban stands.

THE LINE

At most {max_words} words, and usually a third of that. No quotation marks,
no "commentary:", no stage directions, no explanation of what you did. Write
the words a commentator says out loud and nothing else.

If the form carries nothing a commentator would say at all, or it is one of
the moments the silence section names, return an empty line and nothing goes
out. Otherwise say it: a bare surname is a whole line and it beats a line
that describes the picture.\
"""


def phraser_system(*, max_words: int = 28, examples_per_kind: int = 10) -> str:
    """The cacheable prefix: the rules, then real lines by kind.

    Byte-stable for a given pair of arguments, which is what makes sending
    ninety real utterances on every call cost a tenth of sending them once.
    """
    parts = [PHRASER_RULES.format(max_words=max_words), _examples(examples_per_kind)]
    return "\n\n".join(parts)


def _shown(kind: str, per_kind: int) -> int:
    """How many of ``kind`` to print, scaled off ``per_kind``.

    ``per_kind`` is the config's knob and used to be the count for every
    kind. With twenty-one kinds a flat count triples the cached prefix, so
    :data:`SHOWN` sets the weight per kind and this scales the lot together.
    """
    return max(1, round(SHOWN.get(kind, per_kind) * per_kind / 10))


def _examples(per_kind: int) -> str:
    """A representative sample per kind, not the whole file.

    Even strides rather than the first few, because the generated file is in
    broadcast order and the head of each list is one passage of play.
    """
    lines = [
        "REAL COMMENTARY, FROM THE CAPTIONS OF SEVEN MATCHES",
        "Six of them club football — two Premier League, two Leicester, two",
        "Barcelona — and one World Cup final, which is a seventh of them here",
        "because it is a seventh of them in life.",
        "",
        "These are transcripts, not instructions. Nothing in them is about the",
        "match you are calling: do not borrow a name, a score or an incident",
        "from them. What they are for is the shape, the length and the register.",
    ]
    for kind in KINDS:
        sample = _stride(EXAMPLES[kind], _shown(kind, per_kind))
        if not sample:
            continue
        lines.append("")
        lines.append(f"{KIND_LABELS.get(kind, kind)}")
        lines.extend(f"  {text}" for text in sample)
    return "\n".join(lines)


def _stride(items: Sequence[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


#: The moments a note may be dropped into. Two groups, and both of them are
#: lulls: play that is going on and has not resolved into anything, and a
#: dead ball that nobody has struck yet. Everything else — a shot, a save, a
#: goal, a foul, a tackle — is the moment itself, and a commentator who
#: reached for a statistic in the middle of one would be talking over it.
#:
#: The dead-ball four are safe here for the reason the rules already give
#: about the form's event field: a form that says `penalty` is a penalty
#: being *stood over*. The kick has not been taken, which is precisely why
#: there is a gap to fill.
#:
#: A substitution is on the list for the reason the corpus gives rather than
#: for the reason the others are on it: it is not a lull, it is a stoppage
#: with a name attached, and section 3 measures the median window around one
#: at **50 words** — the longest of any event kind in the study. A seat that
#: is offered no clause there says "Change for France." and stops.
QUIET_EVENTS = frozenset(
    {
        Event.BUILD_UP,
        Event.PASS,
        Event.CARRY,
        Event.STOPPAGE,
        Event.KICKOFF,
        Event.FREE_KICK,
        Event.PENALTY,
        Event.CORNER,
        Event.THROW_IN,
        Event.SUBSTITUTION,
    }
)

#: The kinds of moment that get the long line. Section 2.3: at a restart or a
#: stoppage the gap opens to 4.5-4.6 s and the median utterance is 10 words
#: against 7 in an attacking move, with one in five over sixteen. These are
#: the moments where the researched clause, the ledger count and the
#: storyline go, and the reason the word cap is lifted for them.
#:
#: :data:`Event.PENALTY` is not here. A penalty being stood over is quiet,
#: which is why it is in :data:`QUIET_EVENTS`, but it is also about to be the
#: loudest moment in the match and a twenty-word line would still be running
#: when the ball is struck.
LONG_LINE_EVENTS = frozenset(
    {
        Event.KICKOFF,
        Event.THROW_IN,
        Event.FREE_KICK,
        Event.CORNER,
        Event.SUBSTITUTION,
        Event.STOPPAGE,
    }
)

#: What a goal kick looks like from a form that has no word for one. There is
#: no ``Event.GOAL_KICK``: the caller's enum is what can be read off a
#: picture, and a goalkeeper with the ball on the edge of his own area is
#: filed as build-up or as nothing. Section 3 measures goal kicks as 43%
#: silent and, when spoken, the slot the storyline goes in — 1% of them use
#: the words "goal kick" at all — so it is worth catching cheaply rather than
#: not at all.
_GOAL_KICK_WORDS = ("goal kick", "goalkick", "goalkeeper", "the keeper", "his keeper")


def notes_allowed(line: CallerLine) -> bool:
    """Is this a moment quiet enough to drop a researched clause into?

    Asked here rather than by the runtime so that the answer is part of the
    prompt the prompt module builds, and so that a test can drive the builder
    with a penalty and a goal and see the difference without a model.

    Never on a replay, whatever the event. A replay line is a past-tense
    account of one concrete thing the picture is showing, and a number
    dropped into it is the one shape :func:`replay_block` forbids outright —
    so the clauses are not offered rather than offered and refused.

    And always on a goal kick, however the caller filed it. Section 3.1b of
    the study is that a goal kick is the commentator's slot for something
    else: 43% pass in silence and 1% use the words "goal kick", and what goes
    there instead is the storyline. A form filed ``none`` — the emptiest
    picture there is — with the goalkeeper on the ball is that slot, and
    offering it nothing to say is what makes it "Goal kick." or silence.
    """
    if line.scene is Scene.REPLAY:
        return False
    return line.event in QUIET_EVENTS or looks_like_a_goal_kick(line)


def phraser_blocks(
    line: CallerLine,
    state_summary: str,
    recent_lines: Sequence[str],
    *,
    home: str,
    away: str,
    on_the_ball: str | None = None,
    notes: Sequence[Note] = (),
    callbacks: Sequence[bool] = (),
    last_event: Event | None = None,
    followup: str = "",
    ledger: Sequence[LedgerFact] = (),
    replay_first: bool = True,
    dead_ball: DeadBallConfig = DEAD_BALL,
) -> list[Block]:
    """One call's content. Text only, and deliberately small.

    No frames. The phraser is not a second opinion on the picture — it has
    never seen the picture, and giving it one would invite it to describe
    what it saw, which is the failure this whole stage exists to fix. It gets
    the form, the state, and — in a lull — the notes about the people on it.

    ``notes`` are already filtered to the moment by the caller; whether they
    are shown at all is decided here, by the event. ``callbacks`` runs
    alongside them, one flag a note, saying which have already been said once
    in this match — see :mod:`commentary.threads`. Empty means none have, which
    is what every caller of this said before threads existed.

    ``ledger`` is the other kind of clause: counts of what has happened in
    this match, written by :mod:`commentary.ledger` out of the calls already
    made. Shown under the notes, on the same quiet moments and under the same
    rule, and checked afterwards by the gate's ``ledger_claim``.

    ``replay_first`` only means anything when the form is a replay, and it is
    the one thing about a replay the model cannot see: whether an earlier
    line in this same sequence has already said we are watching it again. See
    :func:`replay_block`.

    ``dead_ball`` is the restart slot's numbers — what the long line is asked
    for, in words. At a restart with a clause or a detail attached to it the
    body says THIS IS THE LONG ONE and asks for twice the usual length; with
    nothing attached it asks for silence. See :func:`_restart_block`.
    """
    return [
        text_block(
            _body(
                line,
                state_summary,
                recent_lines,
                home,
                away,
                on_the_ball,
                notes,
                callbacks,
                last_event,
                followup,
                ledger,
                replay_first,
                dead_ball,
            )
        )
    ]


#: The four beats of a goal call, in the order the corpus says them
#: (``docs/research/real-commentary-corpus.md`` section 8.4, and section 2.4
#: for the timing). Beat 1 is the call itself and the phraser is given no
#: block for it. Beats 2 to 4 are the follow-up, and each is one short line of
#: its own two to five seconds after the last, not a clause of a long one.
GOAL_BEATS: dict[int, str] = {
    2: """BEAT 2 — THE CELEBRATION, AND IT IS NOT A SECOND CALL. Present tense,
eight to fourteen words: where he has run, what the keeper did, what the bench
is doing, how long he has known.

DO NOT OPEN ON A NAME AND AN EXCLAMATION MARK. "<Scorer>!" is beat 1. It has
already gone out, with the score written on the end of it by the broadcast,
and a listener who hears that shape a second time hears a second goal. Three
consecutive lines opening "<Scorer>!" went out on this system — the listener
is told he has scored three times in twelve seconds — and this beat is written
the way it is to stop it. No name followed by "!" begins this line: not the
scorer's, not the keeper's, not anybody's. If the first answer does, code
takes the shout off the front and the line goes out without it.

Real ones, and each of these is one beat:

  And look at him go, straight to the corner flag.
  The keeper sent the wrong way, and the whole bench is up.
  He knew it from the moment it left his boot.
  Quick thinking by ter Stegen, and Griezmann celebrates.
  And another standing ovation. It's exhibition stuff.

Eleven words, twelve, eleven, seven, seven. Never fewer than eight: one
fragment on its own is this beat written short, and the shout carries on past
it.

The name may be anywhere in the line — "and away he goes", "<Scorer> wheels
away towards his own bench" are both this beat. What it may not be is the
first word with a shout after it.

WHEN THE CALL NAMED NOBODY, THIS IS WHERE THE NAME GOES, and the block below
says which of the two you are in. A call of "Over the wall, into the top
corner!" leaves a listener who has not been told whose goal it is, so this
beat opens on him — with a comma, not a shout. "<Scorer>, over the wall from
twenty-five yards." Not "<Scorer>! Over the wall."

No number of any kind on this beat. The score went out on the call.""",
    3: """BEAT 3 — ONE NUMBER ABOUT THE SCORER, IN A SENTENCE. This is the beat
the corpus fills about ten seconds in, and it fills it every time: "It's his
third goal of this La Liga campaign.", "Griezmann gets his fifth goal of the
season.", "11 CONSECUTIVE GOALS IN PREMIER League games.", "His first ever
goal for the club."

A WHOLE SENTENCE, WITH A SUBJECT AND A VERB, eight to fourteen words. "Six in
the tournament." is a caption, and it is what this beat keeps writing. "That's
his sixth of the tournament." and "Six goals in this World Cup now for
<Scorer>." are the same fact said by a person. The corpus goes longer still —
"A quick ball out by ter Stegen and Griezmann gets his fifth goal of the
season" is sixteen words — and this is a moment the gaps are open, so the long
line belongs here.

NOT A SHOUT, AND NOT ON HIS NAME. Like beat 2, this line does not open on a
name with an exclamation mark after it. The call was the shout. This is the
man who has just scored being counted, and a count is said, not roared.

**Name him.** A figure with nobody attached to it is not a fact about anyone,
and it is thrown away before it reaches the microphone. The scorer's name, or
a pronoun with his name already in the same line — the number alone is not a
line.

Take the number from the researched clauses below, reworded but never
renumbered. That is the whole of what this beat may contain, and it is
expected, not permitted: if there is a clause about the scorer, say it.

If there is no clause about him, do not invent one and do not reach for the
score. Say what he has done instead — where he has put it, who he beat — in a
sentence of the same length.""",
    4: """BEAT 4 — REBUILD THE MOVE, IN THE PAST TENSE. Ten to twenty seconds
after a goal the corpus goes back over how it happened, naming two or three of
the players:

  It was an excellent cross that was put back into the box by Marcelo. The
  shot rebounded off the post and fell very kindly for Casemiro who slotted
  the ball home.
  Schweinsteiger made the run. He beat Azpilicueta and headed it past a
  stranded Schmeichel.

Twelve to twenty-four words. This is the longest line anybody says about a
goal, and it is the one place a second clause is not padding, because there
are two things to say: what made it and what finished it.

THE JOIN IS THE POINT. Two flat sentences with a full stop between them —
"France drove into the box. <Scorer> came off the ground and buried the
volley." — is a list of two facts, and it is what this beat keeps writing. A
rebuilt move has connective tissue, because the man is describing one thing
happening and not two:

  It was <X>'s ball in, <Y> let it run, and <Z> was there to turn it home.
  <X> won it back on halfway, and by the time the cross came in <Y> had the
  whole six-yard box to himself.
  The shot came back off the post, and it fell for <Z>, who could not miss.

"and", "who", "by the time", "so that", "which is why" — one of those, once.
Not two full stops.

PAST TENSE FROM THE FIRST WORD, not from the second clause. This is the other
thing this beat gets wrong: it opens in the present, as though the move were
still running, and corrects itself halfway through.

  written wrong:  They drive into the box. <Scorer> off the ground, the volley buried.
  written right:  It was a ball driven into the box, and <Scorer> came off the
                  ground to bury the volley.

Same move, same names. The difference is the first verb and the join.

IT HAS TO CARRY SOMETHING THE CALL DID NOT. A rebuild that says the call
again with a past-tense verb on it is not a rebuild — it is the same six
words a third time. What the call never had room for is what goes here: the
run-up, the wall, the keeper's dive, where the ball came from, the second man
in the move. One of those, at least, or there is no line to write and an
empty one is the right answer.

  called:   Over the wall, into the top corner!
    wrong:  <Scorer> took his steps back and whipped it over the wall, into
            the top corner.
    right:  <Scorer> stood over it a long time, and the wall never moved.

Use the move and the names below and nothing else. No score, no tally, no
number.""",
}


def goal_followup_block(
    beat: int,
    *,
    since_s: float,
    scorer: str | None = None,
    notes: Sequence[Note] = (),
    moves: Sequence[str] = (),
    names: Sequence[str] = (),
    said: Sequence[str] = (),
) -> str:
    """What is due next in the thirty seconds after a goal.

    Real commentary's fastest sustained talking is the half-minute after a
    goal: 7 utterances and 60 words, median, with a longest internal gap of
    7.1 s (study section 2.4). This system said two lines of seven words and
    then went quiet, and one of the two was thrown away by the gate. The
    beats are in the corpus and they are in a fixed order, so the order is
    computed in code and the model is told which one it is writing rather
    than being asked to remember four.

    The scorer's notes are printed here rather than left to the ordinary
    `context:` block, which a goal suppresses on the grounds that a goal is no
    moment for a statistic. Ten seconds after one is exactly that moment, and
    the corpus fills it with a statistic every time.
    """
    instruction = GOAL_BEATS.get(beat)
    if instruction is None:
        return ""
    lines = [
        f"THE GOAL HAS BEEN CALLED. {since_s:.0f} seconds ago"
        + (f", by {scorer}" if scorer else "")
        + ". The score has already",
        "gone out on that call, written by the broadcast off the scoreboard. It is",
        "said. Do not say it again, in figures or in words.",
        "",
        "So are the words of the call. The description below is the move as it was",
        "already called; you are not calling it again. This line is the next thing",
        "said about a goal everybody has now seen, and if it could have gone out",
        "as the call itself it is the wrong line.",
        "",
        *_already_said(said, scorer, beat),
        instruction,
    ]
    if beat == 3:
        rows = [f"  - {note.about}: {note.text.strip()}  [{note.kind}]" for note in notes]
        lines += ["", "researched clauses about the scorer:"]
        lines += rows or ["  (none — say the moment instead, and stay off numbers)"]
    if beat == 4:
        lines += ["", "the move, as the eyes saw it:"]
        lines += [f"  - {text.strip()}" for text in moves if text.strip()] or [
            "  (nothing recorded — keep it to the scorer and the finish)"
        ]
        lines += ["", "players seen in it: " + (", ".join(names) if names else "(none read)")]
    return "\n".join(lines)


#: Every way the corpus names a replay as a replay, and the shapes this
#: system's own lines reached for. One list, next to the block that teaches
#: them, because :func:`strip_replay_marker` is the enforcement of the rule
#: :func:`replay_block` states: the sequence is named once, in its first
#: line, and the ones after it go straight at what the picture shows.
#:
#: Sourced from ``docs/research/real-commentary-corpus.md`` section 3.2 —
#: "As we see …", "Having seen the replay …", "Watch this." — plus the two
#: the model actually wrote on ``r1-replay``: at 25-38 s it named the replay
#: in two lines of three, "You see in the replay, …" and "In the replay, …".
REPLAY_MARKERS: tuple[str, ...] = (
    "having seen the replay",
    "as we see it again",
    "as we see that again",
    "as we see this again",
    "as we see it once more",
    "as we see that once more",
    "as we look at it again",
    "looking at it again",
    "seeing it again",
    "you see in the replay",
    "you can see in the replay",
    "in the replay",
    "on the replay",
    "watch the replay",
    "watch this again",
    "watch this",
    "here it is again",
    "let's see it again",
    "we see it again",
)

#: What may sit in front of a marker and still be one: "And in the replay,
#: …" is the same line as "In the replay, …".
_MARKER_LEAD_INS = ("and", "but", "so", "well", "now", "oh")


def strip_replay_marker(text: str) -> tuple[str, str]:
    """Take a leading "as we see it again" off a line, and say what was taken.

    The rule is :func:`replay_block`'s and the corpus's: a replay sequence is
    named as a replay once, in its first line, and then talked through — the
    run at 37:24 is "Watch this." / "Rakitic into Messi." / "Brilliant touch
    … and a fine finish", and only the first of the three says what is on the
    screen. Told this in the prompt, the model named the replay in two lines
    of three, so the second and third have it taken off here.

    Only a *leading* marker, and only a whole one. "the ball actually came
    off his knee in the replay" is a line about the contact with the phrase
    where it belongs, and a line that is nothing *but* the marker is left
    alone — there is no line underneath to uncover.

    Returns the line and the marker removed, ``""`` when nothing was.
    """
    stripped = text.strip()
    if not stripped:
        return text, ""
    lowered = stripped.casefold()
    for lead in _MARKER_LEAD_INS:
        if lowered.startswith(f"{lead} "):
            lowered = lowered[len(lead) + 1 :]
            break
    offset = len(stripped) - len(lowered)
    for marker in REPLAY_MARKERS:
        if not lowered.startswith(marker):
            continue
        rest = stripped[offset + len(marker) :].lstrip(" ,.:;—-")
        if not rest:
            # Nothing under it. A line that is only "Watch the replay." has
            # no second half to promote, and an empty line here would be a
            # beat dropped rather than a phrase removed.
            return text, ""
        return rest[0].upper() + rest[1:], stripped[: offset + len(marker)]
    return text, ""


#: How long a run of words counts as saying the same thing again. Three: two
#: is ordinary English — "and the", "off the" — and four lets "over the wall,
#: into the top corner" through as long as one word in the middle moves.
#: Measured on the free-kick trace, where the call, beat 2 and the rebuild
#: shared "over the wall" and "into the top corner" between them.
REPEAT_RUN = 3


def _already_said(said: Sequence[str], scorer: str | None, beat: int) -> list[str]:
    """The words that have gone out about this goal, and the ban on reusing them.

    Printed on every follow-up beat because the fault it exists for is not
    the model forgetting the call — it is the model being shown the caller's
    description of the move and saying the most vivid phrase in it once per
    beat. On ``runs/rephrased/r2-colour/freekick`` the call at 21.2 s, beat 2
    at 25.2 and the rebuild at 32.5 all carried "over the wall, into the top
    corner": one piece of information, three times, eleven seconds.

    The rule is stated in words here and enforced in code afterwards, the way
    every other rule in this prompt that a model has ignored three times now
    is — a run of three words shared with anything below and the line is
    asked for again, then dropped.
    """
    if not said:
        return []
    rows = [f"  - {text.strip()}" for text in said if text.strip()]
    if not rows:
        return []
    block = [
        "WHAT HAS ALREADY GONE OUT ABOUT THIS GOAL, the call first:",
        *rows,
        "",
        f"NOT ONE RUN OF {REPEAT_RUN} WORDS FROM ANY OF THOSE. Not the detail, not the",
        "phrase you like best in it, not reworded around the edges. Those words are",
        "spent: the listener has them. A line that repeats one is asked for again",
        "and then dropped, and a beat that is dropped is a hole in the loudest",
        "half-minute of the match.",
    ]
    if beat == 2 and scorer and not _names_him(said[0], scorer):
        block += [
            "",
            f"AND THE CALL NAMED NOBODY. {scorer} scored it and nobody listening has",
            "been told. Put his name at the front of this line, with a comma after",
            "it and no exclamation mark.",
        ]
    return [*block, ""]


def _names_him(text: str, scorer: str) -> bool:
    """Is the scorer in this line, by any part of his name?

    The same loose test the rest of the system uses: the form spells him
    "Kylian Mbappé" and the line says "Mbappé".
    """
    lowered = text.casefold()
    parts = [part for part in scorer.split() if len(part) > 2]
    return any(part.casefold() in lowered for part in parts)


def replay_block(event: Event, *, first: bool) -> str:
    """What to write over a replay, and whether this is the first of them.

    The lead used to say nothing at all here. On
    ``runs/trigger/mbappe/file-20260913-185228.jsonl`` the penalty is
    conceded at 12.9 s and the next spoken line is at 49.0 s, and in between
    the caller wrote four accurate replay lines that three separate layers
    refused. Real commentary does the opposite: section 3.2 of
    ``docs/research/real-commentary-corpus.md`` has the whole vocabulary —
    "As we see …", "Having seen the replay …", "Watch this." — and section
    2.4's half-minute after a goal is seven utterances, most of them over a
    replay.

    ``first`` is the thing the model cannot know and the corpus is strict
    about: the replay is named once per sequence and then talked through.
    "Watch this." / "Rakitic into Messi." / "Brilliant touch … and a fine
    finish" is one sequence at 37:24, and only the first of the three says
    it is a replay.

    This block replaces the goal follow-up block when a replay arrives
    inside a goal window, rather than joining it. Beat 4 *is* this line —
    the past-tense rebuild of the move — and beat 2 is a present-tense
    shout, which is the one thing a replay line may never be.
    """
    named = event.value.replace("_", " ")
    lines = [
        f"THIS IS A REPLAY OF A {named.upper()}. The game is not running behind these",
        "pictures: the viewer is watching the incident again, slowed down or from",
        "another angle, and what you say is about something that has already",
        "happened.",
        "",
        "PAST TENSE FROM THE FIRST VERB. Not from the second clause — this is the",
        "one thing this line gets wrong. Real replay talk:",
        "",
        "  It was a good first time pass as well from Ivan Rakitić.",
        "  You see in the replay the ball actually came off Ronaldo's knee.",
        "  Having seen the replay, Suárez played the ball while he was down on the ground.",
        "  Wonderful turn to get beyond the full-back.",
        "",
        "Six to twenty words. The concrete thing on the form is the point of the",
        "line — the contact, the touch, the body part, the finish — because that is",
        "what the replay is showing and the reason it is on screen.",
        "",
        "No score, in figures or in words. No tally, no count, no ordinal. Nothing",
        "in the present tense that could be mistaken for a call: not \"and it's in\",",
        "not \"he scores\", not a shout.",
    ]
    if first:
        lines += [
            "",
            "THIS IS THE FIRST LINE OF THIS REPLAY. You may name it as one, once:",
            '"as we see it again", "having seen the replay", "watch this". Once is',
            "the whole allowance for the sequence.",
        ]
    else:
        lines += [
            "",
            "THIS REPLAY HAS ALREADY BEEN NAMED. A line earlier in this sequence said",
            "we were watching it again, so do not say so a second time. Go straight",
            "at what it shows.",
        ]
    return "\n".join(lines)


#: The three forms that are one moment for the purposes of silence. The
#: corpus counts carries and passes in build-up together and finds 24% of
#: them pass with nothing said at all (study section 3.1a), and the caller
#: files the same passage of play under all three of these words depending
#: on what the ball happened to be doing in six frames.
BUILD_UP_FORMS = frozenset({Event.BUILD_UP, Event.PASS, Event.CARRY})


def looks_like_a_goal_kick(line: CallerLine) -> bool:
    """Is this nothing-much form actually a goalkeeper restarting the game?

    Only asked of ``build_up`` and ``none`` forms, because those are the two
    words the caller has for a picture in which nothing is happening, and a
    goal kick is exactly that picture. Section 3 of the corpus study: 43% of
    goal kicks pass in silence, the words "goal kick" are said at 1% of them,
    and what *does* get said there is the storyline — "been in fine goal
    scoring form for Villa Scott Sinclair with five in four appearances so
    far this season". So the slot is worth finding even by a keyword sniff,
    and the sniff is bounded by the event so that "the keeper gets a hand to
    it" on a save form is not a restart.
    """
    if line.event not in (Event.BUILD_UP, Event.NONE):
        return False
    text = f"{line.line} {line.detail or ''}".casefold()
    return any(word in text for word in _GOAL_KICK_WORDS)


def is_long_line(line: CallerLine, *, followup_beat: int | None = None) -> bool:
    """Is this a moment the corpus gives a long line to?

    Two of them, and they are the two section 2.3 measures the open gaps at:
    a restart or a stoppage (median 10 words, one in five over sixteen), and
    the thirty seconds after a goal (section 2.4: 60 words in 30 seconds,
    with beats 2 to 4 written at 8 to 24 words apiece).

    Never on a replay. A replay line is six to twenty words about one
    concrete thing on the picture, and the pictures change before a long one
    is finished.
    """
    if line.scene is Scene.REPLAY:
        return False
    if followup_beat is not None:
        return True
    return line.event in LONG_LINE_EVENTS or looks_like_a_goal_kick(line)


def _restart_block(
    line: CallerLine,
    *,
    material: bool,
    min_words: int,
    target_words: int,
) -> str:
    """THE LONG ONE, or the silence — which of the two a restart gets.

    Section 2.3 and section 3. At a restart the gap opens and the line gets
    *longer*: median 10 words against 7 in an attacking move, and one in five
    over sixteen. And a third to nearly half of restarts are not called at
    all — 43% of goal kicks, 37% of throw-ins, 33% of free-kick deliveries,
    31% of kickoffs. Which of the two happens turns on one thing: whether
    there is anything to say beyond the restart itself.

    ``material`` is that question answered in code — a researched clause, a
    count off this match, or a detail the eyes picked out. With it, this is
    where the storyline goes and the block says so. Without it, the restart
    is the silence.
    """
    if not material:
        return (
            "\nTHIS IS A RESTART WITH NOTHING ON IT, AND A THIRD OF THOSE ARE NOT\n"
            "CALLED AT ALL. No clause, no count, no detail: 43% of goal kicks, 37% of\n"
            "throw-ins, 33% of free-kick deliveries and 31% of kickoffs pass in\n"
            "silence in real commentary. Return an empty line.\n\n"
        )
    return (
        "\nTHIS IS THE LONG ONE. The ball is dead, the gap is open, and this is where\n"
        f"the context goes: {min_words} to {target_words} words, which is twice the\n"
        "length you would write with the ball moving. The restart is the short half\n"
        "of it and the clause above is the long half.\n"
        "\n"
        "The shape, and the corpus says it this way every time: the restart in a\n"
        "clause, then the storyline.\n"
        "\n"
        "  <Side> to restart, and <Name>, the man who has not been beaten in three.\n"
        "  Throw-in, deep in their own half. This is a back four with one change\n"
        "  in it all season.\n"
        "  been in fine goal scoring form for Villa Scott Sinclair with five in\n"
        "  four appearances so far this season\n"
        "\n"
        "Nothing about the restart itself is worth more than a clause. Nobody\n"
        "listening needs to be told a throw-in is a throw-in; what they cannot see\n"
        "is the fact beside it.\n\n"
    )


def _silence_nudge(line: CallerLine, last_event: Event | None) -> str:
    """One line in the body when this is a moment the corpus usually passes over.

    The rule is in the system prompt as well, where it is one paragraph among
    forty. Four rounds of rephrasing produced one chosen silence in 35 calls
    with the rule in the prompt alone, and the moment it names — the second
    consecutive build-up line about the same player — went past unremarked
    every time. So the condition is computed here, in code, and said again
    where the model is actually looking.

    The restart half of what this used to say has moved to
    :func:`_restart_block`, which has the other half of the answer as well:
    at a restart with a clause attached to it, the line is not shorter than
    usual, it is twice as long.
    """
    if line.scene is Scene.REPLAY:
        # A replay has its own block and its own reason for existing. The
        # nudge would tell it to say nothing about a picture that was put on
        # screen precisely to be talked about.
        return ""
    if line.event in BUILD_UP_FORMS and last_event in BUILD_UP_FORMS:
        return (
            "\nTHE LAST LINE WAS THIS SAME KIND OF MOMENT. If it was about this same\n"
            "player, say nothing: return an empty line. A quarter of build-up touches\n"
            "in real commentary are met with silence, and this is one of them.\n"
        )
    return ""


def _body(
    line: CallerLine,
    state_summary: str,
    recent_lines: Sequence[str],
    home: str,
    away: str,
    on_the_ball: str | None,
    notes: Sequence[Note] = (),
    callbacks: Sequence[bool] = (),
    last_event: Event | None = None,
    followup: str = "",
    ledger: Sequence[LedgerFact] = (),
    replay_first: bool = True,
    dead_ball: DeadBallConfig = DEAD_BALL,
) -> str:
    said = (
        "\n".join(f"  - {text.strip()}" for text in recent_lines if text.strip())
        or "  (nothing said yet)"
    )
    state = state_summary.strip() or "Not established yet."
    if line.scene is Scene.REPLAY:
        # Instead of the goal follow-up, never alongside it: beat 4 is a
        # past-tense rebuild of the move and so is this, and beat 2 is a
        # present-tense shout, which a replay line may never be.
        after = replay_block(line.event, first=replay_first) + "\n\n"
    else:
        after = f"{followup.strip()}\n\n" if followup.strip() else ""
    # The restart slot, and never beside the goal follow-up or the replay
    # block: each of those is already a length instruction, and two of them
    # in one body is the model choosing which to obey.
    if not after and (line.event in LONG_LINE_EVENTS or looks_like_a_goal_kick(line)):
        quiet = notes_allowed(line)
        after = _restart_block(
            line,
            material=bool((quiet and (notes or ledger)) or (line.detail or "").strip()),
            min_words=dead_ball.min_words,
            target_words=dead_ball.target_words,
        )
    return (
        f"THE TEAMS\n  {home} (home) v {away} (away)\n\n"
        f"{_state_heading(line)}\n"
        f"{state}\n\n"
        "WHAT THE EYES SAW — the form, filled in by whoever is watching\n"
        f"{_form(line, home, away, on_the_ball)}\n\n"
        f"{_context(line, notes, callbacks, ledger)}\n\n"
        f"{after}"
        f"{_silence_nudge(line, last_event)}"
        "THE LAST LINES SPOKEN, oldest first, with the kind of moment each was\n"
        "about in brackets after it. Do not repeat or paraphrase these, do not\n"
        "open on a word the last two opened on, and read the kinds: build-up,\n"
        "a pass or a carry twice running about the same player is the moment\n"
        "to say nothing.\n"
        f"{said}\n\n"
        "Say it."
    )


def _state_heading(line: CallerLine) -> str:
    """What MATCH STATE is for on this call, which depends on the event.

    Everywhere else the state is context and saying any of it out loud is a
    score claim. On a goal it is also the source of the third beat, and the
    heading has to say so or the rule about not repeating the score wins the
    argument and the beat never appears.
    """
    # A replay takes the plain heading whatever the event: no score is
    # appended to a replay line, so the goal heading below — which exists to
    # tell the model the broadcast is about to write the number for it —
    # would be promising something that does not happen.
    if line.event is not Event.GOAL or line.scene is Scene.REPLAY:
        return "MATCH STATE — for context only. Never say the score or the clock."
    return (
        "MATCH STATE — for context only, on a goal as much as anywhere else.\n"
        "The score below is going out on this line already: the broadcast\n"
        "appends it after your words, off these numbers. Yours is the name and\n"
        "the how. Write no number at all."
    )


def _context(
    line: CallerLine,
    notes: Sequence[Note],
    callbacks: Sequence[bool] = (),
    ledger: Sequence[LedgerFact] = (),
) -> str:
    """The `context:` block: researched clauses, or an explicit nothing.

    And the `ledger:` block under it, which is the same offer made out of
    counts this match produced rather than out of the pack. Both are held to
    the same moment: a goal or a penalty is too big for an aside of either
    kind, and :func:`notes_allowed` decides that once for the pair.

    Always printed, even when empty, and that is deliberate. A block that
    appears and disappears teaches a model that its absence means "use your
    own knowledge"; a block that is always there and sometimes says none
    teaches it that none means none.

    A clause marked *said before* is a callback, and the corpus is emphatic
    about what a callback is: the same number in a new form. Vardy's record
    goes out as "11 consecutive games", then "the record scorer", then "he
    scored in 11 consecutive games now", then "the 11th consecutive Premier
    League game" (study section 7.1). Four sayings, four shapes, one figure.
    So the mark is on the row and the rule is under it, because a callback
    repeated word for word is not a thread, it is a loop.
    """
    quiet = notes_allowed(line)
    if not notes or not quiet:
        reason = (
            "too big a moment for an aside"
            if notes
            else "nothing researched about anybody on this form"
        )
        block = [f"context:\n  (none — {reason})"]
    else:
        flags = list(callbacks) + [False] * (len(notes) - len(callbacks))
        rows = [
            f"  - {note.about}: {note.text.strip()}  [{note.kind}]"
            + ("  [SAID BEFORE]" if said else "")
            for note, said in zip(notes, flags, strict=False)
        ]
        block = [
            "context: verified notes. Your line MAY carry ONE of these clauses,",
            "reworded but not renumbered, about somebody the line names. Or none.",
            *rows,
        ]
        if any(flags):
            block += [
                "",
                "A clause marked [SAID BEFORE] has already gone out once in this match.",
                "That is not a reason to avoid it — a fact said twice is what makes a",
                "story run — but say it in a different shape from the one it has here:",
                "same number, new words, and shorter than the first time.",
            ]
    return "\n".join([*block, "", _ledger_block(ledger if quiet else ())])


def _ledger_block(ledger: Sequence[LedgerFact]) -> str:
    """The `ledger:` block: counts off this match, or an explicit nothing.

    Always printed, for the reason :func:`_context` gives for always printing
    the notes: a block that comes and goes teaches a model that its absence
    means "use your own knowledge".

    These are not the pack's facts and the difference is worth the model
    knowing, which is why they are a block of their own rather than more rows
    under `context:`. A note was looked up by a person before kickoff; a
    ledger clause was counted by this system out of the calls it has already
    made tonight, and it is true of what the broadcast showed. The clause is
    handed over finished — :mod:`commentary.ledger` wrote the figure and the
    words around it — so the only thing left to do with it is fit it into a
    sentence.
    """
    if not ledger:
        return "ledger:\n  (none — nothing counted yet about anybody on this form)"
    rows = [f"  - {fact.text.strip()}" for fact in ledger]
    return "\n".join(
        [
            "ledger: counts this broadcast has made for itself, so far tonight.",
            "Same rule as above and the same one clause: say it in your own words",
            "about somebody your line names, and do not change the number.",
            *rows,
        ]
    )


def _form(line: CallerLine, home: str, away: str, on_the_ball: str | None) -> str:
    """The caller's form, laid out as facts rather than as prose.

    ``line`` is presented as a description of what was seen and labelled as
    one, because it is the thing the phraser must not copy. Left unlabelled
    it reads as a draft to be lightly edited, and a lightly edited caption is
    still a caption.
    """
    rows = [
        f"  scene: {_scene(line.scene)}",
        # Loud, and second, because it is the field the phrasing has to obey.
        # Every invented outcome the stage produced on its first two traces
        # was the line reaching past this word: a penalty being waited on
        # written as a penalty struck, a foul written as a booking.
        f"  EVENT: {_event(line.event)}  <- your line must be about this and",
        "         nothing later than this",
    ]
    team = line.team or _team_of(line.side, home, away)
    if team:
        rows.append(f"  team in possession: {team}")
    names = _names(line)
    if on_the_ball and on_the_ball not in names:
        names = [on_the_ball, *names]
    if names:
        rows.append(f"  players identified: {', '.join(names)}")
        rows.append("  (these, and any name in the description below, are the only")
        rows.append("   player names you may use)")
    else:
        # Not the same as "name nobody". The caller may name a player it read
        # seconds ago and can no longer see — the carry rule — and that name
        # is in the description with no sighting behind it. Telling the
        # phraser there is nobody would throw it away, and a line that says
        # "Portugal" where the caller said "Ronaldo" is the failure this
        # whole stage is meant to fix in the other direction.
        rows.append("  players identified: none read off the picture this call")
        rows.append("  (you may still use a name that appears in the description below)")
    rows.append(f"  confidence in all of the above: {line.confidence:.2f}")
    detail = (line.detail or "").strip()
    if detail:
        # Named rather than folded into the description, because the whole
        # trouble is that a model compressing a sentence keeps the name and
        # drops everything that made the sentence worth saying. Pulled out
        # and labelled, the thing to keep is not a judgement call any more.
        rows.append(f"  detail: {detail}")
        rows.append("  (the one concrete thing here a listener could not guess. Keep it.)")
    rows.append("")
    rows.append("  what was seen, written down as a description. DO NOT say this back.")
    rows.append("  Take the facts out of it and say them the way a commentator would,")
    rows.append(f"  using only what is here and only about a {_event(line.event)},")
    rows.append("  keeping one concrete detail of the action and dropping the rest:")
    rows.append(f"    {line.line.strip()}")
    return "\n".join(rows)


def _names(line: CallerLine) -> list[str]:
    """Every name the form carries, in order, without repeats."""
    found: list[str] = []
    for sighting in line.sightings:
        name = (sighting.name or "").strip()
        if name and name not in found:
            found.append(name)
    return found


def _team_of(side: Side, home: str, away: str) -> str:
    if side is Side.HOME:
        return home
    if side is Side.AWAY:
        return away
    return ""


def _scene(scene: Scene) -> str:
    return scene.value.replace("_", " ")


def _event(event: Event) -> str:
    return event.value.replace("_", " ")
