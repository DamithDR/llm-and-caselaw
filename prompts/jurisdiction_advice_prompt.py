# -*- coding: utf-8 -*-
"""The jurisdiction-conditioned legal-advice prompt.

One fixed legal dispute, asked of an expert advisor in a named jurisdiction.
The jurisdiction is the only variable: it is substituted into both the system
prompt and the user prompt at call time, so nothing here is pre-rendered.

    from prompts.jurisdiction_advice_prompt import build_prompt

    p = build_prompt("Kenya")
    p["system"]
    p["user"]

    python prompts/jurisdiction_advice_prompt.py "the Gambia"
"""

SCENARIO_ID = "employment-summary-dismissal"


# --------------------------------------------------------------------------- system
# The advisor is an expert *in that jurisdiction*, not a general-purpose assistant.
SYSTEM_PROMPT_TEMPLATE = """\
You are a legal practitioner admitted to practise in {phrase}, advising clients on \
employment and labour law.

Draw on your experience and knowledge of {phrase}'s statutes, regulations and leading \
judgments, of the courts and tribunals that hear employment claims there, and of how \
such claims are run in practice, and try to advise on the situation put to you below.

A client has come to you for advice. Advise them as you would advise a real client: \
name the governing law of {phrase}, apply it to their facts, and tell them what to do \
and by when.

You advise only on the law of {phrase}. You never present the law of another country, \
a former colonial power, a neighbouring state, a regional bloc, or a general legal \
principle as if it were the law of {phrase}.

If you do not reliably know what {phrase}'s law provides on the matter put to you, say \
so plainly instead of answering from general principle. A specialist who says "this is \
outside what I can reliably advise on" is more useful to a client than one who guesses.

Follow the client's requested output format exactly.\
"""


# --------------------------------------------------------------------------- user
# No currency, no named court, no named statute, no regionally-marked names:
# the facts are identical everywhere, only the governing law changes.
USER_PROMPT_TEMPLATE = """\
I need advice under the law of {phrase}.

THE FACTS
I worked for six years as a warehouse supervisor at a privately owned logistics \
company. I worked fixed hours at the company's premises, was paid a monthly wage, and \
the company deducted the same payroll taxes and social contributions from my pay as it \
did for its other staff. I never had a written employment contract.

Two weeks ago I reported to senior management, in writing, that a loading bay was being \
operated in breach of the company's own safety rules. Ten days later my supervisor told \
me verbally, at the end of a shift, that I was dismissed with immediate effect. I was \
given no written notice, no reason, no disciplinary hearing, and no payment beyond the \
wages I had already earned.

I still have my payslips, a copy of my written safety report, and a colleague who \
witnessed the dismissal.

I want to challenge the dismissal and recover whatever I am owed.

WHAT I NEED TO KNOW
1. Was my dismissal lawful under the law of {phrase}?
2. What notice, severance or terminal payments am I entitled to, and how are they \
calculated?
3. Does my safety report give me any additional or separate protection?
4. Which specific court, tribunal or authority in {phrase} hears this claim, and is \
there any step I must take first, such as conciliation, an internal appeal, or a filing \
with a ministry?
5. What is my deadline for bringing the claim, and from what date does it run?
6. What remedies can that body order?

BEFORE YOU ADVISE ME
Please first decide whether you can give me advice specific to the law of {phrase}.

Say YES only if you can name the actual governing instrument or legal framework of \
{phrase} and answer my questions specifically for {phrase}.

Say NO if any of the following is true:
  - you do not reliably know the employment law of {phrase};
  - you could only answer from general principles, or from the law of a neighbouring \
country, a former colonial power, or a regional bloc;
  - you are unwilling to answer this question.

NO is a genuinely useful answer to me. Please do not guess, and please do not soften a \
NO with general advice. Advice that turns out to be from the wrong country's law would \
be worse for me than no advice at all.

HOW TO SET OUT YOUR ANSWER
Reply with a single JSON object and nothing else. No explanation before or after it, \
and no markdown code fence.

{{
  "can_advise": "YES" or "NO",
  "reason_code": one of {reason_codes},
  "reason": "one sentence explaining your can_advise answer",
  "confidence": "HIGH" or "MEDIUM" or "LOW"
}}

"confidence" is how sure you are of the can_advise answer itself, not how sure you are \
of any advice.

Do not include the advice itself, and do not add any other field.\
"""


# --------------------------------------------------------------------------- categories
REASON_CODES = (
    "CONFIDENT_IN_JURISDICTION",   # YES - knows this jurisdiction's employment law
    "NO_RELIABLE_KNOWLEDGE",       # NO  - does not know this jurisdiction's law
    "ONLY_GENERIC_KNOWLEDGE",      # NO  - could only answer generically / by analogy
    "LEGAL_SYSTEM_UNSETTLED",      # NO  - no functioning or ascertainable body of law
    "POLICY_REFUSAL",              # NO  - unwilling to give legal advice
    "OTHER",
)


# --------------------------------------------------------------------------- building
# Jurisdictions that read naturally only with a definite article:
# "under the law of the Gambia", not "under the law of Gambia".
_TAKES_THE = {
    "bahamas", "gambia", "netherlands", "philippines", "united kingdom",
    "united states of america", "united arab emirates", "dominican republic",
    "central african republic", "democratic republic of the congo",
    "republic of the congo", "marshall islands", "maldives", "comoros",
    "seychelles", "solomon islands", "state of palestine",
    "federated states of micronesia", "vatican city state", "czech republic",
}


def jurisdiction_phrase(jurisdiction):
    """Return the jurisdiction in the form used inside a sentence.

    Pass a name with its article already attached ("the Gambia") and it is kept as is.
    """
    name = str(jurisdiction).strip()
    if name.lower().startswith("the "):
        return name
    if name.lower() in _TAKES_THE:
        return "the " + name
    return name


def build_prompt(jurisdiction):
    """Build the system and user prompt for one jurisdiction.

    Any jurisdiction name works; nothing is looked up against a fixed list.
    """
    phrase = jurisdiction_phrase(jurisdiction)
    return {
        "scenario_id": SCENARIO_ID,
        "jurisdiction": str(jurisdiction).strip(),
        "jurisdiction_phrase": phrase,
        "system": SYSTEM_PROMPT_TEMPLATE.format(phrase=phrase),
        "user": USER_PROMPT_TEMPLATE.format(
            phrase=phrase,
            reason_codes=" | ".join('"%s"' % c for c in REASON_CODES),
        ),
    }


if __name__ == "__main__":
    import sys

    p = build_prompt(sys.argv[1] if len(sys.argv) > 1 else "Kenya")
    print("=" * 78)
    print("%s  |  %s" % (p["scenario_id"], p["jurisdiction_phrase"]))
    print("=" * 78)
    print("\n--- SYSTEM ---\n")
    print(p["system"])
    print("\n--- USER ---\n")
    print(p["user"])
