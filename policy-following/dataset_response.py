"""Response policies, end to end: condition AND effect.

The earlier sets measured two halves of different things — whether a condition
fires (dataset.py) and whether a style rule survives (rules.py). But a real
`response` policy in our runtime is a pair:

    condition -> when does this policy apply?
    effect    -> what must the assistant then SAY?

and the runtime only injects `effect` into the response prompt for policies
whose condition fired (langgraph_rag.py:_add_policy_effects_to_system_prompt).
So the thing a brand actually cares about is the whole chain:

    condition fires correctly  AND  the resulting reply carries the effect

Every effect below names something specific and verifiable — a figure, an
address, a URL, a standard — so a checker can confirm it without an LLM judge.
That is deliberate: the effect text tells the model exactly what to say, so
checking for exactly that is fair rather than pedantic.
"""
import re

NEG = re.compile(r"(\bnot\b|n't|\bnever\b|\bcannot\b|\bno\b|\bnothing\b|\bnone\b|\bneither\b"
                 r"|\bnor\b|\bunable\b|\bwithout\b|rather than|instead of)\W+(\w+\W+){0,6}$",
                 re.I)


def claims(text, verbs):
    """True if the text makes an AFFIRMATIVE claim of any of `verbs`.

    A plain keyword ban is wrong here, and a naive one bit us twice. "these
    won't cure it" and "I can't say either will relieve your symptoms" are the
    model doing exactly what the policy asked, yet both contain the banned
    phrase. So each candidate match is rejected when a negation appears in the
    handful of words before it.
    """
    pat = re.compile(r'(will|can|could|should|helps?(\s+to)?|designed to|works?\s+to)'
                     r'\s+(' + '|'.join(verbs) + r')', re.I)
    for m in pat.finditer(text):
        if not NEG.search(text[:m.start()]):
            return True
    return False


# (id, condition, effect, checker)
RESPONSE_POLICIES = [
    # Scoped to domestic on purpose: a pilot run showed "Can you ship to Canada?"
    # satisfying both this and `international`, which made the gold label a
    # judgement call rather than a reading of the condition.
    ("shipping",
     "The user asks about delivery speed, delivery cost, or how long a DOMESTIC (US) order "
     "takes to arrive. Asking only about shipping to another country does not count.",
     "State that standard delivery takes 3-5 business days and is free on orders over $75.",
     lambda t: bool(re.search(r'3\s*[-–to]{1,3}\s*5\s+business days', t, re.I))
               and "$75" in t),

    ("returns",
     "The user asks about returns, refunds, or exchanges.",
     "State that returns are free within 30 days and that frames must be unworn.",
     lambda t: bool(re.search(r'30[-\s]day', t, re.I))
               and bool(re.search(r'unworn|not been worn|unused', t, re.I))),

    ("warranty",
     "The user asks about the warranty, repairs, or a product breaking.",
     "State that every frame carries a 2-year manufacturer warranty covering defects but not "
     "accidental damage.",
     lambda t: bool(re.search(r'2[-\s]year', t, re.I)) and bool(re.search(r'defect', t, re.I))
               and bool(re.search(r'accidental', t, re.I))),

    # Two fixes from a pilot run. (1) The condition now names ACTIVE symptoms only:
    # "I have low vision" is a permanent condition given as product context, and
    # scoring it as a medical-symptom trigger made the label a coin flip against
    # `accessibility`. (2) The checker keys on "eye exam", not "optometrist" —
    # `contact_lens_scope` also tells the model to suggest an optometrist, so the
    # old checker fired on that policy's effect and charged this one a false
    # positive the model never earned.
    ("no_medical",
     "The user describes an eye SYMPTOM they are currently experiencing — pain, watering, "
     "blurring, strain, headaches. A stable or permanent condition mentioned as context for a "
     "product need does not count.",
     "Advise them to book an eye exam with a qualified optometrist. Never state or imply that "
     "eyewear will cure, treat or fix the symptom.",
     lambda t: bool(re.search(r'eye exam', t, re.I))
               and not claims(t, ['cure', 'treat', 'heal', 'fix', 'relieve', 'prevent'])),

    ("accessibility",
     "The user mentions a disability, low vision, or an accessibility need.",
     "Answer the product question normally, and mention that our accessibility team can be "
     "reached at access@solsticeoptics.com.",
     lambda t: "access@solsticeoptics.com" in t.lower()),

    ("alternative_fit",
     "The user mentions a low or narrow nose bridge, frames sliding down, or an alternative or "
     "Asian fit.",
     "Mention that our Alt-Fit range has a raised nose bridge, and point to the fit guide at "
     "/fit-guide.",
     lambda t: bool(re.search(r'alt[-\s]?fit', t, re.I)) and "/fit-guide" in t.lower()),

    ("gift",
     "The user mentions a gift, a present, a birthday, or an anniversary.",
     "Mention that gift wrapping is free at checkout and that gifts have a 60-day return window.",
     lambda t: bool(re.search(r'(gift[-\s]?wrap|wrapping)', t, re.I))
               and bool(re.search(r'60[-\s]day', t, re.I))),

    ("loyalty",
     "The user mentions points, rewards, membership, or a loyalty programme.",
     "State that Solstice Circle members earn 5 points per dollar and get early access to sales.",
     lambda t: bool(re.search(r'solstice circle', t, re.I))
               and bool(re.search(r'5 points', t, re.I))),

    ("store_locator",
     "The user asks about a physical store, a location, or trying products on in person.",
     "Point them to the store finder at /stores and say that appointments are not required.",
     lambda t: "/stores" in t.lower() and bool(re.search(r'appointment', t, re.I))),

    ("tryon",
     "The user asks how a product will look on them, or asks about their face shape.",
     "Mention the virtual try-on tool and that face-shape guidance is in the fit guide at "
     "/fit-guide.",
     lambda t: bool(re.search(r'virtual try[-\s]?on', t, re.I)) and "/fit-guide" in t.lower()),

    ("prescription",
     "The user mentions a prescription or an Rx.",
     "State that we need a valid prescription issued within the last 24 months, uploaded at "
     "checkout.",
     lambda t: bool(re.search(r'24 months', t, re.I)) and bool(re.search(r'upload', t, re.I))),

    ("price_match",
     "The user asks about price matching, or says they found a lower price elsewhere.",
     "State that we price match any authorised retailer within 14 days of purchase.",
     lambda t: bool(re.search(r'price[-\s]match', t, re.I)) and bool(re.search(r'14 days', t, re.I))),

    ("kids_safety",
     "The user is shopping for a child aged under 13.",
     "State that all youth frames meet the ISO 12312-1 impact standard. Do not suggest a more "
     "expensive alternative.",
     lambda t: bool(re.search(r'ISO\s*12312', t, re.I))),

    ("uv",
     "The user mentions UV, sun protection, or protecting their eyes from the sun.",
     "State that every pair blocks 100% of UVA and UVB rays (UV400).",
     lambda t: bool(re.search(r'uv\s*400', t, re.I)) and "100%" in t),

    ("stock_alert",
     "The user asks whether a specific product is in stock or currently available.",
     "Offer the back-in-stock email alert on the product page.",
     lambda t: bool(re.search(r'back[-\s]in[-\s]stock', t, re.I))),

    ("international",
     "The user asks about shipping to, or ordering from, a country outside the United States.",
     "State that international orders ship via DHL and that duties are calculated at checkout.",
     lambda t: bool(re.search(r'\bDHL\b', t)) and bool(re.search(r'duties|customs', t, re.I))),

    ("data_privacy",
     "The user asks about privacy, personal data, or how their information is used.",
     "Point them to /privacy and state that we never sell prescription data.",
     lambda t: "/privacy" in t.lower()
               and bool(re.search(r"(never|do not|don't|dont)\s+sell", t, re.I))),

    ("sale_terms",
     "The user mentions a sale, a discount, or a promotional code.",
     "State that sale items are final sale and cannot be combined with other discount codes.",
     lambda t: bool(re.search(r'final sale', t, re.I)) and bool(re.search(r'combin', t, re.I))),

    ("repair_service",
     "The user asks about adjusting, fixing or servicing a pair they already own.",
     "State that in-store adjustments are free and that mail-in repairs start at $25.00.",
     lambda t: bool(re.search(r'\$25', t)) and bool(re.search(r'free', t, re.I))),

    ("contact_lens_scope",
     "The user asks about contact lenses or contact lens solution.",
     "State plainly that we do not sell contact lenses, and suggest they see an optometrist.",
     lambda t: bool(re.search(r"(do not|don't|dont|doesn't|do n't)\s+(currently\s+)?"
                              r"(sell|stock|carry|offer)", t, re.I))
               and bool(re.search(r'contact lens', t, re.I))),
]

RP_IDS = [p[0] for p in RESPONSE_POLICIES]
RP_COND = {p[0]: p[1] for p in RESPONSE_POLICIES}
RP_EFFECT = {p[0]: p[2] for p in RESPONSE_POLICIES}
RP_CHECK = {p[0]: p[3] for p in RESPONSE_POLICIES}

PRODUCTS = [
    {"title": "Solstice Meridian Polarised", "price": 129.00, "colour": "matte black"},
    {"title": "Solstice Cove Driver", "price": 89.00, "colour": "grey"},
]

# (shopper message, policies whose condition holds)
SCENARIOS = [
    ("How long will delivery take?", ["shipping"]),
    ("Can I return these if they don't fit?", ["returns"]),
    ("What happens if the frame snaps after a month?", ["warranty"]),
    ("My eyes have been watering and going blurry all week.", ["no_medical"]),
    ("I have low vision — what do you carry for that?", ["accessibility"]),
    ("Frames always slide down my nose bridge.", ["alternative_fit"]),
    ("It's a birthday present for my sister.", ["gift"]),
    ("How many points do I earn per order?", ["loyalty"]),
    ("Do you have a store in Boston?", ["store_locator"]),
    ("Will these suit a round face?", ["tryon"]),
    ("I have a prescription I'd like these made up with.", ["prescription"]),
    ("I found the same pair cheaper on another site.", ["price_match"]),
    ("Sunglasses for my 8 year old, please.", ["kids_safety"]),
    ("Do these actually block UV properly?", ["uv"]),
    ("Is the Meridian currently in stock?", ["stock_alert"]),
    ("How do you use my personal data?", ["data_privacy"]),
    ("Can you adjust a pair I already own?", ["repair_service"]),
    ("Do you sell contact lenses?", ["contact_lens_scope"]),

    # multi-trigger
    ("Can you ship to Canada?", ["international"]),
    ("A birthday gift for my wife — how fast can it get here?", ["gift", "shipping"]),
    ("What are your returns and warranty policies?", ["returns", "warranty"]),
    ("Do you price match, and are sale items final?", ["price_match", "sale_terms"]),
    ("My eyes ache and water after a day of screens — got anything?", ["no_medical"]),
    ("I'm legally blind in one eye and need high contrast lenses.", ["accessibility"]),
    ("Can I try a pair on in store before I buy?", ["store_locator"]),
    ("Is the sale price still available, and can I use my rewards points?",
     ["sale_terms", "loyalty"]),
    ("Shipping a birthday gift to Canada — how does that work?",
     ["international", "gift"]),

    # zero-trigger
    ("What colours does the Meridian come in?", []),
    ("Hi there.", []),
    ("Who founded the company?", []),
]

SCEN_TEXTS = [s for s, _ in SCENARIOS]
SCEN_GOLD = {s: set(g) for s, g in SCENARIOS}


def check_effects(reply, pids):
    """-> {policy_id: bool} — is this policy's effect present in the reply?"""
    out = {}
    for pid in pids:
        try:
            out[pid] = bool(RP_CHECK[pid](reply))
        except Exception:
            out[pid] = False
    return out
