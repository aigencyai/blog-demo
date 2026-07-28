"""Response-shaping rules that a brand would actually configure, each written
so a deterministic checker can decide compliance with no LLM in the loop.

This is the `response` policy type: rules that change how the assistant speaks
rather than what it retrieves. Brand voice, legal guardrails and inclusion
requirements all land here, which is why silent non-compliance matters.

Every checker returns True (complied), False (violated), or None (not
applicable to this reply). None is never scored.
"""
import re

COMPETITORS = ["lenscrafters", "warby parker", "sunglass hut", "zenni", "eyebuydirect"]
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF✀-➿️]")
CLOSING = "Ask us anything else about your eyewear."


def _sentences(text):
    # Split on terminal punctuation followed by a capital/quote or end of string,
    # so decimal points inside prices ($129.00) do not create a sentence break.
    parts = re.split(r'(?<=[.!?])\s+(?=["\'“]?[A-Z0-9])', text.strip())
    return [p for p in parts if p.strip()]


def _body(text):
    """Reply with the mandated closing line removed, for checks that should not
    be confounded by it."""
    return text.replace(CLOSING, " ")


# Two things a naive keyword ban gets wrong, both of which flag compliant replies:
# the verb may be aimed at something that is not a medical condition ("prevent
# slipping"), and it may be negated ("they do not treat any vision condition").
# A claim only counts when the verb targets something medical AND is not negated.
_MEDICAL = (r'(condition|conditions|disease|symptom|symptoms|illness|ailment|strain|'
            r'headache|headaches|migraine|migraines|myopia|astigmatism|cataract|'
            r'cataracts|glaucoma|dry eye|eye damage|retinal|vision loss|blindness|fatigue)')
_CLAIM = re.compile(r'\b(cure|cures|treat|treats|prevent|prevents|heal|heals)\b'
                    r'(?:\W+\w+){0,4}?\W+' + _MEDICAL, re.I)
_NEG = re.compile(r"(\bnot\b|n't|\bnever\b|\bcannot\b|\bno\b|\bnothing\b|\bnone\b|"
                  r"\bneither\b|\bnor\b|\bunable\b|\bwithout\b|\bdon'?t\b|\bdoes\s?n'?t\b|"
                  r"rather than|instead of)\W+(\w+\W+){0,6}$", re.I)


def _medical_claim(text):
    for m in _CLAIM.finditer(text):
        if not _NEG.search(text[:m.start()]):
            return True
    return False


RULES = [
    ("R01_sentence_count",
     "Your reply must be between 3 and 5 sentences long, not counting the closing line.",
     lambda t: 3 <= len(_sentences(_body(t))) <= 5),

    ("R02_no_questions",
     "Never ask the shopper a question. Never use a question mark.",
     lambda t: "?" not in t),

    ("R03_closing_line",
     f'End every reply with exactly this sentence: "{CLOSING}"',
     lambda t: t.strip().endswith(CLOSING)),

    ("R04_no_cheap",
     'Never describe anything as "cheap". Use "great value" instead.',
     lambda t: "cheap" not in t.lower()),

    ("R05_price_format",
     "Write every price with a dollar sign and exactly two decimal places, e.g. $129.00.",
     lambda t: (None if not re.search(r'\$\s?\d', t)
                else all(re.fullmatch(r'\d{1,3}(,\d{3})*\.\d{2}', m)
                         for m in re.findall(r'\$\s?([\d,]+(?:\.\d+)?)', t)))),

    ("R06_no_competitors",
     "Never mention a competing retailer by name.",
     lambda t: not any(c in t.lower() for c in COMPETITORS)),

    # Accepts the spelled-out form: a reply saying "a thirty-day return window"
    # states the required fact exactly as plainly as "30-day" does, and the rule
    # asks the model to state a fact, not to pick a numeral.
    ("R07_returns_window",
     "Whenever you recommend a product, state that it ships with a 30-day return window.",
     lambda t: bool(re.search(r'(30|thirty)[-\s]day', t, re.I))),

    ("R08_no_exclamation",
     "Never use an exclamation mark.",
     lambda t: "!" not in t),

    ("R09_second_person",
     'Address the shopper as "you". Never refer to them as "the customer".',
     lambda t: "the customer" not in t.lower()),

    # Stated unconditionally on purpose: a rule whose text is conditional
    # ("when sunglasses are discussed") but whose checker is not would penalise
    # replies the rule never applied to.
    ("R10_uv400",
     'State that the eyewear carries "UV400 protection".',
     lambda t: "uv400" in t.lower().replace(" ", "")),

    ("R11_no_medical_claims",
     "Never say eyewear can cure, treat, prevent or heal any medical condition.",
     lambda t: not _medical_claim(t)),

    ("R12_british_spelling",
     'Use British spelling throughout: "colour", "grey", "customise".',
     lambda t: (None if not re.search(r'\b(colou?rs?|gr[ae]y|customi[sz]e)\b', t, re.I)
                else not re.search(r'\b(colors?|gray|customize)\b', t, re.I))),

    ("R13_no_filler_opener",
     'Never begin a reply with "Sure", "Certainly", "Of course", "Absolutely" or "Great".',
     lambda t: not re.match(r'^\s*(sure|certainly|of course|absolutely|great)\b', t, re.I)),

    ("R14_no_emoji",
     "Never use emoji.",
     lambda t: not EMOJI.search(t)),

    ("R15_brand_once",
     'Mention "Solstice Optics" exactly once.',
     lambda t: len(re.findall(r'solstice optics', t, re.I)) == 1),

    ("R16_fit_tip",
     'Include one line that begins with "Fit tip:" giving a fit or sizing pointer.',
     lambda t: bool(re.search(r'^\s*fit tip:', t, re.I | re.M))),

    ("R17_no_perfect",
     'Never use the word "perfect".',
     lambda t: "perfect" not in t.lower()),

    ("R18_sentence_length",
     "Keep every sentence to 25 words or fewer.",
     lambda t: all(len(s.split()) <= 25 for s in _sentences(_body(t)))),

    ("R19_no_recommend",
     'Never use the word "recommend" or "recommendation".',
     lambda t: not re.search(r'recommend', t, re.I)),

    ("R20_no_first_person_singular",
     'Never write "I". Speak for the store as "we".',
     lambda t: not re.search(r'\bI\b', t)),

    ("R21_no_options",
     'Never use the word "options".',
     lambda t: "options" not in t.lower()),

    ("R22_numbers_as_words",
     'Write whole numbers below ten as words ("two", not "2"). Prices are exempt.',
     lambda t: not re.search(r'(?<![\d$.,\-])[1-9](?![\d.,\-])', _body(t))),

    ("R23_no_stylish",
     'Never use the word "stylish".',
     lambda t: "stylish" not in t.lower()),

    ("R24_open_with_consider",
     'Begin every reply with the word "Consider".',
     lambda t: bool(re.match(r'^\s*consider\b', t, re.I))),
]

RULE_IDS = [r[0] for r in RULES]
RULE_TEXT = {r[0]: r[1] for r in RULES}
RULE_CHECK = {r[0]: r[2] for r in RULES}


def check_all(text, rule_ids):
    """-> {rule_id: True/False/None}"""
    out = {}
    for rid in rule_ids:
        try:
            out[rid] = RULE_CHECK[rid](text)
        except Exception:
            out[rid] = None
    return out


# --- shopper scenarios ----------------------------------------------------
# Each carries retrieved products, mirroring how our response node is fed.

SCENARIOS = [
    ("I need polarized sunglasses for driving.",
     [{"title": "Solstice Meridian Polarised", "price": 129, "colour": "matte black"},
      {"title": "Solstice Cove Driver", "price": 89, "colour": "grey"}]),
    ("Looking for lightweight frames for everyday wear.",
     [{"title": "Solstice Feather Ti", "price": 215, "colour": "gunmetal"},
      {"title": "Solstice Aero Rim", "price": 175, "colour": "silver"}]),
    ("Something for my wife's birthday, she likes classic styles.",
     [{"title": "Solstice Rivera Cat-Eye", "price": 189, "colour": "tortoiseshell"},
      {"title": "Solstice Lumen Round", "price": 149, "colour": "rose gold"}]),
    ("What do you have for cycling?",
     [{"title": "Solstice Ridgeline Wrap", "price": 159, "colour": "neon yellow"},
      {"title": "Solstice Sprint Shield", "price": 139, "colour": "black"}]),
    ("I want the most affordable sunglasses you carry.",
     [{"title": "Solstice Basecamp", "price": 39, "colour": "black"},
      {"title": "Solstice Dune", "price": 45, "colour": "sand"}]),
    ("Show me your premium designer frames.",
     [{"title": "Solstice x Aurelio Milano", "price": 640, "colour": "havana"},
      {"title": "Solstice Atelier 01", "price": 720, "colour": "gold"}]),
    ("Glasses for long days at a computer.",
     [{"title": "Solstice Focus Blue", "price": 119, "colour": "clear"},
      {"title": "Solstice Screen Ease", "price": 99, "colour": "grey"}]),
    ("Sunglasses for a beach holiday.",
     [{"title": "Solstice Lagoon", "price": 109, "colour": "aqua"},
      {"title": "Solstice Shoreline", "price": 95, "colour": "white"}]),
    ("My nose bridge is low and frames slide down. Any suggestions?",
     [{"title": "Solstice Kestrel Alt-Fit", "price": 165, "colour": "matte navy"},
      {"title": "Solstice Haven Alt-Fit", "price": 145, "colour": "black"}]),
    ("I have low vision — do you carry anything high contrast?",
     [{"title": "Solstice Clarity Amber", "price": 179, "colour": "amber"},
      {"title": "Solstice Contrast Pro", "price": 199, "colour": "copper"}]),
    ("Which frames suit a round face?",
     [{"title": "Solstice Angular D-Frame", "price": 155, "colour": "black"},
      {"title": "Solstice Square Line", "price": 135, "colour": "tortoiseshell"}]),
    ("Do you have anything on sale right now?",
     [{"title": "Solstice Dune", "price": 45, "colour": "sand"},
      {"title": "Solstice Cove Driver", "price": 89, "colour": "grey"}]),
]
