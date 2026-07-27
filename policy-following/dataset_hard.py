"""Hard tier: compositional policies over multi-turn conversations.

The easy tier (dataset.py) is close to keyword spotting on a single message,
and every model solves it. That is a useful negative result but it does not
resemble the policies brands actually write, which are compositional and refer
to the conversation rather than to the last message.

What makes this tier hard:

  CONJUNCTION   two or three clauses that must all hold
  NEGATION      "... but not when ..." exception clauses
  STATE         conditions over conversation history, not the latest turn
  COUNTING      "at least two products", "three or more constraints"
  PRECEDENCE    policies that suppress other policies
  WORLD MODEL   a January trip to Buenos Aires is a summer trip

Every hard policy is paired with at least one near-miss conversation that
satisfies all clauses but one, so a model that pattern-matches the topic and
ignores the logic scores badly on precision.
"""

HARD_POLICIES = [
    ("pin_gift_no_budget", "pin",
     "The user is shopping for someone other than themselves AND has mentioned an occasion "
     "(birthday, anniversary, holiday, wedding), AND has NOT stated any budget or price limit "
     "anywhere in the conversation."),

    ("boost_premium_upsell", "boost",
     "The assistant has already shown the user at least TWO products priced under $100 earlier "
     "in this conversation, AND the user's latest message asks for something better, nicer or "
     "higher quality."),

    ("filter_stock_region", "filter",
     "The user has named a delivery destination outside the United States AND is asking about "
     "the availability of a SPECIFIC named product."),

    ("resp_budget_narrowed", "response",
     "The user stated a budget or price limit earlier in the conversation and has since LOWERED "
     "it."),

    ("exclude_rejected_brand", "exclude",
     "The user has explicitly said they dislike, do not want, or reject a particular brand at "
     "any point in the conversation."),

    ("boost_sport_adult", "boost",
     "The user mentions an outdoor sport at any point in the conversation, but NOT when they are "
     "shopping for a child aged under 13."),

    ("resp_repeat_interest", "response",
     "The user has asked about the SAME specific named product in two or more separate turns."),

    ("pin_bundle_offer", "pin",
     "The user has expressed intent to buy at least TWO items, AND no discount, sale or promotion "
     "has been mentioned by either side."),

    ("filter_rx_ready", "filter",
     "The user mentions a prescription or Rx AND asks about delivery timing or how quickly an "
     "order arrives."),

    ("resp_price_sensitive_switch", "response",
     "The user asked for premium, designer or high-end items earlier in the conversation, but "
     "their LATEST message asks for something cheaper or more affordable."),

    ("boost_returning_customer", "boost",
     "The user refers to a previous order or says they have bought from this store before."),

    ("resp_indecision", "response",
     "The assistant has shown products in at least TWO separate turns AND the user's latest "
     "message expresses difficulty choosing between them."),

    ("exclude_kids_marketing", "exclude",
     "The user is shopping for a child aged under 13. Suppresses all promotional and upsell "
     "messaging."),

    ("resp_accessibility_priority", "response",
     "The user mentions a disability, a vision limitation, or an accessibility need. Takes "
     "precedence over any style or upsell policy."),

    ("filter_multi_constraint", "filter",
     "The user has stated THREE OR MORE distinct product constraints across the whole "
     "conversation (for example colour, price, polarisation, shape, material)."),

    # Hardened after a first run: the original wording ("expresses dissatisfaction
    # with the assistant's suggestions") made "these look a bit basic" a defensible
    # fire, so the model was right and the label was wrong. The distinction the
    # brand actually wants — unhappy with the help vs. preferring other products —
    # is now stated explicitly instead of left to judgement.
    ("resp_frustration", "response",
     "The user's LATEST message expresses clear dissatisfaction with the ASSISTANT's help — for "
     "example saying the suggestions are wrong, that it is not listening, or that it is being "
     "unhelpful. Simply preferring different products (\"these look a bit basic\", \"show me "
     "something nicer\") does NOT satisfy this condition."),

    ("pin_seasonal_local", "pin",
     "The user mentions travelling to a destination where it will be SUMMER at the time of "
     "travel, even though it is not summer where the store is based (the northern hemisphere)."),

    ("boost_fast_ship", "boost",
     "The user says they need the item by a specific date that is within 7 days."),

    ("resp_gift_fit_unknown", "response",
     "The user is buying for someone other than themselves AND asks about sizing, fit or "
     "measurements."),

    # Hardened after a first run: "firm maximum" was doing too much work, so the
    # model fired on "budget around $150" and "under $200" and was arguably right.
    # Hard vs. soft ceilings is a real distinction for a brand — a hard ceiling
    # suppresses results, a soft one only reranks — so it is spelled out here.
    ("exclude_over_budget", "exclude",
     "The user has stated a HARD price ceiling using absolute wording such as \"absolute "
     "maximum\", \"nothing over\", \"no more than\", or \"don't show me anything above\". A soft "
     "or approximate budget (\"around $150\", \"under $200\", \"let's keep it under $150\") does "
     "NOT satisfy this condition."),
]

HARD_IDS = [p[0] for p in HARD_POLICIES]
HARD_TYPE = {p[0]: p[1] for p in HARD_POLICIES}
HARD_COND = {p[0]: p[2] for p in HARD_POLICIES}

# --- conversations --------------------------------------------------------
# (turns, gold policy ids firing on the LATEST user turn)
# turns are (role, text); role is "u" or "a".

U, A = "u", "a"

HARD_CONVERSATIONS = [
    # --- H01 gift+occasion+no budget, and its near miss
    ([(U, "I'm looking for a gift for my brother's birthday.")],
     ["pin_gift_no_budget"]),
    ([(U, "I'm looking for a gift for my brother's birthday, budget around $150.")],
     []),                                     # budget stated -> clause fails

    # --- H02 upsell after two sub-$100 products, and its near miss
    ([(U, "Show me sunglasses under $80."),
      (A, "The Solstice Dune at $45.00 and the Solstice Basecamp at $39.00."),
      (U, "These look a bit basic — do you have something nicer?")],
     ["boost_premium_upsell"]),
    ([(U, "Show me a cheap pair."),
      (A, "The Solstice Dune at $45.00."),
      (U, "Anything nicer?")],
     []),                                     # only ONE product shown
    ([(U, "Show me budget sunglasses."),
      (A, "The Solstice Dune at $45.00 and the Solstice Basecamp at $39.00."),
      (U, "Do these come in blue?")],
     []),                                     # two shown, but not asking for better

    # --- H04 budget lowered
    ([(U, "I want frames under $300."),
      (A, "Here are several in that range."),
      (U, "Actually, let's keep it under $150.")],
     ["resp_budget_narrowed"]),

    # --- H10 premium -> cheaper
    ([(U, "Show me your designer collection."),
      (A, "Here is our designer range."),
      (U, "Hmm, these are pricey. Something more affordable?")],
     ["resp_price_sensitive_switch"]),

    # --- H05 rejected brand
    ([(U, "I really don't like Oakley. What else do you have?")],
     ["exclude_rejected_brand"]),

    # --- H06 sport, and the child exception
    ([(U, "Sunglasses for trail running, please.")],
     ["boost_sport_adult"]),
    ([(U, "Sunglasses for my 9 year old, who runs cross country.")],
     ["exclude_kids_marketing"]),             # sport present but child -> H06 suppressed
    ([(U, "Sunglasses for my 14 year old who plays tennis.")],
     ["boost_sport_adult"]),                  # 14 is not under 13

    # --- H07 same product across turns
    ([(U, "Tell me about the Solstice Meridian."),
      (A, "It is a polarised driver's frame at $129.00."),
      (U, "What about the Solstice Cove?"),
      (A, "That one is $89.00."),
      (U, "Can you tell me more about the Solstice Meridian again?")],
     ["resp_repeat_interest"]),

    # --- H03 region + specific product, and its near miss
    ([(U, "I'm in Canada. Is the Solstice Meridian available?")],
     ["filter_stock_region"]),
    ([(U, "Do you ship to Canada?")],
     []),                                     # no specific product named

    # --- H09 rx + timing
    ([(U, "I need these made up with my prescription — how fast can they arrive?")],
     ["filter_rx_ready"]),

    # --- H18 deadline within 7 days
    ([(U, "It's Tuesday and I need a pair before Friday.")],
     ["boost_fast_ship"]),

    # --- H11 returning customer
    ([(U, "I bought a pair from you last year and loved them. Looking for another.")],
     ["boost_returning_customer"]),

    # --- H12 indecision after two product turns
    ([(U, "Show me aviators."),
      (A, "Three aviator styles for you."),
      (U, "Show me a few more."),
      (A, "Three more aviators."),
      (U, "I really can't decide between all of these.")],
     ["resp_indecision"]),

    # --- H16 frustration (sport also present from turn 1)
    ([(U, "Show me something for cycling."),
      (A, "Here are three wrap styles."),
      (U, "None of these are what I asked for. This isn't helpful.")],
     ["boost_sport_adult", "resp_frustration"]),

    # --- H14 accessibility
    ([(U, "I have limited peripheral vision. What frames work best?")],
     ["resp_accessibility_priority"]),

    # --- H15 three constraints accumulated across turns
    ([(U, "I want polarised sunglasses."),
      (A, "Here are our polarised styles."),
      (U, "In black, please."),
      (A, "Black polarised styles."),
      (U, "And under $200.")],
     ["filter_multi_constraint"]),

    # --- H20 firm maximum
    ([(U, "My absolute maximum is $120. Don't show me anything above that.")],
     ["exclude_over_budget"]),

    # --- H08 bundle intent
    ([(U, "I'll take the Meridian, and I also want a pair for my wife.")],
     ["pin_bundle_offer"]),

    # --- H19 gift + fit
    ([(U, "Buying for my dad — how do I work out what size fits him?")],
     ["resp_gift_fit_unknown"]),

    # --- H17 hemisphere reasoning, and its near miss
    ([(U, "I'm going to Buenos Aires in January and need sunglasses.")],
     ["pin_seasonal_local"]),
    ([(U, "I'm going to Madrid in January and need sunglasses.")],
     []),                                     # northern winter

    # --- multi-fire
    ([(U, "Show me your premium range."),
      (A, "Here is the designer collection."),
      (U, "Too expensive. I've bought from you before — nothing over $100 this time.")],
     ["resp_price_sensitive_switch", "boost_returning_customer", "exclude_over_budget"]),
    ([(U, "It's my wife's anniversary present. How do I pick the right frame width for her?")],
     ["pin_gift_no_budget", "resp_gift_fit_unknown"]),

    # --- zero fire
    ([(U, "What are your opening hours?")], []),
    ([(U, "Hi there.")], []),
]


def render(turns):
    """Serialise a conversation the way the gate sees it."""
    lines = []
    for i, (role, text) in enumerate(turns):
        tag = "User" if role == U else "Assistant"
        last = " (latest)" if i == len(turns) - 1 else ""
        lines.append(f"{tag}{last}: {text}")
    return "CONVERSATION\n" + "\n".join(lines)


HARD_KEYS = [render(t) for t, _ in HARD_CONVERSATIONS]
HARD_GOLD = {render(t): set(g) for t, g in HARD_CONVERSATIONS}
