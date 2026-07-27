"""Policy-following benchmark: the policy set and the labelled query set.

Modelled on the five policy types raglib actually supports
(raglib/config/prompts.py, aigency_common/models/policy.py):

    response  - shapes the reply only
    pin       - force-include specific products
    boost     - push specific products up the ranking
    filter    - AND a hard predicate into retrieval
    exclude   - the negative-framing counterpart of filter

Each policy carries a natural-language `condition`, exactly as a brand would
write it in the console. The benchmark task is the one our runtime performs
every turn: given a user message, decide which conditions are triggered.

GROUND TRUTH
Labels are a *literal* reading of the condition text, not a guess at intent.
Conditions were written to be near-mechanical so two careful readers agree.
Near-miss queries are included on purpose: "heading to the pool with the kids,
need shades for myself" mentions kids but is not shopping for a child, so
filter_kids_safety must NOT fire. Precision is as much of the task as recall.
"""

# --- the policy set -------------------------------------------------------

POLICIES = [
    ("pin_new_arrivals", "pin",
     "The user asks for new arrivals, the latest models, or what is new."),
    ("pin_summer_campaign", "pin",
     "The user mentions summer, the beach, a pool, or a vacation."),
    ("pin_designer_collab", "pin",
     "The user asks for designer, luxury, premium, or high-end eyewear."),
    ("boost_house_brand", "boost",
     "The user asks for sunglasses in general terms, without naming a brand or a specific model."),
    ("boost_polarized", "boost",
     "The user mentions polarized lenses or glare."),
    ("boost_sale_items", "boost",
     "The user mentions a sale, a discount, a deal, a coupon, or a promotion."),
    ("boost_sports_wrap", "boost",
     "The user mentions running, cycling, hiking, skiing, or another outdoor sport."),
    ("filter_kids_safety", "filter",
     "The user is shopping for a child aged under 13."),
    ("filter_prescription", "filter",
     "The user mentions a prescription, an Rx, or their eye doctor."),
    ("filter_budget", "filter",
     "The user says cheap, budget, or affordable, or names a price ceiling of $100 or less."),
    ("filter_mens_fit", "filter",
     "The user is shopping for a man or a boy (him, husband, boyfriend, father, son, brother)."),
    ("filter_womens_fit", "filter",
     "The user is shopping for a woman or a girl (her, wife, girlfriend, mother, daughter, sister)."),
    ("filter_blue_light", "filter",
     "The user mentions screens, a computer, gaming, or blue light."),
    ("exclude_discontinued", "exclude",
     "The user asks about a specific product by its model code or style ID."),
    ("exclude_non_uv", "exclude",
     "The user mentions UV, sun protection, or protecting their eyes from the sun."),
    ("resp_warranty", "response",
     "The user asks about the warranty, repairs, or a product breaking."),
    ("resp_shipping", "response",
     "The user asks about shipping, delivery, or how long an order takes to arrive."),
    ("resp_returns", "response",
     "The user asks about returns, refunds, or exchanges."),
    ("resp_no_medical", "response",
     "The user describes an eye symptom, an eye condition, or a vision problem they are experiencing."),
    ("resp_gift", "response",
     "The user mentions a gift, a present, a birthday, or an anniversary."),
    ("resp_tryon", "response",
     "The user asks how a product will look on them, or asks about their face shape."),
    ("resp_store_locator", "response",
     "The user asks about a physical store, a location, or trying products on in person."),
    ("resp_loyalty", "response",
     "The user mentions points, rewards, membership, or a loyalty program."),
    ("resp_accessibility", "response",
     "The user mentions a disability, low vision, or an accessibility need."),
    ("resp_alternative_fit", "response",
     "The user mentions a low or narrow nose bridge, frames sliding down, or an alternative/Asian fit."),
]

POLICY_IDS = [p[0] for p in POLICIES]
POLICY_TYPE = {p[0]: p[1] for p in POLICIES}
POLICY_CONDITION = {p[0]: p[2] for p in POLICIES}

# --- the labelled query set ----------------------------------------------
# (query, set of policy ids whose condition a literal reading triggers)

QUERIES = [
    # -- single-trigger
    ("show me your newest sunglasses",
     ["pin_new_arrivals", "boost_house_brand"]),
    ("what's new this week?",
     ["pin_new_arrivals"]),
    ("I need sunglasses for a beach vacation in July",
     ["pin_summer_campaign", "boost_house_brand"]),
    ("looking for luxury designer frames",
     ["pin_designer_collab"]),
    ("polarized sunglasses for driving",
     ["boost_polarized", "boost_house_brand"]),
    ("anything on sale right now?",
     ["boost_sale_items"]),
    ("I need wraparound glasses for cycling",
     ["boost_sports_wrap"]),
    ("sunglasses for my 8 year old",
     ["filter_kids_safety", "boost_house_brand"]),
    ("I have a prescription from my eye doctor, can you fit these with my Rx?",
     ["filter_prescription"]),
    ("something affordable please",
     ["filter_budget"]),
    ("glasses for staring at a computer all day",
     ["filter_blue_light"]),
    ("do you have RB3025?",
     ["exclude_discontinued"]),
    ("is OO9484D-0949 still available?",
     ["exclude_discontinued"]),
    ("I want frames that block UV and protect my eyes from the sun",
     ["exclude_non_uv"]),
    ("what's your warranty if the frame snaps?",
     ["resp_warranty"]),
    ("how long does shipping take to Chicago?",
     ["resp_shipping"]),
    ("can I return these if they don't fit?",
     ["resp_returns"]),
    ("my eyes water and go blurry after an hour of reading",
     ["resp_no_medical"]),
    ("what frame shape suits a round face?",
     ["resp_tryon"]),
    ("do you have a store in Boston I can visit?",
     ["resp_store_locator"]),
    ("how many points do I earn on a purchase?",
     ["resp_loyalty"]),
    ("I have low vision, do you carry high-contrast lenses?",
     ["resp_accessibility"]),
    ("do you have Asian fit frames?",
     ["resp_alternative_fit"]),

    # -- multi-trigger
    ("cheap polarized sunglasses for my son for the beach",
     ["pin_summer_campaign", "boost_house_brand", "boost_polarized",
      "filter_budget", "filter_mens_fit"]),
    ("a luxury gift for my wife, and how long is shipping?",
     ["pin_designer_collab", "filter_womens_fit", "resp_gift", "resp_shipping"]),
    ("what are your returns and warranty policies?",
     ["resp_returns", "resp_warranty"]),
    ("any new arrivals on sale?",
     ["pin_new_arrivals", "boost_sale_items"]),
    ("a gift for my husband's birthday",
     ["filter_mens_fit", "resp_gift"]),
    ("an anniversary present for my girlfriend",
     ["filter_womens_fit", "resp_gift"]),
    ("cheap sunglasses under $50",
     ["filter_budget", "boost_house_brand"]),
    ("ski goggles under $80",
     ["boost_sports_wrap", "filter_budget"]),
    ("glasses for gaming, my eyes hurt after long sessions",
     ["filter_blue_light", "resp_no_medical"]),
    ("does the loyalty program cover returns?",
     ["resp_loyalty", "resp_returns"]),
    ("premium frames my mother would like, she's 70",
     ["pin_designer_collab", "filter_womens_fit"]),
    ("I need something for hiking that blocks UV",
     ["boost_sports_wrap", "exclude_non_uv"]),
    ("can I try on RB2140 in your NYC store?",
     ["exclude_discontinued", "resp_store_locator"]),
    ("is the summer sale over?",
     ["pin_summer_campaign", "boost_sale_items"]),
    ("my nose bridge is low so frames always slide down — what do you suggest?",
     ["resp_alternative_fit"]),
    ("sunglasses for my wife",
     ["filter_womens_fit", "boost_house_brand"]),

    # -- near misses: a trigger word is present but the condition is not met
    ("heading to the pool with the kids, need shades for myself",
     ["pin_summer_campaign", "boost_house_brand"]),          # NOT filter_kids_safety
    ("my son is 15 and needs new frames",
     ["filter_mens_fit"]),                                   # NOT filter_kids_safety (over 13)
    ("I'm a runner but I want these for everyday wear",
     ["boost_sports_wrap"]),                                 # sport mentioned, literal trigger
    ("do you sell cleaning cloths?",
     []),
    ("what colours do you stock?",
     []),

    # -- zero-trigger
    ("hi", []),
    ("tell me about your company", []),
    ("I need new tires for my car", []),
]

QUERY_TEXTS = [q for q, _ in QUERIES]
GOLD = {q: set(labels) for q, labels in QUERIES}

# --- realistic system-prompt padding -------------------------------------
# Taken from the shape of our production prompts: a site description, catalog
# statistics and tool guidelines. Our real per-tool guideline block is 1,959
# tokens on its own, so "long prompt" here is not a synthetic stress test.

SITE_DESCRIPTION = (
    "Solstice Optics is an online eyewear retailer carrying sunglasses, optical frames and "
    "sports eyewear from 40+ brands, spanning value house-brand styles through designer "
    "collaborations. The catalog holds 18,400 products with prices from $29 to $940."
)

FILLER_BLOCK = """
CATALOG FIELDS AVAILABLE FOR STRUCTURED FILTERING
brand, product_type, frame_shape, frame_material, frame_colour_family, lens_colour_family,
lens_technology, is_polarized, is_photochromic, uv_protection_rating, current_price,
original_price, on_sale, in_stock, rating_value, reviews_count, gender, fit, bridge_width,
lens_width, temple_length, frame_width, weight_grams, warranty_years, country_of_origin.

GENERAL BEHAVIOUR
- Never respond with a follow-up question. Either invoke a tool or explain why you did not.
- If a user query corresponds to multiple product types, invoke one search call per product
  type. Invoke at most 5 search calls per turn; if the user mentions more than 5 product
  types, pick the 5 most relevant.
- The query must always describe ALL relevant attributes in natural language, including any
  attribute you also expressed as a structured_query filter. Never rely on structured_query
  alone to convey an attribute.
- Only add structured_query filters when the user explicitly requests them in the conversation
  history or the current query. Do not infer filters from vague or indirect language.
- If the user references a prior product, extract its attributes (colour, material, style,
  dimensions) and include them in the query.
- You may pass an empty query or structured_query if nothing is specified or can be inferred.
- Always add in_stock == True unless the user explicitly asked for out-of-stock items.
- Use only fields from the catalog's structured filter set. Do not invent field names.
- If the user's constraint cannot be expressed with any field in the allowlist, leave the
  filter out rather than substituting an unrelated field as a stand-in.

RANKING STRATEGY
- price_asc for cheapest / budget-friendly / most affordable requests.
- price_desc for premium / luxury / high-end requests.
- reviews for popular or well-reviewed requests.
- rating is the default when the user expresses no preference.

CLOSE-MATCH HANDLING
When the catalog holds no product satisfying every stated criterion, acknowledge this
explicitly before listing alternatives, and for each alternative name which criterion it
satisfies and which it falls short on. Treat omitting the acknowledgement as a serious error.

OUT-OF-CATALOG RULE
If the user's query is about a product category clearly outside the catalog, do not invoke a
product search tool. When in doubt — synonyms, abbreviations, broader or narrower matches —
invoke the tool. This rule does not apply to specific-but-rare attribute requests, which are
filtered downstream by the critic rather than at tool-selection time.
""".strip()


def context_block(target_tokens: int, enc) -> str:
    """Realistic system-prompt padding of roughly `target_tokens` tokens."""
    if target_tokens <= 0:
        return ""
    body = f"THE STORE\n{SITE_DESCRIPTION}\n\n{FILLER_BLOCK}"
    out = body
    while len(enc.encode(out)) < target_tokens:
        out += "\n\n" + FILLER_BLOCK
    toks = enc.encode(out)[:target_tokens]
    return enc.decode(toks)
