# =============================================================
# PHASE 3 — query_processor.py
# Smart Real Estate Advisor — Mohali
# =============================================================
# SETUP:
#   pip install groq
#   Set environment variable: GROQ_API_KEY=your_key_here
#   Get FREE key at: https://console.groq.com
#
# RUN (standalone test):
#   python query_processor.py
# =============================================================

import os, json, re
from groq import Groq

# ── Groq client ───────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "\n❌ GROQ_API_KEY not set!\n"
        "   Windows CMD   : set GROQ_API_KEY=your_key_here\n"
        "   PowerShell    : $env:GROQ_API_KEY='your_key_here'\n"
        "   Get free key  : https://console.groq.com\n"
    )

client = Groq(api_key=GROQ_API_KEY)

# ── Sector list (all 18 Mohali sectors) ───────────────────────
MOHALI_SECTORS = list(range(66, 86))   # 66 to 85

# ── System prompt for Groq ────────────────────────────────────
SYSTEM_PROMPT = """
You are a real estate query parser for Mohali, Punjab, India.
Your ONLY job is to extract structured filters from a natural language query.

Return ONLY a valid JSON object with these exact keys (use null for unknown):
{
  "min_bhk": int or null,
  "max_bhk": int or null,
  "min_price": float or null,
  "max_price": float or null,
  "sector": int or null,
  "property_type": "Flat" | "House" | "Villa" | null,
  "furnishing": "Unfurnished" | "Semi-Furnished" | "Furnished" | null,
  "near_hospital": true | false | null,
  "near_park": true | false | null,
  "near_mall": true | false | null,
  "max_age": int or null,
  "min_area": float or null,
  "max_area": float or null
}

Rules:
- Prices are ALWAYS stored in Lakhs (₹). Convert everything to Lakhs:
  "80 lakhs"/"80L"/"80 lakh" = 80.0
  "1 crore"/"1 cr"/"1C" = 100.0, "1.5 crore"/"1.5 cr" = 150.0, "2 crore" = 200.0
  "under 1 crore" → max_price: 100.0, "above 1.5 crore" → min_price: 150.0
  "between 1 and 2 crore" → min_price: 100.0, max_price: 200.0
  "50 lakh to 1 crore" → min_price: 50.0, max_price: 100.0
- "apartment / flat / 2BHK / 3BHK" alone (no villa/house) → property_type: "Flat"
- "near hospital / close to hospital / hospital nearby" → near_hospital: true
- "new / newly built / new construction" → max_age: 3
- "spacious" → min_area: 1500
- "budget" or "affordable" → max_price: 120
- "luxury / premium / high-end" → min_price: 200
- Sectors in Mohali are numbered 66 to 85. If user says "sector 7" they likely mean sector 70 or 71 — pick the closest valid sector (66–85).
- Return ONLY the JSON, no explanation, no markdown
"""

def parse_query(user_query: str) -> dict:
    """
    Parse natural language query into structured filters using Groq LLM.
    Returns a dict of filters.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_query}
            ],
            temperature=0.0,
            max_tokens=300
        )
        raw = response.choices[0].message.content.strip()

        # Clean up common LLM formatting issues
        raw = re.sub(r"```json|```", "", raw).strip()

        filters = json.loads(raw)
        return filters

    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error: {e}")
        return {}
    except Exception as e:
        print(f"  ⚠️  Groq API error: {e}")
        return {}


def explain_filters(filters: dict, original_query: str) -> str:
    """
    Generate a plain-English explanation of what the system understood.
    """
    parts = []

    if filters.get("min_bhk") and filters.get("max_bhk"):
        if filters["min_bhk"] == filters["max_bhk"]:
            parts.append(f"{filters['min_bhk']} BHK")
        else:
            parts.append(f"{filters['min_bhk']}–{filters['max_bhk']} BHK")
    elif filters.get("min_bhk"):
        parts.append(f"{filters['min_bhk']}+ BHK")
    elif filters.get("max_bhk"):
        parts.append(f"up to {filters['max_bhk']} BHK")

    if filters.get("property_type"):
        parts.append(filters["property_type"])

    if filters.get("sector"):
        parts.append(f"in Sector {filters['sector']}")

    def _fmt(l): return f"₹{l/100:.2f}Cr" if l >= 100 else f"₹{l:.0f}L"
    price_parts = []
    if filters.get("min_price"):
        price_parts.append(f"above {_fmt(filters['min_price'])}")
    if filters.get("max_price"):
        price_parts.append(f"under {_fmt(filters['max_price'])}")
    if price_parts:
        parts.append("budget " + " and ".join(price_parts))

    if filters.get("furnishing"):
        parts.append(filters["furnishing"].lower())

    nearby = []
    if filters.get("near_hospital"): nearby.append("hospital")
    if filters.get("near_park"):     nearby.append("park")
    if filters.get("near_mall"):     nearby.append("mall")
    if nearby:
        parts.append(f"near {', '.join(nearby)}")

    if filters.get("max_age"):
        parts.append(f"max {filters['max_age']} years old")

    if not parts:
        return f"I understood: '{original_query}' but couldn't extract specific filters. Showing all properties."

    return "I'm looking for: " + " • ".join(parts)


def generate_recommendation_text(property_row: dict, rank: int) -> str:
    """
    Generate a plain-English recommendation for a single property.
    Uses Groq to produce human-friendly explanation.
    """
    prompt = f"""
You are a friendly real estate advisor in Mohali, India.
Write a 2-sentence recommendation for this property (rank #{rank}):

Property details:
- {property_row.get('BHK', '?')} BHK {property_row.get('Property_Type', 'property')}
- Area: {property_row.get('Area_sqft', '?')} sq ft
- Price: ₹{property_row.get('Price_Lakh', '?')} Lakh
- Sector: {property_row.get('Sector', '?')} Mohali
- Furnishing: {property_row.get('Furnishing_Label', property_row.get('Furnishing', '?'))}
- Age: {property_row.get('Age', '?')} years
- Location Score: {property_row.get('Location_Score', '?')}/10
- Amenities Score: {property_row.get('Amenities_Score', '?')}/10

Be specific, mention 2-3 actual details from the data. Do not use bullet points.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return (f"This {property_row.get('BHK','?')} BHK property in Sector "
                f"{property_row.get('Sector','?')} offers great value at "
                f"₹{property_row.get('Price_Lakh','?')}L.")


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 3 — Query Processor Test")
    print("="*60)

    test_queries = [
        "3BHK near hospital under 80 lakhs",
        "luxury villa in sector 7 with 4 bedrooms",
        "affordable 2bhk apartment semi furnished",
        "new construction house near mall under 1.5 crore",
        "spacious 3bhk near park budget 60 to 120 lakhs",
    ]

    for q in test_queries:
        print(f"\n📝 Query: '{q}'")
        filters = parse_query(q)
        print(f"   Filters: {json.dumps(filters, indent=None)}")
        print(f"   📣 {explain_filters(filters, q)}")

    print("\n✅ Phase 3 complete — query_processor.py works!")
    print("   Next: Phase 4 — rag_system.py")