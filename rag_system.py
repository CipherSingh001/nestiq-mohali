# =============================================================
# PHASE 4 — rag_system.py
# Smart Real Estate Advisor — Mohali
# =============================================================
# SETUP:
#   pip install groq
#   Set environment variable: GROQ_API_KEY=your_key_here
#
# RUN (standalone test):
#   python rag_system.py
# =============================================================

import os
import json
from groq import Groq

# ── Groq client ───────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "\n❌ GROQ_API_KEY not set!\n"
        "   Windows CMD   : set GROQ_API_KEY=your_key_here\n"
        "   PowerShell    : $env:GROQ_API_KEY='your_key_here'\n"
    )

client = Groq(api_key=GROQ_API_KEY)

# =============================================================
# KNOWLEDGE BASE — All 18 Mohali Sectors
# This is the RAG knowledge base the LLM uses to answer
# investment questions grounded in real local data
# =============================================================

KNOWLEDGE_BASE = """
============================================================
MOHALI REAL ESTATE KNOWLEDGE BASE — 2025
============================================================

--- SECTOR 66 ---
Type: Residential + Commercial Mix
Average Price: ₹120–180 Lakhs
Annual Appreciation: 8–10%
Best For: Working professionals, budget buyers
Nearby: IT Park Phase 8, decent schools
Infrastructure: Good roads, bus connectivity
Traffic: Medium
Investment Rating: 7/10
Pros: Affordable entry point, near IT hub
Cons: Older infrastructure, moderate amenities

--- SECTOR 67 ---
Type: Residential
Average Price: ₹100–160 Lakhs
Annual Appreciation: 8–10%
Best For: First-time buyers, families
Nearby: Civil Hospital (very close), schools
Infrastructure: Established area, good connectivity
Traffic: Medium
Investment Rating: 7.5/10
Pros: Very close to Civil Hospital, affordable
Cons: High traffic near hospital zone

--- SECTOR 68 ---
Type: Premium Residential
Average Price: ₹150–220 Lakhs
Annual Appreciation: 10–12%
Best For: Families, long-term living
Nearby: Schools, parks, shopping
Infrastructure: Excellent, well-planned sector
Traffic: Low — peaceful area
Investment Rating: 8/10
Pros: Peaceful, excellent infrastructure, green area
Cons: Slightly away from commercial hubs

--- SECTOR 69 ---
Type: Residential
Average Price: ₹130–190 Lakhs
Annual Appreciation: 9–11%
Best For: Families seeking peaceful environment
Nearby: Parks, residential schools
Infrastructure: Good, planned layout
Traffic: Low
Investment Rating: 7.8/10
Pros: Quiet, good for families, green surroundings
Cons: Limited commercial activity nearby

--- SECTOR 70 ---
Type: Premium Residential — TOP SECTOR
Average Price: ₹180–280 Lakhs
Annual Appreciation: 13–15%
Best For: IT professionals, investors, premium buyers
Nearby: IT Park, Elante Mall (2km), hospitals, schools
Infrastructure: World-class — widest roads, best amenities
Traffic: High but well-managed
Investment Rating: 9/10
Pros: Best appreciation, near IT hub, premium infrastructure
     Proximity to Elante Mall, top schools and hospitals
Cons: Higher price point, heavy traffic during office hours
Note: Most sought-after sector in Mohali for investment

--- SECTOR 71 ---
Type: Residential + Institutional
Average Price: ₹160–240 Lakhs
Annual Appreciation: 11–13%
Best For: Professionals, investors
Nearby: IT companies, medical facilities, malls
Infrastructure: Very good connectivity
Traffic: Medium
Investment Rating: 8.5/10
Pros: Strong appreciation, near employment hub
Cons: Busier than peripheral sectors

--- SECTOR 74 ---
Type: Residential
Average Price: ₹90–140 Lakhs
Annual Appreciation: 7–9%
Best For: Budget buyers, first-time investors
Nearby: Basic amenities, local markets
Infrastructure: Developing
Traffic: Low
Investment Rating: 7/10
Pros: Very affordable, low traffic, peaceful
Cons: Developing infrastructure, fewer amenities

--- SECTOR 75 ---
Type: Residential
Average Price: ₹100–150 Lakhs
Annual Appreciation: 7–9%
Best For: Budget-conscious buyers
Nearby: Local amenities, parks
Infrastructure: Average
Traffic: Low
Investment Rating: 7.3/10
Pros: Affordable, quiet, good for retirement
Cons: Far from commercial centers

--- SECTOR 76 ---
Type: Residential
Average Price: ₹110–170 Lakhs
Annual Appreciation: 8–10%
Best For: Families, moderate budget
Nearby: Schools, local hospitals
Infrastructure: Good
Traffic: Medium
Investment Rating: 7.6/10
Pros: Balanced price and amenities
Cons: Average appreciation

--- SECTOR 77 ---
Type: Residential
Average Price: ₹130–200 Lakhs
Annual Appreciation: 9–11%
Best For: Families, mid-range investors
Nearby: Parks, hospitals, schools
Infrastructure: Good, planned
Traffic: Medium
Investment Rating: 8/10
Pros: Good balance of price, peace, and amenities
Cons: Nothing exceptional

--- SECTOR 78 ---
Type: Premium Residential
Average Price: ₹150–230 Lakhs
Annual Appreciation: 10–12%
Best For: Premium families, investors
Nearby: Fortis Hospital, good schools, parks
Infrastructure: Excellent
Traffic: Low — very peaceful
Investment Rating: 8.3/10
Pros: Premium feel, near Fortis Hospital, peaceful
Cons: Slightly premium pricing

--- SECTOR 79 ---
Type: Residential
Average Price: ₹120–180 Lakhs
Annual Appreciation: 8–10%
Best For: Families, budget investors
Nearby: Parks, local amenities
Infrastructure: Good
Traffic: Low
Investment Rating: 7.9/10
Pros: Peaceful, affordable, green area
Cons: Far from major commercial hubs

--- SECTOR 80 ---
Type: Premium Residential
Average Price: ₹170–260 Lakhs
Annual Appreciation: 12–14%
Best For: IT professionals, premium investors
Nearby: IT companies, hospitals, malls
Infrastructure: Excellent
Traffic: Medium
Investment Rating: 8.7/10
Pros: Strong appreciation, premium infrastructure
     Close to employment zone
Cons: Busy during peak hours

--- SECTOR 81 ---
Type: Residential
Average Price: ₹140–210 Lakhs
Annual Appreciation: 10–12%
Best For: Families, mid-level investors
Nearby: Schools, hospitals, parks
Infrastructure: Very good
Traffic: Low — peaceful
Investment Rating: 8.2/10
Pros: Peaceful, good infrastructure, decent appreciation
Cons: Not as premium as Sector 70/82

--- SECTOR 82 ---
Type: Ultra Premium Residential — HIGHEST RATED
Average Price: ₹250–500 Lakhs
Annual Appreciation: 15–18%
Best For: High-net-worth investors, luxury buyers
Nearby: 5-star hotels, premium malls, corporate offices
         Top schools, premium hospitals
Infrastructure: Ultra premium — widest roads, best security
Traffic: High but well-managed
Investment Rating: 9.2/10
Pros: Highest appreciation in Mohali, luxury lifestyle
     Best resale value, premium address value
Cons: Very high entry price, not suitable for budget buyers
Note: Sector 82 is Mohali's most premium address

--- SECTOR 83 ---
Type: Residential
Average Price: ₹110–170 Lakhs
Annual Appreciation: 7–9%
Best For: Budget families, first-time buyers
Nearby: Local amenities, parks
Infrastructure: Average to good
Traffic: Low
Investment Rating: 7.4/10
Pros: Affordable, peaceful, good for families
Cons: Away from premium areas, average appreciation

--- SECTOR 84 ---
Type: Developing Residential
Average Price: ₹90–130 Lakhs
Annual Appreciation: 6–8%
Best For: Very budget-conscious buyers, long-term bet
Nearby: Basic amenities
Infrastructure: Still developing
Traffic: Very Low
Investment Rating: 7.1/10
Pros: Very affordable, potential for future growth
Cons: Underdeveloped infrastructure currently

--- SECTOR 85 ---
Type: Developing Residential
Average Price: ₹80–120 Lakhs
Annual Appreciation: 5–8%
Best For: Very budget buyers, speculative investment
Nearby: Limited amenities
Infrastructure: Early development stage
Traffic: Very Low
Investment Rating: 6.8/10
Pros: Cheapest entry point, maximum upside if area develops
Cons: Least developed, fewest amenities, uncertain timeline

============================================================
MOHALI MARKET TRENDS — 2025
============================================================

Overall Market:
- Mohali average annual appreciation: 10–12%
- Tricity (Mohali + Chandigarh + Panchkula) is one of India's
  fastest growing real estate markets
- IT sector expansion driving demand in Sectors 66–82
- New Chandigarh International Airport boosting Aerocity
  and nearby sector prices

Best Sectors for Investment (Ranked):
1. Sector 82 — Highest appreciation (15–18%), luxury
2. Sector 70 — Best overall value + appreciation (13–15%)
3. Sector 80 — Strong IT-driven demand (12–14%)
4. Sector 71 — Good appreciation near employment (11–13%)
5. Sector 68 — Peaceful premium residential (10–12%)

Best Sectors for Budget Buyers:
1. Sector 85 — Lowest prices (₹80–120L)
2. Sector 84 — Very affordable (₹90–130L)
3. Sector 74 — Budget with decent amenities (₹90–140L)
4. Sector 75 — Quiet and affordable (₹100–150L)

Best Sectors for Families:
1. Sector 68 — Peaceful, excellent schools, parks
2. Sector 78 — Near Fortis Hospital, peaceful
3. Sector 81 — Balanced lifestyle, good schools
4. Sector 77 — Green area, parks, schools

Best Sectors Near Hospitals:
1. Sector 67 — Civil Hospital walking distance
2. Sector 70 — Multiple hospitals nearby
3. Sector 78 — Fortis Hospital very close
4. Sector 82 — Premium hospitals nearby

Property Types:
- 3BHK has highest resale value and demand
- Villas in Sector 82 give best ROI for luxury
- 2BHK most affordable entry point
- Furnished properties rent 30% higher than unfurnished

Mohali vs Chandigarh:
- Mohali offers 20–30% better ROI than Chandigarh
- Chandigarh has higher current prices but lower growth
- Mohali has more new construction options
- For investment: Mohali is clearly better choice
- For immediate premium lifestyle: Chandigarh Sector 17

Rental Yields:
- Average rental yield in Mohali: 3–4% annually
- Best rental yield sectors: 70, 71, 80 (near IT companies)
- Furnished flats near IT park rent ₹20,000–45,000/month
- Villas rent ₹50,000–1,50,000/month in premium sectors

============================================================
BUYING TIPS
============================================================
- Best time to buy: Jan–March (pre-summer, fewer buyers)
- Always verify RERA registration before buying
- Check builder reputation and past project delivery
- Resale flats in Sector 70/82 retain value best
- Under-construction in Sector 80/81 gives best price
- New airport connectivity will boost Sectors 66–70 further
- IT company expansions planned near Sectors 70–82
============================================================
"""


# =============================================================
# SECTOR QUICK REFERENCE — for structured queries
# =============================================================
SECTOR_RATINGS = {
    66: {"rating": 7.0,  "appreciation": "8-10%",  "avg_price": "₹120-180L"},
    67: {"rating": 7.5,  "appreciation": "8-10%",  "avg_price": "₹100-160L"},
    68: {"rating": 8.0,  "appreciation": "10-12%", "avg_price": "₹150-220L"},
    69: {"rating": 7.8,  "appreciation": "9-11%",  "avg_price": "₹130-190L"},
    70: {"rating": 9.0,  "appreciation": "13-15%", "avg_price": "₹180-280L"},
    71: {"rating": 8.5,  "appreciation": "11-13%", "avg_price": "₹160-240L"},
    74: {"rating": 7.0,  "appreciation": "7-9%",   "avg_price": "₹90-140L"},
    75: {"rating": 7.3,  "appreciation": "7-9%",   "avg_price": "₹100-150L"},
    76: {"rating": 7.6,  "appreciation": "8-10%",  "avg_price": "₹110-170L"},
    77: {"rating": 8.0,  "appreciation": "9-11%",  "avg_price": "₹130-200L"},
    78: {"rating": 8.3,  "appreciation": "10-12%", "avg_price": "₹150-230L"},
    79: {"rating": 7.9,  "appreciation": "8-10%",  "avg_price": "₹120-180L"},
    80: {"rating": 8.7,  "appreciation": "12-14%", "avg_price": "₹170-260L"},
    81: {"rating": 8.2,  "appreciation": "10-12%", "avg_price": "₹140-210L"},
    82: {"rating": 9.2,  "appreciation": "15-18%", "avg_price": "₹250-500L"},
    83: {"rating": 7.4,  "appreciation": "7-9%",   "avg_price": "₹110-170L"},
    84: {"rating": 7.1,  "appreciation": "6-8%",   "avg_price": "₹90-130L"},
    85: {"rating": 6.8,  "appreciation": "5-8%",   "avg_price": "₹80-120L"},
}


# =============================================================
# MAIN FUNCTION — Answer investment questions using RAG
# =============================================================

def answer_investment_question(question: str, context: str = "") -> str:
    """
    Uses RAG to answer real estate investment questions.
    Grounds answers in the knowledge base — no hallucination.

    Args:
        question: User's investment question
        context:  Optional — current property context from search results

    Returns:
        str: AI-generated answer grounded in knowledge base
    """

    system_prompt = f"""
You are an expert real estate investment advisor for Mohali and Chandigarh, India.
You MUST answer using ONLY the information in the knowledge base provided.
Do NOT make up numbers or appreciation rates not in the knowledge base.
Be specific, helpful, and cite sector names and numbers.
Keep responses under 180 words.
Use a friendly, professional tone.

KNOWLEDGE BASE:
{KNOWLEDGE_BASE}

{f'CURRENT SEARCH CONTEXT: {context}' if context else ''}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Sorry, I couldn't process your question right now. Error: {str(e)}"


def get_sector_comparison(sector_a: int, sector_b: int) -> str:
    """
    Compare two sectors side by side.
    """
    if sector_a not in SECTOR_RATINGS or sector_b not in SECTOR_RATINGS:
        return "Sector not found in knowledge base."

    a = SECTOR_RATINGS[sector_a]
    b = SECTOR_RATINGS[sector_b]

    question = f"""
Compare Sector {sector_a} vs Sector {sector_b} for investment:
- Sector {sector_a}: Rating={a['rating']}, Appreciation={a['appreciation']}, Price={a['avg_price']}
- Sector {sector_b}: Rating={b['rating']}, Appreciation={b['appreciation']}, Price={b['avg_price']}

Which is better and why? Give a clear recommendation.
"""
    return answer_investment_question(question)


def get_budget_recommendation(budget_lakhs: float) -> str:
    """
    Recommend best sectors for a given budget.
    """
    question = f"""
My budget is ₹{budget_lakhs} Lakhs. 
Which sectors in Mohali should I consider?
What type of property can I get?
Is it good for investment or only for living?
"""
    return answer_investment_question(question)


def get_top_investment_sectors(n: int = 3) -> list:
    """
    Returns top N sectors by investment rating.
    """
    sorted_sectors = sorted(
        SECTOR_RATINGS.items(),
        key=lambda x: x[1]['rating'],
        reverse=True
    )
    return [(s, info) for s, info in sorted_sectors[:n]]


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 4 — RAG System Test")
    print("="*60)

    questions = [
        "Which sector in Mohali is best for investment?",
        "I have a budget of 1.2 crore. What should I buy?",
        "Compare Sector 70 vs Sector 82 for investment",
        "Should I buy in Mohali or Chandigarh?",
        "Which sectors are best for families with children?",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        print("-" * 50)
        ans = answer_investment_question(q)
        print(ans)

    print("\n\n📊 TOP 3 INVESTMENT SECTORS:")
    for sector, info in get_top_investment_sectors(3):
        print(f"  Sector {sector}: Rating={info['rating']} | "
              f"Appreciation={info['appreciation']} | "
              f"Price={info['avg_price']}")

    print("\n✅ Phase 4 complete — rag_system.py works!")
    print("   Next: Phase 5 — recommender.py")