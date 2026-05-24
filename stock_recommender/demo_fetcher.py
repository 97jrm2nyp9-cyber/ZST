"""
Demo data generator — produces realistic synthetic Finviz-style data for the
full ticker list so the recommendation engine can be demonstrated without
network access.

Data is generated with controlled randomness (fixed seed) and mild correlations
between signals to mimic real market dynamics:
  • Trending stocks (high momentum) tend to have better analyst ratings
  • Stocks above key SMAs tend to have better recent performance
  • EPS growth correlates loosely with analyst optimism

All prices and metrics are fictional and for demonstration only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# S&P 500-style ticker universe with known company names / sectors
_UNIVERSE = [
    # Ticker, Company, Sector, Industry, approx price
    ("AAPL",  "Apple Inc.",               "Technology",       "Consumer Electronics",     213.0),
    ("MSFT",  "Microsoft Corp.",           "Technology",       "Software—Infrastructure",  415.0),
    ("NVDA",  "NVIDIA Corp.",              "Technology",       "Semiconductors",           950.0),
    ("AMZN",  "Amazon.com Inc.",           "Consumer Cyclical","Internet Retail",          190.0),
    ("GOOGL", "Alphabet Inc.",             "Communication",    "Internet Content",         175.0),
    ("META",  "Meta Platforms Inc.",       "Communication",    "Internet Content",         515.0),
    ("TSLA",  "Tesla Inc.",                "Consumer Cyclical","Auto Manufacturers",       175.0),
    ("BRK-B", "Berkshire Hathaway B",      "Financial",        "Insurance—Diversified",    436.0),
    ("AVGO",  "Broadcom Inc.",             "Technology",       "Semiconductors",          1620.0),
    ("JPM",   "JPMorgan Chase & Co.",      "Financial",        "Banks—Diversified",        210.0),
    ("LLY",   "Eli Lilly and Co.",         "Healthcare",       "Drug Manufacturers",       780.0),
    ("V",     "Visa Inc.",                 "Financial",        "Credit Services",          280.0),
    ("UNH",   "UnitedHealth Group Inc.",   "Healthcare",       "Healthcare Plans",         490.0),
    ("XOM",   "Exxon Mobil Corp.",         "Energy",           "Oil & Gas Integrated",    115.0),
    ("MA",    "Mastercard Inc.",           "Financial",        "Credit Services",          480.0),
    ("COST",  "Costco Wholesale Corp.",    "Consumer Defensive","Discount Stores",         840.0),
    ("HD",    "Home Depot Inc.",           "Consumer Cyclical","Home Improvement Retail",  370.0),
    ("WMT",   "Walmart Inc.",              "Consumer Defensive","Discount Stores",          95.0),
    ("NFLX",  "Netflix Inc.",              "Communication",    "Entertainment",            675.0),
    ("ABBV",  "AbbVie Inc.",               "Healthcare",       "Drug Manufacturers",       185.0),
    ("JNJ",   "Johnson & Johnson",         "Healthcare",       "Drug Manufacturers",       155.0),
    ("BAC",   "Bank of America Corp.",     "Financial",        "Banks—Diversified",         40.0),
    ("CRM",   "Salesforce Inc.",           "Technology",       "Software—Application",     285.0),
    ("ORCL",  "Oracle Corp.",              "Technology",       "Software—Infrastructure",  120.0),
    ("MRK",   "Merck & Co. Inc.",          "Healthcare",       "Drug Manufacturers",       125.0),
    ("CVX",   "Chevron Corp.",             "Energy",           "Oil & Gas Integrated",    160.0),
    ("TMO",   "Thermo Fisher Scientific", "Healthcare",        "Diagnostics & Research",  530.0),
    ("KO",    "The Coca-Cola Co.",         "Consumer Defensive","Beverages—Non-Alcoholic",  63.0),
    ("CSCO",  "Cisco Systems Inc.",        "Technology",       "Communication Equipment",  51.0),
    ("ACN",   "Accenture PLC",             "Technology",       "Information Technology",  330.0),
    ("ABT",   "Abbott Laboratories",      "Healthcare",        "Medical Devices",          115.0),
    ("LIN",   "Linde PLC",                "Basic Materials",   "Specialty Chemicals",      455.0),
    ("MCD",   "McDonald's Corp.",          "Consumer Cyclical","Restaurants",              285.0),
    ("PEP",   "PepsiCo Inc.",              "Consumer Defensive","Beverages—Non-Alcoholic", 165.0),
    ("IBM",   "IBM Corp.",                 "Technology",       "IT Services",              185.0),
    ("GE",    "GE Aerospace",             "Industrials",       "Aerospace & Defense",      180.0),
    ("TXN",   "Texas Instruments Inc.",    "Technology",       "Semiconductors",          175.0),
    ("ADBE",  "Adobe Inc.",                "Technology",       "Software—Application",    465.0),
    ("AMD",   "Advanced Micro Devices",   "Technology",        "Semiconductors",          155.0),
    ("QCOM",  "QUALCOMM Inc.",             "Technology",       "Semiconductors",          180.0),
    ("PM",    "Philip Morris Intl.",       "Consumer Defensive","Tobacco",                 125.0),
    ("DHR",   "Danaher Corp.",             "Healthcare",       "Diagnostics & Research",  225.0),
    ("WFC",   "Wells Fargo & Co.",         "Financial",        "Banks—Diversified",         55.0),
    ("GS",    "Goldman Sachs Group",       "Financial",        "Capital Markets",          480.0),
    ("BX",    "Blackstone Inc.",           "Financial",        "Asset Management",         125.0),
    ("ISRG",  "Intuitive Surgical Inc.",   "Healthcare",       "Medical Devices",          420.0),
    ("INTU",  "Intuit Inc.",               "Technology",       "Software—Application",    620.0),
    ("CAT",   "Caterpillar Inc.",          "Industrials",      "Farm & Heavy Constr.",     345.0),
    ("NOW",   "ServiceNow Inc.",           "Technology",       "Software—Application",    845.0),
    ("SPGI",  "S&P Global Inc.",           "Financial",        "Financial Data & Exchanges",455.0),
    ("BKNG",  "Booking Holdings Inc.",     "Consumer Cyclical","Travel Services",         3800.0),
    ("UNP",   "Union Pacific Corp.",       "Industrials",      "Railroads",               235.0),
    ("AXP",   "American Express Co.",      "Financial",        "Credit Services",          235.0),
    ("AMGN",  "Amgen Inc.",                "Healthcare",       "Drug Manufacturers",       300.0),
    ("PLD",   "Prologis Inc.",             "Real Estate",      "REIT—Industrial",           115.0),
    ("RTX",   "RTX Corp.",                 "Industrials",      "Aerospace & Defense",      125.0),
    ("NEE",   "NextEra Energy Inc.",       "Utilities",        "Utilities—Diversified",     69.0),
    ("HON",   "Honeywell Intl Inc.",       "Industrials",      "Diversified Industrials",  205.0),
    ("SYK",   "Stryker Corp.",             "Healthcare",       "Medical Devices",          350.0),
    ("LOW",   "Lowe's Companies Inc.",     "Consumer Cyclical","Home Improvement Retail",  240.0),
    ("MS",    "Morgan Stanley",            "Financial",        "Capital Markets",          107.0),
    ("T",     "AT&T Inc.",                 "Communication",    "Telecom Services",          19.0),
    ("VRTX",  "Vertex Pharmaceuticals",   "Healthcare",        "Drug Manufacturers",       460.0),
    ("ELV",   "Elevance Health Inc.",      "Healthcare",       "Healthcare Plans",         415.0),
    ("BSX",   "Boston Scientific Corp.",   "Healthcare",       "Medical Devices",           80.0),
    ("MDT",   "Medtronic PLC",             "Healthcare",       "Medical Devices",           85.0),
    ("BLK",   "BlackRock Inc.",            "Financial",        "Asset Management",         900.0),
    ("C",     "Citigroup Inc.",            "Financial",        "Banks—Diversified",         64.0),
    ("DE",    "Deere & Company",           "Industrials",      "Farm & Heavy Constr.",     395.0),
    ("UBER",  "Uber Technologies Inc.",    "Technology",       "Software—Application",      70.0),
    ("ADP",   "ADP Inc.",                  "Technology",       "Staffing & Employment",    255.0),
    ("PGR",   "Progressive Corp.",         "Financial",        "Insurance—Prop & Cas.",    255.0),
    ("CB",    "Chubb Limited",             "Financial",        "Insurance—Diversified",    265.0),
    ("MMC",   "Marsh & McLennan Cos.",     "Financial",        "Insurance—Diversified",    210.0),
    ("PANW",  "Palo Alto Networks Inc.",   "Technology",       "Software—Infrastructure",  165.0),
    ("ETN",   "Eaton Corp PLC",            "Industrials",      "Electrical Equipment",     315.0),
    ("CI",    "The Cigna Group",           "Healthcare",       "Healthcare Plans",         330.0),
    ("SO",    "Southern Company",          "Utilities",        "Utilities—Regulated Elec.",  88.0),
    ("MO",    "Altria Group Inc.",         "Consumer Defensive","Tobacco",                  44.0),
    ("DUK",   "Duke Energy Corp.",         "Utilities",        "Utilities—Regulated Elec.", 105.0),
    ("ZTS",   "Zoetis Inc.",               "Healthcare",       "Drug Manufacturers",       175.0),
    ("AON",   "Aon PLC",                   "Financial",        "Insurance—Diversified",    355.0),
    ("TJX",   "TJX Companies Inc.",        "Consumer Cyclical","Apparel Retail",           115.0),
    ("ICE",   "Intercontinental Exchange","Financial",         "Financial Exchanges",       155.0),
    ("SCHW",  "Charles Schwab Corp.",      "Financial",        "Capital Markets",           70.0),
    ("REGN",  "Regeneron Pharmaceuticals","Healthcare",        "Drug Manufacturers",       730.0),
    ("CME",   "CME Group Inc.",            "Financial",        "Financial Exchanges",       220.0),
    ("CL",    "Colgate-Palmolive Co.",     "Consumer Defensive","Household Products",       97.0),
    ("ITW",   "Illinois Tool Works Inc.",  "Industrials",      "Specialty Industrial",     245.0),
    ("USB",   "U.S. Bancorp",              "Financial",        "Banks—Regional",            44.0),
    ("WM",    "Waste Management Inc.",     "Industrials",      "Waste Management",         220.0),
    ("PNC",   "PNC Financial Services",   "Financial",        "Banks—Regional",            165.0),
    ("EMR",   "Emerson Electric Co.",      "Industrials",      "Electrical Equipment",     100.0),
    ("GM",    "General Motors Co.",        "Consumer Cyclical","Auto Manufacturers",        48.0),
    ("F",     "Ford Motor Co.",            "Consumer Cyclical","Auto Manufacturers",        11.0),
    ("TGT",   "Target Corp.",              "Consumer Defensive","Discount Stores",          130.0),
    ("FDX",   "FedEx Corp.",               "Industrials",      "Integrated Freight",       275.0),
    ("NSC",   "Norfolk Southern Corp.",    "Industrials",      "Railroads",               235.0),
    ("COF",   "Capital One Financial",     "Financial",        "Credit Services",          155.0),
    ("NOC",   "Northrop Grumman Corp.",    "Industrials",      "Aerospace & Defense",      480.0),
    ("HUM",   "Humana Inc.",               "Healthcare",       "Healthcare Plans",         275.0),
    ("GD",    "General Dynamics Corp.",    "Industrials",      "Aerospace & Defense",      275.0),
    ("LMT",   "Lockheed Martin Corp.",     "Industrials",      "Aerospace & Defense",      445.0),
    ("SLB",   "SLB",                       "Energy",           "Oil & Gas Equipment",       43.0),
    ("ECL",   "Ecolab Inc.",               "Basic Materials",  "Specialty Chemicals",      235.0),
    ("HCA",   "HCA Healthcare Inc.",       "Healthcare",       "Medical Care Facilities",  340.0),
    ("OKE",   "ONEOK Inc.",                "Energy",           "Oil & Gas Midstream",       90.0),
    ("CTAS",  "Cintas Corp.",              "Industrials",      "Staffing & Employment",    195.0),
    ("APD",   "Air Products & Chemicals", "Basic Materials",  "Specialty Chemicals",      305.0),
    ("FTNT",  "Fortinet Inc.",             "Technology",       "Software—Infrastructure",  62.0),
    ("MCO",   "Moody's Corp.",             "Financial",        "Financial Data & Exchanges",415.0),
    ("WELL",  "Welltower Inc.",            "Real Estate",      "REIT—Healthcare",          130.0),
    ("D",     "Dominion Energy Inc.",      "Utilities",        "Utilities—Regulated Elec.", 18.0),
    ("PSX",   "Phillips 66",               "Energy",           "Oil & Gas Refining",       140.0),
    ("MPC",   "Marathon Petroleum Corp.", "Energy",            "Oil & Gas Refining",       165.0),
    ("VLO",   "Valero Energy Corp.",       "Energy",           "Oil & Gas Refining",       135.0),
    ("EOG",   "EOG Resources Inc.",        "Energy",           "Oil & Gas E&P",            130.0),
    ("COP",   "ConocoPhillips",            "Energy",           "Oil & Gas E&P",            105.0),
    ("PSA",   "Public Storage",            "Real Estate",      "REIT—Industrial",          315.0),
    ("AMT",   "American Tower Corp.",      "Real Estate",      "REIT—Specialty",           185.0),
    ("EQIX",  "Equinix Inc.",              "Real Estate",      "REIT—Specialty",           785.0),
    ("O",     "Realty Income Corp.",       "Real Estate",      "REIT—Retail",               56.0),
    ("SPG",   "Simon Property Group",      "Real Estate",      "REIT—Retail",              165.0),
    ("PH",    "Parker-Hannifin Corp.",     "Industrials",      "Specialty Industrial",     545.0),
    ("AFL",   "Aflac Inc.",                "Financial",        "Insurance—Life",            110.0),
    ("KLAC",  "KLA Corp.",                 "Technology",       "Semiconductor Equipment",  780.0),
    ("LRCX",  "Lam Research Corp.",        "Technology",       "Semiconductor Equipment",  870.0),
    ("AMAT",  "Applied Materials Inc.",    "Technology",       "Semiconductor Equipment",  185.0),
    ("MCHP",  "Microchip Technology",      "Technology",       "Semiconductors",            67.0),
    ("MU",    "Micron Technology Inc.",    "Technology",       "Semiconductors",           120.0),
    ("SNPS",  "Synopsys Inc.",             "Technology",       "Software—Application",     515.0),
    ("CDNS",  "Cadence Design Systems",    "Technology",       "Software—Application",     290.0),
    ("NXPI",  "NXP Semiconductors",        "Technology",       "Semiconductors",           210.0),
    ("ON",    "ON Semiconductor Corp.",    "Technology",       "Semiconductors",            51.0),
    ("STZ",   "Constellation Brands Inc.","Consumer Defensive","Beverages—Alcoholic",      230.0),
    ("KMB",   "Kimberly-Clark Corp.",      "Consumer Defensive","Household Products",      130.0),
    ("GIS",   "General Mills Inc.",        "Consumer Defensive","Packaged Foods",            64.0),
    ("HSY",   "Hershey Co.",               "Consumer Defensive","Confectioners",           165.0),
    ("SHW",   "Sherwin-Williams Co.",      "Basic Materials",  "Specialty Chemicals",      345.0),
    ("PPG",   "PPG Industries Inc.",       "Basic Materials",  "Specialty Chemicals",      115.0),
    ("NEM",   "Newmont Corp.",             "Basic Materials",  "Gold",                      46.0),
    ("FCX",   "Freeport-McMoRan Inc.",     "Basic Materials",  "Copper",                    43.0),
    ("DOW",   "Dow Inc.",                  "Basic Materials",  "Chemicals",                 40.0),
    ("LYB",   "LyondellBasell Industries","Basic Materials",   "Chemicals",                 80.0),
    ("DHI",   "D.R. Horton Inc.",          "Consumer Cyclical","Residential Construction",  155.0),
    ("LEN",   "Lennar Corp.",              "Consumer Cyclical","Residential Construction",  135.0),
    ("PYPL",  "PayPal Holdings Inc.",      "Financial",        "Credit Services",           67.0),
    ("PLTR",  "Palantir Technologies",     "Technology",       "Software—Application",      22.0),
    ("SNOW",  "Snowflake Inc.",            "Technology",       "Software—Application",     165.0),
    ("DDOG",  "Datadog Inc.",              "Technology",       "Software—Application",     115.0),
    ("CRWD",  "CrowdStrike Holdings",      "Technology",       "Software—Infrastructure",  360.0),
    ("ZS",    "Zscaler Inc.",              "Technology",       "Software—Infrastructure",  195.0),
    ("NET",   "Cloudflare Inc.",           "Technology",       "Software—Infrastructure",   85.0),
    ("TEAM",  "Atlassian Corp.",           "Technology",       "Software—Application",     190.0),
    ("HUBS",  "HubSpot Inc.",              "Technology",       "Software—Application",     570.0),
    ("VEEV",  "Veeva Systems Inc.",        "Healthcare",       "Health Information Svcs.",  210.0),
    ("WDAY",  "Workday Inc.",              "Technology",       "Software—Application",     250.0),
    ("BILL",  "BILL Holdings Inc.",        "Technology",       "Software—Application",      69.0),
    ("TTD",   "The Trade Desk Inc.",       "Technology",       "Software—Application",      85.0),
    ("ROKU",  "Roku Inc.",                 "Technology",       "Entertainment",              63.0),
    ("COIN",  "Coinbase Global Inc.",      "Financial",        "Capital Markets",           225.0),
    ("RBLX",  "Roblox Corp.",              "Technology",       "Electronic Gaming & Multi", 40.0),
    ("SNAP",  "Snap Inc.",                 "Communication",    "Internet Content",           10.0),
    ("PINS",  "Pinterest Inc.",            "Communication",    "Internet Content",           31.0),
    ("LYFT",  "Lyft Inc.",                 "Technology",       "Software—Application",      13.0),
    ("ABNB",  "Airbnb Inc.",               "Consumer Cyclical","Travel Services",           143.0),
    ("DASH",  "DoorDash Inc.",             "Consumer Cyclical","Internet Retail",           195.0),
    ("DKNG",  "DraftKings Inc.",           "Consumer Cyclical","Gambling",                  40.0),
    ("HOOD",  "Robinhood Markets Inc.",    "Financial",        "Capital Markets",            22.0),
]

# Sector-level signal biases (to make demo data realistic)
_SECTOR_BIAS = {
    "Technology":        {"recom": -0.4, "eps_next_y":  0.08, "momentum": 0.06},
    "Healthcare":        {"recom": -0.2, "eps_next_y":  0.04, "momentum": 0.02},
    "Financial":         {"recom": -0.1, "eps_next_y":  0.02, "momentum": 0.01},
    "Consumer Cyclical": {"recom":  0.0, "eps_next_y":  0.01, "momentum": 0.00},
    "Consumer Defensive":{"recom":  0.1, "eps_next_y": -0.01, "momentum":-0.02},
    "Industrials":       {"recom": -0.1, "eps_next_y":  0.02, "momentum": 0.01},
    "Energy":            {"recom":  0.2, "eps_next_y": -0.03, "momentum":-0.03},
    "Communication":     {"recom": -0.1, "eps_next_y":  0.03, "momentum": 0.03},
    "Real Estate":       {"recom":  0.2, "eps_next_y": -0.02, "momentum":-0.04},
    "Utilities":         {"recom":  0.3, "eps_next_y": -0.02, "momentum":-0.05},
    "Basic Materials":   {"recom":  0.1, "eps_next_y": -0.01, "momentum":-0.01},
}


def _pct_str(v: float) -> str:
    return f"{v * 100:.2f}%"


def fetch_all(max_stocks: int = 100, filters: str = "geo_usa",
              delay: float = 1.5) -> "pd.DataFrame":
    """
    Generate synthetic Finviz-style data for demonstration purposes.
    All values are realistic-range fiction; not real market data.
    """
    import pandas as pd

    rng = np.random.default_rng(seed=42)

    universe = _UNIVERSE[:max_stocks]
    import sys as _sys
    print(f"\n  Demo mode: generating synthetic data for {len(universe)} stocks.",
          file=_sys.stderr)

    rows = []
    for ticker, company, sector, industry, base_price in universe:
        bias = _SECTOR_BIAS.get(sector, {"recom": 0.0, "eps_next_y": 0.0, "momentum": 0.0})

        # Correlated latent "quality" factor −1 → +1
        quality = rng.normal(0.0, 0.4)
        quality = np.clip(quality, -1, 1)

        # Analyst recommendation 1−5 (lower = stronger buy)
        recom_raw = 3.0 + bias["recom"] - quality * 1.2 + rng.normal(0, 0.4)
        recom = round(float(np.clip(recom_raw, 1.0, 5.0)), 1)

        # Price with some volatility around base
        price = float(base_price * (1 + rng.normal(0, 0.15)))
        price = max(price, 1.0)

        # Target price: upside correlates with analyst optimism
        upside_pct = 0.15 + (3 - recom) * 0.08 + rng.normal(0, 0.10)
        target_price = round(price * (1 + upside_pct), 2)

        # SMA deviations: correlated with quality & momentum
        mom_bias = bias["momentum"] + quality * 0.08
        sma20  = mom_bias + rng.normal(0, 0.04)
        sma50  = mom_bias * 1.5 + rng.normal(0, 0.07)
        sma200 = mom_bias * 2.5 + rng.normal(0, 0.12)

        # Performance (momentum)
        p1m = mom_bias + rng.normal(0, 0.04)
        p3m = mom_bias * 3 + rng.normal(0, 0.08)
        p6m = mom_bias * 5 + rng.normal(0, 0.14)
        p1y = mom_bias * 8 + rng.normal(0, 0.22)

        # RSI: high quality → likely overbought
        rsi = 50 + quality * 20 + rng.normal(0, 10)
        rsi = float(np.clip(rsi, 5, 99))

        # EPS Q/Q: recent earnings beat — correlated with quality
        eps_qq = quality * 0.20 + rng.normal(0, 0.15)

        # EPS growth
        eps_ny = bias["eps_next_y"] + quality * 0.15 + rng.normal(0, 0.10)
        eps_5y = bias["eps_next_y"] * 0.8 + quality * 0.12 + rng.normal(0, 0.08)

        # Smart money: insider/inst transactions (3-month change)
        insider_trans = quality * 0.04 + rng.normal(0, 0.03)
        inst_trans    = quality * 0.03 + rng.normal(0, 0.025)

        # Float short: high-quality stocks tend to have lower short interest
        float_short = max(0.005, 0.10 - quality * 0.07 + rng.normal(0, 0.04))
        short_ratio = max(0.1, float_short * 20 + rng.normal(0, 1.5))

        # 52W high proximity: correlated with momentum
        # 0 = at 52W high, negative = below. Strong momentum → near high.
        w52h = min(0.0, mom_bias * 3 - 0.05 + rng.normal(0, 0.08))
        w52l = max(0.0, -mom_bias * 3 + 0.30 + rng.normal(0, 0.12))

        # Beta: growth sectors higher
        beta_base = {"Technology": 1.3, "Consumer Defensive": 0.6,
                     "Utilities": 0.5, "Financial": 1.1}.get(sector, 1.0)
        beta = max(0.1, beta_base + rng.normal(0, 0.2))

        rows.append({
            "Ticker":        ticker,
            "Company":       company,
            "Sector":        sector,
            "Industry":      industry,
            "Market Cap":    f"{rng.integers(1, 3000)}B",
            "Recom":         str(recom),
            "Target Price":  str(target_price),
            "Price":         str(round(price, 2)),
            "SMA20":         _pct_str(sma20),
            "SMA50":         _pct_str(sma50),
            "SMA200":        _pct_str(sma200),
            "52W High":      _pct_str(w52h),
            "52W Low":       _pct_str(w52l),
            "RSI":           str(round(rsi, 1)),
            "Perf Month":    _pct_str(p1m),
            "Perf Quart":    _pct_str(p3m),
            "Perf Half":     _pct_str(p6m),
            "Perf Year":     _pct_str(p1y),
            "Perf YTD":      _pct_str(p3m),
            "EPS Q/Q":       _pct_str(eps_qq),
            "EPS next Y":    _pct_str(eps_ny),
            "EPS next 5Y":   _pct_str(eps_5y),
            "Insider Trans": _pct_str(insider_trans),
            "Inst Trans":    _pct_str(inst_trans),
            "Float Short":   _pct_str(float_short),
            "Short Ratio":   str(round(float(short_ratio), 1)),
            "Beta":          str(round(float(beta), 2)),
        })

    return pd.DataFrame(rows)
