"""
ETF metadata and classification helpers.
"""

from typing import Dict, List


CATEGORY_LABELS = {
    "wide_basis": "宽基",
    "industry": "行业",
    "theme": "主题",
    "commodity": "商品",
    "bond": "债券",
    "cross_border": "跨境",
    "other": "其他",
}

WIDE_BASIS_KEYWORDS = (
    "沪深300", "中证500", "中证1000", "中证a500", "中证a50", "msci中国a50",
    "上证50", "上证180", "上证380", "上证综合", "上证指数", "深证100",
    "深证50", "深证成指", "创业板", "科创50", "北证50",
)

INDUSTRY_KEYWORDS = {
    "financials": ("证券", "券商", "银行", "保险", "金融", "地产", "房地产"),
    "healthcare": ("医药", "医疗", "生物", "创新药"),
    "consumer": ("消费", "食品", "酒", "家电", "零售"),
    "technology": ("半导体", "芯片", "电子", "计算机", "通信", "软件", "5g"),
    "energy": ("能源", "煤炭", "油气", "石油", "天然气"),
    "materials": ("有色", "钢铁", "稀土", "化工", "建材", "资源"),
    "industrial": ("机械", "军工", "装备", "制造", "工业"),
    "utilities": ("电力", "公用事业", "环保"),
    "autos": ("汽车", "新能源车"),
    "media": ("传媒", "游戏"),
    "agriculture": ("农业", "养殖"),
}

THEME_KEYWORDS = {
    "dividend": ("红利",),
    "state_owned": ("央企", "国企"),
    "esg": ("esg",),
    "innovation": ("创新", "升级", "g60"),
    "ai": ("人工智能", "机器人", "算力", "大数据"),
    "commodity": ("黄金", "大宗商品"),
}

SPECIAL_CODE_METADATA: Dict[str, Dict] = {
    "510300": {"category": "wide_basis", "sector": "broad_market", "tags": ["hs300"]},
    "510500": {"category": "wide_basis", "sector": "broad_market", "tags": ["zz500"]},
    "512100": {"category": "wide_basis", "sector": "broad_market", "tags": ["zz1000"]},
    "588000": {"category": "wide_basis", "sector": "broad_market", "tags": ["star50"]},
    "588050": {"category": "wide_basis", "sector": "broad_market", "tags": ["star50"]},
    "588080": {"category": "wide_basis", "sector": "broad_market", "tags": ["star50"]},
    "159915": {"category": "wide_basis", "sector": "broad_market", "tags": ["chinext"]},
    "159922": {"category": "wide_basis", "sector": "broad_market", "tags": ["chinext"]},
    "159949": {"category": "wide_basis", "sector": "broad_market", "tags": ["chinext"]},
    "159952": {"category": "wide_basis", "sector": "broad_market", "tags": ["chinext"]},
    "512010": {"category": "industry", "sector": "financials", "tags": ["broker"]},
    "512880": {"category": "industry", "sector": "financials", "tags": ["broker"]},
    "510660": {"category": "industry", "sector": "healthcare", "tags": ["pharma"]},
    "510650": {"category": "industry", "sector": "financials", "tags": ["real_estate"]},
    "510150": {"category": "industry", "sector": "consumer", "tags": ["consumer"]},
    "510630": {"category": "industry", "sector": "consumer", "tags": ["consumer"]},
    "510170": {"category": "commodity", "sector": "materials", "tags": ["commodity"]},
}


def _lower_name(name: str) -> str:
    return (name or "").strip().lower()


def _tags_from_name(name: str) -> List[str]:
    lowered = _lower_name(name)
    tags: List[str] = []
    for tag, keywords in THEME_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(tag)
    return tags


def infer_etf_metadata(code: str, name: str) -> Dict:
    override = SPECIAL_CODE_METADATA.get(code, {})
    lowered = _lower_name(name)

    if override:
        return {
            "code": code,
            "name": name,
            "category": override.get("category", "other"),
            "category_label": CATEGORY_LABELS.get(override.get("category", "other"), "其他"),
            "sector": override.get("sector", "other"),
            "tags": list(override.get("tags", [])),
        }

    for sector, keywords in INDUSTRY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tags = _tags_from_name(name)
            return {
                "code": code,
                "name": name,
                "category": "industry",
                "category_label": CATEGORY_LABELS["industry"],
                "sector": sector,
                "tags": tags,
            }

    if any(keyword in lowered for keyword in WIDE_BASIS_KEYWORDS):
        tags = _tags_from_name(name)
        return {
            "code": code,
            "name": name,
            "category": "wide_basis",
            "category_label": CATEGORY_LABELS["wide_basis"],
            "sector": "broad_market",
            "tags": tags,
        }

    tags = _tags_from_name(name)
    if "commodity" in tags:
        category = "commodity"
        sector = "materials"
    elif code.startswith(("511",)):
        category = "bond"
        sector = "fixed_income"
    elif code.startswith(("513", "159866", "159920")):
        category = "cross_border"
        sector = "global"
    elif tags:
        category = "theme"
        sector = "theme"
    else:
        category = "other"
        sector = "other"

    return {
        "code": code,
        "name": name,
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, "其他"),
        "sector": sector,
        "tags": tags,
    }
