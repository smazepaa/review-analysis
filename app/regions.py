_REGIONS: dict[str, tuple[str, ...]] = {
    "North America": ("us", "ca"),
    "Latin America & Caribbean": (
        "mx", "br", "ar", "cl", "co", "pe", "ve", "uy", "py", "bo",
        "ec", "gt", "hn", "sv", "ni", "cr", "pa", "do", "jm", "tt",
        "bs", "bb", "bz", "dm", "gd", "kn", "lc", "vc", "ai", "ag",
        "ky", "ms", "tc", "vg", "gy", "sr",
    ),
    "Europe": (
        "gb", "de", "fr", "es", "it", "nl", "se", "no", "dk", "fi",
        "pl", "ru", "ie", "ch", "at", "be", "pt", "gr", "cz", "hu",
        "ro", "ua", "ee", "lv", "lt", "sk", "si", "hr", "rs", "mt",
        "lu", "cy", "is", "by", "md", "mk", "al", "bg", "ba", "me",
        "am", "ge", "az",
    ),
    "Middle East & North Africa": (
        "ae", "sa", "qa", "kw", "bh", "om", "jo", "lb", "il", "eg",
        "ye", "tr", "tn", "dz", "ma",
    ),
    "Sub-Saharan Africa": (
        "za", "ng", "ke", "gh", "ug", "tz", "ci", "sn", "mz", "mu",
        "mg", "mw", "na", "bw", "sc", "ml", "bf", "gm", "gn", "gw",
        "ga", "cg", "cd", "cv", "td", "ne", "sl", "lr", "sz", "zm",
        "zw", "rw", "ao", "bj",
    ),
    "Asia Pacific": (
        "jp", "kr", "cn", "tw", "hk", "sg", "my", "th", "id", "ph",
        "vn", "in", "np", "pk", "lk", "mn", "kz", "kg", "tj", "tm",
        "uz", "bn", "kh", "la", "mm", "bt", "mo", "mv",
    ),
    "Oceania": ("au", "nz", "fj", "pg", "sb", "vu", "fm", "pw"),
}

_REVERSE: dict[str, str] = {
    code: region for region, codes in _REGIONS.items() for code in codes
}


def region_for(country: str | None) -> str:
    if not country:
        return "Other"
    return _REVERSE.get(country.lower(), "Other")


def countries_in_region(region: str) -> tuple[str, ...]:
    return _REGIONS.get(region, ())


def all_regions() -> tuple[str, ...]:
    return tuple(_REGIONS.keys())
