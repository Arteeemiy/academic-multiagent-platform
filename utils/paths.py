def normalize_name(name: str) -> str:
    """
    Normalize model / embedder names for filesystem-safe paths.
    """
    return name.replace("/", "_")
