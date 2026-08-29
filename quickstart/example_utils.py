"""Small utilities used by the quickstart examples (CRBOT DSH migration E2E fixture)."""


def clamp(value, lo, hi):
    """Clamp value into [lo, hi]."""
    if lo > hi:
        raise ValueError("lo must be <= hi")
    return max(lo, min(value, hi))


def parse_ports(spec):
    """Parse a comma-separated port spec string like '8000,8001-8003' into a sorted list."""
    ports = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ports.extend(range(int(a), int(b) + 1))
        else:
            ports.append(int(part))
    return sorted(ports)


def format_duration(seconds):
    """Format seconds into a human readable duration string."""
    # TODO: implement
    return ""
