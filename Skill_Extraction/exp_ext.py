import re

def extract_experience(text):
    text = text.lower()

    # Case: "no prior experience required"
    if "no prior experience" in text:
        return 0

    # Case: "minimum X year maximum Y year"
    match_range = re.search(r"minimum\s+(\d+)\s+year.*maximum\s+(\d+)\s+year", text)
    if match_range:
        return (int(match_range.group(1)), int(match_range.group(2)))

    # Case: "(\d+) year experience" or "(\d+)-(\d+) year"
    match_single = re.search(r"(\d+)\s*[\-to]*\s*(\d*)\s*year", text)
    if match_single:
        start = int(match_single.group(1))
        end = int(match_single.group(2)) if match_single.group(2) else start
        return (start, end)

    return None
