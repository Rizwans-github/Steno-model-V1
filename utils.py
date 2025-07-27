def get_char_map():
    chars = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z",
        "ng", "pee", "bee", "tee", "dee"
    ]
    return {c: i + 1 for i, c in enumerate(chars)}
