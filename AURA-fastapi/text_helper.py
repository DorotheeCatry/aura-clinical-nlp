def wrap_text(text, max_length=80):
    lines = []
    line = ""
    text = str(text)
    for word in text.split():
        if len(line) + len(word) + 1 <= max_length:
            line += (" " if line else "") + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return "\n".join(lines)