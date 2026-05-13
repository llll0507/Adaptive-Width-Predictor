def extract_answer(text):
    def find_matching_brace(s, start):
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == "{":
                count += 1
            elif s[i] == "}":
                count -= 1
            i += 1
        return i if count == 0 else -1

    # Find start position of \boxed{
    start = text.find("\\boxed{")
    if start == -1:
        return None

    # Start from brace position
    content_start = start + 7  # len('\\boxed{') = 7
    end = find_matching_brace(text, content_start)

    if end == -1:
        return None

    return text[content_start : end - 1]


if __name__ == "__main__":
    # Example usage
    text = "The polar coordinates are $\\boxed{\\left(3, \\frac{\\pi}{2} \\right)}.$"
    answer = extract_answer(text)
    print(answer)  # Should output: \left(3, \frac{\pi}{2} \right)
    print(type(answer))

    text = "The polar coordinates are $\\bo{5}.$"
    answer = extract_answer(text)
    print(answer)  # Should output: 5
