import re


def extract_mcqs(text):
    # Regular expression to find patterns like "(a) option text"
    pattern = re.compile(r"\([a-z]\) [^\)]+")

    # Find all occurrences of the pattern
    mcqs = pattern.findall(text)

    # Join all the extracted MCQs into a single string separated by new lines
    result = "\n".join(mcqs)
    return result
