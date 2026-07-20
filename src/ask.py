"""
Quick command-line interface to ask the RAG system any question.
Usage: python3 src/ask.py "your question here"
"""

import sys
from generate import answer_question


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 src/ask.py "your question here"')
        return

    question = sys.argv[1]
    answer, chunks = answer_question(question)

    print("QUESTION:", question)
    print("\nANSWER:\n", answer)
    print("\n\nSOURCES USED:")
    for i, c in enumerate(chunks):
        print(f"[{i+1}] {c['title']} (distance: {c['distance']:.3f})")


if __name__ == "__main__":
    main()