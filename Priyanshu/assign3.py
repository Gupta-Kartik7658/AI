import os
import re

def preprocess_document(text):
    """
    Preprocess a document:
    1. Normalize (lowercase + remove punctuation except periods)
    2. Tokenize into sentences
    3. Join sentences back into a single string with periods
    """
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s.]', '', text)  # Remove punctuation except periods
    # Replace line breaks with space
    text = text.replace('\n', ' ')
    # Split into sentences by period
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    # Join sentences back into a single string separated by periods
    result = '. '.join(sentences) + '.'
    return result

# ---- Main Program ----
if __name__ == "__main__":
    # Get the folder where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build full paths for doc1.txt and doc2.txt
    doc1_path = os.path.join(script_dir, "doc1.txt")
    doc2_path = os.path.join(script_dir, "doc2.txt")

    # Read the files
    with open(doc1_path, "r", encoding="utf-8") as f:
        doc1 = f.read()
    with open(doc2_path, "r", encoding="utf-8") as f:
        doc2 = f.read()

    # Preprocess
    text1 = preprocess_document(doc1)
    text2 = preprocess_document(doc2)

    # Print results
    print("text1 =", repr(text1))
    print("text2 =", repr(text2))
