# semantic_chunker.py

# Embedding-based semantic chunking implementation
# Requires: pip install sentence-transformers numpy

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import re


class SemanticChunker:
    """
    Splits documents into semantic chunks based on embedding similarity.

    The chunker detects topic shifts by comparing embeddings of consecutive
    sentences. When similarity drops below the threshold, a new chunk begins.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.2,
        min_chunk_size: int = 2000,
        max_chunk_size: int = 6000
    ):
        # Load the sentence transformer model for generating embeddings
        self.model = SentenceTransformer(model_name)

        # Threshold below which we consider sentences to be on different topics
        # Lower values = fewer, larger chunks; higher values = more, smaller chunks
        self.similarity_threshold = similarity_threshold

        # Prevent chunks that are too small (noise) or too large (defeats purpose)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        Handles common edge cases like abbreviations and decimal numbers.
        """
        # Protect common abbreviations to prevent false sentence splits.
        # Python's re module requires fixed-width lookbehind, so we use
        # placeholders instead of a variable-width negative lookbehind.
        abbreviations = [
            'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.',
            'Sr.', 'Jr.', 'vs.', 'etc.', 'e.g.', 'i.e.'
        ]
        protected = text
        for i, abbr in enumerate(abbreviations):
            protected = protected.replace(abbr, f'__ABBR{i}__')

        # Split on sentence-ending punctuation followed by space and capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)

        # Restore abbreviations, clean whitespace, and drop empty entries
        result = []
        for sentence in sentences:
            for i, abbr in enumerate(abbreviations):
                sentence = sentence.replace(f'__ABBR{i}__', abbr)
            if sentence.strip():
                result.append(sentence.strip())

        return result

    def compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embedding vectors for all sentences in batch.
        Returns a 2D array where each row is a sentence embedding.
        """
        return self.model.encode(sentences, convert_to_numpy=True)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        Returns a value between -1 and 1, where 1 means identical direction.
        """
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product == 0:
            return 0.0

        return dot_product / norm_product

    def find_breakpoints(
        self,
        sentences: List[str],
        embeddings: np.ndarray
    ) -> List[int]:
        """
        Identify indices where semantic breaks occur.

        Compares each sentence to its neighbor. When similarity drops
        below threshold, that position becomes a chunk boundary.
        """
        breakpoints = []

        for i in range(len(sentences) - 1):
            similarity = self.cosine_similarity(embeddings[i], embeddings[i + 1])

            if similarity < self.similarity_threshold:
                breakpoints.append(i + 1)

        return breakpoints

    def merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks that are too small with their neighbors.
        Prevents fragmentation while respecting max size limits.
        """
        if not chunks:
            return chunks

        merged = []
        buffer = chunks[0]

        for chunk in chunks[1:]:
            if len(buffer) < self.min_chunk_size:
                combined = buffer + " " + chunk

                # Only merge if result stays under max size
                if len(combined) <= self.max_chunk_size:
                    buffer = combined
                else:
                    # Buffer too small but can't merge; keep it anyway
                    merged.append(buffer)
                    buffer = chunk
            else:
                merged.append(buffer)
                buffer = chunk

        # Don't forget the last buffer
        merged.append(buffer)

        return merged

    def chunk(self, text: str) -> List[str]:
        """
        Main entry point: split document into semantic chunks.

        Args:
            text: The document text to chunk

        Returns:
            List of text chunks, each representing a semantic unit
        """
        # Step 1: Break into sentences
        sentences = self.split_into_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Step 2: Generate embeddings for all sentences
        embeddings = self.compute_embeddings(sentences)

        # Step 3: Find where topic shifts occur
        breakpoints = self.find_breakpoints(sentences, embeddings)

        # Step 4: Build chunks from sentences between breakpoints
        chunks = []
        start_idx = 0

        for bp in breakpoints:
            chunk_sentences = sentences[start_idx:bp]
            chunk_text = " ".join(chunk_sentences)
            chunks.append(chunk_text)
            start_idx = bp

        # Add the final chunk
        if start_idx < len(sentences):
            chunk_text = " ".join(sentences[start_idx:])
            chunks.append(chunk_text)

        # Step 5: Merge chunks that are too small
        chunks = self.merge_small_chunks(chunks)

        return chunks