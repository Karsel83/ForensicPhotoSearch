"""Apply the two-stage FAISS integration to image_person_search.py once."""

from pathlib import Path


TARGET = Path("image_person_search.py")


def replace_once(text, old, new):
    if old not in text:
        raise SystemExit("Expected source block was not found; no file was changed.")
    return text.replace(old, new, 1)


def main():
    if not TARGET.exists():
        raise SystemExit("Run this script from the ForensicPhotoSearch project root.")

    text = TARGET.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from image_database import ImageEmbeddingIndex\n",
        "from image_database import ImageEmbeddingIndex\n"
        "from two_stage_faiss import TwoStageImageSearch\n"
    )
    text = replace_once(
        text,
        '        metadata_path="data/embeddings/metadata.json",\n'
        '        top_k=50\n',
        '        metadata_path="data/embeddings/metadata.json",\n'
        '        faiss_index_path="data/faiss/image.index",\n'
        '        candidate_k=1000,\n'
        '        top_k=50\n'
    )
    text = replace_once(
        text,
        "        self.metadata_path = metadata_path\n"
        "        self.top_k = top_k\n",
        "        self.metadata_path = metadata_path\n"
        "        self.faiss_index_path = faiss_index_path\n"
        "        self.candidate_k = candidate_k\n"
        "        self.top_k = top_k\n"
    )

    old_search = '''        # No evidence image is decoded and no YOLO/OSNet inference is run here.
        # The supplied paths remain in the signature for compatibility with the
        # existing unified-search caller.
        index = ImageEmbeddingIndex(
            self.embeddings_path,
            self.metadata_path
        )

        matches = index.search(query_feature, top_k=self.top_k)
        results = []

        for metadata, score in matches:
            result = dict(metadata)
            result["similarity"] = round(float(score), 4)
            results.append(result)

        print(
            f"[*] Vector index entries: {len(index.metadata)} "
            f"(Top-{len(results)} returned)"
        )

        return results
'''
    new_search = '''        # FAISS retrieves candidates; exact cosine re-ranks only those candidates.
        # The compatible arguments remain because forensic_search.py already calls this method.
        searcher = TwoStageImageSearch(
            index_file=self.faiss_index_path,
            embeddings_file=self.embeddings_path,
            metadata_file=self.metadata_path
        )

        results, metrics = searcher.search(
            query_feature,
            candidate_k=self.candidate_k,
            top_k=self.top_k
        )

        print(
            f"[*] Two-stage image search: "
            f"candidates={metrics['candidate_k']}, "
            f"total={metrics['total_ms']:.4f}ms"
        )

        return results
'''
    text = replace_once(text, old_search, new_search)
    TARGET.write_text(text, encoding="utf-8")
    print("[*] Two-stage FAISS integration applied to image_person_search.py")


if __name__ == "__main__":
    main()
