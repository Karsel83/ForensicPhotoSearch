from image_person_search import ImagePersonSearch
from reid_model import PersonReID

QUERY = r".\video\data\query.jpg"
IMAGE_DIR = r".\evidence\images"
EVIDENCE_DIR = r".\results\image_evidence"

print("=" * 60)
print("IMAGE PERSON SEARCH TEST")
print("=" * 60)

print("[1] Loading Re-ID model...")
reid = PersonReID()

print("[2] Extracting query feature...")
query_feature = reid.extract(QUERY)

print("[3] Starting image search...")

searcher = ImagePersonSearch()

results = searcher.search_directory(
    query_feature,
    IMAGE_DIR,
    EVIDENCE_DIR
)

print()
print("=" * 60)
print("TOP 10 IMAGE RESULTS")
print("=" * 60)

for i, result in enumerate(results[:10], start=1):

    print(
        f"{i:02d}. "
        f"{result['image']:<20} "
        f"person={result['person_index']:<2} "
        f"score={result['similarity']:.4f}"
    )

print()
print(f"[*] Total person detections: {len(results)}")
print(f"[*] Evidence directory: {EVIDENCE_DIR}")
