"""Build the policy index and measure retrieval quality (Hit@1, Hit@3, MRR)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.rag.store import PolicyStore


def main():
    store = PolicyStore().build()
    store.save()
    print(f"Indexed {len(store.chunks)} chunks from {len(store.documents)} documents")

    qa_path = config.POLICY_DIR / "eval_questions.json"
    if not qa_path.exists():
        return
    items = json.loads(qa_path.read_text())
    hits1 = hits3 = rr = 0.0
    rows = []
    for item in items:
        docs = [c.doc for c, _ in store.search(item["question"], k=5)]
        rank = docs.index(item["expected_doc"]) + 1 if item["expected_doc"] in docs else None
        hits1 += rank == 1; hits3 += bool(rank and rank <= 3); rr += 1 / rank if rank else 0
        rows.append({**item, "rank": rank, "top_doc": docs[0] if docs else None})
    n = len(items)
    report = {"questions": n, "hit@1": round(hits1 / n, 3), "hit@3": round(hits3 / n, 3),
              "mrr": round(rr / n, 3), "details": rows}
    (config.REPORTS_DIR / "rag_evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"Hit@1={report['hit@1']}  Hit@3={report['hit@3']}  MRR={report['mrr']}")


if __name__ == "__main__":
    main()
