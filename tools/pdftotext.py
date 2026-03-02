from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
except Exception:  # optional dependency
    fitz = None


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"  # separation between pages
    return text


def _resolve(obj: Any) -> Any:
    return obj.get_object() if hasattr(obj, "get_object") else obj


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(v)
    except Exception:
        return ""


def _selected_text_from_annot(page: Any, ann: Any) -> str:
    """Best-effort reconstruction of the text covered by highlight-like annotations."""
    if fitz is None:
        return ""

    subtype = (ann.type[1] or "").lower()
    if subtype not in {"highlight", "underline", "strikeout", "squiggly"}:
        return ""

    quads = getattr(ann, "vertices", None)
    rects = []
    if quads and len(quads) >= 4:
        for i in range(0, len(quads), 4):
            quad = quads[i : i + 4]
            xs = []
            ys = []
            for p in quad:
                if hasattr(p, "x") and hasattr(p, "y"):
                    x, y = p.x, p.y
                else:
                    x, y = p[0], p[1]
                xs.append(x)
                ys.append(y)
            rects.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))
    else:
        rects = [ann.rect]

    # Prefer clip-based extraction (preserves ordering better than word intersection).
    clipped = []
    for rr in rects:
        t = (page.get_text("text", clip=rr) or "").strip()
        if t:
            clipped.append(t)
    if clipped:
        # De-dup identical chunks while preserving order.
        seen = set()
        uniq = []
        for t in clipped:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return "\n".join(uniq).strip()

    # Fallback: intersect words with the annotation rectangle(s).
    words = page.get_text("words")
    picked = []
    for x0, y0, x1, y1, w, *_ in words:
        r = fitz.Rect(x0, y0, x1, y1)
        if any(r.intersects(rr) for rr in rects):
            picked.append((y0, x0, w))

    picked.sort()
    # De-dup consecutive duplicates.
    out = []
    last = None
    for _, _, w in picked:
        if w != last:
            out.append(w)
        last = w
    return " ".join(out).strip()


def extract_comments_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """Extract PDF annotations/comments and (when available) the selected/highlighted text."""
    if fitz is not None:
        doc = fitz.open(pdf_path)
        out: List[Dict[str, str]] = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            ann = page.first_annot
            while ann:
                info = getattr(ann, "info", {}) or {}
                subtype = ann.type[1] or ""
                author = _to_str(info.get("title"))
                subject = _to_str(info.get("subject"))
                created = _to_str(info.get("creationDate"))
                modified = _to_str(info.get("modDate"))
                comment_text = _to_str(info.get("content")).strip()
                selected_text = _selected_text_from_annot(page, ann)

                if comment_text or selected_text:
                    out.append(
                        {
                            "page": str(page_idx + 1),
                            "subtype": subtype,
                            "author": author,
                            "subject": subject,
                            "created": created,
                            "modified": modified,
                            "selected_text": selected_text,
                            "comment": comment_text.replace("\r\n", "\n").replace("\r", "\n"),
                        }
                    )

                ann = ann.next

        doc.close()
        return out

    # Fallback: pypdf can read the comment text fields, but cannot reliably recover selected/highlighted text.
    reader = PdfReader(pdf_path)
    out: List[Dict[str, str]] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        annots = page.get("/Annots")
        if not annots:
            continue

        for a in annots:
            aobj = _resolve(a)
            if not hasattr(aobj, "get"):
                continue

            subtype = _to_str(aobj.get("/Subtype")).lstrip("/")
            author = _to_str(aobj.get("/T"))
            contents = _to_str(aobj.get("/Contents"))
            rich = _to_str(aobj.get("/RC"))
            subject = _to_str(aobj.get("/Subj"))
            created = _to_str(aobj.get("/CreationDate"))
            modified = _to_str(aobj.get("/M"))

            comment_text = (contents or rich).strip()
            if not comment_text:
                continue

            out.append(
                {
                    "page": str(page_idx),
                    "subtype": subtype,
                    "author": author,
                    "subject": subject,
                    "created": created,
                    "modified": modified,
                    "selected_text": "",
                    "comment": comment_text.replace("\r\n", "\n").replace("\r", "\n"),
                }
            )

    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract text (and optionally PDF comments/annotations) from a PDF")
    ap.add_argument("pdf", nargs="?", default="Millimeter_Wave_Radar_Hardware_and_Signal_Processi.pdf")
    ap.add_argument("--text-out", default=None, help="Path for extracted text output")
    ap.add_argument("--extract-comments", action="store_true", help="Also export PDF annotations/comments")
    ap.add_argument(
        "--comments-out",
        default=None,
        help="Comments output path (tab-separated .txt recommended). If omitted, defaults to <pdf>_comments.txt",
    )
    args = ap.parse_args(argv)

    pdf_path = args.pdf
    stem = Path(pdf_path).with_suffix("")

    text_out = Path(args.text_out) if args.text_out else stem.with_suffix(".txt")
    comments_out = Path(args.comments_out) if args.comments_out else stem.with_name(stem.name + "_comments.txt")

    extracted_text = extract_text_from_pdf(pdf_path)
    text_out.write_text(extracted_text, encoding="utf-8")
    print(f"text saved to {text_out}")

    if args.extract_comments or args.comments_out:
        comments = extract_comments_from_pdf(pdf_path)
        with comments_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["page", "subtype", "author", "subject", "created", "modified", "selected_text", "comment"],
                delimiter="\t",
            )
            w.writeheader()
            w.writerows(comments)
        print(f"comments saved to {comments_out} ({len(comments)} items)")
        if fitz is None:
            print("note: install PyMuPDF to populate selected_text for highlight/underline annotations")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
