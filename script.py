import os
import csv
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuración ──────────────────────────────────────────────────────────────
DATASET_DIR = Path(r"C:\Users\ALEJANDRA\OneDrive - UVG\Escritorio\Quinto Año\Semestre 1\Vision\LAB8-VPC\dataset")

CLASSES = ["Bottle", "Food", "Drink"]

LIMITS = {
    "train":      700,
    "validation": 125,
    "test":       125,
}

# IDs de clase en Open Images V7
CLASS_IDS = {
    "Bottle": "/m/04dr76w",
    "Food":   "/m/02wbm",
    "Drink":  "/m/0271t",
}

# URLs de los CSVs oficiales de Open Images V7
CSV_URLS = {
    "train": {
        "bbox":   "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
        "images": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    },
    "validation": {
        "bbox":   "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "images": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    },
    "test": {
        "bbox":   "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
        "images": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return
    print(f"  [download] {dest.name} ...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"  [ok] {dest.name}")


def download_image(row, out_dir: Path):
    """Descarga una imagen dada una fila del CSV de imágenes."""
    image_id  = row.get("ImageID") or row.get("image_id") or list(row.values())[0]
    url       = row.get("OriginalURL") or row.get("original_url") or ""
    if not url:
        return False
    ext  = Path(url).suffix or ".jpg"
    dest = out_dir / f"{image_id}{ext}"
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


# ── Core ───────────────────────────────────────────────────────────────────────
def load_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_image_ids_for_classes(bbox_rows, class_ids, limit):
    """Devuelve hasta `limit` image-ids que contengan alguna de las clases."""
    ids = set()
    for row in bbox_rows:
        if row.get("LabelName") in class_ids:
            ids.add(row["ImageID"])
        if len(ids) >= limit:
            break
    return ids


def get_bbox_for_ids(bbox_rows, image_ids, class_ids):
    """Filtra las anotaciones que corresponden a los ids seleccionados."""
    return [
        r for r in bbox_rows
        if r["ImageID"] in image_ids and r.get("LabelName") in class_ids
    ]


def write_annotations_csv(rows, out_path: Path, class_name_map: dict):
    """Guarda anotaciones en formato CSV con nombre de clase legible."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ImageID", "ClassName", "XMin", "XMax", "YMin", "YMax"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "ImageID":   r["ImageID"],
                "ClassName": class_name_map.get(r["LabelName"], r["LabelName"]),
                "XMin":      r["XMin"],
                "XMax":      r["XMax"],
                "YMin":      r["YMin"],
                "YMax":      r["YMax"],
            })
    print(f"  [annotations] {out_path.name} ({len(rows)} bboxes)")


def process_split(split: str, limit: int, meta_dir: Path):
    print(f"\n{'='*60}")
    print(f"  Split: {split.upper()}  (límite: {limit} imágenes)")
    print(f"{'='*60}")

    # Directorios de salida
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Descargar CSVs de metadata
    urls = CSV_URLS[split]
    bbox_path   = meta_dir / f"{split}_bbox.csv"
    images_path = meta_dir / f"{split}_images.csv"
    download_file(urls["bbox"],   bbox_path)
    download_file(urls["images"], images_path)

    # Leer CSVs
    print("  Leyendo anotaciones...")
    bbox_rows   = load_csv(bbox_path)
    images_rows = load_csv(images_path)

    class_ids     = set(CLASS_IDS.values())
    class_name_map = {v: k for k, v in CLASS_IDS.items()}

    # Seleccionar image IDs
    selected_ids = get_image_ids_for_classes(bbox_rows, class_ids, limit)
    print(f"  Image IDs encontrados: {len(selected_ids)}")

    # Filtrar anotaciones
    selected_bbox = get_bbox_for_ids(bbox_rows, selected_ids, class_ids)

    # Guardar CSV de anotaciones
    write_annotations_csv(selected_bbox, labels_dir / "annotations.csv", class_name_map)

    # Descargar imágenes
    id_to_row = {
        (r.get("ImageID") or r.get("image_id") or list(r.values())[0]): r
        for r in images_rows
    }
    to_download = [id_to_row[i] for i in selected_ids if i in id_to_row]

    print(f"  Descargando {len(to_download)} imágenes...")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(download_image, row, images_dir): row for row in to_download}
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                ok += 1
            else:
                fail += 1
            if i % 50 == 0:
                print(f"    {i}/{len(to_download)} — ok={ok} fail={fail}")

    print(f"  [done] {split}: {ok} descargadas, {fail} fallidas")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    meta_dir = DATASET_DIR / "_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for split, limit in LIMITS.items():
        process_split(split, limit, meta_dir)

    print("\n✓ Descarga completa.")
    print(f"  Dataset en: {DATASET_DIR}")


if __name__ == "__main__":
    main()