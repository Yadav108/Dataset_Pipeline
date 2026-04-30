"""Pre-capture data manager with SQLite persistence."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_VOLUMES = ["1ml", "1.3ml", "2ml", "3ml", "3.5ml", "4ml", "4.5ml", "10ml"]
DEFAULT_CLASSES = [
    ("4ml", "VACUETTE", "VAC_PURPLE"),
    ("4ml", "VACUETTE", "VAC_RED"),
    ("4ml", "VACUETTE", "VAC_GREEN"),
    ("4ml", "VACUETTE", "VAC_LIGHTBLUE"),
    ("4ml", "VACUETTE", "VAC_GRAY"),
    ("4ml", "VACUETTE", "SAR_BLUE"),
    ("3.5ml", "VACUETTE", "VAC_GOLD"),
    ("4.5ml", "SARSTEDT Regular", "SAR_GREEN"),
    ("4.5ml", "SARSTEDT Regular", "SAR_RED"),
    ("3ml", "SARSTEDT Regular", "SAR_LAVENDER"),
    ("10ml", "SARSTEDT Regular", "SAR_YELLOW"),
    ("1.3ml", "SARSTEDT Small", "SAR_SM_ORANGE"),
    ("1.3ml", "SARSTEDT Small", "SAR_SM_GRAY"),
    ("1.3ml", "SARSTEDT Small", "SAR_SM_GREEN"),
    ("1.3ml", "SARSTEDT Small", "SAR_SM_RED"),
    ("1.3ml", "SARSTEDT Small", "SAR_SM_TRANSPARENT"),
    ("1ml", "SARSTEDT Small", "SAR_SM_TRANSPARENT"),
]

FAMILIES = ("VACUETTE", "SARSTEDT Regular", "SARSTEDT Small", "Custom")
CLASS_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
VOLUME_PATTERN = re.compile(r"^\d+(\.\d+)?ml$")


def _now() -> str:
    return datetime.now().isoformat()


def _parse_volume_value(volume: str) -> float:
    return float(volume.lower().replace("ml", ""))


class PreCaptureStore:
    """SQLite store for volumes, classes, and pre-capture records."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._seed_defaults()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tube_classes (
                volume TEXT NOT NULL,
                class_name TEXT NOT NULL,
                family TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (volume, class_name)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS capture_records (
                volume TEXT NOT NULL,
                class_name TEXT NOT NULL,
                family TEXT NOT NULL,
                quantity_count INTEGER NOT NULL,
                batch_number TEXT,
                expiry_date TEXT,
                storage_location TEXT,
                notes TEXT,
                session_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (volume, class_name)
            )
            """
        )
        self.conn.commit()

    def _seed_defaults(self) -> None:
        stamp = _now()
        for volume, family, class_name in DEFAULT_CLASSES:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO tube_classes (volume, class_name, family, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (volume, class_name, family, stamp, stamp),
            )
        self.conn.commit()

    def list_volumes(self) -> List[str]:
        rows = self.conn.execute("SELECT DISTINCT volume FROM tube_classes").fetchall()
        merged = set(DEFAULT_VOLUMES)
        merged.update(row["volume"] for row in rows)
        return sorted(merged, key=_parse_volume_value)

    def list_classes_for_volume(self, volume: str) -> List[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT volume, class_name, family, created_at, updated_at
            FROM tube_classes
            WHERE volume = ?
            ORDER BY class_name
            """,
            (volume,),
        ).fetchall()
        return list(rows)

    def add_class(self, volume: str, class_name: str, family: str) -> bool:
        stamp = _now()
        result = self.conn.execute(
            """
            INSERT OR IGNORE INTO tube_classes (volume, class_name, family, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (volume, class_name, family, stamp, stamp),
        )
        self.conn.commit()
        return result.rowcount > 0

    def upsert_record(
        self,
        volume: str,
        class_name: str,
        quantity_count: int,
        batch_number: str,
        expiry_date: str,
        storage_location: str,
        notes: str,
        session_id: str,
    ) -> str:
        existing = self.conn.execute(
            """
            SELECT created_at FROM capture_records WHERE volume = ? AND class_name = ?
            """,
            (volume, class_name),
        ).fetchone()
        family_row = self.conn.execute(
            "SELECT family FROM tube_classes WHERE volume = ? AND class_name = ?",
            (volume, class_name),
        ).fetchone()
        if family_row is None:
            raise ValueError(f"Class {class_name} is not mapped to volume {volume}.")

        stamp = _now()
        created_at = existing["created_at"] if existing else stamp
        self.conn.execute(
            """
            INSERT INTO capture_records (
                volume, class_name, family, quantity_count, batch_number, expiry_date,
                storage_location, notes, session_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(volume, class_name) DO UPDATE SET
                family=excluded.family,
                quantity_count=excluded.quantity_count,
                batch_number=excluded.batch_number,
                expiry_date=excluded.expiry_date,
                storage_location=excluded.storage_location,
                notes=excluded.notes,
                session_id=excluded.session_id,
                updated_at=excluded.updated_at
            """,
            (
                volume,
                class_name,
                family_row["family"],
                quantity_count,
                batch_number,
                expiry_date,
                storage_location,
                notes,
                session_id,
                created_at,
                stamp,
            ),
        )
        self.conn.commit()
        return "updated" if existing else "created"

    def delete_record(self, volume: str, class_name: str) -> bool:
        result = self.conn.execute(
            "DELETE FROM capture_records WHERE volume = ? AND class_name = ?",
            (volume, class_name),
        )
        self.conn.commit()
        return result.rowcount > 0

    def list_records(
        self, volume: str = "", class_name: str = "", family: str = ""
    ) -> List[sqlite3.Row]:
        query = """
            SELECT volume, class_name, family, quantity_count, batch_number, expiry_date,
                   storage_location, notes, session_id, created_at, updated_at
            FROM capture_records
            WHERE (? = '' OR volume = ?)
              AND (? = '' OR class_name LIKE ?)
              AND (? = '' OR family = ?)
            ORDER BY volume, class_name
        """
        class_filter = f"%{class_name.upper()}%" if class_name else ""
        rows = self.conn.execute(
            query,
            (volume, volume, class_name, class_filter, family, family),
        ).fetchall()
        return list(rows)

    def summary_rows(
        self, volume: str = "", class_name: str = "", family: str = ""
    ) -> List[sqlite3.Row]:
        query = """
            SELECT c.volume, c.class_name, c.family,
                   CASE WHEN r.class_name IS NULL THEN 0 ELSE 1 END AS record_count,
                   COALESCE(r.updated_at, c.updated_at) AS last_updated
            FROM tube_classes c
            LEFT JOIN capture_records r
              ON r.volume = c.volume AND r.class_name = c.class_name
            WHERE (? = '' OR c.volume = ?)
              AND (? = '' OR c.class_name LIKE ?)
              AND (? = '' OR c.family = ?)
            ORDER BY c.volume, c.class_name
        """
        class_filter = f"%{class_name.upper()}%" if class_name else ""
        rows = self.conn.execute(
            query,
            (volume, volume, class_name, class_filter, family, family),
        ).fetchall()
        return list(rows)

    def export_json(self, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "exported_at": _now(),
            "classes": [dict(row) for row in self.conn.execute("SELECT * FROM tube_classes ORDER BY volume, class_name")],
            "records": [dict(row) for row in self.conn.execute("SELECT * FROM capture_records ORDER BY volume, class_name")],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    def export_csv(self, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.conn.execute(
            """
            SELECT volume, class_name, family, quantity_count, batch_number, expiry_date,
                   storage_location, notes, session_id, created_at, updated_at
            FROM capture_records
            ORDER BY volume, class_name
            """
        ).fetchall()
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "volume",
                    "class_name",
                    "family",
                    "quantity_count",
                    "batch_number",
                    "expiry_date",
                    "storage_location",
                    "notes",
                    "session_id",
                    "created_at",
                    "updated_at",
                ]
            )
            for row in rows:
                writer.writerow([row[k] for k in row.keys()])
        return out_path


def _prompt_choice(prompt: str, choices: Sequence[str]) -> Optional[str]:
    if not choices:
        return None
    for idx, item in enumerate(choices, start=1):
        print(f"  [{idx}] {item}")
    while True:
        raw = input(f"{prompt} (or Q to cancel): ").strip()
        if raw.lower() == "q":
            return None
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(choices):
                return choices[index - 1]
        print("  ✗ Invalid selection.")


def _prompt_capture_mode() -> str:
    mode_map = {
        "1": "single_side",
        "2": "single_top",
        "3": "multi_top",
    }
    while True:
        print("\nCapture mode:")
        print("  1: single_side  — side view, hand-held (R1.4 angle variation)")
        print("  2: single_top   — top-down, tube in rack (R1.4 angle variation)")
        print("  3: multi_top    — top-down, multiple tubes per frame (rack grid)")
        print()
        user_input = input("Enter mode number (1, 2, or 3): ").strip()
        if user_input in mode_map:
            return mode_map[user_input]
        print("Invalid input. Enter 1, 2, or 3.")


def _capture_record_flow(
    store: PreCaptureStore, session_id: str
) -> Optional[Tuple[str, str, str]]:
    print("\n[PRE-CAPTURE] Volume Selection")
    volume = _prompt_choice("Select volume", store.list_volumes())
    if volume is None:
        return None

    classes = store.list_classes_for_volume(volume)
    class_labels = [f"{row['class_name']} ({row['family']})" for row in classes]
    if not class_labels:
        print("  ⚠ No classes for this volume. Add a new class first.")
        return None

    picked_label = _prompt_choice("Select class", class_labels)
    if picked_label is None:
        return None
    class_name = picked_label.split(" (", 1)[0]

    batch = input("Batch number (optional): ").strip()
    expiry = input("Expiry date YYYY-MM-DD (optional): ").strip()
    location = input("Storage location (optional): ").strip()
    notes = input("Notes/comments (optional): ").strip()

    if expiry and not re.match(r"^\d{4}-\d{2}-\d{2}$", expiry):
        print("  ✗ Expiry format must be YYYY-MM-DD.")
        return None

    result = store.upsert_record(
        volume=volume,
        class_name=class_name,
        quantity_count=0,
        batch_number=batch,
        expiry_date=expiry,
        storage_location=location,
        notes=notes,
        session_id=session_id,
    )
    print(f"  ✓ Record {result} for {volume} / {class_name}")
    capture_mode = _prompt_capture_mode()
    return (volume, class_name, capture_mode)


def _add_new_class_flow(store: PreCaptureStore) -> None:
    print("\n[PRE-CAPTURE] Add New Class")
    class_name = input("New class name (e.g., VAC_ORANGE): ").strip().upper()
    if not CLASS_PATTERN.match(class_name):
        print("  ✗ Class name must be uppercase and underscores only.")
        return

    mode = input("Use [E]xisting volume or [N]ew volume? ").strip().upper()
    if mode not in {"E", "N"}:
        print("  ✗ Enter E or N.")
        return

    if mode == "E":
        volume = _prompt_choice("Select volume", store.list_volumes())
        if volume is None:
            return
    else:
        volume = input("New volume (e.g., 5ml): ").strip().lower()
        if not VOLUME_PATTERN.match(volume):
            print("  ✗ Volume must look like 1ml, 1.3ml, 10ml.")
            return

    family = _prompt_choice("Select family", FAMILIES)
    if family is None:
        return

    created = store.add_class(volume=volume, class_name=class_name, family=family)
    if not created:
        print("  ✗ Class already exists for this volume.")
        return
    print(f"  ✓ Added {class_name} for {volume} ({family})")


def _manage_records_flow(store: PreCaptureStore) -> None:
    print("\n[PRE-CAPTURE] Summary / Manage")
    volume_filter = input("Filter volume (blank = all): ").strip()
    class_filter = input("Filter class (blank = all): ").strip().upper()
    family_filter = input("Filter family [VACUETTE/SARSTEDT Regular/SARSTEDT Small/Custom] (blank = all): ").strip()

    rows = store.summary_rows(volume_filter, class_filter, family_filter)
    if not rows:
        print("  No classes found with the selected filters.")
    else:
        print("\n  Volume | Class | Family | Records | Last Updated")
        for row in rows:
            print(
                f"  {row['volume']} | {row['class_name']} | {row['family']} | "
                f"{row['record_count']} | {row['last_updated']}"
            )

    records = store.list_records(volume_filter, class_filter, family_filter)
    if records:
        print("\n  Existing Records:")
        for idx, row in enumerate(records, start=1):
            print(
                f"  [{idx}] {row['volume']} / {row['class_name']} | "
                f"batch={row['batch_number'] or '-'} | expiry={row['expiry_date'] or '-'}"
            )

        action = input("Action: [E]dit, [D]elete, [B]ack: ").strip().upper()
        if action not in {"E", "D"}:
            return

        pick_raw = input("Pick record number: ").strip()
        if not pick_raw.isdigit():
            print("  ✗ Invalid record index.")
            return
        pick = int(pick_raw)
        if pick < 1 or pick > len(records):
            print("  ✗ Record index out of range.")
            return
        row = records[pick - 1]

        if action == "D":
            if store.delete_record(row["volume"], row["class_name"]):
                print("  ✓ Record deleted.")
            else:
                print("  ✗ Record not found.")
            return

        batch = input(f"Batch number [{row['batch_number'] or ''}]: ").strip() or (row["batch_number"] or "")
        expiry = input(f"Expiry date [{row['expiry_date'] or ''}]: ").strip() or (row["expiry_date"] or "")
        location = input(f"Storage location [{row['storage_location'] or ''}]: ").strip() or (row["storage_location"] or "")
        notes = input(f"Notes [{row['notes'] or ''}]: ").strip() or (row["notes"] or "")
        if expiry and not re.match(r"^\d{4}-\d{2}-\d{2}$", expiry):
            print("  ✗ Expiry format must be YYYY-MM-DD.")
            return
        store.upsert_record(
            volume=row["volume"],
            class_name=row["class_name"],
            quantity_count=int(row["quantity_count"] or 0),
            batch_number=batch,
            expiry_date=expiry,
            storage_location=location,
            notes=notes,
            session_id=row["session_id"] or "",
        )
        print("  ✓ Record updated.")
    else:
        print("\n  No captured records yet.")


def _export_flow(store: PreCaptureStore, exports_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = store.export_json(exports_dir / f"pre_capture_{timestamp}.json")
    csv_path = store.export_csv(exports_dir / f"pre_capture_{timestamp}.csv")
    print(f"  ✓ Exported JSON: {json_path}")
    print(f"  ✓ Exported CSV:  {csv_path}")


def run_pre_capture_workflow(session_id: str, base_dir: Path) -> Optional[Dict[str, object]]:
    """Run interactive pre-capture workflow and persist data in SQLite."""
    db_path = base_dir / "pre_capture" / "pre_capture.db"
    exports_dir = base_dir / "pre_capture" / "exports"
    store = PreCaptureStore(db_path)
    try:
        print("\n" + "=" * 60)
        print("  PRE-CAPTURE DATA ENTRY")
        print("=" * 60)
        print(f"Database: {db_path}")

        while True:
            print("\n[PRE-CAPTURE MENU]")
            print("  [1] Capture data for existing class")
            print("  [2] Add new class")
            print("  [3] Summary / search / edit / delete")
            print("  [4] Export CSV + JSON")
            print("  [5] Continue to camera capture")
            print("  [Q] Skip and continue")
            choice = input("Choose action: ").strip().upper()

            if choice == "1":
                capture_context = _capture_record_flow(store, session_id=session_id)
                if capture_context is not None:
                    volume, class_name, capture_mode = capture_context
                    print("  ✓ Pre-capture complete.")
                    return {
                        "volume_ml": _parse_volume_value(volume),
                        "class_id": class_name,
                        "capture_mode": capture_mode,
                    }
            elif choice == "2":
                _add_new_class_flow(store)
            elif choice == "3":
                _manage_records_flow(store)
            elif choice == "4":
                _export_flow(store, exports_dir)
            elif choice == "5" or choice == "Q":
                print("  ✓ Pre-capture complete.")
                return None
            else:
                print("  ✗ Invalid choice.")
    finally:
        store.close()
