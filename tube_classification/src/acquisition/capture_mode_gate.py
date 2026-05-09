from loguru import logger


def run_capture_mode_gate() -> str:
    """Prompt user to select capture mode.
    
    Displays mode options and validates user input. Loops until valid
    selection (1, 2, 3, or 4) is provided.
    
    Returns:
        str: One of "single_side", "single_top", "multi_side", or "multi_top"
    """
    mode_map = {
        "1": "single_side",
        "2": "single_top",
        "3": "multi_side",
        "4": "multi_top",
    }
    
    while True:
        print("\nCapture mode:")
        print("  1: single_side  — side view, hand-held (R1.4 angle variation)")
        print("  2: single_top   — top-down, tube in rack (R1.4 angle variation)")
        print("  3: multi_side   — side view, multiple tubes per frame")
        print("  4: multi_top    — top-down, multiple tubes per frame (rack grid)")
        print()
        
        user_input = input("Enter mode number (1, 2, 3, or 4): ").strip()
        
        if user_input in mode_map:
            mode = mode_map[user_input]
            logger.info(f"Capture mode selected: {mode}")
            return mode
        else:
            print("Invalid input. Enter 1, 2, 3, or 4.")


def run_multi_side_slot_gate() -> list[dict]:
    """Declare tube slots for multi_side capture, left-to-right.

    Asks whether tubes are the same class or different, then collects
    per-slot info accordingly.

    Returns:
        List of dicts, one per slot, left-to-right order.
        Each dict has keys: class_id, volume_ml, fill_level_result.
    """
    from src.acquisition.fill_level_detector import FillLevelResult, FillLevel

    def _load_class_options() -> list[dict]:
        try:
            from pathlib import Path
            import yaml

            registry_path = Path("config/registry.yaml")
            if not registry_path.exists():
                return []

            with registry_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            options = []
            for tube in data.get("tubes", []):
                class_id = str(tube.get("class_id", "")).strip().upper()
                if not class_id:
                    continue
                options.append({
                    "class_id": class_id,
                    "volume_ml": float(tube.get("volume_ml", 0.0)),
                    "family": str(tube.get("family", "")),
                })
            return sorted(options, key=lambda item: item["class_id"])
        except Exception as e:
            logger.warning(f"Could not load tube class list: {e}")
            return []

    class_options = _load_class_options()

    def _ask_fill(slot_label: str) -> FillLevelResult:
        while True:
            prompt = f"  Fill level for {slot_label} [F]ull / [H]alf / [E]mpty: "
            try:
                import msvcrt

                print(prompt, end="", flush=True)
                key = msvcrt.getwch().strip().lower()
                print(key.upper())
            except Exception:
                key = input(prompt).strip().lower()
            if key == "f":
                return FillLevelResult(
                    level=FillLevel.FULL,
                    confidence="operator_declared",
                    boundary_ratio=0.0,
                )
            if key == "h":
                return FillLevelResult(
                    level=FillLevel.HALF,
                    confidence="operator_declared",
                    boundary_ratio=0.5,
                )
            if key == "e":
                return FillLevelResult(
                    level=FillLevel.EMPTY,
                    confidence="operator_declared",
                    boundary_ratio=1.0,
                )
            print("  Enter F, H, or E.")

    def _ask_class_volume() -> tuple[str, float]:
        class_id: str | None = None
        if class_options:
            volumes = sorted({option["volume_ml"] for option in class_options})
            print("\n  Volumes:")
            for idx, volume in enumerate(volumes, start=1):
                print(f"  [{idx}] {volume:g}ml")
            print("  [C] Custom volume/class")

            while True:
                raw = input("  Select volume number, volume ml, or C: ").strip().upper()
                if raw == "C":
                    volume_ml = None
                    break
                if raw.isdigit():
                    index = int(raw)
                    if 1 <= index <= len(volumes):
                        volume_ml = volumes[index - 1]
                        break
                    print(f"  Invalid. Enter 1-{len(volumes)}, volume ml, or C.")
                    continue

                try:
                    candidate_volume = float(raw.lower().replace("ml", ""))
                except ValueError:
                    print("  Invalid. Use a listed number, volume ml, or C.")
                    continue

                matched_volume = next(
                    (
                        volume for volume in volumes
                        if abs(float(volume) - candidate_volume) < 1e-6
                    ),
                    None,
                )
                if matched_volume is not None:
                    volume_ml = matched_volume
                    break

                confirm = input(f"  Use custom volume {candidate_volume:g}ml? [y/N]: ").strip().lower()
                if confirm == "y":
                    volume_ml = candidate_volume
                    break

            if volume_ml is not None:
                volume_options = [
                    option for option in class_options
                    if abs(float(option["volume_ml"]) - float(volume_ml)) < 1e-6
                ]
                if volume_options:
                    print(f"\n  Tube classes for {volume_ml:g}ml:")
                    for idx, option in enumerate(volume_options, start=1):
                        print(f"  [{idx}] {option['class_id']}  {option['family']}")
                    print("  [C] Custom class")

                    while True:
                        raw = input("  Select class number, class name, or C: ").strip().upper()
                        if raw == "C":
                            break
                        if raw.isdigit():
                            index = int(raw)
                            if 1 <= index <= len(volume_options):
                                picked = volume_options[index - 1]
                                return picked["class_id"], picked["volume_ml"]
                            print(f"  Invalid. Enter 1-{len(volume_options)}, class name, or C.")
                            continue

                        picked = next(
                            (
                                option for option in volume_options
                                if option["class_id"] == raw
                            ),
                            None,
                        )
                        if picked is not None:
                            return picked["class_id"], picked["volume_ml"]

                        if raw and all(c.isalnum() or c == "_" for c in raw):
                            confirm = input(f"  Use custom class {raw} at {volume_ml:g}ml? [y/N]: ").strip().lower()
                            if confirm == "y":
                                return raw, float(volume_ml)
                        else:
                            print("  Invalid. Use a listed number, class name, or C.")
                else:
                    print(f"\n  No registered tube classes for {volume_ml:g}ml.")

            if volume_ml is not None:
                while True:
                    class_id = input(f"  Class name for {volume_ml:g}ml (e.g. VAC_GREEN): ").strip().upper()
                    if class_id and all(c.isalnum() or c == "_" for c in class_id):
                        return class_id, float(volume_ml)
                    print("  Invalid. Use alphanumeric and underscore only.")

            # Custom volume path: ask volume first, then class manually.
            while True:
                raw_vol = input("  Volume ml: ").strip()
                try:
                    volume_ml = float(raw_vol)
                    if volume_ml > 0:
                        break
                except ValueError:
                    pass
                print("  Enter a positive number.")
        else:
            print("\n  Tube class list unavailable; enter class manually.")
            while True:
                raw_vol = input("  Volume ml: ").strip()
                try:
                    volume_ml = float(raw_vol)
                    if volume_ml > 0:
                        break
                except ValueError:
                    pass
                print("  Enter a positive number.")

        if class_id is None:
            while True:
                class_id = input("  Class name (e.g. VAC_GREEN): ").strip().upper()
                if class_id and all(c.isalnum() or c == "_" for c in class_id):
                    break
                print("  Invalid. Use alphanumeric and underscore only.")
        return class_id, volume_ml

    print("\n" + "=" * 60)
    print("MULTI-SIDE SLOT DECLARATION (left -> right)")
    print("=" * 60)

    while True:
        raw = input("How many tubes in frame? (2-6): ").strip()
        if raw.isdigit() and 2 <= int(raw) <= 6:
            n_slots = int(raw)
            break
        print("Enter a number between 2 and 6.")

    print("\nAre the tubes:")
    print("  1: Same class (same type, ask fill level per slot)")
    print("  2: Different classes (ask class + volume + fill level per slot)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            break
        print("Enter 1 or 2.")

    slots = []

    if choice == "1":
        print("\n--- Tube class (applies to all slots) ---")
        class_id, volume_ml = _ask_class_volume()

        for i in range(n_slots):
            pos = "leftmost" if i == 0 else "rightmost" if i == n_slots - 1 else f"{i + 1} from left"
            fill = _ask_fill(f"slot {i} ({pos})")
            slots.append({
                "class_id": class_id,
                "volume_ml": volume_ml,
                "fill_level_result": fill,
            })
            logger.info(
                f"Slot {i}: class={class_id}, volume={volume_ml}ml, "
                f"fill={fill.level.value}"
            )
    else:
        for i in range(n_slots):
            pos = "leftmost" if i == 0 else "rightmost" if i == n_slots - 1 else f"{i + 1} from left"
            print(f"\n--- Slot {i} ({pos}) ---")
            class_id, volume_ml = _ask_class_volume()
            fill = _ask_fill(f"slot {i}")
            slots.append({
                "class_id": class_id,
                "volume_ml": volume_ml,
                "fill_level_result": fill,
            })
            logger.info(
                f"Slot {i}: class={class_id}, volume={volume_ml}ml, "
                f"fill={fill.level.value}"
            )

    print("\n--- Slot summary ---")
    for i, slot in enumerate(slots):
        print(
            f"  [{i}] {slot['class_id']}  {slot['volume_ml']}ml  "
            f"{slot['fill_level_result'].level.value}"
        )
    print()

    return slots
