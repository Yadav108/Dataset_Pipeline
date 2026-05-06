from loguru import logger


def run_multi_tube_assignment_gate() -> str:
    """Prompt whether multi-capture frame contains same or different tube types."""
    while True:
        print("\nMulti-tube assignment:")
        print("  1: same      — all detected tubes use the preselected class/volume")
        print("  2: different — declare per-slot class/volume")
        print()
        user_input = input("Enter option (1 or 2): ").strip()
        if user_input == "1":
            logger.info("Multi-tube assignment selected: same")
            return "same"
        if user_input == "2":
            logger.info("Multi-tube assignment selected: different")
            return "different"
        print("Invalid input. Enter 1 or 2.")


def _prompt_volume_selection(available_tubes: list[dict]) -> tuple[float, list[dict]]:
    """Prompt operator to declare a tube volume and return matching tube classes.

    Loops until a valid volume is entered. If no registry match exists, offers
    to create a minimal custom entry for that volume.

    Args:
        available_tubes: Full tube registry as a list of tube dicts.

    Returns:
        Tuple of (volume_ml, matched_tubes).
    """
    unique_volumes = sorted(set(t["volume_ml"] for t in available_tubes))

    while True:
        try:
            raw = input(f"  Volume in ml {unique_volumes}: ").strip()
            volume_ml = float(raw)
        except ValueError:
            print("  Invalid input. Enter a number.")
            continue

        matched = [t for t in available_tubes if t["volume_ml"] == volume_ml]

        if not matched:
            print(f"  No tube class for {volume_ml}ml. Available: {unique_volumes}")
            confirm = input(f"  Create custom class for {volume_ml}ml? [y/n]: ").strip().lower()
            if confirm == "y":
                matched = [{"class_id": f"CUSTOM_{volume_ml}ml", "family": "CUSTOM", "volume_ml": volume_ml}]
            else:
                continue

        return (volume_ml, matched)


def _prompt_fill_level() -> dict:
    """Prompt operator for fill level and return a metadata-ready dict."""
    level_map = {
        "f": ("full",  "operator_declared", 0.0),
        "h": ("half",  "operator_declared", 0.5),
        "e": ("empty", "operator_declared", 1.0),
    }
    while True:
        raw = input("  Fill level [F]ull / [H]alf / [E]mpty: ").strip().lower()
        if raw in level_map:
            level, confidence, boundary_ratio = level_map[raw]
            return {"level": level, "confidence": confidence, "boundary_ratio": boundary_ratio}
        print("  Invalid. Enter F, H, or E.")


def _prompt_class_selection(matched_tubes: list[dict]) -> str:
    """Prompt operator to select a tube class from matched candidates.

    If exactly one match, selects it silently. If multiple, presents a numbered
    list and also accepts a freeform custom class name.

    Args:
        matched_tubes: List of tube dicts matching the declared volume.

    Returns:
        Selected class_id string (upper-case).
    """
    if len(matched_tubes) == 1:
        class_id = matched_tubes[0]["class_id"]
        print(f"  Class: {class_id}")
        return class_id

    print(f"  {len(matched_tubes)} classes match:")
    for i, t in enumerate(matched_tubes):
        print(f"    [{i}] {t['class_id']}")

    while True:
        raw = input("  Select class index or enter custom name: ").strip()

        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(matched_tubes):
                class_id = matched_tubes[idx]["class_id"]
                print(f"  Selected: {class_id}")
                return class_id
            print(f"  Invalid index. Enter 0–{len(matched_tubes) - 1}.")
            continue

        if raw and len(raw) >= 3 and all(c.isalnum() or c == "_" for c in raw):
            class_id = raw.upper()
            confirm = input(f"  Confirm custom class '{class_id}'? [y/n]: ").strip().lower()
            if confirm == "y":
                print(f"  Using: {class_id}")
                return class_id
            continue

        print("  Invalid. Use alphanumeric + underscore, min 3 chars.")


def run_multi_slot_gate(
    available_tubes: dict,
    cfg,
) -> list[dict]:
    """Prompt operator to declare tube count and per-slot class/volume.

    Step 1: asks for the number of tube slots in the frame (1–16).
    Step 2: prompt slot direction in view (left→right or right→left).
    Step 3: for each slot, calls _prompt_volume_selection() then
            _prompt_class_selection() to collect (volume_ml, class_id).

    Args:
        available_tubes: Full tube registry (list of tube dicts).
        cfg: App config (passed through; not used for file I/O here).

    Returns:
        List of slot dicts ordered to match detector x-sorted order
        (left-to-right):
        [{"slot": 0, "volume_ml": 4.0, "class_id": "VAC_LIGHT_BLUE",
          "fill_level": {"level": "empty", "confidence": "operator_declared",
                         "boundary_ratio": 1.0}}, ...]

    Raises:
        ValueError: If operator enters an invalid slot count on both attempts.
    """
    # Step 1 — slot count: two chances, then raise
    n_slots: int | None = None
    for attempt in range(2):
        raw = input("Number of tube slots in frame: ").strip()
        if raw.isdigit():
            val = int(raw)
            if 1 <= val <= 16:
                n_slots = val
                break
        if attempt == 0:
            print("  Invalid. Enter an integer between 1 and 16.")

    if n_slots is None:
        raise ValueError("Invalid slot count")

    logger.info(f"Multi-slot declared: {n_slots} slots")

    # Step 2 — slot direction mapping
    slot_direction = "ltr"
    while True:
        raw = input(
            "Slot order in preview? [1] left-to-right (default), [2] right-to-left: "
        ).strip()
        if raw in {"", "1"}:
            slot_direction = "ltr"
            break
        if raw == "2":
            slot_direction = "rtl"
            break
        print("  Invalid. Enter 1 or 2.")

    # Step 3 — per-slot volume, class, and fill level
    slots: list[dict] = []
    for i in range(n_slots):
        print(f"\nSlot {i} —")
        volume_ml, matched = _prompt_volume_selection(available_tubes)
        class_id = _prompt_class_selection(matched)
        fill_level = _prompt_fill_level()
        slots.append({
            "slot": i,
            "volume_ml": volume_ml,
            "class_id": class_id,
            "fill_level": fill_level,
        })
        logger.info(
            f"  Slot {i}: {class_id} / {volume_ml}ml / fill={fill_level['level']}"
        )

    if slot_direction == "rtl":
        slots = list(reversed(slots))
        for i, slot in enumerate(slots):
            slot["slot"] = i
        logger.info("Slot mapping: right-to-left (reversed to match detector order)")
    else:
        logger.info("Slot mapping: left-to-right")

    return slots


def run_capture_mode_gate() -> str:
    """Prompt user to select capture mode.
    
    Displays mode options and validates user input. Loops until valid
    selection (1, 2, 3, or 4) is provided.
    
    Returns:
        str: One of "single_side", "single_top", or "multi_top"
    """
    mode_map = {
        "1": "single_side",
        "2": "single_top",
        "3": "multi_top",
        "4": "multi_side",
    }

    while True:
        print("\nCapture mode:")
        print("  1: single_side  — side view, single tube per frame")
        print("  2: single_top   — top-down, tube in rack grid, single tube per frame")
        print("  3: multi_top    — top-down, multiple tubes per frame (rack grid)")
        print("  4: multi_side   — side view, multiple tubes per frame")
        print()

        user_input = input("Enter mode number (1, 2, 3, or 4): ").strip()
        
        if user_input in mode_map:
            mode = mode_map[user_input]
            logger.info(f"Capture mode selected: {mode}")
            return mode
        else:
            print("Invalid input. Enter 1, 2, 3, or 4.")
