from loguru import logger


def run_capture_mode_gate() -> str:
    """Prompt user to select capture mode.
    
    Displays mode options and validates user input. Loops until valid
    selection (1, 2, or 3) is provided.
    
    Returns:
        str: One of "single_side", "single_top", or "multi_top"
    """
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
            mode = mode_map[user_input]
            logger.info(f"Capture mode selected: {mode}")
            return mode
        else:
            print("Invalid input. Enter 1, 2, or 3.")
