"""
Ugandan National ID Card Capture & OCR Application
====================================================
Captures live webcam video, detects and processes a Ugandan National ID card
(front and back), extracts key text fields, and stores the results in a CSV.

Requirements (see requirements.txt):
    pip install opencv-python pytesseract pandas numpy

Tesseract binary must also be installed:
    Ubuntu/Debian : sudo apt-get install tesseract-ocr
    macOS         : brew install tesseract
    Windows       : https://github.com/UB-Mannheim/tesseract/wiki

Usage:
    python id_capture.py

Key bindings inside the preview window:
    SPACE  – capture the currently displayed side of the ID card
    C      – confirm the extracted data and move to the next step
    R      – retry the current side (discard current extraction result)
    Q      – quit the application
"""

import csv
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_FILE = "ugandan_ids.csv"
CSV_COLUMNS = ["timestamp", "surname", "given_name", "nin", "date_of_birth", "village"]

# NIN regex: starts with CM (male) or CF (female), followed by exactly 12
# alphanumeric characters (total 14 characters).
NIN_PATTERN = re.compile(r"\b(C[MF][A-Z0-9]{12})\b", re.IGNORECASE)

# Date-of-birth: accept common formats dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy
DOB_PATTERN = re.compile(
    r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b"
)

# Tesseract OCR config strings
# PSM 6 = assume a single uniform block of text
TESS_DEFAULT_CONFIG = "--oem 3 --psm 6"
# For the NIN field we restrict to uppercase letters + digits
TESS_NIN_CONFIG = (
    "--oem 3 --psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# Overlay text settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
FONT_THICKNESS = 2
STATUS_COLOR = (0, 255, 0)    # green
WARNING_COLOR = (0, 165, 255) # orange
ERROR_COLOR = (0, 0, 255)     # red
INFO_COLOR = (255, 255, 255)  # white


# ---------------------------------------------------------------------------
# Helper: draw multi-line text onto a frame
# ---------------------------------------------------------------------------

def put_multiline(frame: np.ndarray, lines: list[str], origin: tuple[int, int],
                  color: tuple[int, int, int] = INFO_COLOR,
                  scale: float = FONT_SCALE,
                  thickness: int = FONT_THICKNESS,
                  line_height: int = 28) -> None:
    """Draw multiple lines of text on *frame* starting at *origin*."""
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x, y), FONT, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)   # black shadow
        cv2.putText(frame, line, (x, y), FONT, scale, color,
                    thickness, cv2.LINE_AA)
        y += line_height


# ---------------------------------------------------------------------------
# CSV storage
# ---------------------------------------------------------------------------

def ensure_csv(path: str = CSV_FILE) -> None:
    """Create *path* with the header row if it does not already exist."""
    if not Path(path).exists():
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def save_to_csv(record: dict, path: str = CSV_FILE) -> None:
    """Append a single *record* dict to the CSV at *path*.

    Args:
        record: A dict with keys matching CSV_COLUMNS.
        path:   Destination CSV file path.
    """
    ensure_csv(path)
    row = {col: record.get(col, "") for col in CSV_COLUMNS}
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return *pts* (shape 4×2) ordered: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]   # bottom-right (largest x+y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right  (smallest y-x)
    rect[3] = pts[np.argmax(diff)]  # bottom-left (largest y-x)
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a perspective warp so the four corners of *pts* become a rectangle."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0],
         [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def find_card_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    """Attempt to detect a rectangular card contour in *gray*.

    Returns a 4-point contour (shape 4×2) or None if not found.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            # Accept only contours that occupy at least 10 % of the frame area
            frame_area = gray.shape[0] * gray.shape[1]
            if cv2.contourArea(approx) > 0.10 * frame_area:
                return approx.reshape(4, 2).astype("float32")
    return None


def preprocess_image(frame: np.ndarray) -> np.ndarray:
    """Pre-process *frame* for OCR.

    Steps:
        1. Convert to grayscale.
        2. Detect card contour and apply perspective correction (fallback to
           full frame if contour not found).
        3. Upscale to at least 1 200 px wide (Tesseract accuracy improves on
           higher-resolution images).
        4. Apply Gaussian blur to reduce noise.
        5. Apply adaptive thresholding.

    Returns:
        A grayscale, thresholded image ready for OCR.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- perspective correction ---
    pts = find_card_contour(gray)
    if pts is not None:
        warped = four_point_transform(gray, pts)
    else:
        warped = gray.copy()

    # --- upscale if needed ---
    min_width = 1200
    h, w = warped.shape[:2]
    if w < min_width:
        scale = min_width / w
        warped = cv2.resize(warped, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_CUBIC)

    # --- denoise ---
    denoised = cv2.GaussianBlur(warped, (3, 3), 0)

    # --- adaptive threshold ---
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    return thresh


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_image(processed: np.ndarray, config: str = TESS_DEFAULT_CONFIG) -> str:
    """Run Tesseract on *processed* and return the raw text string."""
    try:
        text = pytesseract.image_to_string(processed, lang="eng", config=config)
        return text
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract binary not found. "
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki "
            "or via your system package manager."
        )
    except Exception as exc:  # noqa: BLE001
        # Re-raise with a user-friendly message; the original exception is
        # chained so the full technical details remain available in tracebacks.
        raise RuntimeError(
            "OCR processing failed. Ensure the image is clear and well-lit."
        ) from exc


# ---------------------------------------------------------------------------
# Data parsing
# ---------------------------------------------------------------------------

def _find_value_after_keyword(lines: list[str], keyword: str) -> str:
    """Return the text that follows *keyword* on the same or next non-empty line."""
    keyword_upper = keyword.upper()
    for i, line in enumerate(lines):
        upper = line.upper().strip()
        if keyword_upper in upper:
            # Try same line first (e.g. "SURNAME: MUKASA")
            remainder = line[upper.find(keyword_upper) + len(keyword_upper):]
            remainder = remainder.strip(" :;-\t")
            if remainder:
                return remainder
            # Otherwise look on the next non-empty line
            for j in range(i + 1, min(i + 4, len(lines))):
                next_line = lines[j].strip()
                if next_line:
                    return next_line
    return ""


def parse_front(text: str) -> dict:
    """Extract front-side fields from OCR *text*.

    Returns:
        Dict with keys: surname, given_name, nin, date_of_birth.
        Missing fields are empty strings.
    """
    lines = [l for l in text.splitlines() if l.strip()]

    result: dict[str, str] = {
        "surname": "",
        "given_name": "",
        "nin": "",
        "date_of_birth": "",
    }

    # --- NIN (most reliable via regex) ---
    nin_match = NIN_PATTERN.search(text)
    if nin_match:
        result["nin"] = nin_match.group(1).upper()

    # --- DATE OF BIRTH ---
    # Look for keyword first, then fall back to any date-like string
    dob_context = _find_value_after_keyword(lines, "DATE OF BIRTH") or \
                  _find_value_after_keyword(lines, "DOB") or \
                  _find_value_after_keyword(lines, "BIRTH")
    dob_match = DOB_PATTERN.search(dob_context) if dob_context else \
                DOB_PATTERN.search(text)
    if dob_match:
        result["date_of_birth"] = dob_match.group(1)

    # --- SURNAME ---
    result["surname"] = _find_value_after_keyword(lines, "SURNAME") or \
                        _find_value_after_keyword(lines, "FAMILY NAME") or \
                        _find_value_after_keyword(lines, "LAST NAME")

    # --- GIVEN NAME ---
    result["given_name"] = _find_value_after_keyword(lines, "GIVEN NAME") or \
                           _find_value_after_keyword(lines, "GIVEN NAMES") or \
                           _find_value_after_keyword(lines, "FIRST NAME") or \
                           _find_value_after_keyword(lines, "FORENAME")

    return result


def parse_back(text: str) -> dict:
    """Extract back-side fields from OCR *text*.

    Returns:
        Dict with key: village.
    """
    lines = [l for l in text.splitlines() if l.strip()]

    village = _find_value_after_keyword(lines, "VILLAGE") or \
              _find_value_after_keyword(lines, "VILLAG")  # handle OCR typos

    return {"village": village}


# ---------------------------------------------------------------------------
# OCR worker (runs in a background thread)
# ---------------------------------------------------------------------------

class OCRWorker(threading.Thread):
    """Processes a captured frame and notifies the main thread via a callback."""

    def __init__(self, frame: np.ndarray, side: str, callback):
        """
        Args:
            frame:    BGR frame captured from the webcam.
            side:     "front" or "back".
            callback: callable(result: dict, error: str | None) called when done.
        """
        super().__init__(daemon=True)
        self.frame = frame.copy()
        self.side = side
        self.callback = callback

    def run(self) -> None:
        try:
            processed = preprocess_image(self.frame)
            raw_text = ocr_image(processed)

            if self.side == "front":
                data = parse_front(raw_text)
            else:
                data = parse_back(raw_text)

            self.callback(data, raw_text, None)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            self.callback({}, "", error_msg)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    """Tracks the state machine of the ID-capture workflow."""

    # Possible states
    FRONT_READY = "front_ready"      # waiting to capture front
    FRONT_PROCESSING = "front_proc"  # OCR running on front
    FRONT_DONE = "front_done"        # front extracted, awaiting confirm/retry
    BACK_READY = "back_ready"        # waiting to capture back
    BACK_PROCESSING = "back_proc"    # OCR running on back
    BACK_DONE = "back_done"          # back extracted, awaiting confirm
    SAVING = "saving"                # about to save & reset

    def __init__(self):
        self.state = self.FRONT_READY
        self.front_data: dict = {}
        self.back_data: dict = {}
        self.raw_front: str = ""
        self.raw_back: str = ""
        self.status_message: str = ""
        self.status_color: tuple = STATUS_COLOR
        self.lock = threading.Lock()

    def reset(self):
        """Reset to initial state for the next ID card."""
        with self.lock:
            self.state = self.FRONT_READY
            self.front_data = {}
            self.back_data = {}
            self.raw_front = ""
            self.raw_back = ""
            self.status_message = ""

    def set_status(self, msg: str, color: tuple = STATUS_COLOR):
        with self.lock:
            self.status_message = msg
            self.status_color = color


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def render_overlay(frame: np.ndarray, app: AppState) -> np.ndarray:
    """Draw guidance text and extracted field values onto *frame*."""
    display = frame.copy()
    h, w = display.shape[:2]

    # --- top instruction bar ---
    bar_height = 36
    cv2.rectangle(display, (0, 0), (w, bar_height), (40, 40, 40), -1)

    state = app.state
    if state == AppState.FRONT_READY:
        instruction = "Press SPACE to capture FRONT  |  Q = Quit"
        color = STATUS_COLOR
    elif state == AppState.FRONT_PROCESSING:
        instruction = "Processing front side... please wait"
        color = WARNING_COLOR
    elif state == AppState.FRONT_DONE:
        instruction = "Front OK? Press C to confirm, R to retry  |  Q = Quit"
        color = WARNING_COLOR
    elif state == AppState.BACK_READY:
        instruction = "Flip card. Press SPACE to capture BACK  |  Q = Quit"
        color = STATUS_COLOR
    elif state == AppState.BACK_PROCESSING:
        instruction = "Processing back side... please wait"
        color = WARNING_COLOR
    elif state == AppState.BACK_DONE:
        instruction = "Back OK? Press C to confirm, R to retry  |  Q = Quit"
        color = WARNING_COLOR
    else:
        instruction = "Saving record..."
        color = STATUS_COLOR

    cv2.putText(display, instruction, (10, 24), FONT, 0.60,
                (0, 0, 0), FONT_THICKNESS + 1, cv2.LINE_AA)
    cv2.putText(display, instruction, (10, 24), FONT, 0.60,
                color, FONT_THICKNESS, cv2.LINE_AA)

    # --- extracted fields (bottom panel) ---
    panel_lines: list[str] = []

    if app.front_data:
        panel_lines.append(f"Surname   : {app.front_data.get('surname', '—')}")
        panel_lines.append(f"Given Name: {app.front_data.get('given_name', '—')}")
        panel_lines.append(f"NIN       : {app.front_data.get('nin', '—')}")
        panel_lines.append(f"DOB       : {app.front_data.get('date_of_birth', '—')}")

    if app.back_data:
        panel_lines.append(f"Village   : {app.back_data.get('village', '—')}")

    if panel_lines:
        panel_top = h - len(panel_lines) * 28 - 16
        cv2.rectangle(display, (0, panel_top - 8), (w, h), (30, 30, 30), -1)
        put_multiline(display, panel_lines, (10, panel_top + 4))

    # --- status message (middle area) ---
    if app.status_message:
        msg_lines = app.status_message.split("\n")
        put_multiline(display, msg_lines,
                      (10, bar_height + 32),
                      color=app.status_color, scale=0.60, line_height=24)

    return display


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point – opens webcam and runs the interactive capture loop."""
    ensure_csv()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open webcam. Ensure a camera is connected and not in use."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    app = AppState()
    window_name = "Ugandan ID Capture — PDMIS Assistant"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # --- OCR completion callback (called from worker thread) ---
    def on_ocr_done(data: dict, raw_text: str, error: Optional[str]):
        with app.lock:
            if error:
                app.status_message = f"OCR Error: {error}\nPress R to retry."
                app.status_color = ERROR_COLOR
                if app.state == AppState.FRONT_PROCESSING:
                    app.state = AppState.FRONT_DONE
                elif app.state == AppState.BACK_PROCESSING:
                    app.state = AppState.BACK_DONE
                return

            if app.state == AppState.FRONT_PROCESSING:
                app.front_data = data
                app.raw_front = raw_text
                app.state = AppState.FRONT_DONE
                missing = [k for k, v in data.items() if not v]
                if missing:
                    app.status_message = (
                        f"Partial extraction — missing: {', '.join(missing)}\n"
                        "Press C to confirm anyway, or R to retry."
                    )
                    app.status_color = WARNING_COLOR
                else:
                    app.status_message = "Front extracted successfully! Press C to confirm."
                    app.status_color = STATUS_COLOR

            elif app.state == AppState.BACK_PROCESSING:
                app.back_data = data
                app.raw_back = raw_text
                app.state = AppState.BACK_DONE
                if not data.get("village"):
                    app.status_message = (
                        "Village not found.\nPress C to confirm anyway, or R to retry."
                    )
                    app.status_color = WARNING_COLOR
                else:
                    app.status_message = "Back extracted successfully! Press C to confirm."
                    app.status_color = STATUS_COLOR

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                app.set_status("Cannot read from webcam.", ERROR_COLOR)
                time.sleep(0.1)
                continue

            display = render_overlay(frame, app)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            # --- Q: quit ---
            if key == ord("q") or key == ord("Q"):
                break

            # --- SPACE: capture ---
            elif key == ord(" "):
                with app.lock:
                    current_state = app.state

                if current_state == AppState.FRONT_READY:
                    app.set_status("Capturing front side...", WARNING_COLOR)
                    app.state = AppState.FRONT_PROCESSING
                    worker = OCRWorker(frame, "front", on_ocr_done)
                    worker.start()

                elif current_state == AppState.BACK_READY:
                    app.set_status("Capturing back side...", WARNING_COLOR)
                    app.state = AppState.BACK_PROCESSING
                    worker = OCRWorker(frame, "back", on_ocr_done)
                    worker.start()

                elif current_state in (AppState.FRONT_DONE, AppState.BACK_DONE):
                    # SPACE does nothing in done-states; user must use C or R
                    pass

            # --- C: confirm ---
            elif key == ord("c") or key == ord("C"):
                with app.lock:
                    current_state = app.state

                if current_state == AppState.FRONT_DONE:
                    app.state = AppState.BACK_READY
                    app.set_status(
                        "Front confirmed! Now flip the card and press SPACE.",
                        STATUS_COLOR,
                    )

                elif current_state == AppState.BACK_DONE:
                    # Merge and save
                    record = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        **app.front_data,
                        **app.back_data,
                    }
                    try:
                        save_to_csv(record)
                        app.set_status(
                            f"Record saved to {CSV_FILE}!\nReady for next ID — capturing FRONT.",
                            STATUS_COLOR,
                        )
                    except OSError as exc:
                        app.set_status(f"Failed to save CSV: {exc}", ERROR_COLOR)
                    app.reset()

            # --- R: retry current side ---
            elif key == ord("r") or key == ord("R"):
                with app.lock:
                    current_state = app.state

                if current_state == AppState.FRONT_DONE:
                    app.front_data = {}
                    app.raw_front = ""
                    app.state = AppState.FRONT_READY
                    app.set_status("Retrying front capture. Press SPACE when ready.", INFO_COLOR)

                elif current_state == AppState.BACK_DONE:
                    app.back_data = {}
                    app.raw_back = ""
                    app.state = AppState.BACK_READY
                    app.set_status("Retrying back capture. Press SPACE when ready.", INFO_COLOR)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
