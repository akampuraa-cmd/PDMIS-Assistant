# PDMIS FIS Assistant — Ugandan National ID Capture

A production‑ready Python application that captures live webcam video,
processes a Ugandan National ID card (front **and** back), extracts key
fields via OCR, and stores them in a structured CSV file compatible with
Excel / PDMIS FIS.

---

## Features

| Feature | Details |
|---|---|
| Live webcam preview | OpenCV window with real‑time feed |
| Perspective correction | Detects card contour, applies warp transform |
| OCR | Tesseract (pytesseract) with adaptive thresholding |
| Front fields | SURNAME, GIVEN NAME, NIN, DATE OF BIRTH |
| Back field | VILLAGE |
| NIN validation | Regex `C[MF][A-Z0-9]{12}` (CM = male, CF = female, 14 chars) |
| CSV output | `ugandan_ids.csv` with timestamp |
| Threading | OCR runs in a background thread – UI stays responsive |
| Error handling | On‑screen feedback; retry without losing partial data |

---

## Prerequisites

### 1. Python 3.8+

```bash
python --version   # should be 3.8 or newer
```

### 2. Tesseract OCR binary

| OS | Command |
|---|---|
| Ubuntu / Debian | `sudo apt-get install tesseract-ocr` |
| macOS (Homebrew) | `brew install tesseract` |
| Windows | Download installer from [UB‑Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) |

Verify: `tesseract --version`

### 3. Python packages

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python id_capture.py
```

### Key bindings

| Key | Action |
|---|---|
| `SPACE` | Capture the currently displayed side of the ID |
| `C` | Confirm the extracted data and proceed to the next step |
| `R` | Retry the current side (discards last extraction result) |
| `Q` | Quit the application |

---

## Workflow

```
Start webcam
    │
    ▼
Press SPACE ──► Capture FRONT ──► Pre‑process ──► OCR ──► Parse fields
                                                              │
                                                     Show fields on screen
                                                              │
                                              Press C (confirm) or R (retry)
                                                              │
                                                              ▼
                                                    Flip card over
                                                              │
                                                Press SPACE ──► Capture BACK
                                                              │
                                                         OCR + parse VILLAGE
                                                              │
                                                     Press C to confirm
                                                              │
                                                    Append to ugandan_ids.csv
                                                              │
                                                       Ready for next ID
```

---

## Output CSV

`ugandan_ids.csv` — created automatically in the working directory.

| Column | Example |
|---|---|
| `timestamp` | `2025-04-01T14:30:00` |
| `surname` | `MUKASA` |
| `given_name` | `JOSEPH` |
| `nin` | `CM12345678J9CX` |
| `date_of_birth` | `01/01/1990` |
| `village` | `NAMUGONGO` |

---

## Project Structure

```
PDMIS-Assistant/
├── id_capture.py      # Main application
├── requirements.txt   # Python dependencies
├── ugandan_ids.csv    # Generated at runtime (gitignored)
└── Readme.md
```

---

## Troubleshooting

* **"Cannot open webcam"** – ensure a camera is connected and not used by another app.
* **"Tesseract binary not found"** – install Tesseract (see Prerequisites §2) and ensure it is on your `PATH`.
* **Poor OCR accuracy** – ensure good lighting and hold the card flat/steady in the frame. The app will overlay a warning if fields are missing and let you retry.
