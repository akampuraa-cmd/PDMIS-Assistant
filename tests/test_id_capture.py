"""
Unit tests for id_capture.py — covers pure logic functions that do not
require a webcam or a running Tesseract installation.
"""

import csv
import os
import tempfile
import textwrap

import numpy as np
import pytest

# ── import the module under test ──────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import id_capture as ic


# ---------------------------------------------------------------------------
# order_points
# ---------------------------------------------------------------------------

class TestOrderPoints:
    def test_canonical_rectangle(self):
        pts = np.array([[10, 10], [110, 10], [110, 60], [10, 60]], dtype="float32")
        result = ic.order_points(pts)
        np.testing.assert_array_equal(result[0], [10, 10])   # top-left
        np.testing.assert_array_equal(result[1], [110, 10])  # top-right
        np.testing.assert_array_equal(result[2], [110, 60])  # bottom-right
        np.testing.assert_array_equal(result[3], [10, 60])   # bottom-left

    def test_shuffled_points(self):
        pts = np.array([[110, 60], [10, 10], [110, 10], [10, 60]], dtype="float32")
        result = ic.order_points(pts)
        np.testing.assert_array_equal(result[0], [10, 10])
        np.testing.assert_array_equal(result[2], [110, 60])


# ---------------------------------------------------------------------------
# four_point_transform
# ---------------------------------------------------------------------------

class TestFourPointTransform:
    def _make_image(self, h=100, w=200):
        img = np.zeros((h, w), dtype="uint8")
        img[10:90, 10:190] = 128
        return img

    def test_output_is_not_empty(self):
        img = self._make_image()
        pts = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype="float32")
        warped = ic.four_point_transform(img, pts)
        assert warped.size > 0

    def test_output_shape_matches_corners(self):
        img = self._make_image(h=100, w=200)
        # Define corners that match a 200×100 rectangle
        pts = np.array([[0, 0], [199, 0], [199, 99], [0, 99]], dtype="float32")
        warped = ic.four_point_transform(img, pts)
        assert warped.shape[0] == 99
        assert warped.shape[1] == 199


# ---------------------------------------------------------------------------
# find_card_contour — smoke test on synthetic images
# ---------------------------------------------------------------------------

class TestFindCardContour:
    def _frame_with_rectangle(self, h=480, w=640):
        """Create a grayscale image with a bright rectangle on a dark background."""
        img = np.zeros((h, w), dtype="uint8")
        img[60:360, 80:480] = 220
        return img

    def test_returns_array_or_none(self):
        img = self._frame_with_rectangle()
        result = ic.find_card_contour(img)
        assert result is None or (isinstance(result, np.ndarray) and result.shape == (4, 2))

    def test_empty_image_returns_none(self):
        img = np.zeros((480, 640), dtype="uint8")
        result = ic.find_card_contour(img)
        assert result is None


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def test_output_is_grayscale_2d(self):
        # Create a synthetic BGR image with some variation
        frame = np.random.randint(0, 256, (480, 640, 3), dtype="uint8")
        result = ic.preprocess_image(frame)
        assert result.ndim == 2, "Expected 2-D (grayscale) output"

    def test_output_at_least_min_width(self):
        frame = np.random.randint(0, 256, (100, 200, 3), dtype="uint8")
        result = ic.preprocess_image(frame)
        assert result.shape[1] >= 1200


# ---------------------------------------------------------------------------
# NIN_PATTERN regex
# ---------------------------------------------------------------------------

class TestNINPattern:
    def test_valid_male_nin(self):
        assert ic.NIN_PATTERN.search("NIN: CM12345678J9CX") is not None

    def test_valid_female_nin(self):
        assert ic.NIN_PATTERN.search("CF98765432ABCD") is not None

    def test_invalid_prefix(self):
        assert ic.NIN_PATTERN.search("CA12345678J9CX") is None

    def test_too_short(self):
        assert ic.NIN_PATTERN.search("CM12345678") is None

    def test_too_long(self):
        # 15 chars with word boundary → should not match the 14-char pattern
        assert ic.NIN_PATTERN.search("CM12345678J9CXZ") is None


# ---------------------------------------------------------------------------
# parse_front
# ---------------------------------------------------------------------------

class TestParseFront:
    SAMPLE_TEXT = textwrap.dedent("""\
        REPUBLIC OF UGANDA
        NATIONAL IDENTIFICATION CARD
        SURNAME: MUKASA
        GIVEN NAME: JOSEPH PETER
        DATE OF BIRTH: 01/01/1985
        NIN: CM12345678J9CX
        SEX: MALE
    """)

    def test_surname_extracted(self):
        result = ic.parse_front(self.SAMPLE_TEXT)
        assert result["surname"] == "MUKASA"

    def test_given_name_extracted(self):
        result = ic.parse_front(self.SAMPLE_TEXT)
        assert result["given_name"] == "JOSEPH PETER"

    def test_nin_extracted(self):
        result = ic.parse_front(self.SAMPLE_TEXT)
        assert result["nin"] == "CM12345678J9CX"

    def test_dob_extracted(self):
        result = ic.parse_front(self.SAMPLE_TEXT)
        assert result["date_of_birth"] == "01/01/1985"

    def test_missing_fields_are_empty_strings(self):
        result = ic.parse_front("REPUBLIC OF UGANDA")
        for key in ("surname", "given_name", "nin", "date_of_birth"):
            assert result[key] == "", f"Expected empty string for '{key}'"

    def test_female_nin(self):
        text = "NIN: CF98765432ABCD"
        result = ic.parse_front(text)
        assert result["nin"] == "CF98765432ABCD"

    def test_dob_dash_separator(self):
        text = "DATE OF BIRTH: 15-06-1990"
        result = ic.parse_front(text)
        assert result["date_of_birth"] == "15-06-1990"

    def test_value_on_next_line(self):
        text = "SURNAME\nOKELLO\nGIVEN NAME\nSARAH\nDATE OF BIRTH\n31/12/2000\n"
        result = ic.parse_front(text)
        assert result["surname"] == "OKELLO"
        assert result["given_name"] == "SARAH"
        assert result["date_of_birth"] == "31/12/2000"


# ---------------------------------------------------------------------------
# parse_back
# ---------------------------------------------------------------------------

class TestParseBack:
    def test_village_extracted(self):
        text = "VILLAGE: NAMUGONGO"
        result = ic.parse_back(text)
        assert result["village"] == "NAMUGONGO"

    def test_village_on_next_line(self):
        text = "VILLAGE\nKITINTALE\n"
        result = ic.parse_back(text)
        assert result["village"] == "KITINTALE"

    def test_missing_village(self):
        result = ic.parse_back("BACK OF ID CARD")
        assert result["village"] == ""

    def test_ocr_typo_villag(self):
        text = "VILLAG: BWAISE"
        result = ic.parse_back(text)
        assert result["village"] == "BWAISE"


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

class TestCSV:
    def _tmp_csv(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        os.unlink(path)   # ensure it does not exist yet
        return path

    def test_ensure_csv_creates_file_with_header(self):
        path = self._tmp_csv()
        try:
            ic.ensure_csv(path)
            assert os.path.exists(path)
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                assert list(reader.fieldnames) == ic.CSV_COLUMNS
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_ensure_csv_does_not_overwrite_existing(self):
        path = self._tmp_csv()
        try:
            ic.ensure_csv(path)
            # Write a data row manually
            with open(path, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=ic.CSV_COLUMNS).writerow(
                    {c: "x" for c in ic.CSV_COLUMNS}
                )
            # Calling again should not truncate the file
            ic.ensure_csv(path)
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == 1
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_to_csv_appends_record(self):
        path = self._tmp_csv()
        try:
            record = {
                "timestamp": "2025-01-01T00:00:00",
                "surname": "MUKASA",
                "given_name": "JOSEPH",
                "nin": "CM12345678J9CX",
                "date_of_birth": "01/01/1985",
                "village": "NAMUGONGO",
            }
            ic.save_to_csv(record, path)
            ic.save_to_csv(record, path)   # second append
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == 2
            assert rows[0]["nin"] == "CM12345678J9CX"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_to_csv_handles_missing_keys(self):
        path = self._tmp_csv()
        try:
            ic.save_to_csv({"surname": "TEST"}, path)
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            assert rows[0]["given_name"] == ""
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------

class TestAppState:
    def test_initial_state(self):
        state = ic.AppState()
        assert state.state == ic.AppState.FRONT_READY
        assert state.front_data == {}
        assert state.back_data == {}

    def test_reset_clears_data(self):
        state = ic.AppState()
        state.front_data = {"surname": "MUKASA"}
        state.back_data = {"village": "NAMUGONGO"}
        state.state = ic.AppState.BACK_DONE
        state.reset()
        assert state.state == ic.AppState.FRONT_READY
        assert state.front_data == {}
        assert state.back_data == {}

    def test_set_status(self):
        state = ic.AppState()
        state.set_status("Hello", (0, 255, 0))
        assert state.status_message == "Hello"
        assert state.status_color == (0, 255, 0)
