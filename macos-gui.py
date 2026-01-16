#!/usr/bin/env python3
"""
macos-gui.py - General-purpose macOS GUI Automation Library

Provides:
- Interactive repl
- Screen capture
- OCR text detection (via Tesseract)
- Template matching (via provided images)
- Mouse/keyboard control (via cliclick)
- High-level operations (click_on_text, find_element, wait_for, etc.)

Usage:
    from macos-gui import Screen, Mouse, Keyboard, GUI

    gui = GUI()
    gui.click_on_text("Submit")
    gui.type_in_field("Search", "hello world")
"""

import subprocess
import tempfile
import time
import random
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path

import numpy as np
from PIL import Image

@dataclass
class Config:
    """Global configuration."""
    tesseract_path: str = "/opt/homebrew/bin/tesseract"
    cliclick_path: str = "/opt/homebrew/bin/cliclick"

    # OCR settings
    ocr_lang: str = "eng"
    ocr_psm: int = 11  # Sparse text - find as much text as possible

    # Mouse behavior
    human_like: bool = True # todo - not human like
    mouse_speed: float = 0.3  # seconds for full movement

    # Timing
    default_timeout: float = 10.0
    poll_interval: float = 0.5

config = Config()

class Screen:
    """Screen capture and image operations."""

    @staticmethod
    def capture(region: Tuple[int, int, int, int] = None,
                path: str = None) -> Image.Image:
        """
        Capture screen or region.

        Args:
            region: (x, y, width, height) or None for full screen
            path: Optional path to save screenshot

        Returns:
            PIL Image
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name

        try:
            cmd = ['screencapture', '-x']  # -x = no sound

            if region:
                x, y, w, h = region
                cmd.extend(['-R', f'{x},{y},{w},{h}'])

            cmd.append(tmp_path)
            subprocess.run(cmd, capture_output=True, check=True)

            img = Image.open(tmp_path)

            if path:
                img.save(path)

            return img.copy()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def capture_window(app_name: str = None, path: str = None) -> Optional[Image.Image]:
        """
        Capture a specific application window.

        Args:
            app_name: Application name (e.g., "Firefox", "Safari")
            path: Optional path to save screenshot
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name

        try:
            if app_name:
                # Get window ID using AppleScript
                script = f'''
                tell application "System Events"
                    tell process "{app_name}"
                        set frontWindow to front window
                        return id of frontWindow
                    end tell
                end tell
                '''
                # For now, just capture the whole screen - window capture is complex
                cmd = ['screencapture', '-x', tmp_path]
            else:
                cmd = ['screencapture', '-x', tmp_path]

            subprocess.run(cmd, capture_output=True, check=True)
            img = Image.open(tmp_path)

            if path:
                img.save(path)

            return img.copy()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def size() -> Tuple[int, int]:
        """Get screen size."""
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True, text=True
        )
        # Parse resolution - this is rough, but works
        for line in result.stdout.split('\n'):
            if 'Resolution' in line:
                match = re.search(r'(\d+)\s*x\s*(\d+)', line)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
        return (1920, 1080)  # fallback


# OCR; text detection
@dataclass
class TextMatch:
    """A detected text region."""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def __repr__(self):
        return f'TextMatch("{self.text}", pos=({self.x},{self.y}), conf={self.confidence:.0f}%)'


class OCR:
    """OCR text detection using Tesseract."""

    @staticmethod
    def find_text(image: Union[Image.Image, str, np.ndarray],
                  pattern: str = None,
                  min_confidence: float = 60.0) -> List[TextMatch]:
        """
        Find all text in image, optionally filtering by pattern.

        Args:
            image: PIL Image, path string, or numpy array
            pattern: Regex pattern to filter results (None = all text)
            min_confidence: Minimum confidence threshold (0-100)

        Returns:
            List of TextMatch objects
        """
        # Convert to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Save to temp file for tesseract
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name
            image.save(tmp_path)

        try:
            # Run tesseract with TSV output for bounding boxes
            cmd = [
                config.tesseract_path,
                tmp_path,
                'stdout',
                '-l', config.ocr_lang,
                '--psm', str(config.ocr_psm),
                'tsv'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Tesseract error: {result.stderr}")

            matches = []
            lines = result.stdout.strip().split('\n')

            if len(lines) < 2:
                return matches

            # Parse TSV output
            # Format: level, page_num, block_num, par_num, line_num, word_num,
            #         left, top, width, height, conf, text
            header = lines[0].split('\t')

            for line in lines[1:]:
                parts = line.split('\t')
                if len(parts) < 12:
                    continue

                try:
                    conf = float(parts[10])
                    text = parts[11].strip()

                    if not text or conf < min_confidence:
                        continue

                    if pattern and not re.search(pattern, text, re.IGNORECASE):
                        continue

                    match = TextMatch(
                        text=text,
                        x=int(parts[6]),
                        y=int(parts[7]),
                        width=int(parts[8]),
                        height=int(parts[9]),
                        confidence=conf
                    )
                    matches.append(match)

                except (ValueError, IndexError):
                    continue

            return matches

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def find_exact(image: Union[Image.Image, str],
                   text: str,
                   min_confidence: float = 70.0) -> Optional[TextMatch]:
        """
        Find exact text match (case-insensitive).

        Returns the highest confidence match, or None.
        """
        matches = OCR.find_text(image, min_confidence=min_confidence)

        text_lower = text.lower()
        exact_matches = [m for m in matches if m.text.lower() == text_lower]

        if exact_matches:
            return max(exact_matches, key=lambda m: m.confidence)
        return None

    @staticmethod
    def find_fuzzy(image: Union[Image.Image, str],
                   text: str,
                   min_confidence: float = 60.0) -> Optional[TextMatch]:
        """
        Find text with fuzzy matching (handles OCR splitting/errors).

        Matches if:
        - Exact match
        - Text starts with query
        - Query is contained in text
        - Levenshtein distance is small

        Returns the best match or None.
        """
        matches = OCR.find_text(image, min_confidence=min_confidence)
        text_lower = text.lower()

        candidates = []

        for m in matches:
            m_lower = m.text.lower()

            # Exact match - best
            if m_lower == text_lower:
                candidates.append((m, 100))
            # Text starts with query
            elif m_lower.startswith(text_lower):
                candidates.append((m, 90))
            # Query contained in text
            elif text_lower in m_lower:
                candidates.append((m, 80))
            # Similar length and some overlap
            elif len(m_lower) >= len(text_lower) * 0.7:
                overlap = sum(1 for c in text_lower if c in m_lower)
                if overlap >= len(text_lower) * 0.6:
                    candidates.append((m, 50 + overlap))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return best[0]

        return None

    @staticmethod
    def find_containing(image: Union[Image.Image, str],
                        text: str,
                        min_confidence: float = 60.0) -> List[TextMatch]:
        """
        Find all text regions containing the given text.
        """
        matches = OCR.find_text(image, min_confidence=min_confidence)
        text_lower = text.lower()
        return [m for m in matches if text_lower in m.text.lower()]

    @staticmethod
    def read_all(image: Union[Image.Image, str]) -> str:
        """
        Read all text from image as a single string.
        """
        if isinstance(image, str):
            image = Image.open(image)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name
            image.save(tmp_path)

        try:
            cmd = [
                config.tesseract_path,
                tmp_path,
                'stdout',
                '-l', config.ocr_lang,
                '--psm', '3'  # Fully automatic page segmentation
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout.strip()

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@dataclass
class TemplateMatch:
    """A template match result."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    template_name: str = ""

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class Templates:
    """Template matching for visual element detection."""

    @staticmethod
    def match(image: Union[Image.Image, str, np.ndarray],
              template: Union[Image.Image, str, np.ndarray],
              threshold: float = 0.8,
              method: str = 'ncc') -> List[TemplateMatch]:
        """
        Find template in image using normalized cross-correlation.

        Args:
            image: Image to search in
            template: Template to find
            threshold: Minimum match score (0-1)
            method: 'ncc' (normalized cross-correlation) or 'sad' (sum of abs diff)

        Returns:
            List of TemplateMatch objects
        """
        # Convert inputs to grayscale numpy arrays
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        elif len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.uint8)

        if isinstance(template, str):
            template = Image.open(template)
        if isinstance(template, Image.Image):
            template = np.array(template.convert('L'))
        elif len(template.shape) == 3:
            template = np.mean(template, axis=2).astype(np.uint8)

        img = image.astype(float)
        tmpl = template.astype(float)

        th, tw = tmpl.shape
        ih, iw = img.shape

        if th > ih or tw > iw:
            return []

        # Normalize template
        tmpl_mean = tmpl.mean()
        tmpl_std = tmpl.std()
        if tmpl_std < 1e-6:
            return []
        tmpl_norm = (tmpl - tmpl_mean) / tmpl_std

        matches = []
        step = max(1, min(th, tw) // 8)  # Adaptive step size

        for y in range(0, ih - th + 1, step):
            for x in range(0, iw - tw + 1, step):
                region = img[y:y+th, x:x+tw]
                region_mean = region.mean()
                region_std = region.std()

                if region_std < 1e-6:
                    continue

                region_norm = (region - region_mean) / region_std

                # Normalized cross-correlation
                ncc = (tmpl_norm * region_norm).mean()

                if ncc >= threshold:
                    matches.append(TemplateMatch(
                        x=x, y=y,
                        width=tw, height=th,
                        confidence=ncc
                    ))

        # Non-maximum suppression
        matches = Templates._nms(matches, tw // 2, th // 2)

        return sorted(matches, key=lambda m: m.confidence, reverse=True)

    @staticmethod
    def _nms(matches: List[TemplateMatch],
             x_thresh: int, y_thresh: int) -> List[TemplateMatch]:
        """Non-maximum suppression to remove overlapping matches."""
        if not matches:
            return []

        matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
        keep = []

        for match in matches:
            overlaps = False
            for kept in keep:
                if (abs(match.x - kept.x) < x_thresh and
                    abs(match.y - kept.y) < y_thresh):
                    overlaps = True
                    break
            if not overlaps:
                keep.append(match)

        return keep

    @staticmethod
    def find_color_region(image: Union[Image.Image, np.ndarray],
                          color: Tuple[int, int, int],
                          tolerance: int = 20,
                          min_area: int = 100) -> List[TemplateMatch]:
        """
        Find regions matching a specific color.

        Args:
            image: Image to search
            color: RGB color tuple
            tolerance: Color matching tolerance
            min_area: Minimum region area in pixels
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))

        target = np.array(color)
        diff = np.abs(image.astype(int) - target)
        mask = np.all(diff <= tolerance, axis=2).astype(np.uint8)

        # Find connected regions (simple approach)
        regions = []
        visited = np.zeros_like(mask, dtype=bool)
        h, w = mask.shape

        for start_y in range(0, h, 5):
            for start_x in range(0, w, 5):
                if mask[start_y, start_x] and not visited[start_y, start_x]:
                    # Flood fill
                    min_x, min_y = start_x, start_y
                    max_x, max_y = start_x, start_y
                    area = 0
                    stack = [(start_x, start_y)]

                    while stack and area < 10000:  # Limit for performance
                        x, y = stack.pop()
                        if x < 0 or y < 0 or x >= w or y >= h:
                            continue
                        if visited[y, x] or not mask[y, x]:
                            continue

                        visited[y, x] = True
                        area += 1
                        min_x, min_y = min(min_x, x), min(min_y, y)
                        max_x, max_y = max(max_x, x), max(max_y, y)

                        stack.extend([(x+1,y), (x-1,y), (x,y+1), (x,y-1)])

                    if area >= min_area:
                        regions.append(TemplateMatch(
                            x=min_x, y=min_y,
                            width=max_x - min_x + 1,
                            height=max_y - min_y + 1,
                            confidence=area / ((max_x-min_x+1) * (max_y-min_y+1))
                        ))

        return regions


class Mouse:
    """Mouse control using cliclick."""

    @staticmethod
    def position() -> Tuple[int, int]:
        """Get current mouse position."""
        result = subprocess.run(
            [config.cliclick_path, 'p'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        return (0, 0)

    @staticmethod
    def move(x: int, y: int, smooth: bool = None):
        """
        Move mouse to position.

        Args:
            x, y: Target coordinates
            smooth: Use human-like movement (default: config.human_like)
        """
        if smooth is None:
            smooth = config.human_like

        if smooth:
            Mouse._smooth_move(x, y)
        else:
            subprocess.run([config.cliclick_path, f'm:{x},{y}'], capture_output=True)

    @staticmethod
    def _smooth_move(target_x: int, target_y: int):
        """Human-like bezier curve movement."""
        start_x, start_y = Mouse.position()

        dist = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
        steps = max(15, int(dist / 20))

        # Bezier control points
        ctrl1_x = start_x + (target_x - start_x) * 0.25 + random.randint(-25, 25)
        ctrl1_y = start_y + (target_y - start_y) * 0.25 + random.randint(-25, 25)
        ctrl2_x = start_x + (target_x - start_x) * 0.75 + random.randint(-15, 15)
        ctrl2_y = start_y + (target_y - start_y) * 0.75 + random.randint(-15, 15)

        for i in range(steps + 1):
            t = i / steps

            # Cubic bezier
            x = int((1-t)**3 * start_x + 3*(1-t)**2*t * ctrl1_x +
                    3*(1-t)*t**2 * ctrl2_x + t**3 * target_x)
            y = int((1-t)**3 * start_y + 3*(1-t)**2*t * ctrl1_y +
                    3*(1-t)*t**2 * ctrl2_y + t**3 * target_y)

            subprocess.run([config.cliclick_path, f'm:{x},{y}'], capture_output=True)

            # Ease-in-out timing
            speed = 4 * t * (1 - t)
            time.sleep((config.mouse_speed / steps) * (0.5 + speed))

    @staticmethod
    def click(x: int = None, y: int = None, button: str = 'left', count: int = 1):
        """
        Click at position (or current position).

        Args:
            x, y: Click position (None = current position)
            button: 'left' or 'right'
            count: Number of clicks (1 = single, 2 = double)
        """
        if x is not None and y is not None:
            Mouse.move(x, y)
            time.sleep(random.uniform(0.03, 0.08))

        pos = Mouse.position()

        if button == 'left':
            cmd = 'dc' if count == 2 else 'c'
        else:
            cmd = 'rc'

        subprocess.run([config.cliclick_path, f'{cmd}:{pos[0]},{pos[1]}'],
                      capture_output=True)

    @staticmethod
    def double_click(x: int = None, y: int = None):
        """Double-click at position."""
        Mouse.click(x, y, count=2)

    @staticmethod
    def right_click(x: int = None, y: int = None):
        """Right-click at position."""
        Mouse.click(x, y, button='right')

    @staticmethod
    def drag(start: Tuple[int, int], end: Tuple[int, int]):
        """Drag from start to end position."""
        Mouse.move(start[0], start[1])
        time.sleep(0.1)

        # Mouse down, move, mouse up
        subprocess.run([config.cliclick_path, f'dd:{start[0]},{start[1]}'],
                      capture_output=True)
        time.sleep(0.05)

        Mouse._smooth_move(end[0], end[1])

        subprocess.run([config.cliclick_path, f'du:{end[0]},{end[1]}'],
                      capture_output=True)

    @staticmethod
    def scroll(amount: int, horizontal: bool = False):
        """
        Scroll by amount.

        Args:
            amount: Positive = down/right, negative = up/left
            horizontal: Horizontal scroll if True
        """
        scroll_dir = -1 if amount > 0 else 1  # Positive amount = scroll down = negative delta

        for _ in range(abs(amount)):
            js_code = f'''
                ObjC.import('Quartz');
                var event = $.CGEventCreateScrollWheelEvent(null, 0, 1, {scroll_dir});
                $.CGEventPost($.kCGHIDEventTap, event);
            '''
            subprocess.run(['osascript', '-l', 'JavaScript', '-e', js_code],
                          capture_output=True)
            time.sleep(0.02)


class Keyboard:
    """Keyboard control using AppleScript."""

    # Key code mapping
    KEY_CODES = {
        'return': 36, 'enter': 36, 'tab': 48, 'space': 49,
        'delete': 51, 'backspace': 51, 'escape': 53, 'esc': 53,
        'up': 126, 'down': 125, 'left': 123, 'right': 124,
        'home': 115, 'end': 119, 'pageup': 116, 'pagedown': 121,
        'f1': 122, 'f2': 120, 'f3': 99, 'f4': 118, 'f5': 96,
        'f6': 97, 'f7': 98, 'f8': 100, 'f9': 101, 'f10': 109,
        'f11': 103, 'f12': 111,
    }

    @staticmethod
    def type(text: str, interval: float = 0.0):
        """
        Type text string.

        Args:
            text: Text to type
            interval: Delay between characters (0 = fast)
        """
        if interval > 0 and config.human_like:
            for char in text:
                escaped = char.replace('\\', '\\\\').replace('"', '\\"')
                subprocess.run([
                    'osascript', '-e',
                    f'tell application "System Events" to keystroke "{escaped}"'
                ], capture_output=True)
                time.sleep(interval * random.uniform(0.7, 1.4))
        else:
            escaped = text.replace('\\', '\\\\').replace('"', '\\"')
            subprocess.run([
                'osascript', '-e',
                f'tell application "System Events" to keystroke "{escaped}"'
            ], capture_output=True)

    @staticmethod
    def press(key: str, modifiers: List[str] = None):
        """
        Press a key with optional modifiers.

        Args:
            key: Key name (e.g., 'return', 'tab', 'a', 'f1')
            modifiers: List of modifiers ['cmd', 'ctrl', 'alt', 'shift']
        """
        key_lower = key.lower()

        mod_str = ''
        if modifiers:
            mod_map = {
                'cmd': 'command down', 'command': 'command down',
                'ctrl': 'control down', 'control': 'control down',
                'alt': 'option down', 'option': 'option down',
                'shift': 'shift down',
            }
            parts = [mod_map.get(m.lower(), '') for m in modifiers if m.lower() in mod_map]
            if parts:
                mod_str = ' using {' + ', '.join(parts) + '}'

        if key_lower in Keyboard.KEY_CODES:
            code = Keyboard.KEY_CODES[key_lower]
            subprocess.run([
                'osascript', '-e',
                f'tell application "System Events" to key code {code}{mod_str}'
            ], capture_output=True)
        else:
            # Single character
            subprocess.run([
                'osascript', '-e',
                f'tell application "System Events" to keystroke "{key}"{mod_str}'
            ], capture_output=True)

    @staticmethod
    def hotkey(*keys):
        """
        Press a keyboard shortcut.

        Example:
            Keyboard.hotkey('cmd', 'c')  # Copy
            Keyboard.hotkey('cmd', 'shift', 'p')  # Some shortcut
        """
        if len(keys) == 1:
            Keyboard.press(keys[0])
        else:
            modifiers = list(keys[:-1])
            key = keys[-1]
            Keyboard.press(key, modifiers)

class GUI:
    """High-level GUI automation operations."""

    def __init__(self):
        self.last_screenshot = None
        self.last_screenshot_time = 0

    def screenshot(self, max_age: float = 0.5) -> Image.Image:
        """Get screenshot, using cached version if recent enough."""
        now = time.time()
        if (self.last_screenshot is None or
            now - self.last_screenshot_time > max_age):
            self.last_screenshot = Screen.capture()
            self.last_screenshot_time = now
        return self.last_screenshot

    def refresh(self):
        """Force refresh screenshot cache."""
        self.last_screenshot = None

    def find_text(self, text: str,
                  timeout: float = None,
                  refresh: bool = True) -> Optional[TextMatch]:
        """
        Find text on screen.

        Args:
            text: Text to find
            timeout: Wait up to this long (None = no wait)
            refresh: Refresh screenshot before searching
        """
        if refresh:
            self.refresh()

        start = time.time()
        timeout = timeout or 0

        while True:
            img = self.screenshot(max_age=0.1)

            # Try exact match first
            match = OCR.find_exact(img, text)
            if match:
                return match

            # Try containing match
            matches = OCR.find_containing(img, text)
            if matches:
                return matches[0]

            if time.time() - start >= timeout:
                break

            time.sleep(config.poll_interval)
            self.refresh()

        return None

    def click_on_text(self, text: str,
                      timeout: float = None,
                      offset: Tuple[int, int] = (0, 0)) -> bool:
        """
        Click on text found via OCR.

        Args:
            text: Text to find and click
            timeout: Wait up to this long for text to appear
            offset: (x, y) offset from text center

        Returns:
            True if clicked, False if not found
        """
        match = self.find_text(text, timeout=timeout)

        if match:
            x, y = match.center
            Mouse.click(x + offset[0], y + offset[1])
            return True

        return False

    def wait_for_text(self, text: str,
                      timeout: float = None) -> bool:
        """
        Wait for text to appear on screen.

        Returns:
            True if found, False if timeout
        """
        return self.find_text(text, timeout=timeout or config.default_timeout) is not None

    def type_in_field(self, label: str, value: str,
                      timeout: float = None) -> bool:
        """
        Find a field by its label and type into it.

        Args:
            label: Field label text
            value: Value to type
            timeout: Wait timeout

        Returns:
            True if successful
        """
        match = self.find_text(label, timeout=timeout)

        if match:
            # Click to the right of the label (where input usually is)
            x = match.x + match.width + 20
            y = match.center[1]
            Mouse.click(x, y)
            time.sleep(0.2)

            # Select all and type
            Keyboard.hotkey('cmd', 'a')
            time.sleep(0.1)
            Keyboard.type(value)
            return True

        return False

    def click_button(self, text: str,
                     timeout: float = None) -> bool:
        """
        Click a button by its text.

        Same as click_on_text but semantically clearer.
        """
        return self.click_on_text(text, timeout=timeout)

    def read_screen(self) -> str:
        """Read all text currently on screen."""
        self.refresh()
        return OCR.read_all(self.screenshot())

    def list_text(self, min_confidence: float = 60) -> List[TextMatch]:
        """List all detected text on screen."""
        self.refresh()
        return OCR.find_text(self.screenshot(), min_confidence=min_confidence)


def repl():
    """Interactive REPL for testing GUI automation."""
    gui = GUI()

    print("\n" + "="*50)
    print("  macos-gui - Interactive REPL")
    print("="*50)
    print("\nCommands:")
    print("  find <text>     - Find text on screen")
    print("  click <text>    - Click on text")
    print("  type <text>     - Type text")
    print("  press <key>     - Press key (e.g., 'return', 'tab')")
    print("  hotkey <k1> <k2> - Press hotkey (e.g., 'cmd c')")
    print("  move <x> <y>    - Move mouse")
    print("  pos             - Show mouse position")
    print("  read            - Read all text on screen")
    print("  list            - List all detected text")
    print("  screenshot [f]  - Save screenshot")
    print("  help            - Show this help")
    print("  quit            - Exit")
    print("-"*50)

    while True:
        try:
            cmd = input("\nmacos-gui> ").strip()

            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            action = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if action in ('q', 'quit', 'exit'):
                break

            elif action in ('h', 'help', '?'):
                print("Commands: find, click, type, press, hotkey, move, pos, read, list, screenshot, quit")

            elif action == 'find':
                match = gui.find_text(arg)
                if match:
                    print(f"Found: {match}")
                else:
                    print(f"Not found: '{arg}'")

            elif action == 'click':
                if gui.click_on_text(arg):
                    print(f"Clicked on '{arg}'")
                else:
                    print(f"Not found: '{arg}'")

            elif action == 'type':
                Keyboard.type(arg, interval=0.05)
                print(f"Typed: '{arg}'")

            elif action == 'press':
                Keyboard.press(arg)
                print(f"Pressed: {arg}")

            elif action == 'hotkey':
                keys = arg.split()
                Keyboard.hotkey(*keys)
                print(f"Hotkey: {' + '.join(keys)}")

            elif action == 'move':
                coords = arg.split()
                if len(coords) >= 2:
                    x, y = int(coords[0]), int(coords[1])
                    Mouse.move(x, y)
                    print(f"Moved to ({x}, {y})")

            elif action == 'pos':
                pos = Mouse.position()
                print(f"Mouse position: {pos}")

            elif action == 'read':
                text = gui.read_screen()
                print("--- Screen text ---")
                print(text[:1000] + ("..." if len(text) > 1000 else ""))

            elif action == 'list':
                matches = gui.list_text()
                print(f"Found {len(matches)} text regions:")
                for m in matches[:20]:
                    print(f"  {m}")
                if len(matches) > 20:
                    print(f"  ... and {len(matches) - 20} more")

            elif action in ('screenshot', 'ss', 'capture'):
                path = arg or f"screenshot_{int(time.time())}.png"
                Screen.capture(path=path)
                print(f"Saved: {path}")

            else:
                print(f"Unknown command: {action}")

        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == 'repl':
            repl()
        elif cmd == 'screenshot':
            path = sys.argv[2] if len(sys.argv) > 2 else 'screenshot.png'
            Screen.capture(path=path)
            print(f"Saved: {path}")
        elif cmd == 'read':
            gui = GUI()
            print(gui.read_screen())
        elif cmd == 'find':
            if len(sys.argv) > 2:
                gui = GUI()
                match = gui.find_text(sys.argv[2])
                if match:
                    print(match)
                else:
                    print("Not found")
        elif cmd == 'click':
            if len(sys.argv) > 2:
                gui = GUI()
                gui.click_on_text(sys.argv[2])
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: macos-gui.py [repl|screenshot|read|find <text>|click <text>]")
    else:
        repl()
