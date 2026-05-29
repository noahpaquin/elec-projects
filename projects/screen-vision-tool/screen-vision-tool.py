import glob
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mss
import pyautogui
from pynput import keyboard

# ================== CONFIG ==================
TOGGLE_KEY = keyboard.Key.insert   # pause/resume (resets to SEARCHING immediately)
EXIT_KEY   = keyboard.Key.f9       # optional clean exit

CLICK_INTERVAL_MS = 120
CLICK_TIMEOUT_S   = 20.0

# After stop/timeout: focus click, wait, press 4 twice, click
POST_WAIT_S         = 1.2
POST_KEY            = "4"
POST_KEY_GAP_MS     = 80
POST_CLICK_AFTER_MS = 50

# Boost (only while SEARCHING): click, 60ms, key5, 500ms, click, 500ms, key4, 500ms, click
BOOST_INTERVAL_MS = 130_000
BOOST_DELAY_MS    = 500
BOOST_PRECLICK_MS = 60
BOOST_KEYS        = ("5", "4")

# Template match thresholds (0..1)
START_THRESHOLD = 0.55
STOP_THRESHOLD  = 0.60

# Stop images appear in bottom-right quarter
STOP_ROI_FRAC = (0.50, 0.50, 0.50, 0.50)

START_GLOB = "templates/start_exclaim*"
STOP_GLOB  = "templates/stop/*"

FORCE_MONITOR_INDEX = 1
# ===========================================

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

paused    = False
stop_flag = False
state     = "SEARCHING"

kb = keyboard.Controller()


@dataclass(slots=True)
class Template:
    name: str
    gray: np.ndarray   # stored as grayscale — faster matchTemplate
    h: int
    w: int


def _ms() -> int:
    return time.perf_counter_ns() // 1_000_000


def on_press(key):
    global paused, stop_flag, state
    if key == TOGGLE_KEY:
        paused = not paused
        print("Paused" if paused else "Resumed (reset to SEARCHING)")
        state = "SEARCHING"
    elif key == EXIT_KEY:
        stop_flag = True
        print("Exiting...")


def interruptible_sleep_ms(total_ms: int) -> bool:
    end_ns = time.perf_counter_ns() + total_ms * 1_000_000
    while time.perf_counter_ns() < end_ns:
        if stop_flag or paused:
            return False
        time.sleep(0.005)
    return True


def load_templates(paths: List[str]) -> List[Template]:
    out: List[Template] = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"Failed to load template: {p}")
        h, w = img.shape[:2]
        out.append(Template(
            name=p.replace("\\", "/").split("/")[-1],
            gray=img, h=h, w=w,
        ))
    return out


def frac_roi(mon: dict, roi_frac: Tuple[float, float, float, float]) -> dict:
    xf, yf, wf, hf = roi_frac
    return {
        "left":   int(mon["left"]  + mon["width"]  * xf),
        "top":    int(mon["top"]   + mon["height"] * yf),
        "width":  int(mon["width"] * wf),
        "height": int(mon["height"] * hf),
    }


def grab_gray(sct: mss.mss, region: dict) -> np.ndarray:
    # mss returns BGRA — convert directly to grayscale, no intermediate copy
    return cv2.cvtColor(np.array(sct.grab(region), dtype=np.uint8), cv2.COLOR_BGRA2GRAY)


def best_match(
    gray_frame: np.ndarray,
    templates: List[Template],
) -> Optional[Tuple[float, Template]]:
    best: Optional[Tuple[float, Template]] = None
    for t in templates:
        if gray_frame.shape[0] < t.h or gray_frame.shape[1] < t.w:
            continue
        res = cv2.matchTemplate(gray_frame, t.gray, cv2.TM_CCOEFF_NORMED)
        max_val = float(cv2.minMaxLoc(res)[1])
        if best is None or max_val > best[0]:
            best = (max_val, t)
    return best


def click(x: int, y: int) -> None:
    pyautogui.moveTo(x, y, duration=0)
    pyautogui.click(button="left")


def press_key(ch: str) -> None:
    key = keyboard.KeyCode.from_char(ch)
    kb.press(key)
    kb.release(key)


def pick_monitor(sct: mss.mss) -> Tuple[int, dict]:
    idx = FORCE_MONITOR_INDEX
    if idx is None:
        return 1, sct.monitors[1]
    if idx <= 0 or idx >= len(sct.monitors):
        raise SystemExit(f"FORCE_MONITOR_INDEX={idx} invalid. Choose 1..{len(sct.monitors)-1}.")
    return idx, sct.monitors[idx]


def do_boost(click_x: int, click_y: int) -> bool:
    print("[BOOST] start")
    click(click_x, click_y)
    if not interruptible_sleep_ms(BOOST_PRECLICK_MS):
        return False
    press_key(BOOST_KEYS[0])
    if not interruptible_sleep_ms(BOOST_DELAY_MS):
        return False
    click(click_x, click_y)
    if not interruptible_sleep_ms(BOOST_DELAY_MS):
        return False
    press_key(BOOST_KEYS[1])
    if not interruptible_sleep_ms(BOOST_DELAY_MS):
        return False
    click(click_x, click_y)
    print("[BOOST] done")
    return True


def do_post_sequence(click_x: int, click_y: int) -> bool:
    print("[POST] start")
    click(click_x, click_y)
    if not interruptible_sleep_ms(50):
        return False
    if not interruptible_sleep_ms(int(POST_WAIT_S * 1000)):
        return False
    press_key(POST_KEY)
    if not interruptible_sleep_ms(POST_KEY_GAP_MS):
        return False
    press_key(POST_KEY)
    if POST_CLICK_AFTER_MS > 0:
        if not interruptible_sleep_ms(POST_CLICK_AFTER_MS):
            return False
    click(click_x, click_y)
    print("[POST] done")
    return True


def main():
    global state, stop_flag

    keyboard.Listener(on_press=on_press, daemon=True).start()
    print("INSERT = pause/resume (resets to SEARCHING) | F9 = exit | failsafe: mouse to top-left")

    start_paths = sorted(glob.glob(START_GLOB))
    stop_paths  = sorted(glob.glob(STOP_GLOB))

    if not start_paths:
        raise SystemExit(f"No start templates found: {START_GLOB}")
    if not stop_paths:
        raise SystemExit(f"No stop templates found: {STOP_GLOB}")

    start_templates = load_templates(start_paths)
    stop_templates  = load_templates(stop_paths)

    print(f"Loaded start templates: {[t.name for t in start_templates]}")
    print(f"Loaded stop templates:  {[t.name for t in stop_templates]}")

    last_click_ms  = _ms() - CLICK_INTERVAL_MS
    last_boost_ms  = _ms()
    click_cycle_ns = 0

    with mss.MSS() as sct:
        print("Monitors (mss order):")
        for i, m in enumerate(sct.monitors):
            print(f"  [{i}] {'(virtual) ' if i == 0 else ''}{m}")

        mon_idx, mon = pick_monitor(sct)
        print(f"Using monitor index {mon_idx}: {mon}")

        start_region = mon
        stop_region  = frac_roi(mon, STOP_ROI_FRAC)
        print(f"Start ROI: {start_region}")
        print(f"Stop ROI:  {stop_region}")

        click_x = mon["left"] + mon["width"]  // 2
        click_y = mon["top"]  + mon["height"] // 2

        while not stop_flag:
            if paused:
                time.sleep(0.01)
                continue

            if state == "SEARCHING":
                now_ms = _ms()

                if (now_ms - last_boost_ms) >= BOOST_INTERVAL_MS:
                    if do_boost(click_x, click_y):
                        last_boost_ms = _ms()
                    continue

                frame = grab_gray(sct, start_region)
                match = best_match(frame, start_templates)
                if match and match[0] >= START_THRESHOLD:
                    score, t = match
                    click_cycle_ns = time.perf_counter_ns()
                    last_click_ms  = _ms() - CLICK_INTERVAL_MS
                    state = "CLICKING"
                    print(f"START: {t.name} score={score:.3f} -> clicking ({click_x},{click_y})")
                else:
                    time.sleep(0.015)   # 15ms between scans (was 30ms)
                continue

            # ── CLICKING ──
            if (time.perf_counter_ns() - click_cycle_ns) / 1e9 >= CLICK_TIMEOUT_S:
                print("DONE: timeout")
                do_post_sequence(click_x, click_y)
                state = "SEARCHING"
                continue

            stop_frame = grab_gray(sct, stop_region)
            sm = best_match(stop_frame, stop_templates)
            if sm and sm[0] >= STOP_THRESHOLD:
                print(f"DONE: {sm[1].name} score={sm[0]:.3f}")
                do_post_sequence(click_x, click_y)
                state = "SEARCHING"
                continue

            now_ms = _ms()
            if (now_ms - last_click_ms) >= CLICK_INTERVAL_MS:
                click(click_x, click_y)
                last_click_ms = now_ms

            time.sleep(0.010)   # 10ms polling in CLICKING state (was 20ms)


if __name__ == "__main__":
    main()
