import asyncio
from datetime import datetime, timezone
from enum import Enum
import os
import cv2
import numpy as np
import pyautogui
import time
import glob
from sklearn.cluster import KMeans
from collections import Counter
import threading
from tracker import add_encounter_to_json, add_shiny_to_json, reset_session_encounters
from dotenv import load_dotenv
from discord_bot import run_bot, get_client, get_bot_loop, image_to_discord_file
load_dotenv()

discord_token = os.environ.get("DISCORD_TOKEN")
search_analysis_lock = threading.Lock()

class State(Enum):
    SEARCHING = "searching"
    BATTLING = "battling"
    RUNNING = "running"
    CATCHING = "catching"
    RESETTING = "resetting"

state = State.SEARCHING

def take_screenshot(region=None):
    """
    Capture a screenshot of the screen or a specific region.
    region: (left, top, width, height)
    """
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot

def save_battle_sprite(sprite_img, folder="./screenshots"):
    """Save the cropped battle sprite to a folder with a timestamped name."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(folder, f"battle_sprite_{timestamp}.png")
    cv2.imwrite(filename, sprite_img)
    print(f"Saved battle sprite to: {filename}")

def save_gray_image(img, filename_prefix, folder="./screenshots"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # If already single channel (grayscale), no need to convert
    if len(img.shape) == 2:
        gray_img = img
    elif len(img.shape) == 3:
        # Could be 3 or 4 channels - handle accordingly
        if img.shape[2] == 1:
            gray_img = img[:, :, 0]
        elif img.shape[2] == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported number of channels: {img.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(folder, f"{filename_prefix}_{timestamp}.png")
    cv2.imwrite(filename, gray_img)
    print(f"Saved grayscale image to: {filename}")

def load_shiny_pngs(folder_path: str = "./Shiny"):
    """
    Load all PNG images in the given folder as BGR OpenCV images.
    Returns a list of images and their file paths.
    """
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    images = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images, image_paths

def crop_white_borders(img, threshold=255):
    """
    Crop out white (#FFFFFF) borders from all sides of the image.
    Works on BGR images with 3 channels.
    
    Args:
        img (np.ndarray): Input BGR image.
        threshold (int): The threshold to consider a pixel as white (default 255).

    Returns:
        Cropped image without white borders.
    """
    # Create a mask of pixels that are white (all channels >= threshold)
    white_mask = np.all(img >= threshold, axis=2)

    # Find rows and columns that are not fully white
    non_white_rows = np.where(~np.all(white_mask, axis=1))[0]
    non_white_cols = np.where(~np.all(white_mask, axis=0))[0]

    if non_white_rows.size == 0 or non_white_cols.size == 0:
        print("[WARNING]: The provided image is completely white!")
        return img

    # Get min and max indices for cropping
    top, bottom = non_white_rows[0], non_white_rows[-1] + 1
    left, right = non_white_cols[0], non_white_cols[-1] + 1

    # Crop the image
    cropped_img = img[top:bottom, left:right]

    return cropped_img

def ensure_bgra(img):
    """Ensure image has an alpha channel (BGRA)."""
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

def prepare_orb_image(img, crop=True):
    """
    Prepare an image for ORB by cropping and converting to grayscale.
    """
    img = ensure_bgra(img)

    if crop:
        img = crop_white_borders(img)

    # Remove white pixels completely by setting them to black and transparent
    white_mask = np.all(img[:, :, :3] >= 255, axis=2)
    img[white_mask] = [0, 0, 0, 0]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return gray

def resize_to_match(img1, img2):
    """
    Resize both images to the same target size (average dimensions).
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_w = int((w1 + w2) / 2)
    target_h = int((h1 + h2) / 2)

    img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return img1_resized, img2_resized

def orb_match_score(img1, img2):
    img1, img2 = resize_to_match(img1, img2)

    orb = cv2.ORB().create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 60]
    return len(good_matches)

def compare_image_to_list_orb(battle_sprite, shiny_frames, top_n=3):
    """
    Compare the battle sprite to a list of shiny frames using ORB feature matching.
    """
    battle_gray = prepare_orb_image(battle_sprite)
    # DEBUG ONLY
    # save_gray_image(battle_gray, "orb_battle_grey") 
    results = []

    for idx, shiny in enumerate(shiny_frames):
        shiny_gray = prepare_orb_image(shiny)
        score = orb_match_score(battle_gray, shiny_gray)
        results.append((idx, score))

    results.sort(key=lambda x: x[1], reverse=True)

    # Debug: save top matches
    # for rank, (idx, score) in enumerate(results[:top_n], start=1):
    #     save_gray_image(prepare_orb_image(shiny_frames[idx]), f"orb_top{rank}_shiny_gray_{idx}")

    return results[:top_n]

def get_pixel_color(img, x, y):
    """
    Get the BGR color value of a pixel at (x, y) in an OpenCV image.

    Args:
        img (np.ndarray): The image (as a NumPy array).
        x (int): X-coordinate (horizontal).
        y (int): Y-coordinate (vertical).

    Returns:
        tuple: (B, G, R) values of the pixel.
    """
    if y >= img.shape[0] or x >= img.shape[1]:
        raise ValueError(f"Coordinates ({x}, {y}) are out of image bounds {img.shape[1]}x{img.shape[0]}")

    return tuple(img[y, x])

def get_dominant_color(image, k=3):
    """
    Find the most dominant BGR color in an image using K-means clustering.

    Args:
        image (np.ndarray): Input image (BGR format).
        k (int): Number of clusters to form; default is 1 for the single most dominant color.

    Returns:
        tuple: Dominant color as (B, G, R).
    """
    # Crop white borders of base image to focus solely on the target
    image = crop_white_borders(image)

    # Resize for efficiency (smaller = faster, but lose accuracy)
    resized_img = cv2.resize(image, (54, 54), interpolation=cv2.INTER_NEAREST)
    # Drop fully transparent pixels
    if resized_img.shape[2] == 4:
        b, g, r, a = cv2.split(resized_img)
        mask = a > 0
        pixels = np.stack((b[mask], g[mask], r[mask]), axis=1)
    else:
        pixels = resized_img.reshape((-1, 3))

    # Remove near-white & black pixels
    pixels = [
        p for p in pixels
        if not all(channel >= 254 for channel in p)
        and not all(channel <= 1 for channel in p) 
    ]
    if not pixels:
        print("[ERROR] FELL BACK TO ALL WHITE")
        return (255, 255, 255)  # fallback
    
    unique_pixels = np.unique(pixels, axis=0)
    k = min(k, len(unique_pixels))
    if k == 0:
        print("[ERROR]No unique pixels some how idk tbh")
        return (255, 255, 255)

    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(pixels)

    counts = Counter(kmeans.labels_)
    dominant_index = counts.most_common(1)[0][0]
    dominant_color = kmeans.cluster_centers_[dominant_index]

    return tuple(map(int, dominant_color))

def color_distance(bgr1, bgr2):
    """
    Calculate the Euclidean distance between two BGR colors.
    
    Args:
        bgr1 (tuple): First BGR color (e.g., (228, 160, 123))
        bgr2 (tuple): Second BGR color (e.g., (230, 181, 139))

    Returns:
        float: Distance between the colors
    """
    b1, g1, r1 = bgr1
    b2, g2, r2 = bgr2
    return ((b1 - b2) ** 2 + (g1 - g2) ** 2 + (r1 - r2) ** 2) ** 0.5

def search():
    hold_keys(["a", "d"], 0.5)

def hold_keys(keys, hold_time=1.0):
    """
    Press and hold one or more keys for a specified duration.

    Args:
        keys (str or list): A single key (e.g. 'a') or a list of keys (e.g. ['ctrl', 'c']).
        hold_time (float): How long to hold the key(s), in seconds.
    """
    if isinstance(keys, str):
        keys = [keys]

    # Press keys down
    for key in keys:
        pyautogui.keyDown(key)
        time.sleep(hold_time)
        pyautogui.keyUp(key)

def main_from_screenshot(shiny_frames, shiny_paths):
    global state
    while(True):
        if state == State.SEARCHING:
            search()
        if state == State.BATTLING:
            add_encounter_to_json()
            time.sleep(2.5)
            hold_keys(["p", "o", "o"], 0.5)
            region = (1032, 64, 458, 360)  # left, top, width, height
            screenshot = take_screenshot(region)

            battle_dominant_color = get_dominant_color(screenshot)
            print("Dominant Color (BGR):", battle_dominant_color)
            # Compare the battle sprite against shiny frames
            top_matches = compare_image_to_list_orb(screenshot, shiny_frames)


            for rank, (idx, score) in enumerate(top_matches, start=1):
                window_name = f"Top {rank} Shiny Match"
                winning_shiny_dominant_color = get_dominant_color(shiny_frames[idx])
                shiny_score = color_distance(battle_dominant_color, winning_shiny_dominant_color)
                print(f"Rank {rank}: {shiny_paths[idx]} | Score: {score:.4f} | dominant_color: {winning_shiny_dominant_color} | score (small = good): {shiny_score} | is shiny: {shiny_score < 10}")
                if shiny_score < 10:
                    discord_file = image_to_discord_file(screenshot)
                    add_shiny_to_json(discord_file)
                    print(f"Found shiny at rank: {rank}... saving image and shutting down")
                    save_battle_sprite(screenshot)
                    return # TODO: Add capture options (maybe controlled and updated by discord message or something)
            time.sleep(1.5)
            hold_keys(["s", "d", "p", "o", "o", "o"], 0.5)
            state = State.SEARCHING

def analyze_for_battle_trigger(check_interval=1):
    """
    Background thread that captures screenshots during SEARCHING state
    and checks for a condition to switch to BATTLING.
    """
    global state
    while True:
        if state != State.SEARCHING:
            time.sleep(check_interval)
            continue

        if search_analysis_lock.locked():
            time.sleep(check_interval)
            continue

        with search_analysis_lock:
            screenshot = take_screenshot()
            # These coords must be something specific to the battle scene (I use a black portion of a text box)
            pixel = get_pixel_color(screenshot, 440, 845)

            if pixel == (0, 0, 0):  # checking if said pixel is pure black
                print("[INFO] Battle trigger detected! Switching state.")
                state = State.BATTLING

        time.sleep(check_interval)

if __name__ == "__main__":
    reset_session_encounters()
    shiny_frames, shiny_paths = load_shiny_pngs("./Shiny")
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    time.sleep(2)
    while get_bot_loop() is None and get_client() is None:
        time.sleep(0.1)
    print("DONE BOT SET UP")
    battle_trigger_thread = threading.Thread(
        target=analyze_for_battle_trigger,
        daemon=True
    )
    battle_trigger_thread.start()
    main_from_screenshot(shiny_frames=shiny_frames, shiny_paths=shiny_paths)
    time.sleep(10) # TODO: LOL 
    asyncio.run_coroutine_threadsafe(get_client().shutdown(), get_bot_loop())
