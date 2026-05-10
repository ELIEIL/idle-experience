#!/usr/bin/env python3
"""
Metaball-based heightmap editor for projection mapping.

Concept:
- You place and manipulate "metaballs" (soft blobs) on a canvas.
- Each metaball contributes to a scalar field (height) using a radial falloff.
- The combined field is visualized as a grayscale heightmap or as contour lines.
- You can save the current view as a PNG to use in projection mapping tools.

Dependencies (install in your venv):
    python -m pip install PyQt5 numpy opencv-python

Controls:
- Left click: add a metaball at mouse position.
- Right click: remove the nearest metaball (if within a small radius).
- Drag with left button over an existing metaball: move that metaball.
- Mouse wheel: change radius of the currently selected metaball.

- 'g' : grayscale heightmap view
- 'c' : contour-line view
- '+' / '-' : increase / decrease global field strength
- 's' : save current view as PNG in ./renders_metaball/
- 'r' : randomize a simple metaball layout
- 'ESC' / 'q' : quit
"""

import os
import signal
import sys
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
import praw
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageDraw, ImageFont
import chromadb
from sentence_transformers import SentenceTransformer
import json
import hashlib
import time

# ==========================
# CONFIGURATION
# ==========================

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

WINDOW_TITLE = "Metaball Heightmap Editor"
RENDER_DIR = "renders_metaball"

# Camera / motion tracking configuration
CAMERA_INDEX = 0
MOTION_BLUR_KERNEL = 15
MOTION_THRESHOLD = 25
MOTION_DILATION_ITER = 2
MOTION_MIN_AREA = 800
MOTION_TRACKING_ENABLED = False  # Disabled for performance

# Field / rendering parameters
BASE_STRENGTH = 1.0      # base contribution factor
FALLOFF_POWER = 2.0      # higher = sharper blobs, lower = softer
ISOVALUE = 0.45          # iso-threshold for metaball outlines (0-1 in field space)
ISO_BAND_WIDTH = 0.14    # moderate band thickness around ISOVALUE for text stroke
LINE_THICKNESS = 2

# GDELT integration (Option A): number of events -> number of metaballs
GDELT_ENABLED = True
GDELT_KEYWORD = "CLIMATE"
GDELT_DEFAULT_METABALLS = 6
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Reddit integration (Option B): live posts as characters
REDDIT_ENABLED = True
REDDIT_SUBREDDITS = ["politics", "worldnews"]  # subreddits to fetch from
REDDIT_REFRESH_INTERVAL = 60  # refresh every 60 seconds
REDDIT_POST_LIMIT = 25  # number of posts to fetch per subreddit
REDDIT_CLIENT_ID = ""  # Set your Reddit app client ID
REDDIT_CLIENT_SECRET = ""  # Set your Reddit app client secret
REDDIT_USER_AGENT = "MetaballEditor/1.0"

# Image metaballs (news images from GDELT) - enabled with improved caching
IMAGE_METABALLS_ENABLED = True
IMAGE_METABALL_COUNT = 2  # number of image metaballs to spawn
IMAGE_REFRESH_INTERVAL = 120  # refresh images every 2 minutes

# Semantic database integration
SEMANTIC_DB_ENABLED = True
SEMANTIC_DB_PATH = "./chroma_db"
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight sentence transformer model

# Text stroke parameters
# Multiple character sets for dynamic per-metaball text styles
TEXT_CHAR_SETS = [
    "0123456789",                                    # Numbers
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",                  # Uppercase letters
    "abcdefghijklmnopqrstuvwxyz",                  # Lowercase letters
    "!@#$%^&*()-_=+[]{}|;:,.<>?",                  # Symbols
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",         # Alphanumeric
    "?!#$%@&",                                      # Punctuation
    "░▒▓█",                                         # Block characters
    "○●◎◌◍",                                        # Circle characters
]
TEXT_COLOR = (240, 240, 240)   # light gray / white (fallback if no images)
TEXT_STEP = 20                 # sampling grid step in pixels (more characters, denser)
TEXT_FONT = cv2.FONT_HERSHEY_PLAIN  # more pixel-like built-in font
TEXT_SCALE = 0.8              # smaller glyphs for Reddit text
TEXT_THICKNESS = 1            # thinner stroke style

# Custom font configuration
USE_CUSTOM_FONT = True
FONT_FOLDER = "Fonts"
FONT_PATH = "Fonts/MD Thermochrome 0.4/Desktop/Regular/MDThermochrome0.4-Regular-Trial.otf"  # path to custom font file
FALLBACK_FONT = cv2.FONT_HERSHEY_PLAIN  # fallback if custom font fails

# Image sampling for text colors (disabled)
IMAGE_SAMPLING_ENABLED = False
IMAGE_FOLDER = "images"  # folder to load images from
IMAGE_SCALE_FACTOR = 0.5  # downscale images for faster sampling

# Custom color palette (hex colors converted to BGR for OpenCV)
USE_COLOR_SPECTRUM = False
CUSTOM_COLORS = [
    (40, 40, 214),    # #d62828 (red) -> BGR
    (254, 0, 0),      # #0000FE (blue) -> BGR
    (91, 152, 63),    # #3F985b (green) -> BGR
    (34, 72, 242),    # #F24822 (orange) -> BGR
    (255, 255, 255),  # #FFFFFF (white) -> BGR
]

# Per-metaball outline colors (cycled)
OUTLINE_COLORS = [
    QtCore.Qt.red,
    QtCore.Qt.green,
    QtCore.Qt.blue,
    QtCore.Qt.cyan,
    QtCore.Qt.magenta,
    QtCore.Qt.yellow,
]

# Metaball interaction
DEFAULT_RADIUS = 150.0   # slightly larger default radius for a meatier look
MIN_RADIUS = 20.0
MAX_RADIUS = 400.0
SELECT_RADIUS_PX = 25.0  # picking distance in pixels

# Floating animation
FLOATING_ENABLED = True
FLOAT_SPEED = 0.6        # speed of floating movement (faster for more movement)
FLOAT_UPDATE_MS = 80     # update interval in milliseconds


@dataclass
class Metaball:
    x: float
    y: float
    radius: float
    strength: float
    char_set_index: int = 0  # index into TEXT_CHAR_SETS
    vx: float = 0.0  # velocity x
    vy: float = 0.0  # velocity y
    subreddit: str = ""  # subreddit for Reddit text content
    reddit_text: str = ""  # fetched Reddit text for this metaball
    type: str = "text"  # "text" or "image"
    image_url: str = ""  # URL for image metaballs
    image_data: Optional[np.ndarray] = None  # loaded image data


class RedditFetcher:
    """Fetches posts from Reddit subreddits using PRAW for better API handling."""

    def __init__(self, subreddits: List[str], post_limit: int = 25):
        self.subreddits = subreddits
        self.post_limit = post_limit
        self._reddit = None
        self._init_reddit()

    def _init_reddit(self):
        """Initialize PRAW Reddit instance with fallback to read-only mode."""
        try:
            if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
                self._reddit = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT,
                )
                print("Initialized Reddit with credentials")
            else:
                # Fallback to read-only mode without credentials
                self._reddit = praw.Reddit(
                    user_agent=REDDIT_USER_AGENT,
                    read_only=True,
                )
                print("Initialized Reddit in read-only mode (no credentials)")
        except praw.exceptions.PRAWException as e:
            print(f"PRAW error initializing Reddit: {e}")
            self._reddit = None
        except Exception as e:
            print(f"Failed to initialize Reddit: {e}")
            self._reddit = None

    def fetch_posts(self, subreddit: str) -> str:
        """Fetch posts from a subreddit and return concatenated text."""
        if self._reddit is None:
            print("Reddit not initialized, skipping fetch")
            return ""

        try:
            sub = self._reddit.subreddit(subreddit)
            text_parts = []

            for post in sub.hot(limit=self.post_limit):
                text_parts.append(post.title)
                if post.selftext:
                    text_parts.append(post.selftext)

            return " ".join(text_parts)
        except praw.exceptions.PRAWException as e:
            print(f"PRAW error fetching from r/{subreddit}: {e}")
            return ""
        except praw.exceptions.RedditAPIException as e:
            print(f"Reddit API error fetching from r/{subreddit}: {e}")
            return ""
        except Exception as e:
            print(f"Failed to fetch from r/{subreddit}: {e}")
            return ""

    def fetch_all_subreddits(self) -> dict:
        """Fetch posts from all configured subreddits."""
        results = {}
        for subreddit in self.subreddits:
            results[subreddit] = self.fetch_posts(subreddit)
        return results


class NewsImageFetcher:
    """Fetches news images from GDELT for image metaballs with improved error handling."""

    def __init__(self):
        self.api_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.events_url = "https://api.gdeltproject.org/api/v2/events/events"
        self._cache = {}  # Simple in-memory cache
        self._cache_ttl = 300  # 5 minutes cache TTL

    def _get_cache_key(self, keyword: str, limit: int) -> str:
        """Generate a cache key for the request."""
        return f"{keyword}_{limit}"

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self._cache_ttl

    def fetch_image_urls(self, keyword: str = "news", limit: int = 10) -> List[str]:
        """Fetch image URLs from GDELT news articles with caching."""
        cache_key = self._get_cache_key(keyword, limit)

        # Check cache
        if cache_key in self._cache and self._is_cache_valid(self._cache[cache_key]['timestamp']):
            print(f"Using cached image URLs for {keyword}")
            return self._cache[cache_key]['data']

        try:
            params = {
                "query": keyword,
                "maxrecords": limit * 5,  # fetch more to find images
                "format": "json",
            }
            response = requests.get(self.api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            image_urls = []

            for article in articles:
                image_url = article.get("image")
                if image_url and image_url.startswith("http"):
                    image_urls.append(image_url)
                    if len(image_urls) >= limit:
                        break

            # Cache the results
            self._cache[cache_key] = {
                'data': image_urls,
                'timestamp': time.time()
            }

            return image_urls
        except requests.exceptions.Timeout:
            print(f"Timeout fetching image URLs from GDELT")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Request error fetching image URLs: {e}")
            return []
        except Exception as e:
            print(f"Failed to fetch image URLs: {e}")
            return []

    def fetch_news_headlines(self, keyword: str = "news", limit: int = 10) -> List[dict]:
        """Fetch news headlines from GDELT events API for semantic linking."""
        try:
            params = {
                "query": keyword,
                "maxrecords": limit,
                "format": "json",
            }
            response = requests.get(self.events_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            events = data.get("events", [])
            headlines = []

            for event in events:
                headline = {
                    'title': event.get('eventdescription', ''),
                    'source': event.get('sourceurl', ''),
                    'date': event.get('date', ''),
                }
                if headline['title']:
                    headlines.append(headline)
                    if len(headlines) >= limit:
                        break

            return headlines
        except Exception as e:
            print(f"Failed to fetch news headlines: {e}")
            return []

    def load_image(self, url: str, target_size: Tuple[int, int] = (200, 200)) -> Optional[np.ndarray]:
        """Load an image from URL and resize to target size with error handling."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                return img
        except requests.exceptions.Timeout:
            print(f"Timeout loading image from {url}")
        except requests.exceptions.RequestException as e:
            print(f"Request error loading image from {url}: {e}")
        except Exception as e:
            print(f"Failed to load image from {url}: {e}")
        return None


class SemanticDatabase:
    """Semantic database using ChromaDB for linking Reddit posts and news content."""

    def __init__(self, db_path: str = SEMANTIC_DB_PATH, model_name: str = SEMANTIC_MODEL_NAME):
        self.db_path = db_path
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._embedding_model = None
        self._init_db()

    def _init_db(self):
        """Initialize ChromaDB client and embedding model."""
        try:
            self._client = chromadb.PersistentClient(path=self.db_path)
            self._collection = self._client.get_or_create_collection(
                name="content_store",
                metadata={"hnsw:space": "cosine"}
            )
            self._embedding_model = SentenceTransformer(self.model_name)
            print(f"Initialized semantic database at {self.db_path}")
        except chromadb.errors.ChromaDBError as e:
            print(f"ChromaDB error initializing: {e}")
            self._client = None
            self._collection = None
            self._embedding_model = None
        except Exception as e:
            print(f"Failed to initialize semantic database: {e}")
            self._client = None
            self._collection = None
            self._embedding_model = None

    def add_content(self, content_id: str, text: str, metadata: dict = None):
        """Add content to semantic database with embeddings."""
        if self._collection is None or self._embedding_model is None:
            return

        try:
            embedding = self._embedding_model.encode(text).tolist()
            self._collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[content_id],
                metadatas=[metadata or {}]
            )
        except chromadb.errors.ChromaDBError as e:
            print(f"ChromaDB error adding content: {e}")
        except Exception as e:
            print(f"Failed to add content to semantic DB: {e}")

    def find_similar(self, query_text: str, n_results: int = 5) -> List[dict]:
        """Find similar content in the database based on semantic similarity."""
        if self._collection is None or self._embedding_model is None:
            return []

        try:
            query_embedding = self._embedding_model.encode(query_text).tolist()
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            similar_items = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    similar_items.append({
                        'text': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0
                    })
            return similar_items
        except chromadb.errors.ChromaDBError as e:
            print(f"ChromaDB error querying: {e}")
            return []
        except Exception as e:
            print(f"Failed to query semantic DB: {e}")
            return []

    def link_reddit_to_news(self, reddit_text: str, news_keyword: str = "climate") -> str:
        """Link Reddit text to semantically similar news content."""
        similar = self.find_similar(reddit_text, n_results=1)
        if similar:
            return similar[0]['text']
        return ""


class MotionTracker:
    """Simple background-subtraction motion tracker running in a thread.

    Tracks the centroid of the largest motion blob from a single camera.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._lock = threading.Lock()
        self._centroid: Optional[Tuple[float, float]] = None
        self._frame_size: Optional[Tuple[int, int]] = None  # (w, h)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_centroid_and_size(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]]]:
        with self._lock:
            return self._centroid, self._frame_size

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._running = False
            return
        self._cap = cap

        ret, prev = cap.read()
        if not ret:
            self._running = False
            cap.release()
            self._cap = None
            return

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        h, w = prev_gray.shape[:2]
        with self._lock:
            self._frame_size = (w, h)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_delta = cv2.absdiff(prev_gray, gray)
            prev_gray = gray

            blurred = cv2.GaussianBlur(frame_delta, (MOTION_BLUR_KERNEL, MOTION_BLUR_KERNEL), 0)
            _, thresh = cv2.threshold(blurred, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            if MOTION_DILATION_ITER > 0:
                thresh = cv2.dilate(thresh, None, iterations=MOTION_DILATION_ITER)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest = None
            largest_area = 0.0
            for c in contours:
                area = cv2.contourArea(c)
                if area > largest_area:
                    largest_area = area
                    largest = c

            centroid = None
            if largest is not None and largest_area >= MOTION_MIN_AREA:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    centroid = (cx, cy)

            with self._lock:
                self._centroid = centroid
                if self._frame_size is None:
                    self._frame_size = (w, h)

        cap.release()
        self._cap = None


class MetaballCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None, motion_tracker=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

        self._motion_tracker = motion_tracker  # Store motion tracker first
        self.metaballs: List[Metaball] = []
        self.field_strength_scale = 1.0
        self.view_mode = "contour"  # "gray" or "contour"
        self._field: Optional[np.ndarray] = None
        self._image: Optional[np.ndarray] = None
        self._pixmap: Optional[QtGui.QPixmap] = None

        self._dragging: bool = False
        self._drag_index: int = -1

        # Image sampling: load images from folder for color sampling
        self._sample_images: List[np.ndarray] = []
        if IMAGE_SAMPLING_ENABLED:
            self._load_sample_images()

        # Load custom font if enabled
        self._pil_font = None
        if USE_CUSTOM_FONT:
            self._load_custom_font()

        # Reddit fetcher
        self._reddit_fetcher = None
        self._reddit_data: dict = {}  # subreddit -> text content
        if REDDIT_ENABLED:
            self._reddit_fetcher = RedditFetcher(REDDIT_SUBREDDITS, REDDIT_POST_LIMIT)
            self._fetch_reddit_posts()

        # News image fetcher
        self._image_fetcher = None
        self._image_urls: List[str] = []
        if IMAGE_METABALLS_ENABLED:
            self._image_fetcher = NewsImageFetcher()
            self._fetch_news_images()

        # Semantic database
        self._semantic_db = None
        if SEMANTIC_DB_ENABLED:
            self._semantic_db = SemanticDatabase()
            print("Semantic database initialized")

        # Timer to apply motion tracking to metaballs (if enabled)
        self._motion_timer = None
        if self._motion_tracker is not None:
            self._motion_timer = QtCore.QTimer(self)
            self._motion_timer.timeout.connect(self._apply_motion_to_metaballs)
            self._motion_timer.start(40)  # ~25 FPS polling

        # Timer for floating animation
        self._float_timer = None
        if FLOATING_ENABLED:
            self._float_timer = QtCore.QTimer(self)
            self._float_timer.timeout.connect(self._animate_floating)
            self._float_timer.start(FLOAT_UPDATE_MS)

        # Timer for Reddit refresh
        self._reddit_timer = None
        if REDDIT_ENABLED:
            self._reddit_timer = QtCore.QTimer(self)
            self._reddit_timer.timeout.connect(self._fetch_reddit_posts)
            self._reddit_timer.start(REDDIT_REFRESH_INTERVAL * 1000)  # convert to milliseconds

        # Timer for image refresh
        self._image_timer = None
        if IMAGE_METABALLS_ENABLED:
            self._image_timer = QtCore.QTimer(self)
            self._image_timer.timeout.connect(self._fetch_news_images)
            self._image_timer.start(IMAGE_REFRESH_INTERVAL * 1000)  # convert to milliseconds

        # Initial render
        self.update_field_and_image()

    # ---------- Image sampling ----------

    def _load_sample_images(self):
        """Load images from IMAGE_FOLDER for color sampling."""
        if not os.path.exists(IMAGE_FOLDER):
            print(f"Image folder '{IMAGE_FOLDER}' not found. Using fallback white text.")
            return

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        loaded_count = 0

        for filename in os.listdir(IMAGE_FOLDER):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                filepath = os.path.join(IMAGE_FOLDER, filename)
                try:
                    img = cv2.imread(filepath)
                    if img is not None:
                        # Downscale for faster sampling
                        if IMAGE_SCALE_FACTOR < 1.0:
                            h, w = img.shape[:2]
                            new_h, new_w = int(h * IMAGE_SCALE_FACTOR), int(w * IMAGE_SCALE_FACTOR)
                            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        self._sample_images.append(img)
                        loaded_count += 1
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

        if loaded_count > 0:
            print(f"Loaded {loaded_count} images from '{IMAGE_FOLDER}' for color sampling.")
        else:
            print(f"No valid images found in '{IMAGE_FOLDER}'. Using fallback white text.")

    def _sample_color_from_images(self, x: int, y: int) -> Tuple[int, int, int]:
        """Sample a color from the loaded images based on position."""
        if not self._sample_images:
            return TEXT_COLOR

        # Cycle through images based on position
        img_idx = (x + y * 13) % len(self._sample_images)
        img = self._sample_images[img_idx]

        # Sample pixel from image based on position
        h, w = img.shape[:2]
        sample_x = x % w
        sample_y = y % h

        # OpenCV is BGR
        b, g, r = img[sample_y, sample_x]
        return (int(b), int(g), int(r))

    def _get_spectrum_color(self, x: int, y: int) -> Tuple[int, int, int]:
        """Get a color from HSV spectrum based on position."""
        if not USE_COLOR_SPECTRUM:
            return TEXT_COLOR

        # Map position to HSV hue (0-179 in OpenCV)
        hue = (x + y) % 180
        saturation = 200  # High saturation
        value = 255      # Full brightness

        # Convert HSV to BGR
        hsv_color = np.uint8([[[hue, saturation, value]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        b, g, r = bgr_color[0, 0]
        return (int(b), int(g), int(r))

    def _get_custom_color(self, x: int, y: int) -> Tuple[int, int, int]:
        """Get a color from the custom palette based on position."""
        if not CUSTOM_COLORS:
            return TEXT_COLOR

        # Cycle through colors based on position with better distribution
        color_idx = ((x // TEXT_STEP) + (y // TEXT_STEP)) % len(CUSTOM_COLORS)
        return CUSTOM_COLORS[color_idx]

    def _load_custom_font(self):
        """Load custom font from FONT_PATH using PIL."""
        if not os.path.exists(FONT_PATH):
            print(f"Font file '{FONT_PATH}' not found. Using fallback OpenCV font.")
            return

        try:
            # Load font at appropriate size
            font_size = int(TEXT_SCALE * 30)  # Scale factor for PIL font size
            self._pil_font = ImageFont.truetype(FONT_PATH, font_size)
            print(f"Loaded custom font: {FONT_PATH}")
        except Exception as e:
            print(f"Failed to load custom font: {e}. Using fallback OpenCV font.")

    def _fetch_reddit_posts(self):
        """Fetch posts from Reddit and update metaball text content."""
        if self._reddit_fetcher is None:
            return

        print("Fetching Reddit posts...")
        self._reddit_data = self._reddit_fetcher.fetch_all_subreddits()

        # Store in semantic database
        if self._semantic_db is not None:
            for subreddit, text in self._reddit_data.items():
                if text:
                    content_id = f"reddit_{subreddit}_{int(time.time())}"
                    self._semantic_db.add_content(
                        content_id=content_id,
                        text=text,
                        metadata={"source": "reddit", "subreddit": subreddit}
                    )
            print("Stored Reddit content in semantic database")

        # Update each metaball with its subreddit's text
        for mb in self.metaballs:
            if mb.subreddit and mb.subreddit in self._reddit_data:
                mb.reddit_text = self._reddit_data[mb.subreddit]

        print(f"Fetched Reddit data for {len(self._reddit_data)} subreddits")
        self.update_field_and_image()

    def _fetch_news_images(self):
        """Fetch news images from GDELT and update image metaballs."""
        if self._image_fetcher is None:
            return

        print("Fetching news images...")
        self._image_urls = self._image_fetcher.fetch_image_urls(limit=IMAGE_METABALL_COUNT)

        # Fetch news headlines for semantic database
        if self._semantic_db is not None:
            print("Fetching news headlines for semantic database...")
            headlines = self._image_fetcher.fetch_news_headlines(
                keyword=GDELT_KEYWORD,
                limit=10
            )
            for headline in headlines:
                content_id = f"news_{hashlib.md5(headline['title'].encode()).hexdigest()}"
                self._semantic_db.add_content(
                    content_id=content_id,
                    text=headline['title'],
                    metadata={
                        "source": "news",
                        "url": headline['source'],
                        "date": headline['date']
                    }
                )
            print(f"Stored {len(headlines)} news headlines in semantic database")

        # Update image metaballs with new images
        image_mb_count = 0
        for mb in self.metaballs:
            if mb.type == "image" and image_mb_count < len(self._image_urls):
                url = self._image_urls[image_mb_count]
                mb.image_url = url
                mb.image_data = self._image_fetcher.load_image(url)
                image_mb_count += 1

        print(f"Fetched {len(self._image_urls)} image URLs")
        self.update_field_and_image()

    # ---------- Metaball management ----------

    def _gdelt_estimate_metaball_count(self, fallback: int) -> int:
        """Fetch a rough event count from GDELT and map to a metaball count.

        This uses the public GDELT v2 events API with a simple keyword query.
        If anything fails (network, parsing, etc.), it returns the fallback.
        """

        if not GDELT_ENABLED:
            return fallback

        url = "https://api.gdeltproject.org/api/v2/events/events"
        params = {
            "query": GDELT_KEYWORD,
            "maxrecords": 50,  # keep it small; we only need an order-of-magnitude
            "format": "json",
        }

        try:
            resp = requests.get(url, params=params, timeout=3.0)
            resp.raise_for_status()
            data = resp.json()
            events = data.get("events") or []
            count = len(events)
        except Exception:
            return fallback

        # Map event count (0..50) into a reasonable metaball range, e.g. 3..12
        if count <= 0:
            return fallback

        min_balls, max_balls = 3, 12
        # Normalize count to [0,1] assuming up to 50 events
        t = min(1.0, count / 50.0)
        est = int(min_balls + t * (max_balls - min_balls))
        return max(min_balls, min(max_balls, est))

    def add_metaball(self, x: float, y: float, radius: float = DEFAULT_RADIUS, strength: float = BASE_STRENGTH, char_set_index: int = 0):
        self.metaballs.append(Metaball(x, y, radius, strength, char_set_index))
        self.update_field_and_image()

    def remove_nearest_metaball(self, x: float, y: float):
        if not self.metaballs:
            return
        idx, dist = self._find_nearest_index(x, y)
        if idx is not None and dist <= SELECT_RADIUS_PX:
            del self.metaballs[idx]
            self.update_field_and_image()

    def _find_nearest_index(self, x: float, y: float) -> Tuple[Optional[int], float]:
        best_idx = None
        best_dist2 = float("inf")
        for i, m in enumerate(self.metaballs):
            dx = m.x - x
            dy = m.y - y
            d2 = dx * dx + dy * dy
            if d2 < best_dist2:
                best_dist2 = d2
                best_idx = i
        return best_idx, best_dist2 ** 0.5 if best_idx is not None else float("inf")

    def change_selected_radius(self, x: float, y: float, delta: float):
        if not self.metaballs:
            return
        idx, dist = self._find_nearest_index(x, y)
        if idx is None or dist > SELECT_RADIUS_PX:
            return
        mb = self.metaballs[idx]
        mb.radius = float(np.clip(mb.radius + delta, MIN_RADIUS, MAX_RADIUS))
        self.update_field_and_image()

    # ---------- Mouse / input handling ----------

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        pos = event.pos()
        x = pos.x()
        y = pos.y()

        if event.button() == QtCore.Qt.LeftButton:
            # Check if we hit an existing metaball to start dragging
            idx, dist = self._find_nearest_index(x, y)
            if idx is not None and dist <= SELECT_RADIUS_PX:
                self._dragging = True
                self._drag_index = idx

        elif event.button() == QtCore.Qt.RightButton:
            self.remove_nearest_metaball(x, y)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._dragging and 0 <= self._drag_index < len(self.metaballs):
            pos = event.pos()
            self.metaballs[self._drag_index].x = pos.x()
            self.metaballs[self._drag_index].y = pos.y()
            self.update_field_and_image()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = False
            self._drag_index = -1

    def wheelEvent(self, event: QtGui.QWheelEvent):
        # Change radius of nearest metaball under cursor
        angle_delta = event.angleDelta().y()
        if angle_delta == 0:
            return
        delta = 10.0 if angle_delta > 0 else -10.0
        pos = event.pos()
        self.change_selected_radius(pos.x(), pos.y(), delta)

    def _apply_motion_to_metaballs(self):
        """If motion tracking is active, move the nearest metaball to the motion centroid."""
        if self._motion_tracker is None or not self.metaballs:
            return

        centroid, frame_size = self._motion_tracker.get_centroid_and_size()
        if centroid is None or frame_size is None:
            return

        cam_w, cam_h = frame_size
        cx, cy = centroid

        # Map camera coordinates into canvas coordinate system
        mapped_x = float(cx / max(cam_w, 1) * CANVAS_WIDTH)
        mapped_y = float(cy / max(cam_h, 1) * CANVAS_HEIGHT)

        idx, _ = self._find_nearest_index(mapped_x, mapped_y)
        if idx is None:
            return

        mb = self.metaballs[idx]
        mb.x = mapped_x
        mb.y = mapped_y
        self.update_field_and_image()

    def _animate_floating(self):
        """Animate metaballs floating with smooth movement and boundary bouncing."""
        if not self.metaballs or self._dragging:
            return

        for mb in self.metaballs:
            # Update position
            mb.x += mb.vx
            mb.y += mb.vy

            # Bounce off boundaries
            if mb.x < mb.radius:
                mb.x = mb.radius
                mb.vx = -mb.vx
            elif mb.x > CANVAS_WIDTH - mb.radius:
                mb.x = CANVAS_WIDTH - mb.radius
                mb.vx = -mb.vx

            if mb.y < mb.radius:
                mb.y = mb.radius
                mb.vy = -mb.vy
            elif mb.y > CANVAS_HEIGHT - mb.radius:
                mb.y = CANVAS_HEIGHT - mb.radius
                mb.vy = -mb.vy

        self.update_field_and_image()

    # ---------- Field computation & rendering ----------

    def set_view_mode(self, mode: str):
        if mode in ("gray", "contour"):
            self.view_mode = mode
            self.update_field_and_image()

    def adjust_strength_scale(self, factor: float):
        self.field_strength_scale = float(np.clip(self.field_strength_scale * factor, 0.2, 5.0))
        self.update_field_and_image()

    def random_layout(self, count: int = 6):
        self.metaballs.clear()
        rng = np.random.default_rng()

        # Determine how many metaballs to spawn, optionally using GDELT.
        mb_count = self._gdelt_estimate_metaball_count(count if count > 0 else GDELT_DEFAULT_METABALLS)

        # Split between text and image metaballs
        image_mb_count = min(IMAGE_METABALL_COUNT, mb_count) if IMAGE_METABALLS_ENABLED else 0
        text_mb_count = mb_count - image_mb_count

        # Create image metaballs
        for i in range(image_mb_count):
            x = float(rng.uniform(0.2, 0.8) * CANVAS_WIDTH)
            y = float(rng.uniform(0.2, 0.8) * CANVAS_HEIGHT)
            r = float(rng.uniform(0.4, 0.7) * DEFAULT_RADIUS)  # slightly larger for images
            s = float(rng.uniform(0.7, 1.3) * BASE_STRENGTH)
            vx = float(rng.uniform(-FLOAT_SPEED, FLOAT_SPEED))
            vy = float(rng.uniform(-FLOAT_SPEED, FLOAT_SPEED))
            image_url = self._image_urls[i] if i < len(self._image_urls) else ""
            image_data = self._image_fetcher.load_image(image_url) if image_url and self._image_fetcher else None
            self.metaballs.append(Metaball(x, y, r, s, 0, vx, vy, "", "", "image", image_url, image_data))

        # Create text metaballs
        for i in range(text_mb_count):
            x = float(rng.uniform(0.2, 0.8) * CANVAS_WIDTH)
            y = float(rng.uniform(0.2, 0.8) * CANVAS_HEIGHT)
            r = float(rng.uniform(0.3, 0.6) * DEFAULT_RADIUS)
            s = float(rng.uniform(0.7, 1.3) * BASE_STRENGTH)
            char_idx = rng.integers(0, len(TEXT_CHAR_SETS))
            vx = float(rng.uniform(-FLOAT_SPEED, FLOAT_SPEED))
            vy = float(rng.uniform(-FLOAT_SPEED, FLOAT_SPEED))
            subreddit = ""
            if REDDIT_ENABLED and REDDIT_SUBREDDITS:
                subreddit = REDDIT_SUBREDDITS[i % len(REDDIT_SUBREDDITS)]
            reddit_text = self._reddit_data.get(subreddit, "") if subreddit else ""
            self.metaballs.append(Metaball(x, y, r, s, char_idx, vx, vy, subreddit, reddit_text, "text"))
        self.update_field_and_image()

    def update_field_and_image(self):
        # Recompute field
        field = self._compute_field()
        self._field = field

        if self.view_mode == "gray":
            img = self._field_to_gray(field)  # 2D uint8
        else:
            img = self._field_to_contours(field)  # 3-channel BGR

        self._image = img

        # Handle grayscale vs color images for Qt
        if img.ndim == 2:
            h, w = img.shape
            qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        else:
            h, w, _ = img.shape
            # OpenCV is BGR, QImage.Format_BGR888 expects BGR byte order
            qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_BGR888)

        self._pixmap = QtGui.QPixmap.fromImage(qimg.copy())  # copy to detach from NumPy buffer
        self.update()

    def _compute_field(self) -> np.ndarray:
        h, w = CANVAS_HEIGHT, CANVAS_WIDTH
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        field = np.zeros((h, w), dtype=np.float32)

        for m in self.metaballs:
            dx = xx - m.x
            dy = yy - m.y
            r2 = dx * dx + dy * dy
            radius2 = max(m.radius * m.radius, 1.0)
            # Smooth falloff: strength / (1 + (d^2/r^2))^(p/2)
            contrib = (m.strength * self.field_strength_scale) / (1.0 + (r2 / radius2)) ** (FALLOFF_POWER / 2.0)
            field += contrib

        # Normalize to [0,1] for visualization
        if len(self.metaballs) == 0:
            return field

        max_val = float(field.max())
        if max_val > 1e-6:
            field = field / max_val
        return field

    def _field_to_gray(self, field: np.ndarray) -> np.ndarray:
        img = (np.clip(field, 0.0, 1.0) * 255.0).astype(np.uint8)
        return img

    def _field_to_contours(self, field: np.ndarray) -> np.ndarray:
        """Create a text-based metaball stroke around the iso-surface.

        - Take the normalized field [0,1].
        - Keep values within ISO_BAND_WIDTH/2 around ISOVALUE.
        - On that band, render keyboard characters so the visible
          outline is made of text instead of a solid band.
        - Use the character set from the dominant metaball at each pixel.
        - Use custom PIL font if available, otherwise fall back to OpenCV font.
        - Return a 3-channel BGR image on black background.
        """

        field_clipped = np.clip(field, 0.0, 1.0)

        band_half = ISO_BAND_WIDTH * 0.5
        low = max(0.0, ISOVALUE - band_half)
        high = min(1.0, ISOVALUE + band_half)
        band_mask = (field_clipped >= low) & (field_clipped <= high)

        # Start with black background
        h, w = field_clipped.shape
        contour_img = np.zeros((h, w, 3), dtype=np.uint8)

        if not np.any(band_mask):
            return contour_img

        # Use PIL for custom font rendering, fall back to OpenCV
        use_pil = self._pil_font is not None

        if use_pil:
            # Create PIL image for rendering
            pil_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
            draw = ImageDraw.Draw(pil_img)

        # Draw characters on a grid wherever the band mask is true.
        # Character choice is deterministic based on local field value + position,
        # and uses the character set from the dominant metaball at that location.
        for y in range(0, h, TEXT_STEP):
            for x in range(0, w, TEXT_STEP):
                if not band_mask[y, x]:
                    continue

                # Find which metaball contributes most at this pixel
                max_contrib = -1.0
                dominant_mb = None
                for m in self.metaballs:
                    dx = x - m.x
                    dy = y - m.y
                    r2 = dx * dx + dy * dy
                    radius2 = max(m.radius * m.radius, 1.0)
                    contrib = (m.strength * self.field_strength_scale) / (1.0 + (r2 / radius2)) ** (FALLOFF_POWER / 2.0)
                    if contrib > max_contrib:
                        max_contrib = contrib
                        dominant_mb = m

                if dominant_mb is None:
                    continue

                # Use Reddit text if available, otherwise fall back to character sets
                if REDDIT_ENABLED and dominant_mb.reddit_text:
                    chars = np.array(list(dominant_mb.reddit_text))
                else:
                    char_set = TEXT_CHAR_SETS[dominant_mb.char_set_index % len(TEXT_CHAR_SETS)]
                    chars = np.array(list(char_set))

                n_chars = len(chars)
                if n_chars == 0:
                    continue

                v = field_clipped[y, x]
                base = int(v * 997)  # spread across value range
                idx = (base + x + 31 * y) % n_chars
                ch = chars[idx]

                # Use custom color palette instead of spectrum
                color = self._get_custom_color(x, y)

                if use_pil:
                    # PIL uses RGB, convert from BGR
                    rgb_color = (color[2], color[1], color[0])
                    draw.text((x, y), ch, font=self._pil_font, fill=rgb_color)
                else:
                    cv2.putText(
                        contour_img,
                        ch,
                        (x, y),
                        FALLBACK_FONT,
                        TEXT_SCALE,
                        color,
                        TEXT_THICKNESS,
                        cv2.LINE_AA,
                    )

        if use_pil:
            # Convert PIL image back to numpy array (RGB -> BGR for OpenCV)
            contour_img = np.array(pil_img)
            contour_img = cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR)

        return contour_img

    def save_current_view(self):
        if self._image is None:
            return
        if not os.path.exists(RENDER_DIR):
            os.makedirs(RENDER_DIR)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RENDER_DIR, f"metaball_{self.view_mode}_{timestamp}.png")
        cv2.imwrite(filename, self._image)
        print(f"Saved {filename}")

    # ---------- Painting ----------

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)

        if self._pixmap is not None:
            # Fit pixmap to widget while preserving aspect ratio
            target_rect = self.rect()
            pix = self._pixmap
            scaled = pix.scaled(target_rect.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            x = (target_rect.width() - scaled.width()) // 2
            y = (target_rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        # Draw images for image metaballs
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        scale_x = self.width() / CANVAS_WIDTH
        scale_y = self.height() / CANVAS_HEIGHT

        for m in self.metaballs:
            if m.type == "image" and m.image_data is not None:
                # Convert numpy image to QPixmap
                h, w = m.image_data.shape[:2]
                bytes_per_line = 3 * w
                q_img = QtGui.QImage(m.image_data.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
                pixmap = QtGui.QPixmap.fromImage(q_img)

                # Scale image to fit inside metaball radius
                cx = m.x * scale_x
                cy = m.y * scale_y
                radius_px = m.radius * min(scale_x, scale_y)
                image_size = int(radius_px * 1.5)
                scaled_pixmap = pixmap.scaled(image_size, image_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

                # Draw circular mask
                painter.setClipRegion(QtGui.QRegion(int(cx - radius_px), int(cy - radius_px), int(radius_px * 2), int(radius_px * 2), QtGui.QRegion.Ellipse))
                painter.drawPixmap(int(cx - image_size // 2), int(cy - image_size // 2), scaled_pixmap)
                painter.setClipping(False)

        # Draw red center markers on top (for editing)
        for m in self.metaballs:
            cx = m.x * self.width() / CANVAS_WIDTH
            cy = m.y * self.height() / CANVAS_HEIGHT

            painter.setPen(QtGui.QPen(QtCore.Qt.red, 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(cx, cy), 8, 8)

            painter.setBrush(QtCore.Qt.red)
            painter.drawEllipse(QtCore.QPointF(cx, cy), 3, 3)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)

        # Only initialize motion tracker if enabled
        self._motion_tracker = None
        if MOTION_TRACKING_ENABLED:
            self._motion_tracker = MotionTracker()
            self._motion_tracker.start()

        self.canvas = MetaballCanvas(self, motion_tracker=self._motion_tracker)
        self.setCentralWidget(self.canvas)

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("g"), self, activated=lambda: self.canvas.set_view_mode("gray"))
        QtWidgets.QShortcut(QtGui.QKeySequence("c"), self, activated=lambda: self.canvas.set_view_mode("contour"))
        QtWidgets.QShortcut(QtGui.QKeySequence("+"), self, activated=lambda: self.canvas.adjust_strength_scale(1.1))
        QtWidgets.QShortcut(QtGui.QKeySequence("-"), self, activated=lambda: self.canvas.adjust_strength_scale(1.0 / 1.1))
        QtWidgets.QShortcut(QtGui.QKeySequence("s"), self, activated=self.canvas.save_current_view)
        QtWidgets.QShortcut(QtGui.QKeySequence("r"), self, activated=self.canvas.random_layout)

        QtWidgets.QShortcut(QtGui.QKeySequence("q"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, activated=self.close)

        # Initialize with a simple random layout
        self.canvas.random_layout()

    def closeEvent(self, event):
        """Handle window close event - stop motion tracker gracefully."""
        if self._motion_tracker is not None:
            self._motion_tracker.stop()
        event.accept()


def main():
    import sys

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        QtWidgets.QApplication.instance().quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(CANVAS_WIDTH, CANVAS_HEIGHT)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
