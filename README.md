# Metaball Heightmap Editor

A metaball-based heightmap editor for projection mapping that integrates real-time data from Reddit and GDELT news sources with semantic database linking.

## Features

- **Metaball Visualization**: Create and manipulate soft blob shapes with radial falloff
- **Reddit Integration**: Fetch live posts from subreddits using PRAW library
- **GDELT News Integration**: Fetch news headlines and images from GDELT API
- **Semantic Database**: Link Reddit posts and news content using ChromaDB and sentence transformers
- **Real-time Updates**: Automatic refresh of content at configurable intervals
- **Motion Tracking**: Optional camera-based motion tracking (currently disabled)
- **Custom Fonts**: Support for custom TTF/OTF fonts
- **Image Metaballs**: Display news images within metaballs
- **Web Version**: JavaScript-based metaballs with news images overlay (using metaballs-js)

## Installation

### Prerequisites

- Python 3.11 or higher
- Virtual environment (recommended)

### Setup

1. **Create and activate virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Reddit API Setup (Optional but Recommended)

For better Reddit API access and higher rate limits:

1. Go to https://www.reddit.com/prefs/apps
2. Create a new application (select "script" type)
3. Set `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` in the configuration section of `metaball_heightmap_editor.py`

If no credentials are provided, the app will fall back to read-only mode with lower rate limits.

## Configuration

Edit the configuration section at the top of `metaball_heightmap_editor.py`:

```python
# Reddit Configuration
REDDIT_ENABLED = True
REDDIT_SUBREDDITS = ["politics", "worldnews"]
REDDIT_REFRESH_INTERVAL = 60
REDDIT_POST_LIMIT = 25
REDDIT_CLIENT_ID = ""  # Your Reddit app client ID
REDDIT_CLIENT_SECRET = ""  # Your Reddit app client secret

# GDELT Configuration
GDELT_ENABLED = True
GDELT_KEYWORD = "CLIMATE"

# Image Metaballs
IMAGE_METABALLS_ENABLED = True
IMAGE_METABALL_COUNT = 2

# Semantic Database
SEMANTIC_DB_ENABLED = True
SEMANTIC_DB_PATH = "./chroma_db"
```

## Usage

Run the application:
```bash
python metaball_heightmap_editor.py
```

## Web Version (JavaScript Metaballs with News Images)

A web-based version using the metaballs-js library with news images overlay is also available.

### Running the Web Version

1. **Start the Flask API server**:
```bash
python server.py
```

2. **Open the web interface**:
   - Open `web/index.html` in a web browser
   - Or serve it with a simple HTTP server:
```bash
cd web
python -m http.server 8000
```
   - Then open http://localhost:8000 in your browser

### Web Version Features

- WebGL-based metaballs using metaballs-js library
- News images fetched from GDELT API via Flask backend
- Images positioned and animated at metaball locations
- Custom metaball position tracker for synchronization
- Refresh and toggle controls for images

### Web Version Architecture

- **server.py**: Flask API that fetches news images from GDELT and serves them as base64 data
- **web/index.html**: Main HTML file with metaballs-js integration
- **web/metaballs-tracker.js**: Custom JavaScript class to track and simulate metaball movement

### Controls

- **Left click**: Add a metaball at mouse position
- **Right click**: Remove the nearest metaball
- **Drag with left button**: Move an existing metaball
- **Mouse wheel**: Change radius of the selected metaball
- **'g'**: Grayscale heightmap view
- **'c'**: Contour-line view with text
- **'+'/'-'**: Increase/decrease global field strength
- **'s'**: Save current view as PNG in `./renders_metaball/`
- **'r'**: Randomize metaball layout
- **ESC/'q'**: Quit

## Architecture

### Data Flow

1. **Reddit Fetcher**: Uses PRAW library to fetch posts from configured subreddits
2. **GDELT Fetcher**: Fetches news headlines and images from GDELT API with caching
3. **Semantic Database**: Stores and links content using ChromaDB with sentence embeddings
4. **Metaball Canvas**: Visualizes data as metaballs with text or image content

### Components

- `RedditFetcher`: Handles Reddit API interactions using PRAW
- `NewsImageFetcher`: Handles GDELT API requests with caching and error handling
- `SemanticDatabase`: ChromaDB integration for semantic content linking
- `MetaballCanvas`: Main visualization widget
- `MotionTracker`: Camera-based motion tracking (optional)

## Semantic Database

The application uses ChromaDB with sentence-transformers to create semantic links between:

- Reddit posts (stored with subreddit metadata)
- News headlines (stored with source URL and date)

This enables semantic similarity search between different content sources.

## Troubleshooting

### Reddit API Issues

- If you see rate limit errors, set up Reddit API credentials
- The app will fall back to read-only mode if credentials are not provided

### GDELT API Issues

- The GDELT API has built-in caching (5-minute TTL)
- Timeout is set to 15 seconds for requests
- Check your internet connection if fetches fail

### Semantic Database Issues

- The database is stored locally in `./chroma_db/`
- First run will download the sentence transformer model (~100MB)
- If initialization fails, the app will continue without semantic features

### Font Issues

- Custom font path: `Fonts/MD Thermochrome 0.4/Desktop/Regular/MDThermochrome0.4-Regular-Trial.otf`
- If the font file is not found, the app falls back to OpenCV's built-in font

## Dependencies

See `requirements.txt` for full list:
- PyQt5: GUI framework
- numpy: Numerical computations
- opencv-python: Image processing
- Pillow: Image handling and font rendering
- requests: HTTP requests
- praw: Reddit API wrapper
- chromadb: Vector database for semantic search
- sentence-transformers: Text embeddings

## License

This is a prototype for academic/diploma purposes.
