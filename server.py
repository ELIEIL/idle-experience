#!/usr/bin/env python3
"""
Flask API server for serving news images and metaball data.
"""

import base64
import json
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
NEWSAPI_API_KEY = "ce989e260bd44119b3f4e043b6989335"  # Get your API key from https://newsapi.org/
NEWSAPI_API_URL = "https://newsapi.org/v2/everything"
IMAGE_CACHE = {}
CACHE_TTL = 300  # 5 minutes


class NewsImageFetcher:
    """Fetches news images from NewsAPI.org for web API."""

    def __init__(self):
        self.api_url = NEWSAPI_API_URL
        self.api_key = NEWSAPI_API_KEY
        self._cache = {}
        self._cache_ttl = 300

    def fetch_articles(self, keyword: str = "news", limit: int = 10):
        """Fetch articles from NewsAPI.org."""
        cache_key = f"{keyword}_{limit}"

        if cache_key in self._cache and time.time() - self._cache[cache_key]['timestamp'] < self._cache_ttl:
            return self._cache[cache_key]['data']

        if not self.api_key:
            print("WARNING: NEWSAPI_API_KEY not set. Please get a free API key from https://newsapi.org/")
            return []

        try:
            params = {
                "apiKey": self.api_key,
                "q": keyword,
                "pageSize": limit,
                "language": "en",
                "sortBy": "publishedAt"
            }
            response = requests.get(self.api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])

            self._cache[cache_key] = {
                'data': articles,
                'timestamp': time.time()
            }

            return articles
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("ERROR: Invalid NewsAPI key. Please check your API key.")
            elif e.response.status_code == 429:
                print("ERROR: NewsAPI rate limit exceeded. Free tier: 100 requests/day.")
            else:
                print(f"HTTP Error fetching articles: {e}")
            return []
        except Exception as e:
            print(f"Failed to fetch articles: {e}")
            return []

    def load_image_as_base64(self, url: str, target_size=(200, 200)):
        """Load an image from URL and return as base64 string."""
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
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return img_base64
        except Exception as e:
            print(f"Failed to load image from {url}: {e}")
        return None


image_fetcher = NewsImageFetcher()


@app.route('/api/images', methods=['GET'])
def get_news_images():
    """Get news images as base64 data."""
    keyword = request.args.get('keyword', 'news')
    limit = int(request.args.get('limit', 10))

    articles = image_fetcher.fetch_articles(keyword=keyword, limit=limit)
    images_data = []

    for article in articles:
        url_to_image = article.get('urlToImage')
        if url_to_image:
            img_base64 = image_fetcher.load_image_as_base64(url_to_image)
            if img_base64:
                images_data.append({
                    'url': url_to_image,
                    'title': article.get('title', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'data': img_base64
                })
        if len(images_data) >= limit:
            break

    return jsonify({
        'images': images_data,
        'count': len(images_data)
    })


@app.route('/api/metaballs', methods=['GET'])
def get_metaballs():
    """Get metaball configuration."""
    num_metaballs = int(request.args.get('count', 5))
    keyword = request.args.get('keyword', 'news')

    # Generate random metaball positions (0-100 scale)
    import random
    metaballs = []
    for i in range(num_metaballs):
        metaballs.append({
            'x': random.uniform(10, 90),
            'y': random.uniform(10, 90),
            'r': random.uniform(3, 7.5)
        })

    return jsonify({
        'metaballs': metaballs,
        'count': len(metaballs)
    })


@app.route('/api/headlines', methods=['GET'])
def get_headlines():
    """Get news headlines without images."""
    keyword = request.args.get('keyword', 'world news')
    limit = int(request.args.get('limit', 3))
    articles = image_fetcher.fetch_articles(keyword=keyword, limit=limit * 4)
    headlines = []
    for article in articles:
        title = article.get('title', '')
        if '[Removed]' in title or not title:
            continue
        if ' - ' in title:
            title = title.rsplit(' - ', 1)[0]
        headlines.append({
            'title': title,
            'source': article.get('source', {}).get('name', ''),
            'url': article.get('url', '')
        })
        if len(headlines) >= limit:
            break
    return jsonify({'headlines': headlines, 'count': len(headlines)})


@app.route('/api/reddit', methods=['GET'])
def get_reddit_posts():
    """Proxy Reddit posts to avoid CORS issues."""
    subreddit = request.args.get('subreddit', 'worldnews')
    limit = int(request.args.get('limit', 8))

    try:
        headers = {'User-Agent': 'MetaballApp/1.0 (educational prototype)'}
        response = requests.get(
            f'https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}',
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        posts = []
        for child in data['data']['children']:
            post = child['data']
            if post.get('stickied'):
                continue
            posts.append({
                'title': post.get('title', ''),
                'author': post.get('author', ''),
                'score': post.get('score', 0),
                'subreddit': post.get('subreddit', ''),
                'num_comments': post.get('num_comments', 0),
                'selftext': post.get('selftext', '')[:200]
            })
            if len(posts) >= limit:
                break

        return jsonify({'posts': posts, 'count': len(posts)})
    except Exception as e:
        print(f"Reddit fetch error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("Starting Flask API server on http://localhost:5001")
    if not NEWSAPI_API_KEY:
        print("WARNING: NEWSAPI_API_KEY not set in server.py")
        print("Get a free API key from https://newsapi.org/")
    app.run(host='0.0.0.0', port=5001, debug=True)
