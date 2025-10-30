#!/usr/bin/env python3
"""
Aletheia Agent - Idea Discovery

Consolidated into agent-mnemosyne for single-process deployment.
Finds and scores content ideas from multiple sources.

Web search for idea discovery (Singapore-compatible).
Since Claude's WebSearch is US-only, this uses alternative search APIs:
- Perplexity API (AI-powered research, excellent for Singapore)
- SerpApi (Google/Bing/DuckDuckGo results)
- Tavily API (AI-focused search)
- DuckDuckGo (free, no API key)
- Direct RSS feeds (always works)

Set your preferred method with environment variable: SEARCH_METHOD
"""

import os
import json
import yaml
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

# Try to import agent-sdk models
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "agent-sdk"))
    from agent_sdk.models.idea import IdeaModel, SourceType
except ImportError:
    print("Warning: agent-sdk not found, using basic data structures")
    IdeaModel = dict
    SourceType = None


class WebSearcher:
    """Web search that works in Singapore (and globally)."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            # Use MNEMOSYNE_DATA_DIR if set, otherwise default to ~/.mnemosyne
            data_dir_env = os.getenv("MNEMOSYNE_DATA_DIR")
            if data_dir_env:
                data_dir = Path(data_dir_env).expanduser()
            else:
                data_dir = Path.home() / ".mnemosyne"
            config_path = data_dir / "config" / "search-config.yaml"

        logger.info(f"Initializing WebSearcher with config: {config_path}")
        self.config = self._load_config(config_path)
        self.search_method = os.getenv("SEARCH_METHOD", "perplexity")
        logger.info(f"Search method: {self.search_method}")

    def _load_config(self, config_path: Path) -> dict:
        """Load search configuration."""
        if not config_path.exists():
            # Return default config
            return {
                "apis": {
                    "perplexity_key": os.getenv("PERPLEXITY_API_KEY", ""),
                    "serpapi_key": os.getenv("SERPAPI_KEY", ""),
                    "tavily_key": os.getenv("TAVILY_API_KEY", ""),
                },
                "rss_feeds": [
                    "https://www.bis.org/doclist/all_cbspeeches.rss",
                    "https://www.federalreserve.gov/feeds/press_all.xml",
                ],
                "search_params": {
                    "max_results": 10,
                    "days_back": 7,
                    "min_relevance": 0.6
                }
            }

        with open(config_path) as f:
            return yaml.safe_load(f)

    def search_topics(self, topics: List[str]) -> List[Dict]:
        """
        Search for content across multiple topics.

        Args:
            topics: List of topic queries

        Returns:
            List of discovered content with metadata
        """
        logger.info(f"Starting search for {len(topics)} topics using method: {self.search_method}")
        all_results = []

        for topic in topics:
            logger.info(f"Searching for topic: {topic}")
            print(f"Searching for: {topic}")

            try:
                if self.search_method == "perplexity":
                    results = self._search_perplexity(topic)
                elif self.search_method == "serpapi":
                    results = self._search_serpapi(topic)
                elif self.search_method == "tavily":
                    results = self._search_tavily(topic)
                elif self.search_method == "rss":
                    results = self._search_rss(topic)
                elif self.search_method == "reddit":
                    results = self._search_reddit(topic)
                elif self.search_method == "google" or self.search_method == "google-custom":
                    results = self._search_google_custom(topic)
                elif self.search_method == "duckduckgo" or self.search_method == "ddg":
                    results = self._search_duckduckgo(topic)
                elif self.search_method == "all":
                    # Use all available methods and combine results
                    logger.info("Using 'all' method - combining multiple search engines")
                    results = []
                    results.extend(self._search_perplexity(topic) if self.config["apis"].get("perplexity_key") else [])
                    results.extend(self._search_reddit(topic) if self.config["apis"].get("reddit_client_id") else [])
                    results.extend(self._search_google_custom(topic) if self.config["apis"].get("google_api_key") else [])
                    results.extend(self._search_rss(topic))
                else:
                    logger.warning(f"Unknown search method: {self.search_method}, defaulting to Perplexity")
                    print(f"Unknown search method: {self.search_method}, defaulting to Perplexity")
                    results = self._search_perplexity(topic)

                logger.info(f"Found {len(results)} results for topic: {topic}")
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error searching topic '{topic}': {str(e)}", exc_info=True)
                print(f"❌ Error searching '{topic}': {str(e)}")

        logger.info(f"Search complete. Total results: {len(all_results)}")
        return all_results

    def _search_perplexity(self, query: str) -> List[Dict]:
        """
        Search using Perplexity API (AI-powered research).

        Perplexity is excellent for research queries and works globally.
        Sign up at https://www.perplexity.ai/settings/api
        Pricing: $5/1M tokens (very affordable)
        """
        api_key = self.config["apis"].get("perplexity_key")
        if not api_key:
            print("Perplexity API key not found, skipping")
            return []

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Craft a research-focused query with 7-hour recency filter (6 hour run interval + 1 hour overlap)
        research_query = f"""Find articles, papers, and news from the LAST 7 HOURS about: {query}

IMPORTANT: Only include content published within the last 7 hours.

Focus on:
- Financial technology and banking
- Treasury management
- Payments and settlements
- Regulatory developments
- Industry analysis

Return the top 5 most relevant and recent sources with:
1. Article title
2. URL
3. Brief summary (2-3 sentences)
4. Publication date (must be within last 7 hours)"""

        payload = {
            "model": "sonar",  # Perplexity Sonar model with real-time web search
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant helping discover high-quality financial technology content. Provide factual, well-sourced information with URLs."
                },
                {
                    "role": "user",
                    "content": research_query
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Parse the response - Perplexity returns citations in the response
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = data.get("citations", [])

            results = []

            # Extract URLs from content (Perplexity includes them inline)
            import re
            url_pattern = r'https?://[^\s\)\]"]+'
            urls = re.findall(url_pattern, content)

            # Also use citations if available
            for citation in citations:
                url = citation if isinstance(citation, str) else citation.get("url", "")
                if url:
                    urls.append(url)

            # Deduplicate URLs
            seen_urls = set()
            for url in urls:
                if url not in seen_urls and url.startswith("http"):
                    seen_urls.add(url)
                    # Extract domain as source
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                    except:
                        domain = "perplexity"

                    results.append({
                        "title": f"Research result for: {query}",
                        "url": url,
                        "snippet": content[:500],  # Use first 500 chars as snippet
                        "source": domain,
                        "date": datetime.now().isoformat(),
                        "perplexity_content": content  # Store full research summary
                    })

            # Limit to max_results
            max_results = self.config["search_params"].get("max_results", 10)
            return results[:max_results]

        except Exception as e:
            print(f"Perplexity search error: {e}")
            return []

    def _search_serpapi(self, query: str) -> List[Dict]:
        """
        Search using SerpApi (Google results API).

        SerpApi works globally and provides Google search results.
        Sign up at https://serpapi.com (free tier: 100 searches/month)
        """
        api_key = self.config["apis"].get("serpapi_key")
        if not api_key:
            print("SerpApi key not found, skipping")
            return []

        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "num": self.config["search_params"]["max_results"],
            "gl": "sg",  # Singapore region
            "hl": "en",  # English language
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("organic_results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google",
                    "date": datetime.now().isoformat(),
                })

            return results

        except Exception as e:
            print(f"SerpApi error: {e}")
            return []

    def _search_tavily(self, query: str) -> List[Dict]:
        """
        Search using Tavily AI search API.

        Tavily is designed for AI agents and works globally.
        Sign up at https://tavily.com
        """
        api_key = self.config["apis"].get("tavily_key")
        if not api_key:
            print("Tavily API key not found, skipping")
            return []

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": self.config["search_params"]["max_results"],
            "search_depth": "advanced",
            "include_raw_content": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "source": "tavily",
                    "date": datetime.now().isoformat(),
                    "score": item.get("score", 0.0),
                })

            return results

        except Exception as e:
            print(f"Tavily error: {e}")
            return []

    def _search_rss(self, query: str) -> List[Dict]:
        """
        Search RSS feeds (no API key needed).

        This is the fallback option - always works but limited to RSS sources.
        """
        import feedparser

        logger.info(f"Starting RSS search for query: {query}")
        logger.info(f"Checking {len(self.config['rss_feeds'])} RSS feeds")

        results = []
        query_lower = query.lower()
        days_back = self.config["search_params"]["days_back"]
        cutoff_date = datetime.now() - timedelta(days=days_back)

        feeds_checked = 0
        feeds_with_results = 0

        for feed_url in self.config["rss_feeds"]:
            try:
                feeds_checked += 1
                print(f"  Checking RSS: {feed_url}")
                logger.debug(f"Parsing RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)

                feed_results = 0
                for entry in feed.entries:
                    # Check if query matches
                    title = entry.get("title", "").lower()
                    summary = entry.get("summary", "").lower()

                    if query_lower in title or query_lower in summary:
                        # Parse date
                        pub_date = entry.get("published_parsed")
                        if pub_date:
                            pub_datetime = datetime(*pub_date[:6])
                            if pub_datetime < cutoff_date:
                                continue

                        results.append({
                            "title": entry.get("title", ""),
                            "url": entry.get("link", ""),
                            "snippet": entry.get("summary", "")[:500],
                            "source": "rss",
                            "feed": feed_url,
                            "date": pub_datetime.isoformat() if pub_date else datetime.now().isoformat(),
                        })
                        feed_results += 1

                if feed_results > 0:
                    feeds_with_results += 1
                    logger.debug(f"Found {feed_results} results in feed: {feed_url}")

            except Exception as e:
                logger.warning(f"RSS error for {feed_url}: {str(e)}")
                print(f"  RSS error for {feed_url}: {e}")
                continue

        logger.info(f"RSS search complete: Checked {feeds_checked} feeds, {feeds_with_results} had matches, {len(results)} total results")
        return results

    def _search_duckduckgo(self, query: str) -> List[Dict]:
        """
        Search using DuckDuckGo (no API key needed).

        Uses duckduckgo-search library.
        Install: pip install duckduckgo-search
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("duckduckgo-search not installed. Run: pip install duckduckgo-search")
            return []

        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    query,
                    region="sg-en",
                    max_results=self.config["search_params"]["max_results"]
                ))

            results = []
            for item in search_results:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                    "source": "duckduckgo",
                    "date": datetime.now().isoformat(),
                })

            return results

        except Exception as e:
            print(f"DuckDuckGo error: {e}")
            return []

    def _search_reddit(self, query: str) -> List[Dict]:
        """
        Search Reddit for discussions and insights.

        Reddit is excellent for community insights, early trends, and technical discussions.
        Requires Reddit API credentials (free).
        """
        try:
            import praw
        except ImportError:
            print("praw not installed. Run: pip install praw")
            return []

        client_id = self.config["apis"].get("reddit_client_id")
        client_secret = self.config["apis"].get("reddit_client_secret")
        user_agent = self.config["apis"].get("reddit_user_agent", "mnemosyne/1.0")

        if not client_id or not client_secret:
            print("Reddit API credentials not found, skipping")
            return []

        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )

            results = []
            subreddits = self.config.get("reddit_subreddits", [])
            time_filter = self.config["search_params"].get("reddit_time_filter", "week")
            sort = self.config["search_params"].get("reddit_sort", "relevance")
            max_results = self.config["search_params"]["max_results"]

            # Search across configured subreddits
            for subreddit_name in subreddits[:5]:  # Limit to 5 subreddits to avoid API limits
                try:
                    subreddit = reddit.subreddit(subreddit_name)

                    # Search the subreddit
                    for submission in subreddit.search(query, sort=sort, time_filter=time_filter, limit=max_results):
                        results.append({
                            "title": submission.title,
                            "url": f"https://reddit.com{submission.permalink}",
                            "snippet": submission.selftext[:500] if submission.selftext else "",
                            "source": f"reddit:r/{subreddit_name}",
                            "date": datetime.fromtimestamp(submission.created_utc).isoformat(),
                            "score": submission.score,
                            "comments": submission.num_comments
                        })

                except Exception as e:
                    print(f"  Error searching r/{subreddit_name}: {e}")
                    continue

            return results[:max_results]

        except Exception as e:
            print(f"Reddit search error: {e}")
            return []

    def _search_google_custom(self, query: str) -> List[Dict]:
        """
        Search using Google Custom Search API with site-specific restrictions.

        This allows targeting specific high-quality sites (Bloomberg, Reuters, etc.)
        Requires Google API key and Custom Search Engine ID.
        Free tier: 100 queries/day
        """
        try:
            from googleapiclient.discovery import build
        except ImportError:
            print("google-api-python-client not installed. Run: pip install google-api-python-client")
            return []

        api_key = self.config["apis"].get("google_api_key")
        cx_id = self.config["apis"].get("google_cx_id")

        if not api_key or not cx_id:
            print("Google Custom Search credentials not found, skipping")
            return []

        try:
            service = build("customsearch", "v1", developerKey=api_key)

            # Determine which sites to search based on query topic
            site_list = self._get_relevant_sites_for_query(query)

            # Build site restriction query
            site_query = " OR ".join([f"site:{site}" for site in site_list[:10]])  # Max 10 sites
            full_query = f"{query} ({site_query})"

            # Execute search
            result = service.cse().list(
                q=full_query,
                cx=cx_id,
                num=self.config["search_params"]["max_results"]
            ).execute()

            results = []
            for item in result.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google:" + item.get("displayLink", ""),
                    "date": datetime.now().isoformat(),
                })

            return results

        except Exception as e:
            print(f"Google Custom Search error: {e}")
            return []

    def _get_relevant_sites_for_query(self, query: str) -> List[str]:
        """Determine which site list to use based on query content."""
        query_lower = query.lower()
        site_config = self.config.get("site_specific_searches", {})

        # Check query for keywords to determine best site list
        if any(word in query_lower for word in ["crypto", "bitcoin", "ethereum", "stablecoin", "defi", "tokenized"]):
            return site_config.get("crypto_sites", [])
        elif any(word in query_lower for word in ["treasury", "payment", "fintech", "banking"]):
            return site_config.get("fintech_sites", [])
        elif any(word in query_lower for word in ["ai", "machine learning", "artificial intelligence"]):
            return site_config.get("tech_sites", [])
        elif any(word in query_lower for word in ["central bank", "federal reserve", "regulation"]):
            return site_config.get("institutional_sites", [])
        else:
            # Default to financial news
            return site_config.get("financial_news_sites", [])


def main():
    """Test the web searcher."""
    # Load topics from config
    topics_file = Path(__file__).parent.parent.parent / "agent-aletheia" / "agent_aletheia" / "config" / "topics.yaml"

    if topics_file.exists():
        with open(topics_file) as f:
            topics_config = yaml.safe_load(f)

        # Extract topic keywords
        topics = []
        for topic in topics_config.get("primary_topics", []):
            topics.append(topic["name"])
    else:
        # Default topics
        topics = [
            "tokenized deposits",
            "stablecoin liquidity",
            "AI in finance",
            "treasury management",
        ]

    print(f"Searching for topics: {topics}")

    searcher = WebSearcher()
    results = searcher.search_topics(topics)

    print(f"\nFound {len(results)} results")

    # Save results
    output_file = Path.home() / ".mnemosyne" / "search-results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "topics": topics,
            "results": results,
            "total": len(results)
        }, f, indent=2)

    print(f"Saved to: {output_file}")

    # Show sample
    if results:
        print("\nSample result:")
        print(f"Title: {results[0]['title']}")
        print(f"URL: {results[0]['url']}")
        print(f"Snippet: {results[0]['snippet'][:200]}...")


# ============================================================================
# Aletheia Agent Wrapper
# ============================================================================

class AletheiaAgent:
    """
    Aletheia - Idea Discovery Agent

    Finds and scores content ideas from multiple sources.
    Consolidated into single process for direct function calls.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize Aletheia agent

        Args:
            data_dir: Path to Mnemosyne data directory
        """
        self.data_dir = data_dir
        self.searcher = WebSearcher()
        self.discovery_dir = data_dir / "discovery"

        # Ensure directories exist
        self.discovery_dir.mkdir(parents=True, exist_ok=True)

        logger.info("✓ Aletheia agent initialized")

    def discover_ideas(self, topics: List[str], num_ideas: int = 50) -> List[Dict]:
        """
        Run discovery across all sources

        Args:
            topics: List of topic queries to search
            num_ideas: Maximum number of ideas to return

        Returns:
            List of discovered ideas with metadata
        """
        logger.info(f"Starting discovery for {len(topics)} topics")

        # Search topics
        results = self.searcher.search_topics(topics)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        discovery_file = self.discovery_dir / f"discovery_{timestamp}.json"

        with open(discovery_file, 'w') as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "topics": topics,
                "results": results[:num_ideas],
                "total": len(results)
            }, f, indent=2)

        logger.info(f"✓ Discovery complete: {len(results)} results saved to {discovery_file}")

        return results[:num_ideas]

    def get_discovery_history(self, limit: int = 10) -> List[Dict]:
        """List recent discovery runs"""
        discoveries = sorted(self.discovery_dir.glob("discovery_*.json"), reverse=True)[:limit]

        history = []
        for f in discoveries:
            with open(f, 'r') as file:
                data = json.load(file)
                history.append({
                    "timestamp": data.get("generated_at"),
                    "topics": data.get("topics", []),
                    "total_results": data.get("total", 0),
                    "file": f.name
                })

        return history


if __name__ == "__main__":
    main()
