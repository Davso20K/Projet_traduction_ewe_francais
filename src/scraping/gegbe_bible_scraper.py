import asyncio
import json
import time
import logging
from pathlib import Path
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler

from src.config.settings import GEGBE_AUDIO_DIR, GEGBE_TEXT_DIR, GEGBE_META_DIR

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GegbeBibleScraper:
    """
    Scraper Bible Gegbe (Mina)
    Sauvegarde UNIQUEMENT les données brutes dans data/raw/gegbe/
    """

    def __init__(self, output_dir=None):
        if output_dir:
            self.root_dir = Path(output_dir)
            self.audio_dir = self.root_dir / "audio"
            self.text_dir = self.root_dir / "texts"
            self.meta_dir = self.root_dir / "metadata"
        else:
            self.audio_dir = GEGBE_AUDIO_DIR
            self.text_dir = GEGBE_TEXT_DIR
            self.meta_dir = GEGBE_META_DIR

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # Updated URLs for Gegbe (Mina) ID 2236 and suffix .GEN
        self.base_text_url = "https://www.bible.com/bible/2236/{book}.{chapter}.GEN"
        self.base_audio_url = "https://www.bible.com/audio-bible/2236/{book}.{chapter}.GEN"

        self.books = {
            "GEN": {"name": "Genèse", "chapters": 50},
            "MAT": {"name": "Matthieu", "chapters": 28},
            "JHN": {"name": "Jean", "chapters": 21},
        }

        self.session = None
        self.records = []

    # -----------------------------------------------------------------
    # HTTP
    # -----------------------------------------------------------------
    async def init_session(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            }
        )

    async def close_session(self):
        if self.session:
            await self.session.close()

    # -----------------------------------------------------------------
    # Text
    # -----------------------------------------------------------------
    async def extract_text(self, url: str):
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(
                    url=url,
                    wait_for="div[class*='Chapter'], .verse",
                    delay_before_return_html=3
                )
                if not result.success:
                    return []

                soup = BeautifulSoup(result.html, "html.parser")
                return self._parse_verses(soup)

        except Exception as e:
            logger.warning(f"Fallback texte {url}: {e}")
            return await self._fallback_text(url)

    async def _fallback_text(self, url):
        if not self.session:
            await self.init_session()

        async with self.session.get(url) as resp:
            if resp.status != 200:
                return []

            soup = BeautifulSoup(await resp.text(), "html.parser")
            return self._parse_verses(soup)

    def _parse_verses(self, soup):
        verses = []
        elements = soup.find_all(["span", "div"], {"data-usfm": True})

        for i, el in enumerate(elements, start=1):
            text = el.get_text(strip=True)
            if len(text) > 10:
                verses.append({
                    "verse": i,
                    "text": text,
                    "usfm": el.get("data-usfm", "")
                })

        return verses

    # -----------------------------------------------------------------
    # Audio
    # -----------------------------------------------------------------
    async def extract_audio_links(self, url: str):
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(
                    url=url,
                    wait_for="audio, source",
                    delay_before_return_html=5
                )
                if not result.success:
                    return []

                soup = BeautifulSoup(result.html, "html.parser")
                return self._parse_audio_links(soup, url)

        except Exception:
            return await self._fallback_audio(url)

    async def _fallback_audio(self, url):
        if not self.session:
            await self.init_session()

        async with self.session.get(url) as resp:
            if resp.status != 200:
                return []

            soup = BeautifulSoup(await resp.text(), "html.parser")
            return self._parse_audio_links(soup, url)

    def _parse_audio_links(self, soup, base_url):
        links = set()
        for el in soup.find_all(["audio", "source"]):
            src = el.get("src")
            if src:
                links.add(urljoin(base_url, src))
        return list(links)

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------
    async def download_audio(self, url: str, filename: str):
        if not self.session:
            await self.init_session()

        async with self.session.get(url) as resp:
            if resp.status != 200:
                return None

            path = self.audio_dir / filename
            path.write_bytes(await resp.read())
            return str(path)

    # -----------------------------------------------------------------
    # Chapter
    # -----------------------------------------------------------------
    async def process_chapter(self, book_code: str, chapter: int):
        logger.info(f"{book_code} chapitre {chapter}")

        text_url = self.base_text_url.format(book=book_code, chapter=chapter)
        audio_url = self.base_audio_url.format(book=book_code, chapter=chapter)

        verses = await self.extract_text(text_url)
        audio_links = await self.extract_audio_links(audio_url)

        for v in verses:
            audio_file = f"{book_code.lower()}_{chapter:02d}_{v['verse']:02d}.mp3"

            audio_path = None
            if audio_links:
                audio_path = await self.download_audio(audio_links[0], audio_file)

            # texte brut
            text_path = self.text_dir / audio_file.replace(".mp3", ".txt")
            text_path.write_text(v["text"], encoding="utf-8")

            self.records.append({
                "book": book_code,
                "chapter": chapter,
                "verse": v["verse"],
                "text": v["text"],
                "audio_path": audio_path,
                "text_url": text_url,
                "audio_url": audio_url,
                "timestamp": time.time()
            })

        await asyncio.sleep(2)

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    def save_corpus_data(self):
        out = self.meta_dir / "gegbe_bible_raw.json"
        out.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"{len(self.records)} versets sauvegardés")


if __name__ == "__main__":
    async def main():
        scraper = GegbeBibleScraper()
        try:
            # Test simple sur un chapitre
            await scraper.process_chapter("GEN", 1)
            scraper.save_corpus_data()
        finally:
            await scraper.close_session()

    asyncio.run(main())
