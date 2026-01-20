import asyncio
import json
import time
import logging
import re
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
            "GEN": {"name": "Gɔ̃mèjèje be Xoma", "chapters": 50},
            "EXO": {"name": "Toto jo", "chapters": 40},
            "LEV": {"name": "Levìwo", "chapters": 27},
            "NUM": {"name": "Àmè Hɛ̃hlɛ̃", "chapters": 36},
            "DEU": {"name": "Èsea gbìgbɔ̀ hlɛ̃", "chapters": 34},
            "JOS": {"name": "Yosùa", "chapters": 24},
            "JDG": {"name": "Kòjoɖotɔwo", "chapters": 21},
            "RUT": {"name": "Rut", "chapters": 4},
            "1SA": {"name": "1Samuɛl", "chapters": 31},
            "2SA": {"name": "2Samuɛl", "chapters": 24},
            "1KI": {"name": "1Èfìɔwo", "chapters": 22},
            "2KI": {"name": "2Èfìɔ", "chapters": 25},
            "1CH": {"name": "1Kronikà", "chapters": 29},
            "2CH": {"name": "2Kronikà", "chapters": 36},
            "EZR": {"name": "Ɛzrà", "chapters": 10},
            "NEH": {"name": "Nèhemìa", "chapters": 13},
            "TOB": {"name": "TOBÌ", "chapters": 14},
            "JDT": {"name": "Yudìt", "chapters": 16},
            "ESG": {"name": "Ɛstà-G", "chapters": 18},
            "1MA": {"name": "1Màkàbeòwo", "chapters": 16},
            "2MA": {"name": "2Màkàbeòwo", "chapters": 15},
            "JOB": {"name": "Yɔb", "chapters": 42},
            "PSA": {"name": "Èhàwo", "chapters": 150},
            "PRO": {"name": "Èlododowo", "chapters": 31},
            "ECC": {"name": "Àɖàŋùɖètɔ", "chapters": 12},
            "SNG": {"name": "Èhàwo be Èhà", "chapters": 8},
            "WIS": {"name": "Ànyasã", "chapters": 19},
            "SIR": {"name": "Sirasid", "chapters": 51},
            "ISA": {"name": "Ezayà", "chapters": 66},
            "JER": {"name": "Yeremìa", "chapters": 52},
            "LAM": {"name": "Àlenanawo", "chapters": 5},
            "BAR": {"name": "Bàruk", "chapters": 5},
            "EZK": {"name": "Ezekìɛl", "chapters": 48},
            "DAG": {"name": "Dàniɛl-G", "chapters": 14},
            "HOS": {"name": "Òzeà", "chapters": 14},
            "JOL": {"name": "Yoɛ̀l", "chapters": 4},
            "AMO": {"name": "Àmos", "chapters": 9},
            "OBA": {"name": "Obadìa", "chapters": 1},
            "JON": {"name": "Yonà", "chapters": 4},
            "MIC": {"name": "Mikà", "chapters": 7},
            "NAM": {"name": "Nàhum", "chapters": 3},
            "HAB": {"name": "Hàbakuk", "chapters": 3},
            "ZEP": {"name": "Sèfanìa", "chapters": 3},
            "HAG": {"name": "Hagài", "chapters": 2},
            "ZEC": {"name": "Zakarìa", "chapters": 14},
            "MAL": {"name": "Malakìa", "chapters": 3},
            "MAT": {"name": "Màteo", "chapters": 28},
            "MRK": {"name": "Markò", "chapters": 16},
            "LUK": {"name": "Lukà", "chapters": 24},
            "JHN": {"name": "Yòhanɛ̀s", "chapters": 21},
            "ACT": {"name": "Èdɔwɔ̀wɔwo", "chapters": 28},
            "ROM": {"name": "Romàtɔwo", "chapters": 16},
            "1CO": {"name": "1Kòrɛ̃tòtɔwo", "chapters": 16},
            "2CO": {"name": "2Kòrɛ̃tòtɔwo", "chapters": 13},
            "GAL": {"name": "Galatìatɔwɔ", "chapters": 6},
            "EPH": {"name": "Efesòtɔwo", "chapters": 6},
            "PHP": {"name": "Fìlipìtɔwo", "chapters": 4},
            "COL": {"name": "Kolosètɔwo", "chapters": 4},
            "1TH": {"name": "1Tɛsalonikà", "chapters": 5},
            "2TH": {"name": "2Tɛsalonikà", "chapters": 3},
            "1TI": {"name": "1Timòteo", "chapters": 6},
            "2TI": {"name": "2Timòteo", "chapters": 4},
            "TIT": {"name": "Titò", "chapters": 3},
            "PHM": {"name": "Filemɔ̀n", "chapters": 1},
            "HEB": {"name": "Hebrùwo", "chapters": 13},
            "JAS": {"name": "Yakobò", "chapters": 5},
            "1PE": {"name": "1Petrò", "chapters": 5},
            "2PE": {"name": "2Petrò", "chapters": 3},
            "1JN": {"name": "1Yòhanɛ̀s", "chapters": 5},
            "2JN": {"name": "2Yòhanɛ̀s", "chapters": 1},
            "3JN": {"name": "3Yòhanɛ̀s", "chapters": 1},
            "JUD": {"name": "Yudà", "chapters": 1},
            "REV": {"name": "Àvìmènu", "chapters": 22},
        }

        self.session = None
        self.records = []
        self._load_existing_records()

    def _load_existing_records(self):
        meta_file = self.meta_dir / "gegbe_bible_raw.json"
        if meta_file.exists():
            try:
                self.records = json.loads(meta_file.read_text(encoding="utf-8"))
                logger.info(f"Chargé {len(self.records)} enregistrements existants.")
            except Exception as e:
                logger.warning(f"Erreur chargement metadata: {e}")

    # -----------------------------------------------------------------
    # HTTP
    # -----------------------------------------------------------------
    async def init_session(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
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
        verses_map = {}
        # Bible.com uses spans with class 'verse' or data-usfm
        elements = soup.find_all(["span", "div"], {"data-usfm": True})

        for el in elements:
            usfm = el.get("data-usfm", "")
            parts = usfm.split(".")
            if len(parts) < 3:
                continue
            
            v_num = parts[2]
            v_key = v_num.split("-")[0] # Segment support (e.g. 1-2)
            
            # Use only content spans, skip verse numbers
            text = ""
            for content in el.find_all("span", class_="content"):
                text += content.get_text(strip=True) + " "
            
            if not text:
                text = el.get_text(strip=True)
            
            # Cleanup: remove leading numbers and extra spaces
            text = re.sub(r'^\d+', '', text).strip()
            
            if text:
                if v_key not in verses_map:
                    verses_map[v_key] = text
                else:
                    verses_map[v_key] += " " + text

        # Final cleanup and formatting
        sorted_verses = []
        for v_num in sorted(verses_map.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0):
            # Clean up double spaces
            v_text = re.sub(r'\s+', ' ', verses_map[v_num]).strip()
            sorted_verses.append({
                "verse": v_num,
                "text": v_text,
                "usfm": v_num
            })

        return sorted_verses

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
        try:
            # Check if already in records
            if any(r["book"] == book_code and r["chapter"] == chapter for r in self.records):
                logger.info(f"Sauter {book_code} {chapter} (déjà dans metadata)")
                return

            logger.info(f"Traitement {book_code} chapitre {chapter}")

            audio_filename = f"{book_code.lower()}_{chapter:02d}.mp3"
            audio_path_local = self.audio_dir / audio_filename

            text_url = self.base_text_url.format(book=book_code, chapter=chapter)
            audio_url = self.base_audio_url.format(book=book_code, chapter=chapter)

            verses = await self.extract_text(text_url)
            if not verses:
                logger.warning(f"Pas de texte trouvé pour {text_url}")
                return

            # Audio
            audio_links = await self.extract_audio_links(audio_url)
            downloaded_audio_path = None
            
            if audio_links:
                if not audio_path_local.exists():
                    try:
                        downloaded_audio_path = await self.download_audio(audio_links[0], audio_filename)
                    except Exception as e:
                        logger.error(f"Erreur download audio {audio_links[0]}: {e}")
                else:
                    downloaded_audio_path = str(audio_path_local)

            for v in verses:
                v_id = v["verse"]
                safe_v_id = re.sub(r'\D', '_', v_id)
                text_file = f"{book_code.lower()}_{chapter:02d}_{safe_v_id}.txt"
                (self.text_dir / text_file).write_text(v["text"], encoding="utf-8")

                self.records.append({
                    "book": book_code,
                    "chapter": chapter,
                    "verse": v_id,
                    "text": v["text"],
                    "audio_path": downloaded_audio_path,
                    "text_url": text_url,
                    "audio_url": audio_url,
                    "timestamp": time.time()
                })

            # Save metadata incrementally
            self.save_corpus_data()
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Erreur fatale sur {book_code} {chapter}: {e}")
            # Ne pas re-lever pour permettre de continuer sur les autres chapitres

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
        target_books = ["GEN", "MAT", "JHN"]
        try:
            for book_code in target_books:
                if book_code in scraper.books:
                    book_info = scraper.books[book_code]
                    logger.info(f"Début du scraping pour {book_info['name']} ({book_code})")
                    for chapter in range(1, book_info["chapters"] + 1):
                        await scraper.process_chapter(book_code, chapter)
            
            scraper.save_corpus_data()
        finally:
            await scraper.close_session()

    asyncio.run(main())
