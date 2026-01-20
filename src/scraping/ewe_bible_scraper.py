import asyncio
import json
import time
import re
import logging
from pathlib import Path
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler

from src.config.settings import AUDIO_DIR, TEXT_DIR, META_DIR

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EweBibleScraper:
    """
    Scraper Bible Ewe
    Sauvegarde UNIQUEMENT les données brutes dans data/raw/
    """

    def __init__(self, output_dir=None):
        if output_dir:
            self.root_dir = Path(output_dir)
            self.audio_dir = self.root_dir / "audio"
            self.text_dir = self.root_dir / "texts"
            self.meta_dir = self.root_dir / "metadata"
        else:
            self.audio_dir = AUDIO_DIR
            self.text_dir = TEXT_DIR
            self.meta_dir = META_DIR

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.base_text_url = "https://www.bible.com/fr/bible/3306/{book}.{chapter}.EB14"
        self.base_audio_url = "https://www.bible.com/fr/audio-bible/3306/{book}.{chapter}.EB14"
        self.books = {
            "GEN": {"name": "Genèse", "chapters": 50},
            "MAT": {"name": "Matthieu", "chapters": 28},
            "JHN": {"name": "Jean", "chapters": 21},
        }
        # self.books = {
        #     "GEN": {"name": "Mose I", "chapters": 50},
        #     "EXO": {"name": "Mose II", "chapters": 40},
        #     "LEV": {"name": "Mose III", "chapters": 27},
        #     "NUM": {"name": "Mose IV", "chapters": 36},
        #     "DEU": {"name": "Mose V", "chapters": 34},
        #     "JOS": {"name": "Yosua", "chapters": 24},
        #     "JDG": {"name": "Ɖelawo", "chapters": 21},
        #     "RUT": {"name": "Rut", "chapters": 4},
        #     "1SA": {"name": "Samuel I", "chapters": 31},
        #     "2SA": {"name": "Samuel II", "chapters": 24},
        #     "1KI": {"name": "Fiawo I", "chapters": 22},
        #     "2KI": {"name": "Fiawo II", "chapters": 25},
        #     "1CH": {"name": "Kronika I", "chapters": 29},
        #     "2CH": {"name": "Kronika II", "chapters": 36},
        #     "EZR": {"name": "Ezra ƒe Agbalẽ", "chapters": 10},
        #     "NEH": {"name": "Nexemya ƒe Agbalẽ", "chapters": 13},
        #     "TOB": {"name": "Tobit", "chapters": 14},
        #     "JDT": {"name": "Yudit", "chapters": 16},
        #     "ESG": {"name": "Esta G", "chapters": 11},
        #     "1MA": {"name": "1Makabeowo", "chapters": 16},
        #     "2MA": {"name": "2Makabeowo", "chapters": 15},
        #     "JOB": {"name": "Hiob ƒe Agbalẽ", "chapters": 42},
        #     "PSA": {"name": "Psalmowo", "chapters": 150},
        #     "PRO": {"name": "Salomo ƒe Lododowo", "chapters": 31},
        #     "ECC": {"name": "Nyagblɔla Salomo", "chapters": 12},
        #     "SNG": {"name": "Hawo ƒe ha", "chapters": 8},
        #     "WIS": {"name": "Salomo ƒe Nunya", "chapters": 19},
        #     "SIR": {"name": "EKLESIASTIKO", "chapters": 52},
        #     "ISA": {"name": "Yesaya", "chapters": 66},
        #     "JER": {"name": "Yeremya", "chapters": 52},
        #     "LAM": {"name": "Konyifahawo", "chapters": 5},
        #     "BAR": {"name": "Barux", "chapters": 7},
        #     "EZK": {"name": "Xezekiel", "chapters": 48},
        #     "DAG": {"name": "Daniɛl (Greek)", "chapters": 3},
        #     "HOS": {"name": "Hosea", "chapters": 14},
        #     "JOL": {"name": "Yoel", "chapters": 3},
        #     "AMO": {"name": "Amos", "chapters": 9},
        #     "OBA": {"name": "Obadya", "chapters": 1},
        #     "JON": {"name": "Yona", "chapters": 4},
        #     "MIC": {"name": "Mixa", "chapters": 7},
        #     "NAM": {"name": "Naxum", "chapters": 3},
        #     "HAB": {"name": "Xabakuk", "chapters": 3},
        #     "ZEP": {"name": "Zefanya", "chapters": 3},
        #     "HAG": {"name": "Xagai", "chapters": 2},
        #     "ZEC": {"name": "Zaxarya", "chapters": 14},
        #     "MAL": {"name": "Maleaxi", "chapters": 4},
        #     "MAT": {"name": "Mateo", "chapters": 28},
        #     "MRK": {"name": "Marko", "chapters": 16},
        #     "LUK": {"name": "Luka", "chapters": 24},
        #     "JHN": {"name": "Yohanes", "chapters": 21},
        #     "ACT": {"name": "Amedɔdɔawo ƒe Dɔwɔwɔwo", "chapters": 28},
        #     "ROM": {"name": "Romatɔwo", "chapters": 16},
        #     "1CO": {"name": "Korintotɔwo I", "chapters": 16},
        #     "2CO": {"name": "Korintotɔwo II", "chapters": 13},
        #     "GAL": {"name": "Galatiatɔwo", "chapters": 6},
        #     "EPH": {"name": "Efesotɔwo", "chapters": 6},
        #     "PHP": {"name": "Filipitɔwo", "chapters": 4},
        #     "COL": {"name": "Kolosetɔwo", "chapters": 4},
        #     "1TH": {"name": "Tesalonikatɔwo I", "chapters": 5},
        #     "2TH": {"name": "Tesalonikatɔwo II", "chapters": 3},
        #     "1TI": {"name": "Timoteo I", "chapters": 6},
        #     "2TI": {"name": "Timoteo II", "chapters": 4},
        #     "TIT": {"name": "Tito", "chapters": 3},
        #     "PHM": {"name": "Filemon", "chapters": 1},
        #     "HEB": {"name": "Hebritɔwo", "chapters": 13},
        #     "JAS": {"name": "Yakobo", "chapters": 5},
        #     "1PE": {"name": "Petro I", "chapters": 5},
        #     "2PE": {"name": "Petro II", "chapters": 3},
        #     "1JN": {"name": "Yohanes I", "chapters": 5},
        #     "2JN": {"name": "Yohanes II", "chapters": 1},
        #     "3JN": {"name": "Yohanes III", "chapters": 1},
        #     "JUD": {"name": "Yuda", "chapters": 1},
        #     "REV": {"name": "Nyaɖeɖefia", "chapters": 22},
        # }
        
        self.session = None
        self.records = []
        self._load_existing_records()

    def _load_existing_records(self):
        meta_file = self.meta_dir / "ewe_bible_raw.json"
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
        elements = soup.find_all(["span", "div"], {"data-usfm": True})

        for el in elements:
            usfm = el.get("data-usfm", "")
            parts = usfm.split(".")
            if len(parts) < 3:
                continue
            
            v_num = parts[2]
            v_key = v_num.split("-")[0]
            
            # Use only content spans, skip verse numbers
            text = ""
            for content in el.find_all("span", class_="content"):
                text += content.get_text(strip=True) + " "
            
            if not text:
                text = el.get_text(strip=True)
            
            text = re.sub(r'^\d+', '', text).strip()
            
            if text:
                if v_key not in verses_map:
                    verses_map[v_key] = text
                else:
                    verses_map[v_key] += " " + text

        # Final cleanup and formatting
        sorted_verses = []
        for v_num in sorted(verses_map.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0):
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

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    def save_corpus_data(self):
        out = self.meta_dir / "ewe_bible_raw.json"
        out.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"{len(self.records)} versets sauvegardés")
