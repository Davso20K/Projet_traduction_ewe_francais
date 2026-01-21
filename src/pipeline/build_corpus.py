import asyncio
import sys
from src.scraping.ewe_bible_scraper import EweBibleScraper
from src.scraping.gegbe_bible_scraper import GegbeBibleScraper

async def scrape_language(scraper_class, lang_name):
    print(f"--- Scraping {lang_name} ---")
    scraper = scraper_class()  # Utilise les chemins par défaut de settings.py
    await scraper.init_session()

    try:
        for book_code, book_info in scraper.books.items():
            for chapter in range(1, book_info["chapters"] + 1):
                await scraper.process_chapter(book_code, chapter)
        
        scraper.save_corpus_data()
    finally:
        await scraper.close_session()

async def run(lang=None):
    if lang == "ewe":
        await scrape_language(EweBibleScraper, "Ewe")
    elif lang == "gegbe":
        await scrape_language(GegbeBibleScraper, "Gegbe")
    else:
        # Scrape les deux
        await scrape_language(EweBibleScraper, "Ewe")
        await scrape_language(GegbeBibleScraper, "Gegbe")
        
        # Lance l'alignement parallèle
        from src.preprocessing.parallel_aligner import ParallelAligner
        print("--- Alignement du corpus parallèle ---")
        aligner = ParallelAligner()
        aligner.align()

if __name__ == "__main__":
    # Permet de passer la langue en argument : python -m src.pipeline.build_corpus ewe
    target_lang = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run(target_lang))

