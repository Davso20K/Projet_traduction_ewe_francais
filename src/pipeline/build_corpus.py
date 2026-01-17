import asyncio
from src.scraping.ewe_bible_scraper import EweBibleScraper

async def run():
    scraper = EweBibleScraper(output_dir="data/raw/ewe_bible")
    await scraper.init_session()

    for book_code, book_info in scraper.books.items():
        for chapter in range(1, book_info["chapters"] + 1):
            await scraper.process_chapter(book_code, chapter)

    scraper.save_corpus_data()
    await scraper.close_session()

if __name__ == "__main__":
    asyncio.run(run())
