# src/PFR_scraper/pipelines.py
# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import io


class HistoricalPFRWriteItemPipeline:
    """Scrapy item pipeline that writes historical PFR game data to CSV (write mode)."""

    def __init__(self) -> None:
        self.filename = "data/raw/NFL_wk_by_wk.csv"
        self.file: io.TextIOWrapper | None = None

    def open_spider(self, spider) -> None:
        """Open the output CSV file for writing."""
        self.file = open(self.filename, "w")  # noqa: SIM115

    def close_spider(self, spider) -> None:
        """Close the output CSV file."""
        if self.file is not None:
            self.file.close()

    def process_item(self, item, spider):
        """Write a scraped item as a CSV row."""
        line: str = ",".join(item.values()) + "\n"
        if self.file is not None:
            self.file.write(line)
        return item


class AppendNewPFRWriteItemPipeline:
    """Scrapy item pipeline that appends new PFR game data to CSV (append mode)."""

    def __init__(self) -> None:
        self.filename = "data/raw/NFL_wk_by_wk.csv"
        self.file: io.TextIOWrapper | None = None

    def open_spider(self, spider) -> None:
        """Open the output CSV file for appending."""
        self.file = open(self.filename, "a")  # noqa: SIM115

    def close_spider(self, spider) -> None:
        """Close the output CSV file."""
        if self.file is not None:
            self.file.close()

    def process_item(self, item, spider):
        """Write a scraped item as a CSV row."""
        line: str = ",".join(item.values()) + "\n"
        if self.file is not None:
            self.file.write(line)
        return item


class UpcomingScheduleWriteItemPipeline:
    """Scrapy item pipeline that writes upcoming schedule data to CSV (write mode)."""

    def __init__(self) -> None:
        self.filename = "data/raw/NFL_upcoming_schedule.csv"
        self.file: io.TextIOWrapper | None = None

    def open_spider(self, spider) -> None:
        """Open the output CSV file for writing."""
        self.file = open(self.filename, "w")  # noqa: SIM115

    def close_spider(self, spider) -> None:
        """Close the output CSV file."""
        if self.file is not None:
            self.file.close()

    def process_item(self, item, spider):
        """Write a scraped item as a CSV row."""
        line: str = ",".join(item.values()) + "\n"
        if self.file is not None:
            self.file.write(line)
        return item
