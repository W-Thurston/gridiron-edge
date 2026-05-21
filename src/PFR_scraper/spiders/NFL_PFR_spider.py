from datetime import date, datetime
import re
from typing import ClassVar

import pandas as pd
import scrapy
from scrapy import Spider
from scrapy.selector import Selector

from PFR_scraper.item import NFL_PFR_Upcoming_Schedule, NFL_ProFootballReference


class Historical_PFR_Spider(Spider):
    """This spider is used to pull historical NFL data from PFR."""

    name = "NFL_PFR_Historical_Bot"

    # pyrefly: ignore [bad-override]
    custom_settings: ClassVar[dict] = {
        "ITEM_PIPELINES": {"PFR_scraper.pipelines.HistoricalPFRWriteItemPipeline": 100}
    }

    allowed_urls: ClassVar[list[str]] = ["https://www.pro-football-reference.com"]
    start_urls: list[str] = [  # noqa: RUF012
        f"https://www.pro-football-reference.com/years/{x}/games.htm"
        for x in range(1991, date.today().year)
    ]

    # pyrefly: ignore [bad-override-mutable-attribute]
    def parse(self, response):
        rows = response.xpath("//tbody/tr")
        for row in rows:
            week_num = (
                row.xpath('./th[@data-stat="week_num"]/text()').extract_first()
                if row.xpath('./th[@data-stat="week_num"]/text()').extract_first()
                else "NULL_VALUE"
            )
            game_day_of_week = (
                row.xpath('./td[@data-stat="game_day_of_week"]/text()').extract_first()
                if row.xpath('./td[@data-stat="game_day_of_week"]/text()').extract_first()
                else "NULL_VALUE"
            )
            game_date = (
                row.xpath('./td[@data-stat="game_date"]/text()').extract_first()
                if row.xpath('./td[@data-stat="game_date"]/text()').extract_first()
                else "NULL_VALUE"
            )
            gametime = (
                row.xpath('./td[@data-stat="gametime"]/text()').extract_first()
                if row.xpath('./td[@data-stat="gametime"]/text()').extract_first()
                else "NULL_VALUE"
            )
            winner = (
                row.xpath('./td[@data-stat="winner"]/strong/a/text()').extract_first()
                if row.xpath('./td[@data-stat="winner"]/strong/a/text()').extract_first()
                else row.xpath('./td[@data-stat="winner"]/a/text()').extract_first()
                if row.xpath('./td[@data-stat="winner"]/a/text()').extract_first()
                else "NULL_VALUE"
            )
            game_location = (
                row.xpath('./td[@data-stat="game_location"]/text()').extract_first()
                if row.xpath('./td[@data-stat="game_location"]/text()').extract_first()
                else "NULL_VALUE"
            )
            loser = (
                row.xpath('./td[@data-stat="loser"]/a/text()').extract_first()
                if row.xpath('./td[@data-stat="loser"]/a/text()').extract_first()
                else "NULL_VALUE"
            )
            boxscore_link = (
                row.xpath('./td[@data-stat="boxscore_word"]/a/@href').extract_first()
                if row.xpath('./td[@data-stat="boxscore_word"]/a/@href').extract_first()
                else "NULL_VALUE"
            )
            pts_win = (
                row.xpath('./td[@data-stat="pts_win"]/strong/text()').extract_first()
                if row.xpath('./td[@data-stat="pts_win"]/strong/text()').extract_first()
                else "NULL_VALUE"
            )
            pts_lose = (
                row.xpath('./td[@data-stat="pts_lose"]/text()').extract_first()
                if row.xpath('./td[@data-stat="pts_lose"]/text()').extract_first()
                else "NULL_VALUE"
            )
            yards_win = (
                row.xpath('./td[@data-stat="yards_win"]/text()').extract_first()
                if row.xpath('./td[@data-stat="yards_win"]/text()').extract_first()
                else "NULL_VALUE"
            )
            to_win = (
                row.xpath('./td[@data-stat="to_win"]/text()').extract_first()
                if row.xpath('./td[@data-stat="to_win"]/text()').extract_first()
                else "NULL_VALUE"
            )
            yards_lose = (
                row.xpath('./td[@data-stat="yards_lose"]/text()').extract_first()
                if row.xpath('./td[@data-stat="yards_lose"]/text()').extract_first()
                else "NULL_VALUE"
            )
            to_lose = (
                row.xpath('./td[@data-stat="to_lose"]/text()').extract_first()
                if row.xpath('./td[@data-stat="to_lose"]/text()').extract_first()
                else "NULL_VALUE"
            )
            year = str(response.url)[-14:-10] + "-" + str(int(str(response.url)[-14:-10]) + 1)

            ## If the game has not been played yet, it will show 'preview' instead of boxscore, we don't want to waste time going to that page
            if row.xpath('./td[@data-stat="boxscore_word"]/a/text()').extract_first() == "preview":
                return
            elif (
                row.xpath('./td[@data-stat="boxscore_word"]/a/text()').extract_first() == "boxscore"
            ):
                boxscore_url = "https://www.pro-football-reference.com" + boxscore_link

                yield scrapy.Request(
                    boxscore_url,
                    callback=self.parse_boxscores,
                    meta={
                        "week_num_meta": week_num,
                        "game_day_of_week_meta": game_day_of_week,
                        "game_date_meta": game_date,
                        "gametime_meta": gametime,
                        "winner_meta": winner,
                        "game_location_meta": game_location,
                        "loser_meta": loser,
                        "boxscore_link_meta": boxscore_link,
                        "pts_win_meta": pts_win,
                        "pts_lose_meta": pts_lose,
                        "yards_win_meta": yards_win,
                        "to_win_meta": to_win,
                        "yards_lose_meta": yards_lose,
                        "to_lose_meta": to_lose,
                        "year_meta": year,
                    },
                )

    def parse_boxscores(self, response):
        item = NFL_ProFootballReference()
        item["week_num"] = response.meta["week_num_meta"]  # Week number in season
        item["game_day_of_week"] = response.meta["game_day_of_week_meta"]  # Day of week
        item["game_date"] = response.meta["game_date_meta"]  # Month & day
        item["gametime"] = response.meta["gametime_meta"]  # Game Time, EST
        item["winner"] = response.meta["winner_meta"]  # Winning team name
        item["game_location"] = response.meta["game_location_meta"]  # "@" if winning team was away
        item["loser"] = response.meta["loser_meta"]  # Losing team name
        item["boxscore_link"] = response.meta["boxscore_link_meta"]  # Link to boxscores
        item["pts_win"] = response.meta["pts_win_meta"]  # Points scored by winning team
        item["pts_lose"] = response.meta["pts_lose_meta"]  # Points scored by losing team
        item["yards_win"] = response.meta["yards_win_meta"]  # Yards gained by winning team
        item["to_win"] = response.meta["to_win_meta"]  # Turnovers by the winning team
        item["yards_lose"] = response.meta["yards_lose_meta"]  # Yards gained by losing team
        item["to_lose"] = response.meta["to_lose_meta"]  # Turnovers by the losing team
        item["year"] = response.meta["year_meta"]  # Ex: '91-92'
        if response.xpath('//div[@class="scorebox_meta"]/div/a/text()') == [] or (
            re.findall(
                r"\d+",
                response.xpath('//div[@class="scorebox_meta"]/div/a/text()').extract_first(),
            )
            != []
        ):
            item["stadium"] = "NULL_VALUE"
        else:
            item["stadium"] = (
                response.xpath(
                    '//body/div[1]/div[@class="box"]/div[@class="scorebox"]/div[@class="scorebox_meta"]/div/a/text()'
                ).extract_first()
                if response.xpath(
                    '//body/div[1]/div[@class="box"]/div[@class="scorebox"]/div[@class="scorebox_meta"]/div/a/text()'
                ).extract_first()
                else "NULL_VALUE"
            )

        ## Game info data -- Pull data from the "Game Info" grid on the boxscore pages
        ## Initialize thes values
        item["roof"] = None
        item["surface"] = None
        item["vegas_line"] = None
        item["over_under"] = None

        ## regex to remove HTML comments strings
        regex = re.compile(r"<!--(.*)-->", re.DOTALL)

        ## pull all comments from webpage and use above regex
        comments = response.xpath("//comment()").re(regex)

        ## Pull out the one comment we'd like from the above response
        comment_list = [x for x in comments if "div_game_info" in x]
        if len(comment_list) > 0:
            comment = comment_list[0]

            ## Makes it so we can use .xpath() on the above html string
            commentsel = Selector(text=comment, type="html")

            ## Pulls out all the row values for the "Game Info" grid on the boxscore pages (_info==left column, _stat==right column)
            game_data_info = [
                x.strip()
                for x in commentsel.xpath(
                    '//div[@id="div_game_info"]//th[@data-stat="info"]//text()'
                ).extract()
            ]
            game_data_stat = [
                x.strip()
                for x in commentsel.xpath(
                    '//div[@id="div_game_info"]//td[@data-stat="stat"]//text()'
                ).extract()
                if x not in ["(over)", "(under)"]
            ]

            ## Loop over "titles" for each row in the game_info grid and assign the appropriate value from the right column
            for idx, raw_val in enumerate(game_data_info):
                val: str = raw_val.lower()
                if val == "roof":
                    item["roof"] = game_data_stat[idx]
                elif val == "surface":
                    item["surface"] = game_data_stat[idx]
                elif val == "vegas line":
                    item["vegas_line"] = game_data_stat[idx]
                elif val == "over/under":
                    item["over_under"] = game_data_stat[idx]
                elif val in {
                    "weather*",
                    "weather",
                    "attendance",
                    "won toss",
                    "won ot toss",
                    "super bowl mvp",
                    "duration",
                }:
                    continue
                else:
                    print(f"This was an un-caught info in all_game_stats: {val}")

        ## If any of the values we want did not appear on the page, set it to "NULL_VALUE"
        for val in ["roof", "surface", "vegas_line", "over_under"]:
            if item[val] is None:
                item[val] = "NULL_VALUE"

        yield item


class Append_New_PFR_Spider(Spider):
    """This spider is used to pull historical NFL data from PFR."""

    name = "NFL_PFR_Append_New_Bot"

    # pyrefly: ignore [bad-override]
    custom_settings: ClassVar[dict] = {
        "ITEM_PIPELINES": {"PFR_scraper.pipelines.AppendNewPFRWriteItemPipeline": 100}
    }

    allowed_urls: ClassVar[list[str]] = ["https://www.pro-football-reference.com"]
    # start_urls: ClassVar[list[str]] = [f'https://www.pro-football-reference.com/years/{x}/games.htm' for x in range(1991,2023)]

    def __init__(self, year: str = "2023"):
        self.year = year
        self.start_urls = [f"https://www.pro-football-reference.com/years/{self.year}/games.htm"]
        self.cleaned_historical_data_file = "data/cleaned/NFL_wk_by_wk_cleaned.csv"

    # pyrefly: ignore [bad-override-mutable-attribute]
    def parse(self, response):
        df = pd.read_csv(self.cleaned_historical_data_file)

        pulled = df.loc[df["YEAR"].astype(str).str.startswith(self.year), "WEEK_NUM"]

        # numeric weeks present (1..22)
        # pyrefly: ignore [missing-attribute]
        pulled_int = pd.to_numeric(pulled, errors="coerce").dropna().astype(int).tolist()
        pulled_week_nums = set(pulled_int)

        # playoff labels to skip if their numeric week is present
        playoff_map: dict[int, set[str]] = {
            19: {"WildCard", "Wild Card"},
            20: {"Division", "Divisional"},
            21: {"ConfChamp", "Conf Champ", "Conference Championship"},
            22: {"SuperBowl", "Super Bowl"},
        }

        pulled_playoff_labels: set[str] = set()
        for wk, labels in playoff_map.items():
            if wk in pulled_week_nums:
                pulled_playoff_labels |= labels

        rows = response.xpath("//tbody/tr")
        for row in rows:
            week_num = (
                row.xpath('./th[@data-stat="week_num"]/text()').extract_first()
                if row.xpath('./th[@data-stat="week_num"]/text()').extract_first()
                else "NULL_VALUE"
            )
            week_num = week_num.strip()
            # Skip numeric weeks already pulled
            if week_num.isdigit() and int(week_num) in pulled_week_nums:
                continue

            # Skip playoff rounds already pulled
            if week_num in pulled_playoff_labels:
                continue

            game_day_of_week = (
                row.xpath('./td[@data-stat="game_day_of_week"]/text()').extract_first()
                if row.xpath('./td[@data-stat="game_day_of_week"]/text()').extract_first()
                else "NULL_VALUE"
            )
            game_date = (
                row.xpath('./td[@data-stat="game_date"]/text()').extract_first()
                if row.xpath('./td[@data-stat="game_date"]/text()').extract_first()
                else "NULL_VALUE"
            )
            gametime = (
                row.xpath('./td[@data-stat="gametime"]/text()').extract_first()
                if row.xpath('./td[@data-stat="gametime"]/text()').extract_first()
                else "NULL_VALUE"
            )
            winner = (
                row.xpath('./td[@data-stat="winner"]/strong/a/text()').extract_first()
                if row.xpath('./td[@data-stat="winner"]/strong/a/text()').extract_first()
                else row.xpath('./td[@data-stat="winner"]/a/text()').extract_first()
                if row.xpath('./td[@data-stat="winner"]/a/text()').extract_first()
                else "NULL_VALUE"
            )
            game_location = (
                row.xpath('./td[@data-stat="game_location"]/text()').extract_first()
                if row.xpath('./td[@data-stat="game_location"]/text()').extract_first()
                else "NULL_VALUE"
            )
            loser = (
                row.xpath('./td[@data-stat="loser"]/a/text()').extract_first()
                if row.xpath('./td[@data-stat="loser"]/a/text()').extract_first()
                else "NULL_VALUE"
            )
            boxscore_link = (
                row.xpath('./td[@data-stat="boxscore_word"]/a/@href').extract_first()
                if row.xpath('./td[@data-stat="boxscore_word"]/a/@href').extract_first()
                else "NULL_VALUE"
            )
            pts_win = (
                row.xpath('./td[@data-stat="pts_win"]/strong/text()').extract_first()
                if row.xpath('./td[@data-stat="pts_win"]/strong/text()').extract_first()
                else "NULL_VALUE"
            )
            pts_lose = (
                row.xpath('./td[@data-stat="pts_lose"]/text()').extract_first()
                if row.xpath('./td[@data-stat="pts_lose"]/text()').extract_first()
                else "NULL_VALUE"
            )
            yards_win = (
                row.xpath('./td[@data-stat="yards_win"]/text()').extract_first()
                if row.xpath('./td[@data-stat="yards_win"]/text()').extract_first()
                else "NULL_VALUE"
            )
            to_win = (
                row.xpath('./td[@data-stat="to_win"]/text()').extract_first()
                if row.xpath('./td[@data-stat="to_win"]/text()').extract_first()
                else "NULL_VALUE"
            )
            yards_lose = (
                row.xpath('./td[@data-stat="yards_lose"]/text()').extract_first()
                if row.xpath('./td[@data-stat="yards_lose"]/text()').extract_first()
                else "NULL_VALUE"
            )
            to_lose = (
                row.xpath('./td[@data-stat="to_lose"]/text()').extract_first()
                if row.xpath('./td[@data-stat="to_lose"]/text()').extract_first()
                else "NULL_VALUE"
            )
            year = str(response.url)[-14:-10] + "-" + str(int(str(response.url)[-14:-10]) + 1)

            if row.xpath('./td[@data-stat="boxscore_word"]/a/text()').extract_first() == "preview":
                return
            elif (
                row.xpath('./td[@data-stat="boxscore_word"]/a/text()').extract_first() == "boxscore"
            ):
                boxscore_url = "https://www.pro-football-reference.com" + boxscore_link

                yield scrapy.Request(
                    boxscore_url,
                    callback=self.parse_boxscores,
                    meta={
                        "week_num_meta": week_num,
                        "game_day_of_week_meta": game_day_of_week,
                        "game_date_meta": game_date,
                        "gametime_meta": gametime,
                        "winner_meta": winner,
                        "game_location_meta": game_location,
                        "loser_meta": loser,
                        "boxscore_link_meta": boxscore_link,
                        "pts_win_meta": pts_win,
                        "pts_lose_meta": pts_lose,
                        "yards_win_meta": yards_win,
                        "to_win_meta": to_win,
                        "yards_lose_meta": yards_lose,
                        "to_lose_meta": to_lose,
                        "year_meta": year,
                    },
                )

    def parse_boxscores(self, response):
        item = NFL_ProFootballReference()
        item["week_num"] = response.meta["week_num_meta"]  # Week number in season
        item["game_day_of_week"] = response.meta["game_day_of_week_meta"]  # Day of week
        item["game_date"] = response.meta["game_date_meta"]  # Month & day
        item["gametime"] = response.meta["gametime_meta"]  # Game Time, EST
        item["winner"] = response.meta["winner_meta"]  # Winning team name
        item["game_location"] = response.meta["game_location_meta"]  # "@" if winning team was away
        item["loser"] = response.meta["loser_meta"]  # Losing team name
        item["boxscore_link"] = response.meta["boxscore_link_meta"]  # Link to boxscores
        item["pts_win"] = response.meta["pts_win_meta"]  # Points scored by winning team
        item["pts_lose"] = response.meta["pts_lose_meta"]  # Points scored by losing team
        item["yards_win"] = response.meta["yards_win_meta"]  # Yards gained by winning team
        item["to_win"] = response.meta["to_win_meta"]  # Turnovers by the winning team
        item["yards_lose"] = response.meta["yards_lose_meta"]  # Yards gained by losing team
        item["to_lose"] = response.meta["to_lose_meta"]  # Turnovers by the losing team
        item["year"] = response.meta["year_meta"]  # Ex: '91-92'
        if response.xpath('//div[@class="scorebox_meta"]/div/a/text()') == [] or (
            re.findall(
                r"\d+",
                response.xpath('//div[@class="scorebox_meta"]/div/a/text()').extract_first(),
            )
            != []
        ):
            item["stadium"] = "NULL_VALUE"
        else:
            item["stadium"] = (
                response.xpath(
                    '//body/div[1]/div[@class="box"]/div[@class="scorebox"]/div[@class="scorebox_meta"]/div/a/text()'
                ).extract_first()
                if response.xpath(
                    '//body/div[1]/div[@class="box"]/div[@class="scorebox"]/div[@class="scorebox_meta"]/div/a/text()'
                ).extract_first()
                else "NULL_VALUE"
            )

        ## Game info data -- Pull data from the "Game Info" grid on the boxscore pages
        ## Initialize thes values
        item["roof"] = None
        item["surface"] = None
        item["vegas_line"] = None
        item["over_under"] = None

        ## regex to remove HTML comments strings
        regex = re.compile(r"<!--(.*)-->", re.DOTALL)

        ## pull all comments from webpage and use above regex
        comments = response.xpath("//comment()").re(regex)

        ## Pull out the one comment we'd like from the above response
        comment_list = [x for x in comments if "div_game_info" in x]
        if len(comment_list) > 0:
            comment = comment_list[0]

            ## Makes it so we can use .xpath() on the above html string
            commentsel = Selector(text=comment, type="html")

            ## Pulls out all the row values for the "Game Info" grid on the boxscore pages (_info==left column, _stat==right column)
            game_data_info = [
                x.strip()
                for x in commentsel.xpath(
                    '//div[@id="div_game_info"]//th[@data-stat="info"]//text()'
                ).extract()
            ]
            game_data_stat = [
                x.strip()
                for x in commentsel.xpath(
                    '//div[@id="div_game_info"]//td[@data-stat="stat"]//text()'
                ).extract()
                if x not in ["(over)", "(under)"]
            ]

            ## Loop over "titles" for each row in the game_info grid and assign the appropriate value from the right column
            for idx, raw_val in enumerate(game_data_info):
                val: str = raw_val.lower()
                if val == "roof":
                    item["roof"] = game_data_stat[idx]
                elif val == "surface":
                    item["surface"] = game_data_stat[idx]
                elif val == "vegas line":
                    item["vegas_line"] = game_data_stat[idx]
                elif val == "over/under":
                    item["over_under"] = game_data_stat[idx]
                elif val in {
                    "weather*",
                    "weather",
                    "attendance",
                    "won toss",
                    "won ot toss",
                    "super bowl mvp",
                    "duration",
                }:
                    continue
                else:
                    print(f"This was an un-caught info in all_game_stats: {val}")

        ## If any of the values we want did not appear on the page, set it to "NULL_VALUE"
        for val in ["roof", "surface", "vegas_line", "over_under"]:
            if item[val] is None:
                item[val] = "NULL_VALUE"

        yield item


class Upcoming_Schedule_NFLSpider(Spider):
    """This spider is used to the upcoming schedule for the current year from NFL data hosted on PFR."""

    name = "NFL_PFR_Upcoming_Bot"

    # pyrefly: ignore [bad-override]
    custom_settings: ClassVar[dict] = {
        "ITEM_PIPELINES": {"PFR_scraper.pipelines.UpcomingScheduleWriteItemPipeline": 100}
    }

    CURR_YEAR = datetime.now().year

    allowed_urls: ClassVar[list[str]] = ["https://www.pro-football-reference.com"]
    # pyrefly: ignore [bad-override]
    start_urls: ClassVar[list[str]] = [
        f"https://www.pro-football-reference.com/years/{CURR_YEAR}/games.htm"
    ]

    # pyrefly: ignore [bad-override-mutable-attribute]
    def parse(self, response):
        rows = response.xpath("//tbody/tr")
        for row in rows:
            item = NFL_PFR_Upcoming_Schedule()
            item["week_num"] = (
                row.xpath('./th[@data-stat="week_num"]/text()').extract_first()
                if row.xpath('./th[@data-stat="week_num"]/text()').extract_first()
                else "NULL_VALUE"
            )
            item["game_day_of_week"] = (
                row.xpath('./td[@data-stat="game_day_of_week"]/text()').extract_first()
                if row.xpath('./td[@data-stat="game_day_of_week"]/text()').extract_first()
                else "NULL_VALUE"
            )
            item["game_date"] = (
                row.xpath('./td[@data-stat="boxscore_word"]/a/text()').extract_first()
                if row.xpath('./td[@data-stat="boxscore_word"]/a/text()').extract_first()
                else "NULL_VALUE"
            )
            item["away_team"] = (
                row.xpath('./td[@data-stat="visitor_team"]/a/text()').extract_first()
                if row.xpath('./td[@data-stat="visitor_team"]/a/text()').extract_first()
                else "NULL_VALUE"
            )
            item["home_team"] = (
                row.xpath('./td[@data-stat="home_team"]/a/text()').extract_first()
                if row.xpath('./td[@data-stat="home_team"]/a/text()').extract_first()
                else "NULL_VALUE"
            )
            item["gametime"] = (
                row.xpath('./td[@data-stat="gametime"]/text()').extract_first()
                if row.xpath('./td[@data-stat="gametime"]/text()').extract_first()
                else "NULL_VALUE"
            )
            item["year"] = (
                str(response.url)[-14:-10] + "-" + str(int(str(response.url)[-14:-10]) + 1)
            )

            yield item
