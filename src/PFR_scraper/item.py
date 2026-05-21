# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

# class NflPfrItem(scrapy.Item):
from scrapy import Field, Item


class NFL_ProFootballReference(Item):
    week_num = Field()  # Week number in season
    game_day_of_week = Field()  # Day of week
    game_date = Field()  # Month & day
    gametime = Field()  # Game Time, EST
    winner = Field()  # Winning team name
    game_location = Field()  # "@" if winning team was away
    loser = Field()  # Losing team name
    boxscore_link = Field()  # Link to boxscores
    pts_win = Field()  # Points scored by winning team
    pts_lose = Field()  # Points scored by losing team
    yards_win = Field()  # Yards gained by winning team
    to_win = Field()  # Turnovers by the winning team
    yards_lose = Field()  # Yards gained by losing team
    to_lose = Field()  # Turnovers by the losing team
    year = Field()  # Year (ex: '18-'19)
    stadium = Field()  # The name of the stadium the game was played in
    roof = Field()  # Type of roof the stadium has
    surface = Field()  # Type of surface the field has (ie. grass, turf, etc)
    vegas_line = Field()  # Vegas betting line at kickoff
    over_under = Field()  # Vegas over/under at kickoff


class NFL_PFR_Upcoming_Schedule(Item):
    week_num = Field()  # Week number in season
    game_day_of_week = Field()  # Day of week
    game_date = Field()  # Month & day
    home_team = Field()  # Winning team name
    away_team = Field()  # Losing team name
    gametime = Field()  # Game Time, EST
    year = Field()  # Year (ex: '18-'19)
