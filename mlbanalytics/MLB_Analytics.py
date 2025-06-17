# ## Getting Data Via Scraping

#%% Setting up the notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import requests
from bs4 import BeautifulSoup #BeautifulSoup is a class in bs4
import pickle
import smtplib
import openpyxl
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from pretty_html_table import build_table
import datetime as dt


#from bs4 import exceptions as bs_exceptions
#import time
#import html5lib
#%% Defining variables used throughout the code

# Getting team abbreviations from mlb.com for hyperlinks
# order: [city abbrev, nickname]
team_abbrevs_dict = {
    'Cleveland Guardians':['CLE','guardians'],'Baltimore Orioles':['BAL','orioles'],'New York Yankees':['NYY','yankees'],
    'Boston Red Sox':['BOS','redsox'],'Toronto Blue Jays':['TOR','bluejays']
    }

# City abbreviations for data scraping
city_abbrevs = ['CLE','BAL','NYY','BOS','TBR']




# Set to True if retraining model
train_model = False

# Set to True if wanting to visualize stats using dist_visualizer function
viz_stats = False

# Getting the year of this season
this_season = dt.datetime.now().year

# Logic for getting the years we want - last 20 seasons
start_year = this_season - 20
increment = 1
years_sequence = range(start_year, this_season + 1, increment)

#Converting years_sequence to a list of strings
years_to_list = list(years_sequence)
years = [str(y) for y in years_to_list]

# %% Defining webscraping function
def html_scraper(url, tag_name, tbl_num, df_list):
    try:
        response_test = requests.get(url)

        if response_test.status_code == 200:
            html_contents = response_test.text
            print(f"Successfully retrieved HTML from {url}. Status code: {response_test.status_code}.")
        else:
            print(f"Failed to retrieve HTML from {url}. Status code: {response_test.status_code}.")

        # Now use BeautifulSoup to create the soup and begin cleaning the file
        # BeautifulSoup takes "markup" as argument (the contents) 
        # and another "parser" argument to specify the language of contents

        # soup is an object with certain attributes that can access various tags in the HTML code
        # The soup object contents are equivalent to the 

        soup = BeautifulSoup(html_contents, "html.parser")

        #soup = soup.prettify()
        #print(soup)

        ashtml = soup.select(tag_name)
        ashtml_tag = ashtml[0]

        tbl = ashtml_tag.contents
        tbl_string = str(tbl[tbl_num])

        dfs = pd.read_html(tbl_string)
        df_list.append(dfs[0])

    except:
        print(f"Skipped {url}. No data found.")

#%%

# Scraping the site for the data in the range of the years provided.
for year in years:
    try:
        url = f"https://www.baseball-reference.com/leagues/majors/{year}.shtml"
        # print(url)
        
        response_test = requests.get(url)

        if response_test.status_code == 200:
            html_contents = response_test.text
            print(f"Successfully retrieved HTML from url for the year {year}. Status code: {response_test.status_code}.")
        else:
            print(f"Failed to retrieve HTML from url for the year {year}. Status code: {response_test.status_code}.")

        # Now use BeautifulSoup to create the soup and begin cleaning the file
        # BeautifulSoup takes "markup" as argument (the contents) 
        # and another "parser" argument to specify the language of contents

        # soup is an object with certain attributes that can access various tags in the HTML code
        # The soup object contents are equivalent to the 

        soup = BeautifulSoup(html_contents, "html.parser")

        #soup = soup.prettify()
        #print(soup)

        ashtml_batting = soup.select("#all_teams_standard_batting")
        ashtml_batting_tag = ashtml_batting[0]
        tbl_batting = ashtml_batting_tag.contents
        tbl_batting_string = str(tbl_batting[3])

        ashtml_pitching = soup.select("#all_teams_standard_pitching")
        ashtml_pitching_tag = ashtml_pitching[0]
        tbl_pitching = ashtml_pitching_tag.contents
        tbl_pitching_string = str(tbl_pitching[4])

        dfs_bat = pd.read_html(tbl_batting_string)
        df_list_bat.append(dfs_bat[0])

        dfs_pitch = pd.read_html(tbl_pitching_string)
        df_list_pitch.append(dfs_pitch[0])
    except:
        print(f"Skipped the year {year}. The page has no data.")


#%% Scraping data using html_scraper

# Initializing a list to collect data frames at league level
leauge_bat_df_list = []
league_pitch_df_list = []
team_bat_df_list = []
team_pitch_df_list = []

# Urls for to scrape (pre-formatting)
team_bat_url = 'https://www.baseball-reference.com/teams/{team}/{this_season}-batting.shtml'
team_pitch_url = 'https://www.baseball-reference.com/teams/{team}/{this_season}-pitching.shtml'
league_url = "https://www.baseball-reference.com/leagues/majors/{year}.shtml"
league_urls = [league_url.format(year = y) for y in years]

# HTML tag names
team_bat_tag = '#all_team_batting'
team_pitch_tag = '#all_team_pitching'
all_teams_bat_tag = '#all_teams_standard_batting'
all_teams_pitch_tag = '#all_teams_standard_pitching'

# Index to grab from html contents
index_team_bat = 3
index_team_pitch = 0
index_all_teams_bat = 0
index_all_teams_pitch = 4

html_scraper(team_bat_url, team_bat_tag, index_team_bat, team_bat_df_list)
html_scraper(team_pitch_url, team_pitch_tag, index_team_pitch, team_pitch_df_list)
html_scraper(league_url, '#all_team_batting', 3, team_bat_df_list)
html_scraper(team_bat_url, '#all_team_batting', 3, team_bat_df_list)