# Will house modules for NHL analytics
from dataclasses import dataclass

@dataclass
class NHLSeason:
    year: int

    def get_team_stats(self, cleaned = True):
        import requests
        import pandas as pd
        from bs4 import BeautifulSoup #BeautifulSoup is a class in bs4

        # Scraping the site for the data in the range of the years provided.

        # NHL lockout 2004-2005 season (no data)
        # if self.year != 2005:
        url = f"https://www.hockey-reference.com/leagues/NHL_{str(self.year)}.html#all_stats"

        try:
            # Importing HTML data from Hockey Reference using requests
            response_test = requests.get(url)

            # response_test.status_code == 200:
            nhl_html_contents = response_test.text

            # Now use BeautifulSoup to create the soup and begin cleaning the file
            # BeautifulSoup takes "markup" as argument (the contents) 
            # and another "parser" argument to specify the language of contents

            # soup is an object with certain attributes that can access various tags in the HTML code
            # The soup object contents are equivalent to the 

            soup = BeautifulSoup(nhl_html_contents, "html.parser")

            #soup = soup.prettify()
            #print(soup)

            ashtml = soup.select("#all_stats")
            ashtml_var = ashtml[0]
            tbl = ashtml_var.contents
            tbl_string = str(tbl[4])

            df_season = pd.read_html(tbl_string)[0]
            msg= f"Successfully retrieved HTML from url for the year {self.year}. Status code: {response_test.status_code}."
        
            if cleaned:
                # Clean the DataFrame
                df_season = clean_nhl(df_season, [str(self.year)])
        except:
            df_season = pd.DataFrame() # Empty DataFrame for unsuccessful imports
            msg = f"Failed to retrieve HTML from url for the year {self.year}. Status code: {response_test.status_code}."

        # else:
            # print(f"Skipped the year {self.year}. The page has no data.")
        
        return df_season, msg
            

@dataclass
class TeamSeasonData:
    year: str
    team: str
    rank: int
    games_played: int
    wins: int
    losses: int
    ot_losses: int
    points: int
    goals_for: int
    goals_against: int
    goal_difference_per_game: float
    power_play_percentage: float
    penalty_kill_percentage: float
    shots_for_per_game: float
    shots_against_per_game: float
    save_percentage: float
    faceoff_percentage: float
    hits_per_game: float
    blocked_shots_per_game: float

def hock_ref_team_play_data(start_year, end_year):
    import requests
    import pandas as pd
    from bs4 import BeautifulSoup #BeautifulSoup is a class in bs4

    # Initializing a list to collect data frames
    df_list = []

    # Error list
    scrape_errors = []

    # Generating a list of years to scrape data for
    years = [str(year) for year in range(int(start_year), int(end_year) + 1)]

    # Scraping the site for the data in the range of the years provided.
    for year in years:
        # NHL lockout 2004-2005 season (no data)
        if year != '2005':
            url = f"https://www.hockey-reference.com/leagues/NHL_{year}.html#all_stats"

            try:
                # Importing HTML data from Hockey Reference using requests
                response_test = requests.get(url)

                # response_test.status_code == 200:
                nhl_html_contents = response_test.text

                # Now use BeautifulSoup to create the soup and begin cleaning the file
                # BeautifulSoup takes "markup" as argument (the contents) 
                # and another "parser" argument to specify the language of contents

                # soup is an object with certain attributes that can access various tags in the HTML code
                # The soup object contents are equivalent to the 

                soup = BeautifulSoup(nhl_html_contents, "html.parser")

                #soup = soup.prettify()
                #print(soup)

                ashtml = soup.select("#all_stats")
                ashtml_var = ashtml[0]
                tbl = ashtml_var.contents
                tbl_string = str(tbl[4])

                dfs = pd.read_html(tbl_string)
                df_list.append(dfs[0])
                print(f"Successfully retrieved HTML from url for the year {year}. Status code: {response_test.status_code}.")
            
            except:
                years.remove(year) # Removes year for unsuccessful imports
                scrape_errors.append(f"Failed to retrieve HTML from url for the year {year}. Status code: {response_test.status_code}.")

        else:
            print(f"Skipped the year {year}. The page has no data.")
        
        return years, df_list, scrape_errors
    
# Function to clean, manipulate, and create data frame for analysis and model.
# Cleans team data from Hockey Reference and returns a cleaned data frame.
def clean_nhl(frame, years):

    # Getting team abbreviations from NHL.com for hyperlinks
    team_abbrevs_dict = {'Carolina Hurricanes':'hurricanes', 'Boston Bruins':'bruins', 'Columbus Blue Jackets':'bluejackets', 'New Jersey Devils':'devils',
                     'New York Islanders':'islanders', 'New York Rangers':'rangers', 'Philadelphia Flyers':'flyers', "Washington Capitals":'capitals',
                     'Pittsburgh Penguins':'penguins', 'Buffalo Sabres':'sabres', 'Detroit Red Wings':'redwings', 'Florida Panthers':'panthers',
                     'Montreal Canadiens':'canadiens', 'Tampa Bay Lightning':'lightning', 'Toronto Maple Leafs':'mapleleafs', 'Arizona Coyotes':'coyotes',
                     'Chicago Blackhawks':'blackhawks', 'Colorado Avalanche':'avalanche', 'Dallas Stars':'stars', 'Minnesota Wild':'wild', 'Nashville Predators':'predators',
                     'St. Louis Blues':'blues', 'Winnipeg Jets':'jets', 'Anaheim Ducks':'ducks', 'Calgary Flames':'flames', 'Edmonton Oilers':'oilers',
                     'Los Angeles Kings':'kings', 'San Jose Sharks':'sharks', 'Seattle Kraken':'kraken', 'Vegas Golden Knights':'goldenknights', 'Vancouver Canucks':'canucks',
                     'Ottawa Senators':'senators', 'Mighty Ducks of Anaheim':'ducks', 'Phoenix Coyotes':'coyotes', 'Atlanta Thrashers':'jets', 'Utah Hockey Club':'utah'
                    }
    
    # Creating new column names
    multi_index_list = list(frame.columns)
    new_col_names = [pair[1] for pair in multi_index_list]
    new_col_names[1] = "Team"
    frame.columns = new_col_names

    # Deleting the last row (league average data)
    frame = frame.iloc[:-1]

    # Creating new columns
    frame.loc[:,"POff"] = frame["Team"].apply(lambda x: 1 if x[-1] == "*" else 0)
    frame.loc[:,"GDiff/G"] = frame["GF/G"] - frame["GA/G"]
    frame.loc[:,"Team"] = frame.loc[:,"Team"].apply(lambda name: name[0:-1] if name[-1] == "*" else name) 
    frame.loc[:,'Nickname'] = frame.loc[:,'Team'].apply(lambda x: team_abbrevs_dict[x])
    frame = frame.astype({'Rk': int})
    frame.rename(columns = {'Rk':'Rank'}, inplace = True)

    # Reformatting SOW and SOL columns
    frame["SOW"].fillna(0, inplace = True)
    frame["SOL"].fillna(0, inplace = True)
    frame = frame.astype({'SOW':int, 'SOL':int})

    # Creating the year columns (no data for 2005 due to NHL lockout)
    for y in years:
        if y != '2005':
            frame.loc[:,"Year"] = y
    
    return frame

# Enter stats as strings in stat_list
def team_stat_visualizer(df, stat_list):
    import matplotlib.pyplot as plt
    import pandas as pd
    season = df.Year.unique()[0]
    for stat in stat_list:
        # Computes the skewness and kurtosis of the distribution
        k = df[stat].kurt()
        plt.hist(df[stat])
        plt.title(f"Distribution of {stat} in {str(season)} NHL Season, kurtosis = {k:.2f}")
        plt.xlabel(f"{stat}")
        plt.ylabel("Count")
        plt.show()

# Use for skewed distributions
def normalizer(x, df, feature):
    # Thought on using scikit-learn for this...
    # scaler = sklearn.preprocessing.MinMaxScaler()
    min = df[feature].min()
    max = df[feature].max()
    n = (x - min)/(max - min)
    return n

# Use for symmetric distributions
def standardizer(x, df, feature):
    # Thought on using scikit-learn for this...
    # scaler = sklearn.preprocessing.StandardScaler()
    avg = df[feature].mean()
    sd = df[feature].std(ddof = 0) # ddof - Delta Degrees of Freedom = 0 to compute pop stdev
    z = (x - avg)/sd
    return z