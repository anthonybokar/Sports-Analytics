# ## Getting Data Via Scraping
# Source: Sports Reference - https://www.hockey-reference.com/leagues/NHL_2024.html#all_stats

#%% Setting up the notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from datetime import timedelta

#from bs4 import exceptions as bs_exceptions
#import time
#import html5lib
#%% Defining variables used throughout the code

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

# team_abbrev = team_abbrevs_dict['Carolina Hurricanes']

# Set to True if retraining model
train_model = False

# Set to True if wanting to visualize stats using dist_visualizer function
viz_stats = False

# Logic for getting the years we want - last 20 seasons

today = dt.datetime.now()
todays_year = today.year
if today >= dt.datetime(todays_year, 11, 1):
    this_season = todays_year + 1
else:
    this_season = todays_year
# Create a sequence of years, default starting year is 2000.

start_year = todays_year - 20
increment = 1
years_sequence = range(start_year, this_season + 1, increment)

#Converting years_sequence to a list of strings
years_to_list = list(years_sequence)
years = [str(y) for y in years_to_list]

# No data for 2005 in Hockey Reference
years.remove('2005') 

#%% Initializing a list to collect data frames
df_list = []

# Scraping the site for the data in the range of the years provided.
for year in years:
    # NHL lockout 2004-2005 season (no data)
    if year != '2005':
        url = f"https://www.hockey-reference.com/leagues/NHL_{year}.html#all_stats"

        # Importing HTML data from Hockey Reference using requests
        response_test = requests.get(url)

        if response_test.status_code == 200:
            nhl_html_contents = response_test.text
            print(f"Successfully retrieved HTML from url for the year {year}. Status code: {response_test.status_code}.")
        else:
            print(f"Failed to retrieve HTML from url for the year {year}. Status code: {response_test.status_code}.")

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
    else:
        print(f"Skipped the year {year}. The page has no data.")
       
    
# ------------------------------------------------------------------------------------------------------
## Cleaning the Data

#%% Function to clean, manipulate, and create data frame for analysis and model.
def clean_nhl(frame):
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
    return frame

#%% Cleaning the dataframes

cleaned_nhl_dfs = [clean_nhl(frame = x) for x in df_list]

# Creating the year columns
for i in range(len(years)):
    if years[i] != '2005':
        cleaned_nhl_dfs[i].loc[:,"Year"] = years[i]
    else:
        print(f"Error in {years[i]}.")
    
#%% Aggregating the dataframes to one dataframe to subset by year

nhl_data_all_years = pd.DataFrame()
for frame in cleaned_nhl_dfs:
   nhl_data_all_years = pd.concat([nhl_data_all_years, frame], ignore_index=True)

nhl_data_all_years["T"].fillna(0, inplace = True)
nhl_data_all_years["SOW"].fillna(0, inplace = True)
nhl_data_all_years["SOL"].fillna(0, inplace = True)

nhl_data_all_years = nhl_data_all_years.astype({'T': int, 'SOW':int, 'SOL':int})

# Getting data for this season (for attached Excel and Tableau for comparisons)
nhl_data_this_season = nhl_data_all_years[nhl_data_all_years["Year"] == str(this_season)]

################### PREPARING MODEL DATA ###################
#%% Visualizing distribution of statistics to determine whether to normalize or standardize

# Enter stats as strings in stat_list
def dist_visualizer(df, stat_list):
    for stat in stat_list:
        plt.hist(df[stat])
        plt.title(f"Distribution of {stat} since {start_year}")
        plt.xlabel(f"{stat}")
        plt.ylabel("Count")
        plt.show()

# Visualizes some distributions if viz_stats == True
if viz_stats:

    # Looking at numerical statistics' distributions to determine normalization or standardization
    ignore_list = ['Rank', 'Team', 'GP', 'T', 'OL', 'POff', 'Year', 'SOW', 'SOL', 'Nickname']
    stats_to_viz = [stat for stat in nhl_data_all_years.columns if stat not in ignore_list]

    # Visualizing historic data
    dist_visualizer(nhl_data_all_years[nhl_data_all_years["Year"] != str(this_season)], stats_to_viz)
else:
    print("Not visualizing data.")

#%% Normalizing and standardizing variables for model
# For skewed distributions:
normalize_features = ['W', 'L', 'PTS', 'PTS%', 'SRS',
                       'PPO', 'PP%', 'PPA', 'PPOA', 'PK%',
                         'SH', 'SHA', 'PIM/G', 'oPIM/G', 'S',
                         'SA', 'SV%', 'SO', 'S%']

# For approximately symmetric distributions:
standardize_features = ['AvAge', 'GF', 'GA', 'SOS', 
                        'GF/G', 'GA/G', 'PP', 'GDiff/G']

def normalizer(x, df, feature):
    # Thought on using scikit-learn for this...
    # scaler = sklearn.preprocessing.MinMaxScaler()
    min = df[feature].min()
    max = df[feature].max()
    n = (x - min)/(max - min)
    return n
def standardizer(x, df, feature):
    # Thought on using scikit-learn for this...
    # scaler = sklearn.preprocessing.StandardScaler()
    avg = df[feature].mean()
    sd = df[feature].std(ddof = 0) # ddof - Delta Degrees of Freedom = 0 to compute pop stdev
    z = (x - avg)/sd
    return z

#%% Initializing empty dataframe to contain normalized/standardized features and other model features
nhl_model_data = pd.DataFrame()

for y in years:
    df_temp = nhl_data_all_years[nhl_data_all_years["Year"] == y]
    for f in normalize_features:
        df_temp.loc[:,f] = df_temp[f].apply(lambda x: normalizer(x, df = df_temp, feature = f))
    for f in standardize_features:
        df_temp.loc[:,f] = df_temp[f].apply(lambda x: standardizer(x, df = df_temp, feature = f))
    nhl_model_data = pd.concat([nhl_model_data, df_temp], ignore_index = True)

#%% Counting the number of times each team has made the playoffs in a certain number of years prior to that season
years_window = 3
nhl_model_data_final = pd.DataFrame()

for nickname in nhl_model_data["Nickname"].unique():
    df = nhl_model_data[nhl_model_data["Nickname"] == nickname]
    df.loc[:,f"POff_Last_{str(years_window)}Yr_Count"] = df.loc[:,"POff"].shift(1).rolling(window = years_window).sum()
    nhl_model_data_final = pd.concat([nhl_model_data_final,df], ignore_index = True)

nhl_model_data_final = nhl_model_data_final.sort_values(by = ["Year", "Rank"])

nhl_model_data_final[f"POff_Last_{str(years_window)}Yr_Count"].fillna(0, inplace = True)

# %% Model construction 

# Getting training/testing data - all previous seasons except first three and this season
exclude_years = list(range(start_year, start_year + years_window))
exclude_years.append(this_season)
exclude_years = [str(y) for y in exclude_years]
model_training_testing_data = nhl_model_data_final[~nhl_model_data_final["Year"].isin(exclude_years)]

# Data for predicting in this season
model_prediction_data = nhl_model_data_final[nhl_model_data_final["Year"] == str(this_season)]

#%% Splitting into design matrix and target variable
drop_columns = ['Rank', 'Team', 'T', 'OL', 'Nickname', 'Year', 'SOW', 'SOL', 'POff']
y = model_training_testing_data["POff"]
X = model_training_testing_data.drop(drop_columns, axis = 1)

if train_model == True:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # Using a logistic model
    model = LogisticRegression()

    # Training the model
    fitted_model = model.fit(X_train, y_train)
    y_predictions = fitted_model.predict(X_test)
    y_probabilities = fitted_model.predict_proba(X_test)
    acc_score = accuracy_score(y_test, y_predictions)
    print(f"The accuracy score is: {acc_score}.")
    conf_matrix = confusion_matrix(y_test, y_predictions)
    print(conf_matrix)

    # Saving and exporting the model
    file = 'NHL_Playoff_Logistic_Model.pk1'
    pickle.dump(model, open(file, 'wb'))

#%% Loading the pickled model to predict playoff probabilities for this season.
logistic_model = pickle.load(open('NHL_Playoff_Logistic_Model.pk1', 'rb'))

X = model_prediction_data.drop(drop_columns, axis = 1)
y_probabilities = logistic_model.predict_proba(X)

#%% Getting playoff probabilities and attaching them to this season's data.
playoff_probs = []
for p in y_probabilities:
    playoff_probs.append(p[1])

playoff_probs = pd.DataFrame({'POff%':playoff_probs})

nhl_data_this_season = nhl_data_this_season.reset_index(drop = True)
nhl_data_this_season = nhl_data_this_season.join(playoff_probs)
nhl_data_this_season.loc[:,'POff%'] = nhl_data_this_season.loc[:,'POff%'].apply(lambda x: x*100).round(2)

# %% Splitting into tables for Excel export and html email

# Setting columns for html and Excel files
html_cols = ['Team', 'GP', 'W', 'L', 'OL', 'PTS', 'PTS%', 'GF/G', 'GA/G', 'GDiff/G','POff%']
excel_drop_cols = ['Rank', 'Nickname', 'Year', 'T']

# Creating Excel export file for email attachment
this_season_excel_df = nhl_data_this_season.drop(columns = excel_drop_cols).sort_values(by = 'PTS%', ascending = False)
# this_season_excel_df.to_excel("this_season_excel.xlsx", index = False)
nhl_data_all_years.to_excel(f"nhl_team_data_{str(start_year)}_{str(this_season)}.xlsx", index = False, sheet_name = "Last 20 Seasons")

# %% Creating Excel file for attachment and Tableau data source
from openpyxl.utils.dataframe import dataframe_to_rows

def update_sheet(df, file, sheet):
    workbook = openpyxl.load_workbook(file)
    ws = workbook[sheet]
    table = ws.tables["NHL_DATA"]
    num_cols = len(df.columns) # Need function to convert to Excel column letters
    num_rows = len(df) + 1 # +1 for header row?

    if num_cols > 26:
        col = chr(96 + num_cols - 26).upper()
        table.ref = "A1:" + "A" + f"{col}" + f"{num_rows}"
        print("A1:" + "A" + f"{col}" + f"{num_rows}")
    else:
        col = chr(96 + num_cols).upper()
        table.ref = "A1:" + f"{col}" + f"{num_rows}"
        print("A1:" + f"{col}" + f"{num_rows}")
    rows = dataframe_to_rows(df, index = False, header = True)

    for r_idx, row in enumerate(rows, start = 0):
        for c_idx, value in enumerate(row, start = 0):
            ws.cell(row = r_idx + 1, column = c_idx + 1, value = value)

    workbook.save(file)

# %% Writing the Excel 
update_sheet(this_season_excel_df, '/Users/anthony/Desktop/VSFolder/Sports Analytics/NHL Analytics/this_season_excel.xlsx', 'Current Season Data')

# Hyperlinking Team column in html table to f"nhl.com/{team_abbrev}/"
nhl_data_this_season.loc[:,"Team"] = nhl_data_this_season.apply(lambda row: f"<a href='https://www.nhl.com/{row.Nickname}/'>{row.Team}</a>", axis = 1)

#%% Creating html table for email/web app/GUI

this_season_html_df = nhl_data_this_season[html_cols].sort_values(by = 'PTS%', ascending = False)
this_season_html_df = build_table(this_season_html_df, color = 'blue_light', font_size = 'medium', font_family = 'sans-serif', text_align = 'center', index = False, width = "1000px", escape = False, padding = "0px")

# Getting data recency to display in email.
today = dt.datetime.now().strftime('%A, %m/%d/%Y %I:%M %p')

#%%
html_text = f"""
<b><u><h2 style="font-family: sans-serif; text-align: center; color: maroon;">NHL Playoff Predictions</h2></u></b>
<p style="font-size: medium; font-family: sans-serif;">
Hi hockey fans!<br> 
The model is back in action with some improvements. These predictions are based on data from <a href = "https://www.hockey-reference.com/">Hockey Reference</a> (data as of {today}).
You can find all the statistics for this season in the attached Excel file.
</p>
<p style="font-size: x-small; font-family: sans-serif;">
  <b>Key for uncommon statistic abbreviations:</b><br>
  SOW: Shootout wins, SOL: Shootout losses, SRS: Simple rating system based on avg. goal differential and strength of schedule (SOS) PTS%: Points percentage (i.e., points divided by maximum points, similar to winning percentage in other sports) PP%: Power play percentage, PK%: Penalty kill percentage, PIM/G: Penalties in minutes per game oPIM/G: Opponent PIM/G, SH: Short-handed goals, SHA: SH against (See Hockey Reference for more detailed descriptions.)
</p>
"""

# Final html for email attachment
this_season_html_df = this_season_html_df.replace('text-align: center;padding: 0px;width: 1000px"><a href', 'text-align: left;padding: 0px;width: 1000px"><a href')
email_html = html_text + this_season_html_df

with open('nhl_testing.html', 'w') as table:
    table.write(email_html)

# %% Sending emails with attachment.

from NHL_vars import my_email as sender_email, my_email_pw as password, receiver_list as receiver_emails

# Email and SMTP configuration
smtp_server = 'smtp.gmail.com'
smtp_port = 587

# Create a multipart message
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = ''
msg['Bcc'] = ';'.join(receiver_emails)
msg['Subject'] = f"NHL Playoff Predictions - Week of {dt.datetime.now().strftime('%B %d, %Y')}"

# Add HTML body to the email
msg.attach(MIMEText(email_html, 'html'))

# Attach the Excel file
file_path = '/Users/anthony/Desktop/VSFolder/Sports Analytics/NHL Analytics/this_season_excel.xlsx'
with open(file_path, 'rb') as attachment:
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {file_path}")
    msg.attach(part)

# Connect to the SMTP server and send the email
try:
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_emails, msg.as_string())
    print("Email sent successfully!")
except Exception as e:
    print(f"Failed to send email. Error: {str(e)}")
finally:
    server.quit()

# %%
