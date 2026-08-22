#%%
import requests
import os
import pandas as pd

from dotenv import load_dotenv, find_dotenv
import os

dotenv_path = find_dotenv(usecwd=True)
print("Resolved path:", dotenv_path)

result = load_dotenv(dotenv_path, override=True)
print("load_dotenv success:", result)
print("Key found:", os.getenv("FANTASY_PROS_API_KEY"))
#%%
if not os.getenv("FANTASY_PROS_API_KEY"):
    raise EnvironmentError("FANTASY_PROS_API_KEY not found in environment variables. Please set it before running the script.")
else:
    print("FANTASY_PROS_API_KEY found. Proceeding..")
    api_key = os.getenv("FANTASY_PROS_API_KEY")

resp = requests.get(
    "https://api.fantasypros.com/public/v2/json/nfl/players",
    headers={"x-api-key": api_key},
    params={"ecr": "included", "show": "pos_rank"},
)
data = resp.json()
print(data["players"])
print(type(data["players"]))
print(type(data))

# Convert the list of players to a DataFrame
df = pd.DataFrame(data["players"])
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
print(df.shape)

# %%
