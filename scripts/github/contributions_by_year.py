from fastapi import FastAPI, HTTPException
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()
GIT_TOKEN = os.getenv("GIT_TOKEN")
AGENT_LINK = os.getenv("AGENT_LINK")

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://portfolio-phi-mocha-72.vercel.app/", AGENT_LINK],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

QUERY_TEMPLATE = """
query ($username: String!, $from: DateTime, $to: DateTime) {
  user(login: $username) {
    contributionsCollection(from: $from, to: $to) {
      contributionCalendar {
        totalContributions
      }
    }
  }
}
"""

def fetch_contributions_for_year(username: str, year: int):
    from_date = f"{year}-01-01T00:00:00Z"
    to_date = f"{year}-12-31T23:59:59Z"

    headers = {
        "Authorization": f"bearer {GIT_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": QUERY_TEMPLATE,
        "variables": {
            "username": username,
            "from": from_date,
            "to": to_date,
        },
    }

    response = requests.post(GITHUB_GRAPHQL_URL, json=payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch data from GitHub API")

    data = response.json()

    if "errors" in data:
        raise HTTPException(status_code=400, detail=data["errors"])

    return data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["totalContributions"]

@app.get("/contributions/{username}")
def get_contributions(username: str):
    current_year = datetime.utcnow().year
    years = [2024, current_year] if current_year != 2024 else [2024]

    contributions_by_year = {}

    for year in years:
        total = fetch_contributions_for_year(username, year)
        contributions_by_year[year] = total

    return {
        "username": username,
        "contributions": contributions_by_year
    }
