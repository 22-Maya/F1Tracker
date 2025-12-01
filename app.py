from flask import Flask, render_template, url_for
import fastf1
import pandas as pd

app = Flask(__name__)

# Mapping of country names to ISO 3166-1 alpha-2 codes for flag emojis
COUNTRY_TO_FLAG = {
    "Azerbaijan": "🇦🇿", "Bahrain": "🇧🇭", "Saudi Arabia": "🇸🇦", "Australia": "🇦🇺", "Japan": "🇯🇵",
    "China": "🇨🇳", "Miami": "🇺🇸", "Monaco": "🇲🇨", "Canada": "🇨🇦",
    "Spain": "🇪🇸", "Austria": "🇦🇹", "United Kingdom": "🇬🇧", "Hungary": "🇭🇺",
    "Belgium": "🇧🇪", "Netherlands": "🇳🇱", "Italy": "🇮🇹", "Germany": "🇩🇪",
    "Singapore": "🇸🇬", "Mexico": "🇲🇽", "Brazil": "🇧🇷", "United Arab Emirates": "🇦🇪",
    "France": "🇫🇷", "Portugal": "🇵🇹", "Turkey": "🇹🇷", "USA": "🇺🇸",
    "United States": "🇺🇸", "UAE": "🇦🇪", "South Africa": "🇿🇦", "Qatar": "🇶🇦"
}

def get_flag(country):
    """Get flag emoji for a country, with fallback"""
    return COUNTRY_TO_FLAG.get(country, "🏁")

def load_calendar():
    year = pd.Timestamp.today().year
    return fastf1.get_event_schedule(year), year

def get_next_event(schedule):
    schedule["date"] = pd.to_datetime(schedule["EventDate"], errors="coerce")
    today = pd.Timestamp.today().date()
    upcoming = schedule[schedule["date"].dt.date >= today]
    if not upcoming.empty:
        return upcoming.iloc[0]
    return schedule.iloc[-1]

@app.route("/")
def index():
    schedule, year = load_calendar()
    return render_template("index.html", schedule=schedule.to_dict(orient="records"), year=year, get_flag=get_flag)

@app.route("/schedule")
def schedule():
    schedule, year = load_calendar()
    schedule["date_utc"] = pd.to_datetime(schedule["Session1DateUtc"], utc=True)
    return render_template("schedule.html", year=year, schedule=schedule.assign(
        date_iso=schedule["date_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_dict(orient="records"), get_flag=get_flag
    )

@app.route("/next")
def next_race():
    schedule, year = load_calendar()
    next_event = get_next_event(schedule)
    # Convert to UTC ISO format for JavaScript
    event_dict = next_event.to_dict()
    if "Session1DateUtc" in event_dict:
        event_dict["date_utc"] = pd.to_datetime(event_dict["Session1DateUtc"], utc=True).isoformat()
    return render_template("next_race.html", event=event_dict, year=year, get_flag=get_flag)

if __name__ == "__main__":
    app.run(debug=True)