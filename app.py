from flask import Flask, render_template, url_for
import fastf1
import pandas as pd

app = Flask(__name__)

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
    return render_template("index.html", schedule=schedule.to_dict(orient="records"), year=year)

@app.route("/schedule")
def schedule():
    schedule, year = load_calendar()
    return render_template("schedule.html", schedule=schedule.to_dict(orient="records"), year=year)

@app.route("/next")
def next_race():
    schedule, year = load_calendar()
    next_event = get_next_event(schedule)
    return render_template("next_race.html", event=next_event.to_dict(), year=year)

if __name__ == "__main__":
    app.run(debug=True)