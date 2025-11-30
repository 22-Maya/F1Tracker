import argparse
import datetime
import sys
pip install pandas as pd
pip install fastf1 as ff1
from fastf1 import plotting
from matplotlib import pyplot as plt


def pick_next_event(schedule_df):
    # try to find a date-like column and normalize it
    date_col = None
    for col in schedule_df.columns:
        try:
            dates = pd.to_datetime(schedule_df[col], errors="coerce")
            if dates.notna().any():
                schedule_df["_date"] = dates
                date_col = col
                break
        except Exception:
            continue

    if date_col is None:
        raise RuntimeError("Could not find a date column in the FastF1 schedule DataFrame.")

    today = pd.to_datetime(datetime.date.today()).date()
    upcoming = schedule_df[schedule_df["_date"].dt.date >= today].sort_values("_date")
    if not upcoming.empty:
        return upcoming.iloc[0]
    # if no upcoming events this year, return the last event (fallback)
    return schedule_df.sort_values("_date").iloc[-1]


def find_event_name(row):
    for key in ("EventName", "Event", "Name", "Event Full Name", "Grand Prix", "GrandPrix"):
        if key in row.index and pd.notna(row[key]):
            return str(row[key])
    # fallback: first non-date string column
    for col in row.index:
        if col == "_date":
            continue
        if isinstance(row[col], str) and row[col].strip():
            return row[col]
    raise RuntimeError("Could not determine event name from schedule row.")


def main():
    p = argparse.ArgumentParser(description="Show the next F1 track using FastF1 calendar")
    p.add_argument("--year", type=int, default=None, help="Season year (default: current year)")
    p.add_argument("--session", default="Q", help="Session short name: 'P1','P2','P3','Q','R'")
    args = p.parse_args()

    year = args.year or datetime.date.today().year

    # enable local cache for faster repeated runs
    ff1.Cache.enable_cache("cache")

    # load schedule and pick next event
    schedule = ff1.get_event_schedule(year)
    try:
        next_event_row = pick_next_event(schedule)
        event_name = find_event_name(next_event_row)
        event_date = pd.to_datetime(next_event_row["_date"]).date()
    except Exception as e:
        print("Failed to pick next event from calendar:", e, file=sys.stderr)
        sys.exit(1)

    print(f"Selected event: {event_name} on {event_date} (year {year}) — session {args.session}")

    # load session and plot track
    try:
        session = ff1.get_session(year, event_name, args.session)
        session.load()
    except Exception as e:
        print("Failed to load session:", e, file=sys.stderr)
        sys.exit(1)

    plotting.setup_mpl()
    fig, ax = plt.subplots(figsize=(10, 7))
    ff1.plotting.plot_track(session, ax=ax)
    ax.set_title(f"{event_name} {year} — {args.session} ({event_date})")
    plt.show()


if __name__ == "__main__":
    main()