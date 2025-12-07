# how to install: pip install flask fastf1 pandas matplotlib numpy, then run: .venv/bin/python app.py
from flask import Flask, render_template, url_for, redirect, flash
import fastf1
import pandas as pd
from fastf1 import plotting
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import io
import base64
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- FastF1/Matplotlib Configuration ---
# Enable caching for FastF1
fastf1.Cache.enable_cache('cache') 
plotting.setup_mpl()
colormap = mpl.cm.plasma

# Define plot colors for track layout
BACKGROUND_COLOR = '#0f1724' # Dark background
CIRCLE_COLOR = '#FFFFFF'     # White circle around corner number
LINE_COLOR = '#9aa6b2'       # Muted line to corner number
TEXT_COLOR = '#000000'       # Black text for corner number (for contrast on white circle)
TITLE_COLOR = '#FFFFFF'      # White title text
# --- End Config ---

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
    # Get flag emoji for a country, with fallback
    return COUNTRY_TO_FLAG.get(country, "🏁")

def load_calendar():
    # Loads the F1 event schedule for the current year.
    year = pd.Timestamp.today().year
    # FIXED: Removed 'force_download' argument
    return fastf1.get_event_schedule(year), year 

def get_next_event(schedule):
    # Finds the next upcoming race based on the schedule.
    schedule["date"] = pd.to_datetime(schedule["EventDate"], errors="coerce")
    today = pd.Timestamp.today().date()
    upcoming = schedule[schedule["date"].dt.date >= today]
    if not upcoming.empty:
        return upcoming.iloc[0]
    return schedule.iloc[-1]

# --- Track Visualization Functions ---

def rotate(xy, *, angle):
    """Rotates a 2D array of points (track) by a given angle in radians."""
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)


# app.py (Modified draw_f1_circuit loop)

def draw_f1_circuit(year, gp_name, event_type='R', max_years_back=1):
    """Generate a track layout visualization for a given F1 Grand Prix.
    
    Returns:
        tuple: (base64_image_data, title_string, year_used)
    """
    current_year = year
    session_event = None
    
    for y in range(current_year, current_year - max_years_back - 1, -1):
        try:
            logging.info(f"Attempting to load data for {y} {gp_name}...")
            
            session_event = fastf1.get_session(y, gp_name, event_type)
            session_event.load(laps=True, telemetry=True, weather=False, messages=False)
            
            # Check if the session actually loaded any laps
            if session_event.laps.empty:
                raise ValueError(f"Session data for {y} {gp_name} contains no laps.")
                
            lap = session_event.laps.pick_fastest()

            # Check if the fastest lap has positional data
            pos = lap.get_pos_data() 
            
            if pos.empty:
                raise ValueError(f"Fastest lap for {y} {gp_name} has no positional data.")

            # Data loaded successfully!
            year = y
            break 
            
        except fastf1._api.SessionNotAvailableError:
            if y == current_year and max_years_back > 0:
                logging.warning(f"Data for {y} {gp_name} not available. Falling back to previous year.")
                continue
            else:
                raise
                
        except ValueError as e:
            logging.warning(f"Data quality error for {y} {gp_name}: {e}")
            if y == current_year and max_years_back > 0:
                continue
            else:
                raise
        
        except Exception as e:
            logging.error(f"Non-API error loading data for {y} {gp_name}: {e}")
            raise e
    else:
        raise fastf1._api.SessionNotAvailableError(f"No data found for {gp_name} from {current_year} to {current_year - max_years_back}")
    
    # --- Generate the track plot ---
    try:
        lap = session_event.laps.pick_fastest()
        pos = lap.get_pos_data()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
        
        # Plot the track
        ax.plot(pos['X'], pos['Y'], color=LINE_COLOR, linewidth=2, label='Track')
        
        # Mark corners (simplified: mark every 5th point as a corner)
        corner_indices = np.arange(0, len(pos), max(1, len(pos) // 8))
        for idx, i in enumerate(corner_indices):
            x, y = pos['X'].iloc[i], pos['Y'].iloc[i]
            ax.add_patch(plt.Circle((x, y), 50, color=CIRCLE_COLOR, fill=False, linewidth=2))
            ax.text(x, y, str(idx + 1), color=TEXT_COLOR, fontsize=8, ha='center', va='center', weight='bold')
        
        ax.set_aspect('equal')
        ax.set_title(f'{gp_name} Track Layout ({year})', color=TITLE_COLOR, fontsize=14, weight='bold')
        ax.set_xlabel('X (meters)', color=TITLE_COLOR)
        ax.set_ylabel('Y (meters)', color=TITLE_COLOR)
        ax.tick_params(colors=TITLE_COLOR)
        ax.spines['bottom'].set_color(TITLE_COLOR)
        ax.spines['top'].set_color(TITLE_COLOR)
        ax.spines['right'].set_color(TITLE_COLOR)
        ax.spines['left'].set_color(TITLE_COLOR)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        
        title = f'{gp_name} ({year})'
        return image_data, title, year
        
    except Exception as e:
        logging.error(f"Error generating track plot for {gp_name}: {e}")
        raise
# --- Flask Routes ---

@app.route("/")
def index():
    schedule, year = load_calendar()
    return render_template("index.html", schedule=schedule.to_dict(orient="records"), year=year, get_flag=get_flag)

@app.route("/schedule")
def schedule_page():
    schedule, year = load_calendar()
    schedule["date_utc"] = pd.to_datetime(schedule["Session1DateUtc"], utc=True)
    return render_template("schedule.html", year=year, schedule=schedule.assign(
        date_iso=schedule["date_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_dict(orient="records"), get_flag=get_flag
    )

@app.route("/next")
def next_race():
    schedule, year = load_calendar()
    next_event = get_next_event(schedule)
    
    event_dict = next_event.to_dict()
    if "Session1DateUtc" in event_dict:
        event_dict["date_utc"] = pd.to_datetime(event_dict["Session1DateUtc"], utc=True).isoformat()
    
    # Add the FastF1-compatible name for the track link
    event_dict["FastF1Name"] = next_event["EventName"].replace(' Grand Prix', '').replace(' ', '_')

    return render_template("next_race.html", event=event_dict, year=year, get_flag=get_flag)

@app.route("/track/<int:year>/<string:gp_name>")
def track_layout(year, gp_name):
    gp_name_sanitized = gp_name.replace(' Grand Prix', '').replace(' ', '_')
    
    source_year = year 

    try:
        # Try to generate the image, with fallback to previous year
        image_data, title, source_year = draw_f1_circuit(year, gp_name_sanitized, max_years_back=1)
        data_available = bool(image_data)
        
    except fastf1._api.SessionNotAvailableError:
        image_data = ""
        title = f"Track Data Unavailable for {gp_name} ({year} or {year-1})"
        data_available = False
        source_year = year # Use the requested year for the error message

    except Exception as e:
        logging.error(f"Critical error in track_layout route for {year} {gp_name}: {e}")
        image_data = ""
        title = f"Error loading track data for {gp_name}."
        data_available = False
        source_year = year
    
    return render_template('track_layout.html', 
                           image_data=image_data, 
                           title=title,
                           gp_name=gp_name, 
                           year=year, 
                           source_year=source_year,
                           data_available=data_available)

if __name__ == "__main__":
    if not os.path.exists('cache'):
        os.makedirs('cache')
    app.run(debug=True)