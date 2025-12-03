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
import logging # Used for better error handling/logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Enable caching for FastF1
fastf1.Cache.enable_cache('cache') 
plotting.setup_mpl()
colormap = mpl.cm.plasma

# Define plot colors for track layout
BACKGROUND_COLOR = '#0f1724' # Dark background
CIRCLE_COLOR = '#FFFFFF'     # White circle around corner number
LINE_COLOR = '#9aa6b2'       # Muted line to corner number
TEXT_COLOR = '#000000'       # Black text for corner number
TITLE_COLOR = '#FFFFFF'      # White title text

app = Flask(__name__)

# track country to flag mapping
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
    year = pd.Timestamp.today().year
    return fastf1.get_event_schedule(year, force_download=False), year

def get_next_event(schedule):
    schedule["date"] = pd.to_datetime(schedule["EventDate"], errors="coerce")
    today = pd.Timestamp.today().date()
    upcoming = schedule[schedule["date"].dt.date >= today]
    if not upcoming.empty:
        return upcoming.iloc[0]
    return schedule.iloc[-1]


# Track Visualization Functions
def rotate(xy, *, angle):
    """Rotates a 2D array of points (track) by a given angle in radians."""
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

def draw_f1_circuit(year, gp_name, event_type='R'): # Use 'R' for Race, often fastest lap data is here
    """
    Generates the F1 circuit layout image using FastF1 and Matplotlib.
    Returns the image data encoded in base64.
    """
    try:
        # Load the session data. Use 'Race' or 'R' as it typically has the full track map data.
        # We only need laps for position data, so set all others to False
        session_event = fastf1.get_session(year, gp_name, event_type)
        session_event.load(laps=True, telemetry=False, weather=False, messages=False)
        
        # Get required data
        circuit_info = session_event.get_circuit_info()
        lap = session_event.laps.pick_fastest() 
        pos = lap.get_pos_data()

        # Prepare for plotting
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=BACKGROUND_COLOR)
        
        # --- Track Data Preparation ---
        track = pos.loc[:, ('X', 'Y')].to_numpy()
        # Rotation is applied to align the track with north up or a visually pleasing orientation
        track_angle = circuit_info.rotation / 180 * np.pi 
        offset_vector = [500, 0] # Used to position corner markers
        
        # Rotate and plot the track map outline
        rotated_track = rotate(track, angle=track_angle)
        ax.plot(rotated_track[:, 0], rotated_track[:, 1], color ='tab:orange', linewidth=4)
        
        # --- Plotting Corners ---
        # The code iterates over all corners and places a circle, line, and number
        for _, corner in circuit_info.corners.iterrows():
            txt = f"{corner['Number']}{corner['Letter'] if corner['Letter'] else ''}"

            offset_angle = corner['Angle'] / 180 * np.pi

            # Rotate the offset vector to point sideways from the track
            offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

            # Add the offset to the corner position
            text_x = corner['X'] + offset_x
            text_y = corner['Y'] + offset_y

            # Rotate the text position and the corner center to match the track map
            text_x, text_y = rotate([text_x, text_y], angle=track_angle)
            track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

            # Draw circle, line, and text
            ax.scatter(text_x, text_y, color=CIRCLE_COLOR, s=140, zorder=3)
            ax.plot([track_x, text_x], [track_y, track_y], color=LINE_COLOR, linewidth=1, zorder=2) # Note: Changed text_y to track_y based on the official example structure
            ax.text(text_x, text_y, txt,
                    va='center', ha='center', size='small', color=TEXT_COLOR, zorder=4)

        # --- Final Plot Styling ---
        plot_title = f"{year} {gp_name} Grand Prix Circuit"
        ax.set_title(plot_title, color=TITLE_COLOR, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('equal') # Essential for a correct track aspect ratio
        ax.set_facecolor(BACKGROUND_COLOR)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        
        # --- Save to Bytes Buffer and Encode ---
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig) # Close the figure to free memory
        
        # Encode the image data
        image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return image_data, plot_title

    except Exception as e:
        logging.error(f"Error drawing circuit for {year} {gp_name}: {e}")
        return "", f"Error loading track data for {gp_name}"

# --- New Track Visualization Route ---
@app.route("/track/<int:year>/<string:gp_name>")
def track_layout(year, gp_name):
    # The 'gp_name' from the URL needs to be the one FastF1 expects (e.g., 'Singapore', not 'Singapore Grand Prix')
    # Since we're passing it from the event data, it should be correct.
    gp_name_sanitized = gp_name.replace(' Grand Prix', '').replace(' ', '_')

    # Try to generate the image
    image_data, title = draw_f1_circuit(year, gp_name_sanitized)
    
    # Render the new template
    return render_template('track_layout.html', 
                           image_data=image_data, 
                           title=title,
                           gp_name=gp_name, # Use the original name for display
                           year=year)

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
    
    # Add the FastF1-compatible name for the track link
    event_dict["FastF1Name"] = next_event["EventName"].replace(' Grand Prix', '').replace(' ', '_')

    return render_template("next_race.html", event=event_dict, year=year, get_flag=get_flag)


if __name__ == "__main__":
    # Ensure a 'cache' directory exists for FastF1
    if not os.path.exists('cache'):
        os.makedirs('cache')
    app.run(debug=True)