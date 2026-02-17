# how to install: .venv/bin/pip install flask fastf1 pandas matplotlib numpy requests && .venv/bin/python app.py
from flask import Flask, render_template, url_for, redirect, flash, Response, request, abort
import hashlib
import fastf1
import requests
import pandas as pd
from fastf1 import plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import matplotlib as mpl
import io
import base64
import os
import logging
import sys
import subprocess

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

app = Flask(__name__)

# Team Data for 2026 F1 Season (current drivers)
TEAM_DATA = {
    "Mercedes": {"color": "#00D2BE", "logo": "mercedes.png"},
    "Red Bull": {"color": "#1E41FF", "logo": "redbull.png"},
    "Ferrari": {"color": "#DC0000", "logo": "ferrari.png"},
    "McLaren": {"color": "#FF8700", "logo": "mclaren.png"},
    "Alpine": {"color": "#2293D1", "logo": "alpine.png"},
    "Aston Martin": {"color": "#006F62", "logo": "aston.png"},
    "Williams": {"color": "#0066FF", "logo": "williams.png"},
    "Audi": {"color": "#00D4BE", "logo": "audi.png"},
    "Racing Bulls": {"color": "#3671C6", "logo": "racing_bulls.png"},
    "Cadillac": {"color": "#FF00FF", "logo": "cadillac.png"},
    "Haas": {"color": "#B6BABD", "logo": "haas.png"},
}

# Driver Data for 2026 F1 Season (current drivers)
DRIVERS_DATA = [
    {"name": "Charles Leclerc", "number": "16", "team": "Ferrari"},
    {"name": "Lewis Hamilton", "number": "44", "team": "Ferrari"},
    {"name": "Lando Norris", "number": "4", "team": "McLaren"},
    {"name": "Oscar Piastri", "number": "81", "team": "McLaren"},
    {"name": "Max Verstappen", "number": "1", "team": "Red Bull"},
    {"name": "Sergio Perez", "number": "11", "team": "Red Bull"},
    {"name": "George Russell", "number": "63", "team": "Mercedes"},
    {"name": "Andrea Kimi Antonelli", "number": "12", "team": "Mercedes"},
    {"name": "Yuki Tsunoda", "number": "22", "team": "Racing Bulls"},
    {"name": "Isack Hadjar", "number": "7", "team": "Racing Bulls"},
    {"name": "Fernando Alonso", "number": "14", "team": "Aston Martin"},
    {"name": "Lance Stroll", "number": "18", "team": "Aston Martin"},
    {"name": "Nico Hulkenberg", "number": "27", "team": "Haas"},
    {"name": "Gabriel Bortoleto", "number": "5", "team": "Haas"},
    {"name": "Alex Albon", "number": "23", "team": "Williams"},
    {"name": "Carlos Sainz", "number": "55", "team": "Williams"},
    {"name": "Esteban Ocon", "number": "31", "team": "Alpine"},
    {"name": "Jack Doohan", "number": "38", "team": "Alpine"},
    {"name": "Oliver Bearman", "number": "50", "team": "Cadillac"},
    {"name": "Juri Vips", "number": "6", "team": "Cadillac"},
]

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

def draw_f1_circuit(year, gp_name, event_type='R', max_years_back=1, *, angle_rad=None, show_axes=False, figsize=(16, 9), require_official=True):
    """Generate a track layout visualization for a given F1 Grand Prix.
    
    Returns:
        tuple: (base64_image_data, title_string, year_used)
    """
    current_year = year
    session_event = None

    # Try each year starting from the requested one
    for y in range(current_year, current_year - max_years_back - 1, -1):
        logging.info(f"Attempting to load telemetry for {y} {gp_name}...")

        # Sessions ordered by most reliable for track maps
        sessions_to_try = ["FP2", "FP1", "Q", "R"]

        for sess in sessions_to_try:
            try:
                logging.info(f"Trying session {sess}...")
                se = fastf1.get_session(y, gp_name, sess)
                se.load(laps=True, telemetry=True)

                # Skip if no lap or no positional data
                if se.laps.empty:
                    continue

                fastest = se.laps.pick_fastest()
                pos = fastest.get_pos_data()

                if pos.empty:
                    continue

                # Success
                session_event = se
                year = y
                logging.info(f"Loaded {sess} telemetry for {gp_name} {y}")
                break

            except Exception as e:
                logging.warning(f"{sess} session failed for {y} {gp_name}: {e}")
                continue

        if session_event:
            break

    if not session_event:
        raise fastf1._api.SessionNotAvailableError(
            f"No usable telemetry found for {gp_name} from {current_year} back to {current_year - max_years_back}"
        )

    # --- Track Plot Generation ---
    try:
        lap = session_event.laps.pick_fastest()
        pos = lap.get_pos_data()

        xy = np.column_stack((pos['X'].values, pos['Y'].values))

        center = xy.mean(axis=0)
        xy_centered = xy - center

        # Determine official FastF1 rotation
        rotation_deg = None
        try:
            circ = session_event.get_circuit_info()
            rotation_deg = getattr(circ, 'rotation', None)
        except Exception:
            pass

        # Apply rotation choice
        if angle_rad is not None:
            xy_rot = rotate(xy_centered, angle=angle_rad)

        elif rotation_deg is not None:
            xy_rot = rotate(xy_centered, angle=np.deg2rad(rotation_deg))
            logging.info(f"Applied official rotation: {rotation_deg} deg")

        else:
            if require_official:
                logging.info("No official rotation available, strict mode uses no rotation")
                xy_rot = xy_centered.copy()
            else:
                xy_rot = rotate(xy_centered, angle=-np.pi/2)

        xy_rot += center
        pos['X_rot'] = xy_rot[:, 0]
        pos['Y_rot'] = xy_rot[:, 1]

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)

        ax.plot(pos['X_rot'], pos['Y_rot'], color=LINE_COLOR, linewidth=3)

        corner_indices = np.arange(0, len(pos), max(1, len(pos) // 8))
        for idx, i in enumerate(corner_indices):
            x, y = pos['X_rot'].iloc[i], pos['Y_rot'].iloc[i]
            ax.add_patch(plt.Circle((x, y), 50, color=CIRCLE_COLOR, fill=False, linewidth=2))
            ax.text(x, y, str(idx + 1), color=TEXT_COLOR, fontsize=10, ha='center', va='center', weight='bold')

        ax.set_aspect('equal')

        if show_axes:
            ax.axis('on')
            ax.set_title(f'{gp_name} Track Layout ({year})', color=TITLE_COLOR, fontsize=14, weight='bold', pad=20)
            ax.tick_params(colors=TITLE_COLOR)
        else:
            ax.axis('off')

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

# --- OPENF1 API ---
def get_openf1_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# --- Flask Routes ---

@app.route("/")
def index():
    schedule, year = load_calendar()
    return render_template("index.html", schedule=schedule.to_dict(orient="records"), year=year, current_year=year, get_flag=get_flag)

@app.route("/schedule")
def schedule_page():
    schedule, year = load_calendar()
    
    # Filter out testing events - keep only official Grand Prix races
    schedule = schedule[~schedule["EventName"].str.contains("Testing", case=False, na=False)]
    
    # Parse UTC date for the template
    schedule["date_utc"] = pd.to_datetime(schedule["Session1DateUtc"], utc=True)

    # --- Precompute FastF1-compatible safe name ---
    schedule["FastF1Name"] = (
        schedule["EventName"]
        .fillna("")                                        # avoid NaN
        .str.strip()                                       # trim whitespace
        .str.replace(r"\s+", "_", regex=True)              # spaces -> underscores
        .str.replace(r"[^A-Za-z0-9_]", "", regex=True)     # drop punctuation
    )

    schedule = schedule.assign(
        date_iso=schedule["date_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    return render_template(
        "schedule.html",
        year=year,
        schedule=schedule.to_dict(orient="records"),
        get_flag=get_flag,
    )

@app.route("/next")
def next_race():
    schedule, year = load_calendar()
    next_event = get_next_event(schedule)
    
    event_dict = next_event.to_dict()
    if "Session1DateUtc" in event_dict:
        event_dict["date_utc"] = pd.to_datetime(event_dict["Session1DateUtc"], utc=True).isoformat()
    
    # Add the FastF1-compatible name for the track link (with underscores for URL)
    event_dict["FastF1Name"] = next_event["EventName"].replace(' ', '_')
    
    # Convert underscores back to spaces for FastF1 API compatibility
    gp_name_for_fastf1 = event_dict["FastF1Name"].replace('_', ' ')

    # --- Server-side: generate the track image for the Next Race page ---
    try:
        image_data, title, source_year = draw_f1_circuit(year, gp_name_for_fastf1, max_years_back=5)
        data_available = bool(image_data)

    except fastf1._api.SessionNotAvailableError:
        image_data = ""
        title = f"Track Data Unavailable for {event_dict['FastF1Name']} ({year} or {year-5})"
        data_available = False
        source_year = year

    except Exception as e:
        logging.error(f"Error generating track for Next Race: {e}")
        image_data = ""
        title = f"Error loading track data for {event_dict['FastF1Name']}."
        data_available = False
        source_year = year

    # Basic track info for the About section
    track_info = event_dict.get('Location', '') or event_dict.get('EventName', '')

    return render_template("next_race.html",
                           event=event_dict,
                           year=year,
                           get_flag=get_flag,
                           image_data=image_data,
                           title=title,
                           gp_name=event_dict["FastF1Name"],
                           data_available=data_available,
                           track_info=track_info)

@app.route("/race/<int:year>/<string:gp_name>")
def race_view(year, gp_name):
    # Sanitize and find the event in the schedule
    schedule, sched_year = load_calendar()
    def _san(s): return s.replace(' ', '_')
    matches = schedule[schedule['EventName'].apply(lambda s: _san(s) == gp_name)]
    if matches.empty:
        # not found -> 404
        return abort(404, f"Race '{gp_name}' not found in {year}")

    event = matches.iloc[0].to_dict()
    
    if "Session1DateUtc" in event:
        event["date_utc"] = pd.to_datetime(event["Session1DateUtc"], utc=True).isoformat()
    event["FastF1Name"] = gp_name
    
    # Convert underscores back to spaces for FastF1 API compatibility
    gp_name_for_fastf1 = gp_name.replace('_', ' ')

    # Generate the same track variables as in next_race()
    try:
        image_data, title, source_year = draw_f1_circuit(year, gp_name_for_fastf1, max_years_back=5)
        data_available = bool(image_data)
    except fastf1._api.SessionNotAvailableError:
        image_data = ""
        title = f"Track Data Unavailable for {gp_name} ({year} or {year-5})"
        data_available = False
        source_year = year
    except Exception as e:
        logging.error(f"Error generating track for race view: {e}")
        image_data = ""
        title = f"Error loading track data for {gp_name}."
        data_available = False
        source_year = year

    return render_template("next_race.html",
                           event=event,
                           year=year,
                           get_flag=get_flag,
                           image_data=image_data,
                           title=title,
                           gp_name=gp_name,
                           data_available=data_available,
                           track_info=event.get('Location', ''))


@app.route("/replays")
def replays():
    """List past races found in the local FastF1 cache folder.
    This scans `cache/` for event folders like `2025-03-16_Australian_Grand_Prix`.
    """
    cache_root = 'cache'
    events = []
    if os.path.exists(cache_root):
        # Walk year directories (expect structure cache/<year>/<YYYY-MM-DD_EventName>/...)
        for year_dir in sorted(os.listdir(cache_root), reverse=True):
            year_path = os.path.join(cache_root, year_dir)
            if not os.path.isdir(year_path):
                continue
            # look for event folders inside the year dir
            for entry in sorted(os.listdir(year_path), reverse=True):
                entry_path = os.path.join(year_path, entry)
                if not os.path.isdir(entry_path):
                    continue
                # Expect folder name starts with date and underscore
                # e.g. 2025-03-16_Australian_Grand_Prix
                if len(entry) > 11 and entry[4] == '-' and entry[7] == '-' and entry[10] == '_':
                    date_part = entry[0:10]
                    gp_slug = entry[11:]
                    display_name = gp_slug.replace('_', ' ')
                    ev = {
                        'year': year_dir,
                        'gp_slug': gp_slug,
                        'display_name': display_name,
                        'date': date_part
                    }
                    # Attach metadata if present in replay/computed
                    meta_path = os.path.join('replay', 'computed', f"{year_dir}_{gp_slug}_meta.json")
                    if os.path.exists(meta_path):
                        try:
                            import json
                            with open(meta_path, 'r', encoding='utf-8') as mf:
                                ev['meta'] = json.load(mf)
                        except Exception:
                            ev['meta'] = None
                    events.append(ev)
    # sort by year/date descending
    events = sorted(events, key=lambda e: (e['year'], e['date']), reverse=True)
    return render_template('replays.html', events=events)


@app.route('/replays/compute', methods=['POST'])
def compute_replay_metadata():
    year = request.form.get('year')
    gp = request.form.get('gp')
    if not year or not gp:
        flash('Missing year or gp', 'error')
        return redirect(url_for('replays'))

    # Start background process to compute metadata
    cmd = [sys.executable, os.path.join('replay', 'compute_metadata.py'), '--year', str(year), '--gp', gp]
    try:
        subprocess.Popen(cmd)
        flash('Metadata computation started — refresh this page in a minute.', 'info')
    except Exception as e:
        flash(f'Failed to start computation: {e}', 'error')
    return redirect(url_for('replays'))


@app.route('/replay/<int:year>/<string:gp_name>')
def replay_detail(year, gp_name):
    # simple detail page that re-uses race_view info where possible
    # gp_name here is expected to be the FastF1-style underscore name
    return redirect(url_for('race_view', year=year, gp_name=gp_name))

@app.route("/track_image/<int:year>/<string:gp_name>")
def track_image(year, gp_name):
    """Return the PNG bytes for a track image. Supports optional query params:
       - angle: rotation in degrees (positive CCW)
       - show_axes: 1 to show axes, 0 (default) to hide
       - w, h: figsize width and height in inches (floats)
    """
    # Convert underscores back to spaces for FastF1 compatibility
    gp_name_for_fastf1 = gp_name.replace('_', ' ')
    gp_name_sanitized = gp_name

    angle = request.args.get('angle', None)
    show_axes_flag = request.args.get('show_axes', '0')
    w = request.args.get('w', None)
    h = request.args.get('h', None)
    refresh_flag_raw = request.args.get('refresh', '0')

    try:
        angle_rad = None if angle is None else np.deg2rad(float(angle))
    except Exception:
        return abort(400, "Invalid angle parameter")

    try:
        show_axes = bool(int(show_axes_flag))
    except Exception:
        show_axes = False

    figsize = None
    try:
        if w is not None and h is not None:
            figsize = (float(w), float(h))
    except Exception:
        return abort(400, "Invalid w/h parameters")

    cache_dir = os.path.join('cache', 'generated')
    os.makedirs(cache_dir, exist_ok=True)

    cache_key = f"{year}:{gp_name_sanitized}:{angle}:{show_axes}:{figsize}"
    cache_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_hash}.png")

    # If not forcing refresh and cache exists, return cached image
    try:
        refresh_flag = bool(int(refresh_flag_raw))
    except Exception:
        refresh_flag = False

    if os.path.exists(cache_file) and not refresh_flag:
        with open(cache_file, 'rb') as f:
            img_bytes = f.read()
        resp = Response(img_bytes, mimetype='image/png')
        resp.headers['Cache-Control'] = 'public, max-age=3600'
        return resp

    # If refresh requested and cache exists, remove it and regenerate
    if os.path.exists(cache_file) and refresh_flag:
        try:
            os.remove(cache_file)
        except Exception:
            logging.warning(f"Unable to remove cache file {cache_file}")

    # Generate fresh image
    try:
        # In strict-only mode we do not allow manual rotation via query params.
        if angle is not None:
            return abort(400, "Manual angle parameter is not allowed: server enforces strict rotation only.")

        kw = {}
        # Always require official rotation (strict-only)
        kw['require_official'] = True
        if figsize is not None:
            kw['figsize'] = figsize
        kw['show_axes'] = show_axes

        image_b64, title, used_year = draw_f1_circuit(year, gp_name_for_fastf1, max_years_back=5, **kw)
        img_bytes = base64.b64decode(image_b64)

        # Save to cache for future reuse (overwrite if refresh)
        try:
            with open(cache_file, 'wb') as f:
                f.write(img_bytes)
        except Exception:
            logging.warning(f"Unable to write cache file {cache_file}")

        resp = Response(img_bytes, mimetype='image/png')
        # If refresh, mark as no-cache for immediate client fetch
        if refresh_flag:
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        else:
            resp.headers['Cache-Control'] = 'public, max-age=3600'
        return resp

    except fastf1._api.SessionNotAvailableError:
        return abort(404, "Track data not available")
    except Exception as e:
        logging.error(f"Error generating track image: {e}")
        return abort(500, "Error generating image")

@app.route("/championship")
def championship():
    current_year = pd.Timestamp.today().year
    selected_year = request.args.get('year', current_year, type=int)
    
    # Try to fetch standings for the selected year
    drivers = []
    constructors = []
    data_year = None
    show_grid = False
    available_years = []
    
    # Check what years have standings available (for the selector)
    for check_year in range(2025, 2015, -1):
        try:
            drv_url = f"https://ergast.com/api/f1/{check_year}/driverStandings.json"
            drv_data = get_openf1_json(drv_url)
            if drv_data['MRData']['StandingsTable']['StandingsLists']:
                available_years.append(check_year)
        except Exception:
            pass
    
    # Try to fetch standings for the selected year
    try:
        # Drivers
        drv_url = f"https://ergast.com/api/f1/{selected_year}/driverStandings.json"
        drv_data = get_openf1_json(drv_url)
        drivers_raw = drv_data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']

        drivers = [
            {
                "name": f"{d['Driver']['givenName']} {d['Driver']['familyName']}",
                "team": d['Constructors'][0]['name'],
                "team_color": TEAM_DATA.get(d['Constructors'][0]['name'], {}).get('color', '#9aa6b2'),
                "points": d['points']
            }
            for d in drivers_raw
        ]

        # Constructors
        cons_url = f"https://ergast.com/api/f1/{selected_year}/constructorStandings.json"
        cons_data = get_openf1_json(cons_url)
        constructors_raw = cons_data['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']

        constructors = [
            {
                "team": c['Constructor']['name'],
                "team_color": TEAM_DATA.get(c['Constructor']['name'], {}).get('color', '#9aa6b2'),
                "points": c['points']
            }
            for c in constructors_raw
        ]
        
        data_year = selected_year
        logging.info(f"Successfully fetched standings for {selected_year}")
        
    except Exception as e:
        logging.warning(f"Could not fetch standings for {selected_year}: {e}")
        
        # If requested year is current year and has no data, show grid list
        if selected_year == current_year:
            show_grid = True
            # Create driver list with team colors
            drivers = sorted([
                {
                    "name": d['name'],
                    "number": d['number'],
                    "team": d['team'],
                    "team_color": TEAM_DATA.get(d['team'], {}).get('color', '#9aa6b2')
                }
                for d in DRIVERS_DATA
            ], key=lambda x: x['name'])
            
            constructors = sorted([
                {
                    "team": team,
                    "team_color": TEAM_DATA.get(team, {}).get('color', '#9aa6b2')
                }
                for team in TEAM_DATA.keys()
            ], key=lambda x: x['team'])
            
            data_year = None
        else:
            # If a previous year was selected and has no data, try falling back
            for try_year in range(selected_year - 1, selected_year - 5, -1):
                try:
                    drv_url = f"https://ergast.com/api/f1/{try_year}/driverStandings.json"
                    drv_data = get_openf1_json(drv_url)
                    drivers_raw = drv_data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']

                    drivers = [
                        {
                            "name": f"{d['Driver']['givenName']} {d['Driver']['familyName']}",
                            "team": d['Constructors'][0]['name'],
                            "team_color": TEAM_DATA.get(d['Constructors'][0]['name'], {}).get('color', '#9aa6b2'),
                            "points": d['points']
                        }
                        for d in drivers_raw
                    ]

                    cons_url = f"https://ergast.com/api/f1/{try_year}/constructorStandings.json"
                    cons_data = get_openf1_json(cons_url)
                    constructors_raw = cons_data['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']

                    constructors = [
                        {
                            "team": c['Constructor']['name'],
                            "team_color": TEAM_DATA.get(c['Constructor']['name'], {}).get('color', '#9aa6b2'),
                            "points": c['points']
                        }
                        for c in constructors_raw
                    ]
                    
                    data_year = try_year
                    logging.info(f"Successfully fetched standings for fallback year {try_year}")
                    break
                except Exception as inner_e:
                    logging.warning(f"Could not fetch standings for {try_year}: {inner_e}")
                    continue

    return render_template(
        "championship.html",
        drivers=drivers,
        constructors=constructors,
        current_year=current_year,
        selected_year=selected_year,
        data_year=data_year,
        show_grid=show_grid,
        available_years=available_years
    )

if __name__ == "__main__":
    if not os.path.exists('cache'):
        os.makedirs('cache')
    # Use port 5001 to avoid macOS ControlCenter/AirPlay on 5000
    app.run(debug=True, port=5001)