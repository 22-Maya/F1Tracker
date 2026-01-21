# F1 Tracker - Formula 1 Season Companion

A sleek, modern web application for tracking Formula 1 season schedules, races, and championship standings. Built with Flask, FastF1, and beautiful glass-morphism UI design.

## 🏁 Project Overview

**F1 Tracker** is a comprehensive F1 season companion that provides:
- **Live F1 Calendar**: Browse all races in the current season
- **Track Visualizations**: View 3D track layouts for each Grand Prix
- **Championship Standings**: Track driver and constructor championships across multiple seasons
- **Timezone Support**: Convert race times to your local timezone
- **Next Race Information**: Quick overview of upcoming races with detailed circuit maps

## 📋 Features

- 🗓️ **F1 Race Schedule** - Complete season calendar with timezone conversion
- 🏆 **Championship Standings** - Driver and constructor rankings from past seasons
- 🗺️ **Track Layouts** - Auto-generated circuit visualizations from telemetry data
- 🌍 **Global Timezone Support** - Convert race times to your local time zone
- 🎨 **Modern UI Design** - Glass-morphism aesthetic with smooth animations
- ⚡ **Performance Optimized** - Caching system for track images and telemetry data
- 📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile

## 🛠️ Technology Stack

### Backend
- **Flask** - Python web framework for routing and template rendering
- **FastF1** - Python library for F1 data retrieval and telemetry analysis
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing and array operations
- **Matplotlib** - Track visualization and chart generation
- **Requests** - HTTP client for API calls (OpenF1/Ergast API)

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Glass-morphism design and responsive layouts
- **JavaScript (Vanilla)** - Client-side interactivity
- **Luxon.js** - Timezone handling and date formatting

### Data Sources
- **FastF1** - Official F1 telemetry and session data
- **Ergast F1 API** - Historical standings and championship data

## 📂 File Structure

```
F1Tracker/
├── app.py                          # Main Flask application
├── README.md                       # This file
├── LEARNING_GUIDE.md              # In-depth educational guide
│
├── templates/                      # HTML templates
│   ├── index.html                 # Home page
│   ├── schedule.html              # F1 season schedule
│   ├── next_race.html             # Next race details & track view
│   ├── championship.html          # Championship standings
│   ├── nav.html                   # Navigation component
│   └── _macros.html               # Reusable template macros
│
├── static/                         # Static assets
│   ├── style.css                  # Global styling (glass-morphism)
│   └── images/                    # Team and driver logos
│       ├── teams/                 # F1 team logos
│       └── drivers/               # Driver logos
│
├── cache/                          # Data caching
│   ├── 2024/, 2025/, 2026/        # Telemetry data by year
│   └── generated/                 # Generated track images
│
└── documentation/                  # Additional docs
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   cd /Users/mayaitskovich/Desktop/GitHub/F1Tracker
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   .venv/bin/pip install flask fastf1 pandas matplotlib numpy requests
   ```

4. **Run the application**
   ```bash
   .venv/bin/python app.py
   ```

5. **Access the app**
   - Open your browser and navigate to `http://localhost:5001`
   - The app runs on port 5001 (avoids conflict with macOS ControlCenter)

## 📄 File Descriptions

### Backend

#### `app.py` (611 lines)
The main Flask application containing:
- **Route Handlers**: 6 main routes for different pages
  - `@app.route("/")` - Home page with season overview
  - `@app.route("/schedule")` - Complete race schedule
  - `@app.route("/next")` - Next race details
  - `@app.route("/race/<year>/<gp_name>")` - Individual race view
  - `@app.route("/track_image/<year>/<gp_name>")` - Track image endpoint
  - `@app.route("/championship")` - Championship standings

- **Data Functions**:
  - `load_calendar()` - Fetches F1 season schedule
  - `get_next_event()` - Identifies upcoming race
  - `draw_f1_circuit()` - Generates track visualizations
  - `get_openf1_json()` - Queries external APIs

- **Configuration**:
  - Team and driver data for 2026 season
  - Country-to-flag emoji mapping
  - Color schemes for UI elements

### Frontend

#### `index.html` (85 lines)
Home page with:
- Season overview and quick links
- Call-to-action buttons to other sections
- Timezone selector for race times

#### `schedule.html` (120+ lines)
Displays:
- Complete season race schedule in table format
- Country and date information
- Links to individual race pages
- Timezone conversion functionality

#### `next_race.html` (239 lines)
Shows:
- Next upcoming race details
- Dynamically generated track layout image
- Race location, date, and country info
- Regenerate button for fresh track data
- Side-by-side two-column layout

#### `championship.html` (160+ lines)
Features:
- Driver championship standings
- Constructor championship standings
- Year selector for viewing past seasons
- Team logos and color-coded team display
- Fallback to current-year grid if future data unavailable

#### `nav.html` (28 lines)
Navigation component:
- Fixed top navigation bar
- Active link indicators
- Hamburger menu for mobile
- Links to all major pages

#### `_macros.html` (18 lines)
Jinja2 template macro:
- `track_img()` macro for rendering track images
- Handles query parameters and caching

### Styling

#### `style.css` (362 lines)
Comprehensive styling including:
- **Glass-morphism Design**: Frosted glass cards with backdrop blur
- **Color Scheme**: Dark theme with accent colors
- **Responsive Layout**: Mobile-first with breakpoints at 880px and 768px
- **Components**: Navigation pills, cards, tables, forms
- **Animations**: Smooth transitions, hover effects, loading spinner
- **Typography**: Inter font family with custom font sizes

## 🔌 API Integration

### External APIs Used
1. **FastF1 API** - F1 telemetry and session data
   - Session loading
   - Track information
   - Position data

2. **Ergast F1 API** - Championship standings
   - `https://ergast.com/api/f1/{year}/driverStandings.json`
   - `https://ergast.com/api/f1/{year}/constructorStandings.json`

## ⚙️ Configuration

### Caching System
- FastF1 telemetry cached in `cache/` directory
- Generated track images cached with SHA256 hash filenames
- Automatic cache refresh via URL parameter
- 1-hour browser cache on images

### Port Configuration
- Application runs on **port 5001** (default Flask uses 5000)
- This avoids conflicts with macOS ControlCenter/AirPlay

### Data Season
- **Current Season**: 2026
- **Historical Data**: 2024, 2025, 2026
- **Driver/Constructor Data**: 22 drivers across 11 teams

## 🎨 Design Features

- **Glass-Morphism**: Modern frosted glass effect with backdrop blur
- **Dark Theme**: Eye-friendly dark background (#0f1724)
- **Accent Color**: Vibrant red (#ff4655) for highlights
- **Responsive Grid**: Auto-adapting layout for all screen sizes
- **Smooth Animations**: 180ms transitions throughout UI
- **Mobile First**: Hamburger menu and flexible layouts

## 🔄 Workflow Example

1. **User visits homepage** → Loads current season overview
2. **Clicks "Next Race"** → Fetches upcoming race from schedule
3. **Server generates track image** → Uses FastF1 telemetry data
4. **Image cached locally** → Future requests load instantly
5. **User changes timezone** → JavaScript converts all times client-side
6. **Clicks championship link** → Fetches from Ergast API
7. **Fallback to current grid** → If no standings available yet

## 📊 Season Data

### 2026 F1 Grid
- **11 Teams**: Mercedes, Red Bull, Ferrari, McLaren, Alpine, Aston Martin, Williams, Audi, Racing Bulls, Cadillac, Haas
- **22 Drivers**: Current F1 2026 roster
- **Multiple Sessions**: Practice, Qualifying, Race data available

## 🐛 Troubleshooting

### Track Image Won't Load
- Check Python console for FastF1 errors
- Ensure telemetry data is available for that session
- Try the "Regenerate" button on the race page

### Timezone Issues
- Timezone selector in top-right corner
- Times stored in UTC, converted client-side
- Requires JavaScript enabled

### Port Already in Use
- Change port in `app.py` line 600: `app.run(debug=True, port=YOUR_PORT)`

## 📚 For Learning

See `LEARNING_GUIDE.md` for:
- Deep dive into each Python library
- Detailed Flask concepts and routing
- HTML/CSS/JavaScript explanations
- How to extend the application

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

This is a personal project. Feel free to fork and customize for your needs!

---

**Made with ❤️ for Formula 1 fans**
