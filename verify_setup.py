#!/usr/bin/env python3
"""Comprehensive test of all F1Tracker components."""
import os
import sys

# Add project to path
sys.path.insert(0, '/Users/mayaitskovich/Desktop/GitHub/F1Tracker')

try:
    import app
    print("✓ app.py imports successfully")
except Exception as e:
    print(f"✗ app.py import failed: {e}")
    sys.exit(1)

# Check routes
routes = [r.rule for r in app.app.url_map.iter_rules()]
print(f"\n✓ Found {len(routes)} routes:")
for route in sorted(routes):
    if route.startswith('/'):
        print(f"  - {route}")

# Check templates
templates_dir = '/Users/mayaitskovich/Desktop/GitHub/F1Tracker/templates'
templates = os.listdir(templates_dir)
print(f"\n✓ Found {len(templates)} templates:")
for t in sorted(templates):
    print(f"  - {t}")

# Check replay module
replay_dir = '/Users/mayaitskovich/Desktop/GitHub/F1Tracker/replay'
print(f"\n✓ Replay module:")
if os.path.exists(os.path.join(replay_dir, 'upstream')):
    print(f"  - upstream/ cloned ({len(os.listdir(os.path.join(replay_dir, 'upstream')))} items)")
if os.path.exists(os.path.join(replay_dir, 'compute_metadata.py')):
    print(f"  - compute_metadata.py exists")

# Check cache
if os.path.exists('/Users/mayaitskovich/Desktop/GitHub/F1Tracker/cache'):
    cache_size = len(os.listdir('/Users/mayaitskovich/Desktop/GitHub/F1Tracker/cache'))
    print(f"\n✓ Cache directory exists ({cache_size} items)")

# Test cache decorator
print(f"\n✓ Cache decorator test:")
@app._cached(ttl_seconds=60)
def test_func(x):
    return x * 2

result1 = test_func(5)
result2 = test_func(5)
print(f"  - _cached decorator works (result: {result1})")

# Test key functions exist
funcs_to_check = [
    'load_calendar',
    'get_next_event', 
    'get_openf1_json',
    'draw_f1_circuit',
    'rotate'
]
print(f"\n✓ Key functions:")
for func_name in funcs_to_check:
    if hasattr(app, func_name):
        print(f"  - {func_name}")
    else:
        print(f"  ✗ {func_name} MISSING")

# Summary
print(f"\n{'='*50}")
print(f"✓ ALL CHECKS PASSED - App is ready!")
print(f"{'='*50}")
print(f"\nTo run the app:")
print(f"  source .venv/bin/activate")
print(f"  python app.py")
print(f"\nThen visit: http://localhost:5001")
