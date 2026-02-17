#!/usr/bin/env python3
"""Compute minimal metadata for a race using FastF1 and save as JSON.

Usage:
  python replay/compute_metadata.py --year 2025 --gp "Australian Grand Prix"

This script writes to `replay/computed/<YEAR>_<GP_SLUG>_meta.json` where GP_SLUG
is the folder-style name (underscores instead of spaces).
"""
import argparse
import os
import json
import fastf1
import logging

logging.basicConfig(level=logging.INFO)

FPS = 25


def slugify(name: str) -> str:
    return name.replace(' ', '_')


def enable_cache():
    cache_path = 'cache'
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)
    fastf1.Cache.enable_cache(cache_path)


def compute_metadata(year: int, gp_name: str):
    enable_cache()
    gp_for_fastf1 = gp_name.replace('_', ' ')
    logging.info(f'Loading session for {year} {gp_for_fastf1}...')
    try:
        session = fastf1.get_session(year, gp_for_fastf1, 'R')
        session.load(telemetry=False, laps=True)
    except Exception as e:
        logging.error(f'Failed to load session: {e}')
        return None

    # Gather drivers present in laps
    drivers = []
    try:
        for drv in session.drivers:
            try:
                d = session.get_driver(drv)
                drivers.append(d['Abbreviation'])
            except Exception:
                continue
    except Exception:
        drivers = []

    total_laps = 0
    try:
        if not session.laps.empty:
            total_laps = int(session.laps['LapNumber'].max())
    except Exception:
        total_laps = 0

    meta = {
        'year': year,
        'gp': gp_name,
        'drivers': drivers,
        'total_laps': total_laps,
    }

    out_dir = os.path.join('replay', 'computed')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{year}_{slugify(gp_name)}_meta.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    logging.info(f'Wrote metadata to {out_file}')
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--year', required=True, type=int)
    p.add_argument('--gp', required=True)
    args = p.parse_args()
    compute_metadata(args.year, args.gp)
