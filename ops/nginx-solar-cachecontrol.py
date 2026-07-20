#!/usr/bin/env python3
"""nginx aufraeumen: Cache-Control fuer /solar-system/ + Backups aus
sites-enabled entfernen.

SCHRITT 1 — Cache-Control
    Ohne das liefert der Browser die im Betrieb geaenderte index.html aus
    dem Cache; jede Fehlersuche laeuft dann gegen alten Code. /haemotrace/
    und /ai-atc/ machen es bereits so, hier wird genau deren Muster
    uebernommen. Betrifft BEIDE /solar-system/-Bloecke (HTTP :80 und
    HTTPS :443). Nur index.html, Assets bleiben cachebar.

SCHRITT 2 — Backups aus sites-enabled heraus
    nginx laedt `include /etc/nginx/sites-enabled/*`, also AUCH
    narnia.bak-solar-cuda und narnia.bak-solar-lan. Beide deklarieren
    denselben server_name wie die aktive Config; nginx nimmt alphabetisch
    die erste (narnia) und verwirft die anderen mit einer Warnung. Es
    funktioniert also nur zufaellig. Geprueft: beide Backups enthalten
    KEINE Zeile, die nicht auch in der aktiven Config steht — es geht
    nichts verloren. Sie werden nach /etc/nginx/sites-backup/ VERSCHOBEN,
    nicht geloescht.

Aufruf:
    sudo python3 nginx-solar-cachecontrol.py --pruefen   # nur anzeigen
    sudo python3 nginx-solar-cachecontrol.py             # anwenden
"""
from __future__ import annotations

import argparse
import difflib
import pathlib
import shutil
import subprocess
import sys
import time

# Die AKTIVE Datei. sites-enabled/narnia ist hier eine echte Datei, kein
# Symlink auf sites-available (siehe Schlusshinweis).
CONFIG = pathlib.Path("/etc/nginx/sites-enabled/narnia")
ENABLED = CONFIG.parent
ABLAGE = pathlib.Path("/etc/nginx/sites-backup")
BACKUPS = ["narnia.bak-solar-cuda", "narnia.bak-solar-lan"]

ALT = """    location /solar-system/ {
        alias /var/www/html/solar-system/;
        index index.html;
        try_files $uri $uri/ /solar-system/index.html;
    }
"""

NEU = """    location /solar-system/ {
        alias /var/www/html/solar-system/;
        index index.html;
        try_files $uri $uri/ /solar-system/index.html;

        # index.html nie cachen: die Datei wird im laufenden Betrieb
        # geaendert, und ein alter Stand im Browser-Cache verfaelscht
        # jede Fehlersuche. Gleiches Muster wie /haemotrace/ und
        # /ai-atc/. Assets bleiben cachebar.
        location = /solar-system/index.html {
            alias /var/www/html/solar-system/index.html;
            add_header Cache-Control "no-cache, no-store, must-revalidate";
        }
    }
"""


def cache_control(trocken: bool) -> str | None:
    """Rueckgabe: neuer Dateiinhalt, oder None wenn nichts zu tun ist."""
    text = CONFIG.read_text()
    treffer = text.count(ALT)
    if treffer == 0:
        if "location = /solar-system/index.html" in text:
            print("Schritt 1: Cache-Control steht bereits — nichts zu tun.")
            return None
        raise SystemExit(
            f"ABBRUCH: Muster in {CONFIG} nicht gefunden. Die Config hat "
            f"sich geaendert; bitte von Hand pruefen.")
    neu = text.replace(ALT, NEU)
    print(f"Schritt 1: {treffer} /solar-system/-Block/Bloecke aendern\n")
    print("".join(difflib.unified_diff(
        text.splitlines(keepends=True), neu.splitlines(keepends=True),
        fromfile=str(CONFIG), tofile=str(CONFIG) + " (neu)", n=2)))
    return None if trocken else neu


def backups_raus(trocken: bool) -> list[pathlib.Path]:
    vorhanden = [ENABLED / b for b in BACKUPS if (ENABLED / b).exists()]
    if not vorhanden:
        print("Schritt 2: keine Backups in sites-enabled — nichts zu tun.")
        return []
    print(f"\nSchritt 2: {len(vorhanden)} Datei(en) nach {ABLAGE} "
          f"verschieben:")
    for p in vorhanden:
        print(f"    {p}  ->  {ABLAGE / p.name}")
    if trocken:
        return []
    ABLAGE.mkdir(exist_ok=True)
    verschoben = []
    for p in vorhanden:
        ziel = ABLAGE / p.name
        shutil.move(str(p), str(ziel))
        verschoben.append(ziel)
    return verschoben


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pruefen", action="store_true",
                    help="nur anzeigen, nichts aendern")
    args = ap.parse_args()

    neu_text = cache_control(args.pruefen)
    verschoben = backups_raus(args.pruefen)
    if args.pruefen:
        print("\n--pruefen: nichts geaendert.")
        return 0
    if neu_text is None and not verschoben:
        return 0

    alt_text = CONFIG.read_text()
    sicherung = None
    if neu_text is not None:
        # Sicherung NICHT nach sites-enabled — genau das ist ja das
        # Problem, das Schritt 2 beseitigt.
        ABLAGE.mkdir(exist_ok=True)
        sicherung = ABLAGE / f"narnia.vor-cachecontrol-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(CONFIG, sicherung)
        print(f"\nSicherung: {sicherung}")
        CONFIG.write_text(neu_text)

    pruef = subprocess.run(["nginx", "-t"], capture_output=True, text=True)
    print(pruef.stderr.strip())
    if pruef.returncode != 0:
        if neu_text is not None:
            CONFIG.write_text(alt_text)
        for z in verschoben:
            shutil.move(str(z), str(ENABLED / z.name))
        print("nginx -t fehlgeschlagen — alles zurueckgenommen.",
              file=sys.stderr)
        return 1

    nach = subprocess.run(["systemctl", "reload", "nginx"],
                          capture_output=True, text=True)
    if nach.returncode != 0:
        print(nach.stderr.strip(), file=sys.stderr)
        return 1
    print("\nFertig. nginx neu geladen.")
    print("  - /solar-system/index.html wird nicht mehr gecacht")
    print("  - sites-enabled enthaelt nur noch aktive Configs")
    print("\nOffen bleibt: sites-available/narnia (Mai) ist NICHT identisch")
    print("mit der aktiven sites-enabled/narnia (Juli) — zwei Wahrheiten.")
    print("Das ist eine eigene Entscheidung, hier bewusst nicht angefasst.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
