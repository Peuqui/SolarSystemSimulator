"""Persistenter GPU-Leistungs-Cache — misst einmal, speichert nach Hardware.

Wie der Shader-Cache eines Spiels: Beim ersten Lauf auf einer Bestueckung
werden die Karten vermessen; das Ergebnis liegt danach in
``~/.cache/solar-system/gpu_bench.json``. Wird eine Karte getauscht,
umgesteckt oder die Anzahl geaendert, passt die HARDWARE-SIGNATUR nicht mehr
und es wird automatisch neu gemessen.

Warum ueberhaupt gemessen statt Datenblatt: schon `selfgrav_kernel.miss_gewicht`
haelt fest, dass eine FLOP-Tabelle hier falsch herum liegt (die V100 schlaegt
eine RTX 8000, obwohl deren f32-Peak hoeher ist — es entscheidet die
Speicherbandbreite). Und verschiedene LASTEN kroenen verschiedene Karten:
ein kleiner all-pairs-Kernel ist compute-bound (RTX vorn), Particle-Mesh ist
FFT- und bandbreiten-bound (V100 vorn). Darum cachet dieses Modul die Scores
PRO LAST getrennt.

Hardware-agnostisch: eine einzelne GPU oder fuenf gemischte — die Messung
laeuft ueber genau die Karten, die stecken. Bei einer Karte ist jede Wahl
ohnehin trivial; der Cache haelt trotzdem ihren Score fest, damit der Start
nach dem ersten Mal ohne Messung auskommt.

Dieses Modul kennt WEDER `selfgrav_kernel` NOCH `pm_kernel` — die eigentlichen
Mess-Schleifen werden als Callback hereingereicht (`hole_gewichte`). So bleibt
die Persistenz frei von Import-Zyklen und dient beiden Lasten gleich.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import cupy as cp


def _cache_pfad() -> Path:
    basis = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(basis) / "solar-system" / "gpu_bench.json"


def _karten_roh() -> list[str]:
    """Je Karte ein stabiler Steckplatz-String: Name + PCI-Bus. Die
    REIHENFOLGE ist die CUDA-Ordnung und damit Teil der Identitaet — wird
    umgesteckt, aendert sich die Signatur, und das ist gewollt (die Scores
    haengen am CUDA-Index)."""
    teile = []
    for d in range(cp.cuda.runtime.getDeviceCount()):
        p = cp.cuda.runtime.getDeviceProperties(d)
        teile.append(f"{p['name'].decode()}@"
                     f"{p['pciBusID']:02x}:{p['pciDeviceID']:02x}")
    return teile


def hardware_signatur() -> str:
    """Kurzer Hash ueber alle CUDA-Karten (Name + PCI-Bus, in CUDA-Ordnung).
    Aendert sich, sobald eine Karte getauscht, umgesteckt oder die Anzahl
    veraendert wird."""
    roh = ";".join(_karten_roh())
    return hashlib.sha1(roh.encode()).hexdigest()[:16]


def _lade_roh() -> dict:
    """Der gespeicherte Cache, wenn seine Signatur zur JETZIGEN Hardware
    passt — sonst ein frisches, leeres Geruest (alte Scores verworfen)."""
    sig = hardware_signatur()
    pfad = _cache_pfad()
    if pfad.exists():
        try:
            daten = json.loads(pfad.read_text())
            if daten.get("signatur") == sig:
                daten.setdefault("scores", {})
                return daten
        except (json.JSONDecodeError, OSError):
            pass
    return {"signatur": sig, "karten": _karten_roh(), "scores": {}}


def _speichere(daten: dict) -> None:
    """Atomar schreiben (tmp + replace), damit ein Abbruch mitten im Schreiben
    keinen halben, unlesbaren Cache hinterlaesst."""
    pfad = _cache_pfad()
    pfad.parent.mkdir(parents=True, exist_ok=True)
    daten["gemessen"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = pfad.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(daten, indent=2, ensure_ascii=False))
    tmp.replace(pfad)


def hole_gewichte(art: str, mess_fn, devices: list[int]) -> dict[int, float]:
    """Scores der Karten fuer eine Last-``art`` ("allpairs_f32", "pm_fft", …),
    aus dem Cache oder frisch gemessen.

    ``mess_fn(device) -> float`` wird NUR fuer Karten aufgerufen, die noch
    nicht im (gueltigen) Cache stehen; neue Werte werden zurueckgeschrieben.
    Rueckgabe ist auf die angefragten ``devices`` beschraenkt.
    """
    daten = _lade_roh()
    art_scores: dict = daten["scores"].setdefault(art, {})
    fehlt = [d for d in devices if str(d) not in art_scores]
    for d in fehlt:
        art_scores[str(d)] = float(mess_fn(d))
    if fehlt:
        _speichere(daten)
    return {d: art_scores[str(d)] for d in devices}
