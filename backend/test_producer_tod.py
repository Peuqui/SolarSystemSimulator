"""Stirbt der Producer, muss der Stream das melden statt ewig zu warten.

Der Producer laeuft als eigener Prozess. Sein Traceback landet im
Server-Log — der Client sieht davon nichts. `stream()` schleift auf
`running_val`, das aber NUR der Server beim `stop()` zuruecksetzt: ein
abgestuerzter Producer laesst `head_val` einfach stehen, und die Schleife
wartet in 30-ms-Schritten auf Samples, die nie kommen. Der Film startet
dann nicht, ohne eine einzige Meldung.

Ausgeloest wurde das im Betrieb durch mehr als `M_MAX` massive Koerper:
`load_state` lehnt sie ab (zu Recht — sie liegen im Shared Memory des
Kernels), der Producer stirbt beim Start, und der Film hing. Genau diese
Szene stellt der Test her.

Gegenprobe ist der Normalbetrieb: ein gesunder Producer darf keinen
Fehler ausloesen. (Der Zerberst-Fall — Producer stoppt sich absichtlich
selbst — ist ueber `shatter_flag` ausgenommen; ihn zu provozieren
verlangt eine echte Kollision und bleibt `test_film_golden.py`.)

Aufruf: ./venv/bin/python backend/test_producer_tod.py
"""
from __future__ import annotations

import asyncio
import struct
import sys
import time

import numpy as np

import film_producer
import server
from nbody_kernel import M_MAX

FRIST_S = 20.0      # so lange darf stream() hoechstens brauchen


class WSAttrappe:
    """Sammelt, was der Stream sendet. Mehr braucht stream() nicht."""

    def __init__(self):
        self.pakete = []

    async def send(self, paket):
        self.pakete.append(paket)


def szene(m_massiv: int, n_ast: int = 200):
    n = m_massiv + n_ast
    rng = np.random.default_rng(1)
    x = rng.uniform(-50, 50, n)
    y = rng.uniform(-50, 50, n)
    mass = np.zeros(n)
    mass[:m_massiv] = 1e-6
    real_r = np.full(n, 5.79e-7)
    is_ast = np.ones(n, np.uint8)
    is_ast[:m_massiv] = 0
    return {
        "x": x, "y": y, "vx": np.zeros(n), "vy": np.zeros(n),
        "mass": mass, "real_r": real_r,
        "visible": np.ones(n, np.uint8), "is_ast": is_ast,
        "is_star_bh": np.zeros(n, np.uint8),
        "injiziert": np.zeros(n, np.uint8),
    }


def sitzung(s: dict) -> server.FilmSession:
    return server.FilmSession(
        0.0, 0.5, s["x"], s["y"], s["vx"], s["vy"], s["mass"],
        s["real_r"], s["visible"], s["is_ast"], s["is_star_bh"],
        s["injiziert"], False, film_producer.SUB_SAMPLES_DEFAULT)


def fehlerpakete(ws: WSAttrappe) -> list[str]:
    """Pakete mit Status != 0 als Text — build_error legt ihn ab Byte 8."""
    raus = []
    for p in ws.pakete:
        if len(p) >= 8 and struct.unpack_from("<I", p, 0)[0] == 1:
            raus.append(p[8:].decode(errors="replace"))
    return raus


async def fall_absturz() -> bool:
    """Producer stirbt beim Start (M_MAX ueberschritten)."""
    print(f"\n--- Absturz: {M_MAX + 1} massive Koerper (M_MAX = {M_MAX})")
    film = sitzung(szene(M_MAX + 1))
    ws = WSAttrappe()
    try:
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(film.stream(ws), FRIST_S)
        except asyncio.TimeoutError:
            print(f"FEHLGESCHLAGEN — stream() haengt noch nach "
                  f"{FRIST_S:.0f} s (head={film.head_val.value}, "
                  f"proc lebt={film.proc.is_alive()})")
            return False
        dauer = time.monotonic() - t0
        fehler = fehlerpakete(ws)
        if not fehler:
            print(f"FEHLGESCHLAGEN — stream() kehrte nach {dauer:.1f} s "
                  f"zurueck, aber OHNE Fehlermeldung "
                  f"({len(ws.pakete)} Pakete)")
            return False
        print(f"BESTANDEN — nach {dauer:.1f} s gemeldet: {fehler[0]!r}")
        return True
    finally:
        film.stop()


async def fall_normal() -> bool:
    """Gesunder Producer: kein Fehler, Samples fliessen."""
    print("\n--- Normalbetrieb: 1 massiver Koerper")
    film = sitzung(szene(1))
    ws = WSAttrappe()
    try:
        film.resubscribe(0.0, 5.0, jump=True)
        aufgabe = asyncio.create_task(film.stream(ws))
        t0 = time.monotonic()
        while time.monotonic() - t0 < FRIST_S and not aufgabe.done():
            film.playhead_val.value = film.t0
            if len(ws.pakete) >= 3:
                break
            await asyncio.sleep(0.2)
        fehler = fehlerpakete(ws)
        aufgabe.cancel()
        if fehler:
            print(f"FEHLGESCHLAGEN — Fehler im Normalbetrieb: {fehler[0]!r}")
            return False
        if len(ws.pakete) < 3:
            print(f"FEHLGESCHLAGEN — nur {len(ws.pakete)} Pakete in "
                  f"{FRIST_S:.0f} s, der Stream laeuft nicht")
            return False
        print(f"BESTANDEN — {len(ws.pakete)} Pakete, keine Fehlermeldung")
        return True
    finally:
        film.stop()


async def main() -> int:
    server.FilmSession.MAX_BYTES = 64 << 20
    ergebnisse = [await fall_absturz(), await fall_normal()]
    print()
    if all(ergebnisse):
        print("ALLE BESTANDEN")
        return 0
    print("FEHLGESCHLAGEN")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
