"""Stirbt der Server hart, muss der Producer die GPU freigeben.

Der Producer laeuft als eigener Prozess und haelt CUDA-Kontexte. Stirbt
der Server, ohne aufzuraeumen, wird das Kind von init adoptiert und haelt
den GPU-Speicher weiter — ohne je wieder zu rechnen. Beobachtet: 1,35 GB
ueber vier Karten, 14 Stunden lang, nachdem ein Testlauf per `timeout`
abgewuergt worden war.

`daemon=True` allein schuetzt davor NICHT: Python beendet Daemon-Kinder
ueber einen atexit-Handler, und der laeuft bei SIGTERM oder SIGKILL nicht.
Deshalb prueft der Producer in seiner Hauptschleife, ob sich seine
Eltern-PID geaendert hat.

Der Test startet sich selbst als Server-Ersatz (`--kind`), schiesst ihn
mit SIGKILL ab und prueft, ob der Producer von sich aus geht.

Aufruf: ./venv/bin/python backend/test_waisen.py
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import numpy as np

import film_producer
import server

FRIST_S = 30.0      # so lange darf das Kind hoechstens brauchen


def szene(n=2000):
    rng = np.random.default_rng(1)
    x = np.concatenate([[0.0], rng.uniform(-50, 50, n - 1)])
    y = np.concatenate([[0.0], rng.uniform(-50, 50, n - 1)])
    mass = np.zeros(n)
    mass[0] = 1.0
    real_r = np.full(n, 5.79e-7)
    real_r[0] = 0.00465
    is_ast = np.ones(n, np.uint8)
    is_ast[0] = 0
    star = np.zeros(n, np.uint8)
    star[0] = 1
    return (x, y, np.zeros(n), np.zeros(n), mass, real_r, is_ast, star)


def kind_rolle():
    """Server-Ersatz: Session starten, PID melden, dann nur noch warten."""
    x, y, vx, vy, mass, real_r, is_ast, star = szene()
    n = len(x)
    sess = server.FilmSession(0.0, 0.5, x, y, vx, vy, mass, real_r,
                              np.ones(n, np.uint8), is_ast, star,
                              np.zeros(n, np.uint8), False,
                              film_producer.SUB_SAMPLES_DEFAULT)
    t0 = time.time()
    while sess.head_val.value < 5 and time.time() - t0 < 90:
        sess.playhead_val.value = sess.head
        time.sleep(0.2)
    print(f"PRODUCER_PID={sess.proc.pid}", flush=True)
    while True:                      # bewusst kein Aufraeumen
        sess.playhead_val.value = sess.head
        time.sleep(0.5)


def lebt(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main():
    proc = subprocess.Popen([sys.executable, __file__, "--kind"],
                            stdout=subprocess.PIPE, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
    producer_pid = None
    t0 = time.time()
    while time.time() - t0 < 120:
        zeile = proc.stdout.readline()
        if not zeile:
            break
        if zeile.startswith("PRODUCER_PID="):
            producer_pid = int(zeile.split("=")[1])
            break
    if producer_pid is None:
        print("FEHLGESCHLAGEN — Producer ist nie angelaufen")
        proc.kill()
        return 1

    print(f"Server-Ersatz {proc.pid}, Producer {producer_pid}")
    if not lebt(producer_pid):
        print("FEHLGESCHLAGEN — Producer war schon tot")
        proc.kill()
        return 1

    print("SIGKILL auf den Server-Ersatz (keine atexit-Handler)")
    os.kill(proc.pid, signal.SIGKILL)

    t0 = time.time()
    while time.time() - t0 < FRIST_S:
        if not lebt(producer_pid):
            dauer = time.time() - t0
            print(f"\nBESTANDEN — Producer hat sich nach {dauer:.1f} s "
                  f"selbst beendet")
            return 0
        time.sleep(0.5)

    print(f"\nFEHLGESCHLAGEN — Producer lebt nach {FRIST_S:.0f} s immer "
          f"noch und haelt die GPU (PID {producer_pid}, PPID "
          f"{os.stat(f'/proc/{producer_pid}').st_uid and ''}"
          f"{open(f'/proc/{producer_pid}/stat').read().split()[3]})")
    try:
        os.kill(producer_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return 1


if __name__ == "__main__":
    if "--kind" in sys.argv:
        kind_rolle()
    else:
        sys.exit(main())
