"""
separation.py — Hand-/Rollen-Trennung (heuristisch).

Keine notengenaue Links/Rechts-Zuordnung noetig, sondern 'Comping/Voicing'
vs. 'Melodielinie'. Staerkste Heuristik: Onset-Gleichzeitigkeit.
Ehrliche Grenzen: Stride, Locked-Hands-Blockakkorde und Handkreuzungen
tricksen das aus -> Auswertung bleibt deshalb statistisch, nicht per Note.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from .core import Note

SIMUL_WINDOW = 0.045    # s: Onsets innerhalb -> gemeinsam angeschlagen


@dataclass
class Cluster:
    """Gleichzeitig angeschlagene Noten = Akkord/Voicing (Comping)."""
    notes: list[Note]
    @property
    def onset(self) -> float: return min(n.onset for n in self.notes)
    @property
    def pcs(self) -> set[int]: return {n.pc for n in self.notes}
    @property
    def pitches(self) -> list[int]: return sorted(n.pitch for n in self.notes)


@dataclass
class Separated:
    line: list[Note]          # einstimmige Melodie/Solo-Linie
    clusters: list[Cluster]   # Begleit-Voicings
    line_role: str = "rh"     # vermutete Hand
    comp_role: str = "lh"


def _group_by_onset(notes: list[Note], window: float = SIMUL_WINDOW) -> list[list[Note]]:
    groups, cur = [], []
    for n in sorted(notes, key=lambda x: x.onset):
        if not cur or n.onset - cur[0].onset <= window:
            cur.append(n)
        else:
            groups.append(cur)
            cur = [n]
    if cur:
        groups.append(cur)
    return groups


def separate(notes: list[Note]) -> Separated:
    """Trennt Voicings (>=3 gleichzeitige Toene) von der Melodielinie.
    Zweiklaenge werden dem hoeheren Ton (Linie) + tieferem (Comp) zugeordnet."""
    line, clusters = [], []
    for grp in _group_by_onset(notes):
        if len(grp) >= 3:
            clusters.append(Cluster(grp))
        elif len(grp) == 2:
            hi = max(grp, key=lambda n: n.pitch)
            lo = min(grp, key=lambda n: n.pitch)
            line.append(hi)
            clusters.append(Cluster([lo]))     # tiefer Ton als (Teil-)Begleitung
        else:
            line.append(grp[0])
    # Plausibilitaet: liegt die "Linie" im Schnitt tiefer als die Voicings,
    # ist die Rollen-Annahme evtl. invertiert (z.B. Bass-Solo). Nur markieren.
    line = sorted(line, key=lambda n: n.onset)
    sep = Separated(line=line, clusters=clusters)
    if line and clusters:
        line_avg = sum(n.pitch for n in line) / len(line)
        comp_avg = sum(p for c in clusters for p in c.pitches) / max(
            1, sum(len(c.pitches) for c in clusters))
        if line_avg < comp_avg - 2:
            sep.line_role, sep.comp_role = "lh?", "rh?"
    return sep
