# HydroMod Schwalm

## Kurzbeschreibung

HydroMod ist ein tägliches, vollständig rasterverteiltes Niederschlag-Abfluss-Modell für die Schwalm in Mittelhessen. Das Modell arbeitet auf einem 20-m-Raster und simuliert den Tagesabfluss an den Pegeln Alsfeld, Heidelbach und Röllshausen. Für alle drei verschachtelten Pegeleinzugsgebiete wird ein gemeinsamer Parametersatz verwendet.

Berücksichtigt werden Niederschlag, potenzielle und tatsächliche Evapotranspiration, Bodenwasserspeicherung, Schnellabfluss, Zwischenabfluss, Grundwasserneubildung, schneller und langsamer Basisabfluss sowie ein D8-Routing mit zellspezifischen Reisezeiten.

## Voraussetzungen

- entwickelt und getestet mit Python 3.14 (der Großteil der Packages ab 3.10 oder höher verfügbar)
- benötigte Pythonpakete gemäß `requirements.txt`

Installation:

```bash
python -m pip install -r requirements.txt
```

## Eingangsdaten

Die Eingangsdaten werden unter `data/GeoDaten` erwartet.

| Datensatz | Dateiformat | Verwendung |
|---|---|---|
| Digitales Geländemodell | GeoTIFF (`.tif`, `.tiff`) | Modellraster, Hangneigung und D8-Routing |
| Modelldomäne | Shapefile (`.shp`) | Begrenzung und Maskierung des Modellgebiets |
| Schwalm-Flussnetz | Shapefile (`.shp`) | Stream Burning und Gewässerrouting |
| Nutzbare Feldkapazität | Shapefile (`.shp`) | Maximale Größe des Bodenwasserspeichers |
| Bodeneinheiten | Shapefile (`.shp`) | Eingelesen, im Endmodell neutral behandelt |
| LBM-DE2021 | GeoPackage (`.gpkg`) | Landbedeckung, Versiegelung und Vegetation |
| Pegelpunkte | GeoPackage (`.gpkg`) | Snapping und Ableitung der Pegeleinzugsgebiete |
| Beobachteter Tagesabfluss | CSV oder TXT (`.csv`, `.txt`) | Kalibrierung und Validierung |
| HYRAS-Tagesniederschlag | NetCDF (`.nc`) | Täglicher räumlicher Niederschlag |
| Potenzielle Evapotranspiration | ASCII-Raster (`.asc`) | Täglicher Verdunstungsanspruch |
| Reale Evapotranspiration | ASCII-Raster, teilweise in TAR-Archiven | Räumliche Plausibilisierung |
| Grundwasserneubildung | GeoTIFF (`.tif`) | Räumlicher Referenzvergleich |
| Sickerwasserrate | GeoTIFF (`.tif`) | Plausibilisierung der Perkolation |

Die Eingangsdaten sind aufgrund ihres Umfangs von etwa 20 GB nicht
Bestandteil dieses Repositorys. Die benötigten Datensätze und Dateiformate
sind hier aufgeführt. Quellen, räumliche und zeitliche Auflösungen sowie
die durchgeführten Aufbereitungsschritte sind in Kapitel 3 des
Abschlussberichts dokumentiert.

## Konfiguration und Start

Vor dem Start müssen im Skript mindestens

- `BASE_DIR`
- die Pegeldateien und
- die Pfade zu den beobachteten Durchflussdaten in `DAILY_GAUGE_SPECS`

angepasst werden. Projektrelative Pfade werden empfohlen.

Start:

```bash
python HydroMod.py
```

## Modellzeiträume

- Warm-up: 2021, acht Wiederholungen
- Kalibrierung: 2022–2023
- Validierung: 2024–2025

Die Kalibrierung erfolgt gemeinsam an allen drei Pegeln mit Differential Evolution und anschließender lokaler Verfeinerung.

## Modellablauf

Zu Beginn werden die räumlichen Eingangsdaten auf das gemeinsame
20-m-Modellraster übertragen. Das digitale Geländemodell wird konditioniert
und zur Ableitung der D8-Fließrichtungen sowie der Einzugsgebiete der drei
Pegel verwendet.

Für jeden Tag wird die Wasserbilanz rasterzellenweise berechnet. Der
Niederschlag wird auf Evapotranspiration, Bodenwasserspeicherung,
Schnellabfluss, Zwischenabfluss und Grundwasserneubildung aufgeteilt.
Die Grundwasserneubildung speist einen schnellen und einen langsamen
Grundwasserspeicher.

Die erzeugten Abflusskomponenten werden anschließend anhand
zellspezifischer Reisezeiten über das D8-Fließnetz zu den Pegeln Alsfeld,
Heidelbach und Röllshausen geroutet. Kalibrierung und Bewertung erfolgen
gemeinsam für die drei verschachtelten Pegeleinzugsgebiete.

## Ergebnisse

Die Ausgaben werden im Ordner

```text
results/daily_v9_4_nested_gauge_diagnostic_calibration/
```

gespeichert. Erzeugt werden unter anderem:

- Qobs- und Qsim-Zeitreihen für alle drei Pegel
- NSE, KGE und PBIAS
- kalibrierte Parameter und Kalibrierungsverlauf
- jährliche Wasserbilanzen
- Karten von Recharge, Zwischenabfluss und AET
- Pegel-, Snapping- und Routingdiagnosen
- D8-Einzugsgebiete der drei Pegel

## Einschränkungen

Das Modell enthält kein Schnee- oder Frostmodul. Das D8-Routing ist kein hydraulisches Flussmodell; Kanalquerschnitte, Rückstau, Überflutungen und Auenretention werden nicht explizit simuliert. Die meteorologischen Daten liegen räumlich gröber als das 20-m-Modellraster vor.

## Autoren

Erik Gemeinhardt  
Till Ferneding  
Jakob Glesmer
