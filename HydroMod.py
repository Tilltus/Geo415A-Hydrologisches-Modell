"""
HydroMod v9.3 – flächenbewusstes Multi-Pegel- und Rastermodell
für GEO415A / Mittelhessen.

Die Datei enthält weiterhin bewährte V8-Geodatenhilfen, startet aber
standardmäßig ausschließlich den täglichen V9.1-Workflow.

Wesentliche Verbesserungen gegenüber V9.0:
1. Tägliche räumliche HYRAS-Niederschlagswerte pro Modellzelle.
2. Bilineare Übertragung der täglichen PET auf Modellzellzentren.
3. Expliziter Zwischenabflussspeicher statt vollständiger Zuordnung des
   nicht schnellen Sättigungsüberschusses zur Grundwasserneubildung.
4. Feuchteabhängige, nichtlineare Bodenperkolation.
5. Schneller und langsamer Grundwasserspeicher sowie D8-Tagesrouting.
6. Konsistenter Mehrfach-Spin-up in Kalibrierung und finaler Simulation.
7. Mehrzielkalibrierung aus NSE, KGE, Q-PBIAS, mittlerer Recharge-Abweichung
   und räumlicher Recharge-Korrelation.
8. Zusätzliche Karten und Tabellen für Zwischenabfluss, Recharge-Komponenten,
   Speicherzustände und Wasserbilanzresiduen.

Methodische Einschränkungen:
- Ohne Temperaturdaten gibt es kein Schnee-/Frostmodul.
- Bodeneinheiten bleiben neutral, wenn nur BN_ID ohne fachliche Legende vorliegt.
- Die Recharge-Referenz kann einen anderen Bezugszeitraum oder eine andere
  Produktdefinition als die Modellsimulation besitzen; deshalb wird eine
  Toleranzzone verwendet.
- Das D8-Reisezeitrouting ist kein hydraulisches 1D-Flussmodell.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings
import heapq
import json
import shutil
import tarfile
from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.transform import rowcol, xy
from rasterio.io import MemoryFile
from affine import Affine
from shapely.geometry import Point


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # Projektstamm: Der Code ergänzt selbst \"data/GeoDaten\" und \"results\".
    BASE_DIR: Path = Path(
        r"C:\Users\E\Downloads\Projekt_HydroMod\Projekt_HydroMod"
    )

    START: str = "2021-01-01"
    END: str = "2025-12-31"

    # Zeiträume
    WARMUP_END: str = "2021-12-01"
    CALIB_START: str = "2022-01-01"
    CALIB_END: str = "2023-12-01"
    VALID_START: str = "2024-01-01"
    VALID_END: str = "2025-12-01"

    # Datenordner
    CATCHMENT_DIRNAME: str = "einzugsgebiet_shp"
    DGM_DIRNAME: str = "dgm_tif"
    NFK_DIRNAME: str = "Nutzbare_Feldkapazitaet100cm_50000__epsg25832_shp"
    FC_FALLBACK_DIRNAME: str = "Feldkapazitaet100cm_50000__epsg25832_shp"
    PRECIP_DIRNAME: str = "Niederschlag_5nc"
    PET_DIRNAME: str = "evaporation_potenziell"
    QOBS_DIRNAME: str = "Heidelbach durchfluss_5csv"
    RIVER_DIRNAME: str = "schwalm fluss_shp"

    # Die Datei liegt laut aktueller Ordnerstruktur direkt unter GeoDaten.
    GAUGE_FILENAME: str = "Heidelbach_Pegel.gpkg"
    GAUGE_LAYER: Optional[str] = None
    PRIMARY_GAUGE_NAME: str = "Heidelbach"
    PRIMARY_STATION_ID: str = "42880550"
    PRIMARY_GAUGE_OFFICIAL_AREA_KM2: float = 162.19

    # LBM-DE kann entweder eine echte GPKG-Datei oder ein Ordner sein, in dem
    # sich die entpackte Datei befindet.
    LBM_PATHNAME: str = "lbm-de2021.utm32s.gpkg"
    LBM_LAYER: Optional[str] = None
    LBM_SIE_FIELD: Optional[str] = "SIE_AKT"
    LBM_VEG_FIELD: Optional[str] = "VEG_AKT"
    LBM_CLC_FIELD: Optional[str] = "CLC21"

    # Weitere Pegel: station_id -> Qobs-Unterordner. Neue Einträge erst dann
    # ergänzen, wenn Punkt und Durchflussdaten wirklich vorliegen.
    GAUGE_QOBS_FOLDERS: tuple[tuple[str, str], ...] = (
        ("42880550", "Heidelbach durchfluss_5csv"),
        # ("WEITERE_ID", "Weiterer Pegel durchfluss_5csv"),
    )

    # Manuelle Spaltenwahl
    NFK_COLUMN: Optional[str] = None
    P_VAR: Optional[str] = None
    QOBS_DATE_COLUMN: Optional[str] = None
    QOBS_VALUE_COLUMN: Optional[str] = None

    # Einheiten
    P_UNIT_FACTOR: float = 1.0
    PET_UNIT_FACTOR: float = 0.1
    PET_ASC_CRS_FALLBACK: str = "EPSG:31467"
    FC_UNIT_FACTOR: float = 1.0
    DEFAULT_FC_MM: float = 150.0

    # Kalibrierung
    USE_CALIBRATION: bool = True
    RANDOM_SEED: int = 42

    # DGM und Routing
    ROUTING_ENABLED: bool = True
    CROP_PADDING_CELLS: int = 3
    GAUGE_SNAP_RADIUS_M: float = 250.0
    GAUGE_SNAP_DISTANCE_PENALTY_M: float = 50.0
    STREAM_BURN_DEPTH_M: float = 2.0
    ROUTING_MIN_SLOPE: float = 0.0005
    ROUTING_SLOPE_EXPONENT: float = 0.5
    CHANNEL_SPEED_MULTIPLIER: float = 8.0
    ROUTING_VELOCITY_M_S: float = 0.20
    CALIBRATE_ROUTING_VELOCITY: bool = True
    ROUTING_VELOCITY_CANDIDATES: tuple[float, ...] = (
        0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 0.75, 1.00
    )
    MAX_ROUTING_LAG_MONTHS: int = 6

    # Hangneigung
    SLOPE_QUICKFLOW_WEIGHT: float = 0.75

    # LBM-DE / Landbedeckung
    LANDUSE_ENABLED: bool = True
    DEFAULT_SEALED_PERCENT: float = 5.0
    DEFAULT_VEGETATION_PERCENT: float = 70.0
    LANDUSE_QUICKFLOW_IMPERVIOUS_WEIGHT: float = 1.50
    LANDUSE_QUICKFLOW_LOW_VEG_WEIGHT: float = 0.35
    LANDUSE_PET_VEGETATION_WEIGHT: float = 0.30
    LANDUSE_PERC_IMPERVIOUS_REDUCTION: float = 0.85

    # Eingangsmittelwerte werden für die primäre Pegelfläche berechnet.
    FORCING_USE_PRIMARY_GAUGE_CATCHMENT: bool = True

    # V8: Bodeneinheiten und räumliche Referenzprodukte
    SOIL_UNITS_DIRNAME: str = "Bodeneinheiten_50000__epsg25832_shp"
    SOIL_CLASS_FIELD: Optional[str] = None  # Feld aus Bodeneinheiten_50000..., nicht aus LBM-DE
    GWN_REFERENCE_DIRNAME: str = "jährliche grundwasserneubildung_tif"
    SEEPAGE_REFERENCE_DIRNAME: str = "jährliche sickerwasserrate mm_tif"
    AET_REFERENCE_DIRNAME: str = "Reale_Evotranspiration"
    GWN_REFERENCE_UNIT_FACTOR: float = 1.0
    SEEPAGE_REFERENCE_UNIT_FACTOR: float = 1.0
    AET_REFERENCE_UNIT_FACTOR: float = 0.1
    AET_REFERENCE_CRS_FALLBACK: str = "EPSG:31467"

    # Tatsächliche Ordnerstruktur unter BASE_DIR/data/GeoDaten:
    # evaporation_potenziell
    # dgm_tif
    # Reale_Evotranspiration
    # Nutzbare_Feldkapazitaet100cm_50000__epsg25832_shp
    # Niederschlag_5nc
    # Feldkapazitaet100cm_50000__epsg25832_shp
    # Heidelbach durchfluss_5csv
    # Bodeneinheiten_50000__epsg25832_shp
    # Heidelbach_Pegel.gpkg
    # lbm-de2021.utm32s.gpkg
    # schwalm fluss_shp
    # jährliche grundwasserneubildung_tif
    # jährliche sickerwasserrate mm_tif
    # einzugsgebiet_shp

    # V8: Prozessstruktur und Kalibrierung
    ET_STRESS_EXPONENT: float = 0.75
    SPINUP_CYCLES: int = 1
    CALIBRATION_SAMPLE_CELLS: int = 12000
    CALIBRATION_GLOBAL_MAXITER: int = 28
    CALIBRATION_GLOBAL_POPSIZE: int = 6
    FULL_GRID_REFINEMENT_FRACTIONS: tuple[float, ...] = (0.12, 0.05)
    OBJECTIVE_WEIGHT_NSE: float = 0.50
    OBJECTIVE_WEIGHT_KGE: float = 0.30
    OBJECTIVE_WEIGHT_PBIAS: float = 0.20


CFG = Config()


# =============================================================================
# HILFSFUNKTIONEN: DATEIEN UND RASTER
# =============================================================================

def data_dir() -> Path:
    return CFG.BASE_DIR / "data" / "GeoDaten"


def out_dir() -> Path:
    p = CFG.BASE_DIR / "results" / "monthly_multigauge_routing_landuse"
    (p / "maps").mkdir(parents=True, exist_ok=True)
    (p / "plots").mkdir(parents=True, exist_ok=True)
    (p / "tables").mkdir(parents=True, exist_ok=True)
    return p


def find_files(folder: Path, suffixes: Iterable[str]) -> list[Path]:
    suffixes = tuple(s.lower() for s in suffixes)
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes])


def find_first(folder: Path, suffixes: Iterable[str]) -> Path:
    files = find_files(folder, suffixes)
    if not files:
        raise FileNotFoundError(f"Keine Datei mit {suffixes} gefunden in: {folder}")
    return files[0]


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return arr, profile


def write_geotiff(path: Path, arr: np.ndarray, ref_profile: dict, nodata: float = -9999.0) -> None:
    profile = ref_profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="lzw"
    )
    out = np.where(np.isfinite(arr), arr, nodata).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def reproject_to_ref(
    src_path: Path,
    ref_profile: dict,
    src_crs_fallback: Optional[str] = None,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype("float64")
        if src.nodata is not None:
            src_arr[src_arr == src.nodata] = np.nan

        src_crs = src.crs if src.crs is not None else src_crs_fallback
        if src_crs is None:
            raise ValueError(
                f"{src_path} hat kein CRS. Setze z.B. PET_ASC_CRS_FALLBACK='EPSG:25832'."
            )

        dst_arr = np.full(
            (ref_profile["height"], ref_profile["width"]),
            np.nan,
            dtype="float64"
        )

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst_arr



def crop_raster_to_mask(
    arr: np.ndarray,
    mask: np.ndarray,
    profile: dict,
    padding: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict, tuple[int, int]]:
    """
    Schneidet ein Raster auf die Bounding Box des Einzugsgebiets zu.

    Das reduziert den Speicherbedarf deutlich, weil das DGM für Deutschland
    wesentlich größer als das eigentliche Einzugsgebiet sein kann.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        raise ValueError("Die Einzugsgebietsmaske enthält keine gültigen Zellen.")

    r0 = max(int(rows.min()) - padding, 0)
    r1 = min(int(rows.max()) + padding + 1, mask.shape[0])
    c0 = max(int(cols.min()) - padding, 0)
    c1 = min(int(cols.max()) + padding + 1, mask.shape[1])

    arr_crop = arr[r0:r1, c0:c1]
    mask_crop = mask[r0:r1, c0:c1]

    window = Window(c0, r0, c1 - c0, r1 - r0)
    crop_profile = profile.copy()
    crop_profile.update(
        width=c1 - c0,
        height=r1 - r0,
        transform=window_transform(window, profile["transform"]),
    )

    return arr_crop, mask_crop, crop_profile, (r0, c0)


def fill_missing_dem_nearest(dgm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Füllt fehlende DGM-Werte innerhalb des Einzugsgebiets mit dem räumlich
    nächstgelegenen gültigen DGM-Wert.
    """
    from scipy.ndimage import distance_transform_edt

    valid = mask & np.isfinite(dgm)
    if not valid.any():
        raise ValueError("Im Einzugsgebiet wurden keine gültigen DGM-Werte gefunden.")

    missing_inside = mask & ~np.isfinite(dgm)
    if not missing_inside.any():
        return dgm.copy()

    # Für jedes Pixel Indizes des nächstgelegenen gültigen Pixels bestimmen.
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    filled = dgm.copy()
    filled[missing_inside] = dgm[nearest[0][missing_inside], nearest[1][missing_inside]]
    return filled


def calculate_slope_fraction(
    dgm: np.ndarray,
    mask: np.ndarray,
    profile: dict,
) -> np.ndarray:
    """
    Berechnet die Hangneigung aus dem DGM als dimensionslose Steigung
    (m Höhenunterschied pro m Horizontalentfernung).

    Für Karten kann das Ergebnis mit 100 multipliziert als Prozent dargestellt
    werden.
    """
    from scipy.ndimage import distance_transform_edt

    valid = mask & np.isfinite(dgm)
    if not valid.any():
        raise ValueError("Keine gültigen DGM-Zellen zur Hangneigungsberechnung.")

    # Außerhalb der Maske mit nächstem gültigen Wert füllen, damit np.gradient
    # an der Einzugsgebietsgrenze keine extremen künstlichen Kanten erzeugt.
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    filled = dgm[nearest[0], nearest[1]]

    dx = abs(float(profile["transform"].a))
    dy = abs(float(profile["transform"].e))

    dz_dy, dz_dx = np.gradient(filled.astype("float64"), dy, dx)
    slope = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope = np.where(mask, slope, np.nan)

    # Extremwerte aus DGM-Artefakten begrenzen, ohne normale steile Hänge
    # zu verändern.
    finite = slope[mask & np.isfinite(slope)]
    if finite.size:
        upper = float(np.nanpercentile(finite, 99.9))
        if upper > 0:
            slope = np.where(mask, np.minimum(slope, upper), np.nan)

    return slope


def make_slope_quickflow_factor(
    slope_fraction: np.ndarray,
    mask: np.ndarray,
    weight: float,
) -> np.ndarray:
    """
    Erzeugt einen dimensionslosen Faktor für den schnellen Abfluss.

    Steilere Zellen erhalten einen höheren schnellen Abflussanteil. Der Faktor
    wird so zentriert, dass sein Gebietsmittel ungefähr 1 bleibt. Dadurch wird
    der bereits kalibrierte mittlere alpha_fast-Wert nicht unnötig verschoben.
    """
    factor = np.ones_like(slope_fraction, dtype="float64")

    vals = slope_fraction[mask & np.isfinite(slope_fraction)]
    if vals.size == 0 or weight == 0:
        return np.where(mask, factor, np.nan)

    p10, p90 = np.nanpercentile(vals, [10, 90])
    if not np.isfinite(p10) or not np.isfinite(p90) or p90 <= p10:
        return np.where(mask, factor, np.nan)

    normalized = np.clip((slope_fraction - p10) / (p90 - p10), 0.0, 1.0)
    mean_normalized = float(np.nanmean(normalized[mask]))
    factor = 1.0 + weight * (normalized - mean_normalized)
    factor = np.clip(factor, 0.35, 1.80)
    return np.where(mask, factor, np.nan)


def _normalize_factor(
    arr: np.ndarray,
    mask: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """Normiert einen räumlichen Faktor auf Gebietsmittel 1."""
    out = np.asarray(arr, dtype="float64")
    vals = out[mask & np.isfinite(out)]
    if vals.size == 0:
        return np.where(mask, 1.0, np.nan)
    mean = float(np.nanmean(vals))
    if not np.isfinite(mean) or mean <= 0:
        return np.where(mask, 1.0, np.nan)
    out = np.clip(out / mean, minimum, maximum)
    return np.where(mask, out, np.nan)


def load_river_mask(ref_profile: dict, mask: np.ndarray) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    """Lädt die Schwalm-Flusslinie und rastert sie auf das Modellgrid."""
    river_dir = data_dir() / CFG.RIVER_DIRNAME
    river_path = find_first(river_dir, [".shp", ".gpkg", ".geojson"])
    print(f"[OK] Flussnetz für Routing: {river_path}")

    river = gpd.read_file(river_path)
    if river.empty:
        raise ValueError("Das Schwalm-Flussnetz ist leer.")
    if river.crs is None:
        warnings.warn("Flussnetz hat kein CRS. Referenz-CRS wird angenommen.")
        river = river.set_crs(ref_profile["crs"])
    river = river.to_crs(ref_profile["crs"])
    river = river[river.geometry.notna()].copy()

    shapes_river = [(geom, 1) for geom in river.geometry if not geom.is_empty]
    river_mask = rasterize(
        shapes=shapes_river,
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    river_mask &= mask

    print(f"[OK] Flusszellen im Modellgebiet: {int(river_mask.sum())}")
    return river_mask, river


def _find_lbm_vector_path() -> Path:
    candidate = data_dir() / CFG.LBM_PATHNAME
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        files = find_files(candidate, [".gpkg", ".shp"])
        if files:
            return files[0]

    files = [
        p for p in find_files(data_dir(), [".gpkg", ".shp"])
        if "lbm" in p.name.lower() or "landbedeck" in p.name.lower()
    ]
    if not files:
        raise FileNotFoundError(
            "Kein LBM-DE-Vektordatensatz gefunden. Erwartet wurde z. B. "
            f"{candidate}"
        )
    return files[0]


def _choose_polygon_layer(path: Path, manual_layer: Optional[str]) -> Optional[str]:
    if path.suffix.lower() != ".gpkg":
        return None
    try:
        import fiona
        layers = fiona.listlayers(path)
        if manual_layer is not None:
            if manual_layer not in layers:
                raise ValueError(
                    f"LBM_LAYER={manual_layer!r} nicht gefunden. Layer: {layers}"
                )
            return manual_layer

        candidates = []
        for layer in layers:
            try:
                with fiona.open(path, layer=layer) as src:
                    geom_type = str(src.schema.get("geometry", ""))
                    if "Polygon" not in geom_type:
                        continue
                    score = 0
                    low = layer.lower()
                    if "lbm" in low:
                        score += 5
                    if "land" in low or "bedeck" in low:
                        score += 3
                    score += min(len(src), 1_000_000) / 1_000_000
                    candidates.append((score, layer))
            except Exception:
                continue
        if not candidates:
            raise ValueError(f"Kein Polygonlayer im GeoPackage gefunden: {path}")
        candidates.sort(reverse=True)
        return candidates[0][1]
    except ImportError as exc:
        raise ImportError("Für GeoPackage-Layerauswahl wird fiona benötigt.") from exc


def _find_column(columns: Iterable[str], manual: Optional[str], tokens: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    if manual is not None:
        if manual not in cols:
            raise ValueError(f"Spalte {manual!r} nicht vorhanden. Spalten: {cols}")
        return manual
    lower = {str(c).lower(): c for c in cols}
    for token in tokens:
        if token.lower() in lower:
            return lower[token.lower()]
    for token in tokens:
        for c in cols:
            if token.lower() in str(c).lower():
                return c
    return None


def _numeric_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    finite = values[np.isfinite(values)]
    if len(finite) and float(finite.max()) <= 1.0:
        values = values * 100.0
    return values.clip(0.0, 100.0)


def _infer_sie_veg_from_clc(value) -> tuple[float, float]:
    """Fallback aus CLC-Code oder Klassenbezeichnung."""
    s = str(value).lower()
    nums = re.findall(r"\d+", s)
    code = int(nums[0]) if nums else None

    if code is not None:
        if 100 <= code < 200:
            return 65.0, 25.0
        if 200 <= code < 300:
            return 5.0, 70.0
        if 300 <= code < 400:
            return 2.0, 90.0
        if 400 <= code < 500:
            return 0.0, 80.0
        if 500 <= code < 600:
            return 0.0, 5.0

    if any(t in s for t in ["sied", "industrie", "verkehr", "bebau", "versiegel"]):
        return 70.0, 20.0
    if any(t in s for t in ["wald", "forst", "gehölz", "gehoelz"]):
        return 2.0, 92.0
    if any(t in s for t in ["acker", "landwirtschaft", "grünland", "gruenland"]):
        return 5.0, 70.0
    if any(t in s for t in ["wasser", "gewässer", "gewaesser"]):
        return 0.0, 5.0
    if any(t in s for t in ["moor", "sumpf", "feucht"]):
        return 0.0, 85.0
    return CFG.DEFAULT_SEALED_PERCENT, CFG.DEFAULT_VEGETATION_PERCENT


def load_landuse_rasters(
    ref_profile: dict,
    mask: np.ndarray,
    catchment: gpd.GeoDataFrame,
) -> dict[str, np.ndarray]:
    """Liest LBM-DE2021 und erzeugt hydrologisch nutzbare Rasterfaktoren."""
    defaults = {
        "sealed_percent": np.where(mask, CFG.DEFAULT_SEALED_PERCENT, np.nan),
        "vegetation_percent": np.where(mask, CFG.DEFAULT_VEGETATION_PERCENT, np.nan),
        "quickflow_factor": np.where(mask, 1.0, np.nan),
        "pet_factor": np.where(mask, 1.0, np.nan),
        "percolation_factor": np.where(mask, 1.0, np.nan),
    }
    if not CFG.LANDUSE_ENABLED:
        print("[INFO] LBM-DE/Landnutzung ist deaktiviert.")
        return defaults

    try:
        path = _find_lbm_vector_path()
        layer = _choose_polygon_layer(path, CFG.LBM_LAYER)
        print(f"[OK] LBM-DE-Datensatz: {path}")
        if layer is not None:
            print(f"[OK] LBM-DE-Layer: {layer}")

        # Bounding Box im Layer-CRS für schnelles Teilgebietlesen.
        bbox = None
        if path.suffix.lower() == ".gpkg":
            try:
                import fiona
                with fiona.open(path, layer=layer) as src:
                    layer_crs = src.crs_wkt or src.crs
                c = catchment.to_crs(layer_crs) if layer_crs else catchment
                bbox = tuple(c.total_bounds)
            except Exception:
                bbox = None

        gdf = gpd.read_file(path, layer=layer, bbox=bbox)
        if gdf.empty:
            raise ValueError("LBM-DE enthält im Modellgebiet keine Objekte.")
        if gdf.crs is None:
            warnings.warn("LBM-DE hat kein CRS. Referenz-CRS wird angenommen.")
            gdf = gdf.set_crs(ref_profile["crs"])
        gdf = gdf.to_crs(ref_profile["crs"])
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

        sie_col = _find_column(
            gdf.columns, CFG.LBM_SIE_FIELD,
            ["SIE", "versiegel", "impervious", "sealed"],
        )
        veg_col = _find_column(
            gdf.columns, CFG.LBM_VEG_FIELD,
            ["VEG", "vegetation", "vegetationsgrad"],
        )
        clc_col = _find_column(
            gdf.columns, CFG.LBM_CLC_FIELD,
            ["CLC21", "CLC", "landbedeckung", "LB", "LN", "klasse"],
        )

        print(f"[INFO] LBM-Spalten: SIE={sie_col}, VEG={veg_col}, Klasse={clc_col}")

        if sie_col is not None:
            sie = _numeric_percent(gdf[sie_col])
        else:
            sie = pd.Series(np.nan, index=gdf.index, dtype="float64")

        if veg_col is not None:
            veg = _numeric_percent(gdf[veg_col])
        else:
            veg = pd.Series(np.nan, index=gdf.index, dtype="float64")

        if clc_col is not None:
            inferred = gdf[clc_col].apply(_infer_sie_veg_from_clc)
            inf_sie = inferred.apply(lambda x: x[0])
            inf_veg = inferred.apply(lambda x: x[1])
            sie = sie.fillna(inf_sie)
            veg = veg.fillna(inf_veg)

        sie = sie.fillna(CFG.DEFAULT_SEALED_PERCENT).clip(0, 100)
        veg = veg.fillna(CFG.DEFAULT_VEGETATION_PERCENT).clip(0, 100)
        gdf["_SIE"] = sie
        gdf["_VEG"] = veg

        def burn(column: str, fill: float) -> np.ndarray:
            shapes_values = [
                (geom, float(value))
                for geom, value in zip(gdf.geometry, gdf[column])
                if geom is not None and not geom.is_empty and np.isfinite(value)
            ]
            arr = rasterize(
                shapes=shapes_values,
                out_shape=(ref_profile["height"], ref_profile["width"]),
                transform=ref_profile["transform"],
                fill=float(fill),
                dtype="float32",
                all_touched=True,
            ).astype("float64")
            return np.where(mask, arr, np.nan)

        sealed = burn("_SIE", CFG.DEFAULT_SEALED_PERCENT)
        vegetation = burn("_VEG", CFG.DEFAULT_VEGETATION_PERCENT)
        imp = np.clip(sealed / 100.0, 0.0, 1.0)
        vegf = np.clip(vegetation / 100.0, 0.0, 1.0)

        quick_raw = (
            1.0
            + CFG.LANDUSE_QUICKFLOW_IMPERVIOUS_WEIGHT * imp
            + CFG.LANDUSE_QUICKFLOW_LOW_VEG_WEIGHT * (1.0 - vegf)
        )
        pet_raw = 1.0 + CFG.LANDUSE_PET_VEGETATION_WEIGHT * (vegf - np.nanmean(vegf[mask]))
        perc_raw = 1.0 - CFG.LANDUSE_PERC_IMPERVIOUS_REDUCTION * imp

        quick = _normalize_factor(quick_raw, mask, 0.45, 2.20)
        pet = _normalize_factor(pet_raw, mask, 0.65, 1.35)
        perc = _normalize_factor(perc_raw, mask, 0.15, 1.40)

        print(
            f"[OK] Mittlere Versiegelung: {np.nanmean(sealed):.1f} %, "
            f"mittlerer Vegetationsgrad: {np.nanmean(vegetation):.1f} %"
        )
        return {
            "sealed_percent": sealed,
            "vegetation_percent": vegetation,
            "quickflow_factor": quick,
            "pet_factor": pet,
            "percolation_factor": perc,
        }
    except Exception as exc:
        warnings.warn(
            f"LBM-DE konnte nicht verarbeitet werden ({exc}). "
            "Es werden neutrale Landnutzungsfaktoren verwendet."
        )
        return defaults


def condition_dem_priority_flood(
    dgm: np.ndarray,
    mask: np.ndarray,
    river_mask: np.ndarray,
    profile: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Brennt das Gewässer ein und füllt Senken mit Priority-Flood."""
    from scipy.ndimage import binary_erosion

    base = fill_missing_dem_nearest(dgm, mask)
    burned = base.copy()
    burned[river_mask] -= float(CFG.STREAM_BURN_DEPTH_M)

    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    rows_b, cols_b = np.where(boundary)
    if len(rows_b) == 0:
        raise ValueError("Keine Randzellen für DGM-Konditionierung gefunden.")

    filled = burned.copy()
    visited = np.zeros(mask.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for r, c in zip(rows_b, cols_b):
        visited[r, c] = True
        heapq.heappush(heap, (float(filled[r, c]), int(r), int(c)))

    dx = abs(float(profile["transform"].a))
    dy = abs(float(profile["transform"].e))
    diag = float(np.hypot(dx, dy))
    neighbors = (
        (-1, -1, diag), (-1, 0, dy), (-1, 1, diag),
        (0, -1, dx),                   (0, 1, dx),
        (1, -1, diag),   (1, 0, dy),   (1, 1, diag),
    )
    epsilon_per_m = 1e-6

    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc, distance in neighbors:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= mask.shape[0] or cc < 0 or cc >= mask.shape[1]:
                continue
            if not mask[rr, cc] or visited[rr, cc]:
                continue
            visited[rr, cc] = True
            zn = float(burned[rr, cc])
            minimum_upstream = z + epsilon_per_m * distance
            if zn <= z:
                zn = minimum_upstream
            filled[rr, cc] = zn
            heapq.heappush(heap, (zn, rr, cc))

    if int(visited[mask].sum()) != int(mask.sum()):
        warnings.warn("Nicht alle DGM-Zellen wurden durch Priority-Flood erreicht.")

    return np.where(mask, burned, np.nan), np.where(mask, filled, np.nan)


def build_d8_flow_network(
    conditioned_dgm: np.ndarray,
    mask: np.ndarray,
    profile: dict,
    river_mask: np.ndarray,
) -> dict:
    """Erzeugt ein D8-Netz mit mehreren natürlichen Rand-Auslässen."""
    rows, cols = np.where(mask)
    n = len(rows)
    grid_id = np.full(mask.shape, -1, dtype=np.int32)
    grid_id[rows, cols] = np.arange(n, dtype=np.int32)
    z = conditioned_dgm[rows, cols].astype("float64")

    downstream = np.full(n, -1, dtype=np.int32)
    edge_distance = np.zeros(n, dtype="float64")
    route_slope = np.full(n, CFG.ROUTING_MIN_SLOPE, dtype="float64")

    dx = abs(float(profile["transform"].a))
    dy = abs(float(profile["transform"].e))
    diag = float(np.hypot(dx, dy))
    neighbors = (
        (-1, -1, diag), (-1, 0, dy), (-1, 1, diag),
        (0, -1, dx),                   (0, 1, dx),
        (1, -1, diag),   (1, 0, dy),   (1, 1, diag),
    )

    for cid in range(n):
        r, c = int(rows[cid]), int(cols[cid])
        z0 = float(conditioned_dgm[r, c])
        best_slope = 0.0
        best_id = -1
        best_dist = 0.0
        for dr, dc, distance in neighbors:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= mask.shape[0] or cc < 0 or cc >= mask.shape[1]:
                continue
            nid = int(grid_id[rr, cc])
            if nid < 0:
                continue
            dz = z0 - float(conditioned_dgm[rr, cc])
            slope = dz / distance
            if slope > best_slope:
                best_slope = slope
                best_id = nid
                best_dist = distance
        if best_id >= 0:
            downstream[cid] = best_id
            edge_distance[cid] = best_dist
            route_slope[cid] = max(best_slope, CFG.ROUTING_MIN_SLOPE)

    # Höchste Zellen zuerst -> Abflussakkumulation nach unten weitergeben.
    order = np.argsort(z)[::-1].astype(np.int32)
    accumulation = np.ones(n, dtype="float64")
    for cid in order:
        pid = int(downstream[cid])
        if pid >= 0:
            accumulation[pid] += accumulation[cid]

    # Verkettete Kinderlisten für schnelle Upstream-Suche.
    first_child = np.full(n, -1, dtype=np.int32)
    next_sibling = np.full(n, -1, dtype=np.int32)
    for child, parent in enumerate(downstream):
        if parent >= 0:
            next_sibling[child] = first_child[parent]
            first_child[parent] = child

    accumulation_raster = np.full(mask.shape, np.nan, dtype="float64")
    accumulation_raster[rows, cols] = accumulation
    flow_to_raster = np.full(mask.shape, -1, dtype="int32")
    flow_to_raster[rows, cols] = downstream

    print("[OK] D8-Fließnetz erstellt:")
    print(f"     Modellzellen: {n}")
    print(f"     natürliche Rand-Auslässe: {int(np.sum(downstream < 0))}")
    print(f"     maximale Akkumulation: {float(np.nanmax(accumulation)):.0f} Zellen")

    return {
        "rows": rows,
        "cols": cols,
        "grid_id": grid_id,
        "downstream": downstream,
        "edge_distance": edge_distance,
        "route_slope": route_slope,
        "order": order,
        "accumulation": accumulation,
        "accumulation_raster": accumulation_raster,
        "first_child": first_child,
        "next_sibling": next_sibling,
        "river_compact": river_mask[rows, cols].astype(bool),
        "conditioned_dgm": conditioned_dgm,
    }


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return slug or "gauge"


def _choose_point_layer(path: Path, manual_layer: Optional[str]) -> Optional[str]:
    if path.suffix.lower() != ".gpkg":
        return None
    import fiona
    layers = fiona.listlayers(path)
    if manual_layer is not None:
        if manual_layer not in layers:
            raise ValueError(f"Pegel-Layer {manual_layer!r} nicht gefunden: {layers}")
        return manual_layer
    for layer in layers:
        with fiona.open(path, layer=layer) as src:
            if "Point" in str(src.schema.get("geometry", "")):
                return layer
    return layers[0] if layers else None


def load_gauge_points(ref_crs) -> gpd.GeoDataFrame:
    path = data_dir() / CFG.GAUGE_FILENAME
    if not path.exists():
        candidates = [
            p for p in find_files(data_dir(), [".gpkg", ".shp", ".geojson"])
            if "pegel" in p.name.lower() or "gauge" in p.name.lower()
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Pegeldatei nicht gefunden: {path}. "
                "Speichere Heidelbach_Pegel.gpkg unter data/GeoDaten."
            )
        path = candidates[0]

    layer = _choose_point_layer(path, CFG.GAUGE_LAYER)
    gauges = gpd.read_file(path, layer=layer)
    if gauges.empty:
        raise ValueError("Pegeldatei enthält keine Punkte.")
    if gauges.crs is None:
        warnings.warn("Pegeldatei hat kein CRS. EPSG:25832 wird angenommen.")
        gauges = gauges.set_crs("EPSG:25832")
    gauges = gauges.to_crs(ref_crs)

    if not gauges.geometry.geom_type.isin(["Point", "MultiPoint"]).all():
        warnings.warn("Nicht alle Pegelgeometrien sind Punkte; Zentroid wird verwendet.")
        gauges["geometry"] = gauges.geometry.centroid

    name_col = _find_column(gauges.columns, None, ["name", "pegel", "station_name"])
    id_col = _find_column(
        gauges.columns, None,
        ["station_id", "messstellennummer", "station", "id", "nummer"],
    )

    if name_col is None:
        gauges["gauge_name"] = [
            CFG.PRIMARY_GAUGE_NAME if i == 0 else f"Pegel_{i+1}"
            for i in range(len(gauges))
        ]
    else:
        gauges["gauge_name"] = gauges[name_col].astype(str)

    if id_col is None:
        gauges["station_id"] = [
            CFG.PRIMARY_STATION_ID if i == 0 else f"unknown_{i+1}"
            for i in range(len(gauges))
        ]
    else:
        gauges["station_id"] = gauges[id_col].astype(str).str.replace(".0", "", regex=False)

    gauges["gauge_key"] = [
        _safe_slug(f"{sid}_{name}")
        for sid, name in zip(gauges["station_id"], gauges["gauge_name"])
    ]

    primary_match = (
        gauges["station_id"].astype(str).eq(str(CFG.PRIMARY_STATION_ID))
        | gauges["gauge_name"].str.contains(CFG.PRIMARY_GAUGE_NAME, case=False, na=False)
    )
    gauges["is_primary"] = primary_match
    if not primary_match.any():
        gauges.loc[gauges.index[0], "is_primary"] = True
        warnings.warn("Primärer Pegel nicht eindeutig erkannt; erstes Objekt wird verwendet.")
    elif primary_match.sum() > 1:
        first = gauges.index[primary_match][0]
        gauges["is_primary"] = False
        gauges.loc[first, "is_primary"] = True

    print(f"[OK] Pegeldatei: {path}")
    print(gauges[["gauge_name", "station_id", "is_primary"]].to_string(index=False))
    return gauges


def snap_gauge_to_network(
    point,
    network: dict,
    mask: np.ndarray,
    profile: dict,
    river_mask: np.ndarray,
    official_area_km2: Optional[float] = None,
    gauge_name: str = "Pegel",
) -> dict:
    """Hydrologisch robustes Snapping mit Distanz-, Fluss- und Flächenprüfung.

    Bei bekannter amtlicher Einzugsgebietsfläche wird nicht mehr automatisch
    die nächstgelegene Flussmaskenzelle verwendet. Stattdessen werden alle
    aktiven D8-Zellen im erweiterten Suchradius bewertet. Die amtliche Fläche
    ist das wichtigste Kriterium; Punktnähe und Lage auf der Flussmaske dienen
    als zusätzliche Plausibilitätsmerkmale.
    """
    r0, c0 = rowcol(profile["transform"], point.x, point.y)
    r0, c0 = int(r0), int(c0)
    cell_x = abs(float(profile["transform"].a))
    cell_y = abs(float(profile["transform"].e))
    cell = max(cell_x, cell_y)
    cell_area_km2 = cell_x * cell_y / 1e6
    max_radius = float(V93_GAUGE_SNAP_MAX_RADIUS_M)
    radius_cells = max(1, int(np.ceil(max_radius / cell)))

    rmin = max(0, r0 - radius_cells)
    rmax = min(mask.shape[0] - 1, r0 + radius_cells)
    cmin = max(0, c0 - radius_cells)
    cmax = min(mask.shape[1] - 1, c0 + radius_cells)

    target_area = (
        float(official_area_km2)
        if official_area_km2 is not None and np.isfinite(official_area_km2)
        and float(official_area_km2) > 0
        else np.nan
    )
    candidates: list[dict] = []
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if not mask[r, c]:
                continue
            x, y = xy(profile["transform"], r, c, offset="center")
            dist = float(np.hypot(x - point.x, y - point.y))
            if dist > max_radius:
                continue
            cid = int(network["grid_id"][r, c])
            if cid < 0:
                continue
            accumulation_cells = float(network["accumulation"][cid])
            area_km2 = accumulation_cells * cell_area_km2
            if np.isfinite(target_area):
                area_relative_error = abs(area_km2 - target_area) / target_area
                area_log_error = abs(np.log(max(area_km2, 1e-9) / target_area))
            else:
                area_relative_error = np.nan
                area_log_error = np.nan
            on_river = bool(river_mask[r, c])
            candidates.append({
                "gauge_id": cid,
                "row": int(r),
                "col": int(c),
                "x": float(x),
                "y": float(y),
                "distance_m": dist,
                "on_river_mask": on_river,
                "accumulation_cells": accumulation_cells,
                "area_km2": area_km2,
                "area_relative_error": area_relative_error,
                "area_log_error": area_log_error,
            })

    if not candidates:
        raise ValueError(
            f"{gauge_name}: keine aktive D8-Zelle im Suchradius von "
            f"{max_radius:.0f} m."
        )

    if np.isfinite(target_area):
        # Fläche dominiert. Distanz und Flussmaske verhindern unnötige
        # Verschiebungen, wenn mehrere Zellen dieselbe plausible Fläche haben.
        def candidate_cost(item: dict) -> float:
            distance_term = float(item["distance_m"]) / max(max_radius, 1.0)
            river_penalty = 0.0 if item["on_river_mask"] else V93_NON_RIVER_PENALTY
            return (
                V93_AREA_ERROR_WEIGHT * float(item["area_log_error"])
                + V93_DISTANCE_WEIGHT * distance_term
                + river_penalty
            )

        best = min(candidates, key=candidate_cost)
        best["candidate_cost"] = candidate_cost(best)
        best["snap_method"] = "official_area_aware"
        best["official_area_km2"] = target_area
        best["area_difference_percent"] = (
            100.0 * (best["area_km2"] - target_area) / target_area
        )
        within_tolerance = (
            best["area_relative_error"] <= V93_GAUGE_AREA_TOLERANCE_FRACTION
        )
        best["within_area_tolerance"] = bool(within_tolerance)
        if V93_STRICT_GAUGE_AREA_CHECK and not within_tolerance:
            nearest_area = min(candidates, key=lambda item: item["area_relative_error"])
            raise ValueError(
                f"{gauge_name}: keine hydrologisch plausible Pegelzelle gefunden. "
                f"Beste Zelle={best['area_km2']:.3f} km² bei {best['distance_m']:.1f} m, "
                f"amtlich={target_area:.3f} km², Abweichung="
                f"{best['area_difference_percent']:+.1f} %. "
                f"Flächen-nächster Kandidat={nearest_area['area_km2']:.3f} km² "
                f"bei {nearest_area['distance_m']:.1f} m. DGM, Flusseinbrand oder "
                "Einzugsgebietsmaske prüfen."
            )
    else:
        # Fallback ohne Flächenreferenz: Akkumulation fördern, große Distanz
        # und fehlende Flussmaskenlage bestrafen, aber keine Kandidatenklasse
        # vollständig ausschließen.
        penalty = max(float(CFG.GAUGE_SNAP_DISTANCE_PENALTY_M), 1.0)
        def fallback_score(item: dict) -> float:
            river_bonus = V93_RIVER_MASK_BONUS if item["on_river_mask"] else 0.0
            return (
                np.log1p(item["accumulation_cells"])
                - item["distance_m"] / penalty
                + river_bonus
            )
        best = max(candidates, key=fallback_score)
        best["candidate_cost"] = -fallback_score(best)
        best["snap_method"] = "accumulation_distance_fallback"
        best["official_area_km2"] = np.nan
        best["area_difference_percent"] = np.nan
        best["within_area_tolerance"] = True

    # Kandidatenliste für die spätere Diagnose, nach fachlicher Güte sortiert.
    if np.isfinite(target_area):
        ranked = sorted(
            candidates,
            key=lambda item: (
                item["area_relative_error"],
                0 if item["on_river_mask"] else 1,
                item["distance_m"],
            ),
        )
    else:
        ranked = sorted(
            candidates,
            key=lambda item: (-item["accumulation_cells"], item["distance_m"]),
        )
    best["candidate_diagnostics"] = ranked[:V93_SNAP_DIAGNOSTIC_TOP_N]
    best["snap_distance_m"] = float(best["distance_m"])

    # Ursprungspunkt explizit mitführen. Frühere Versionen erwarteten diese
    # Schlüssel in Diagnose- und Exportfunktionen, erzeugten sie beim
    # flächenbasierten Snapping jedoch nicht.
    best["original_x"] = float(point.x)
    best["original_y"] = float(point.y)

    if best["snap_distance_m"] > V93_LARGE_SNAP_WARNING_M:
        warnings.warn(
            f"{gauge_name}: hydrologische Zielzelle liegt "
            f"{best['snap_distance_m']:.1f} m vom amtlichen Punkt entfernt. "
            "Das deutet auf einen Versatz oder eine Unterbrechung zwischen "
            "Flussvektor, DGM und D8-Netz hin; die Fläche ist jedoch plausibel."
        )
    return best


def build_gauge_routing(
    gauge_row: pd.Series,
    network: dict,
    mask: np.ndarray,
    profile: dict,
    river_mask: np.ndarray,
) -> dict:
    official_area = gauge_row.get("official_area_km2", np.nan)
    snapped = snap_gauge_to_network(
        gauge_row.geometry,
        network,
        mask,
        profile,
        river_mask,
        official_area_km2=(
            float(official_area) if np.isfinite(official_area) else None
        ),
        gauge_name=str(gauge_row["gauge_name"]),
    )
    gauge_id = snapped["gauge_id"]

    # Alle Kinder des Pegels rekursiv einsammeln.
    upstream_ids: list[int] = []
    travel_cost = []
    stack: list[tuple[int, float]] = [(gauge_id, 0.0)]
    exponent = float(CFG.ROUTING_SLOPE_EXPONENT)

    while stack:
        cid, cost = stack.pop()
        upstream_ids.append(cid)
        travel_cost.append(cost)
        child = int(network["first_child"][cid])
        while child >= 0:
            slope = max(float(network["route_slope"][child]), CFG.ROUTING_MIN_SLOPE)
            channel_factor = (
                CFG.CHANNEL_SPEED_MULTIPLIER
                if network["river_compact"][child]
                else 1.0
            )
            edge_cost = (
                float(network["edge_distance"][child])
                / (slope ** exponent)
                / channel_factor
            )
            stack.append((child, cost + edge_cost))
            child = int(network["next_sibling"][child])

    upstream_ids_arr = np.asarray(upstream_ids, dtype=np.int32)
    travel_cost_arr = np.asarray(travel_cost, dtype="float64")
    rows = network["rows"][upstream_ids_arr]
    cols = network["cols"][upstream_ids_arr]

    upstream_mask = np.zeros(mask.shape, dtype=bool)
    upstream_mask[rows, cols] = True
    cell_area = abs(profile["transform"].a * profile["transform"].e)
    area_km2 = float(len(upstream_ids_arr) * cell_area / 1e6)

    # Die rekursiv gezählte Fläche muss mit der Akkumulationsfläche der
    # gewählten Zelle übereinstimmen. Größere Differenzen weisen auf einen
    # Netzaufbaufehler hin.
    accumulation_area_km2 = float(snapped["area_km2"])
    topology_difference = area_km2 - accumulation_area_km2
    if abs(topology_difference) > max(0.05, 0.001 * max(area_km2, 1.0)):
        warnings.warn(
            f"{gauge_row['gauge_name']}: rekursive Upstream-Fläche "
            f"({area_km2:.3f} km²) weicht von der Akkumulationsfläche "
            f"({accumulation_area_km2:.3f} km²) ab."
        )

    result = {
        **snapped,
        "name": str(gauge_row["gauge_name"]),
        "station_id": str(gauge_row["station_id"]),
        "key": str(gauge_row["gauge_key"]),
        "is_primary": bool(gauge_row["is_primary"]),
        "upstream_ids": upstream_ids_arr,
        "rows": rows,
        "cols": cols,
        "travel_cost": travel_cost_arr,
        "upstream_mask": upstream_mask,
        "area_km2": area_km2,
        "accumulation_area_km2": accumulation_area_km2,
        "topology_area_difference_km2": float(topology_difference),
        "point_source": gauge_row.get("point_source", "unknown"),
        "configured_point_x": float(gauge_row.get("configured_point_x", np.nan)),
        "configured_point_y": float(gauge_row.get("configured_point_y", np.nan)),
        "official_point_x": float(gauge_row.get("official_point_x", np.nan)),
        "official_point_y": float(gauge_row.get("official_point_y", np.nan)),
        "configured_to_official_distance_m": float(
            gauge_row.get("configured_to_official_distance_m", np.nan)
        ),
    }

    official = result.get("official_area_km2", np.nan)
    area_text = ""
    if np.isfinite(official) and official > 0:
        area_text = (
            f", amtlich={official:.2f} km², "
            f"Abweichung={result['area_difference_percent']:+.2f}%"
        )
    print(
        f"[OK] Pegel {result['name']} ({result['station_id']}): "
        f"Methode={result['snap_method']}, Snap={result['snap_distance_m']:.1f} m, "
        f"Upstream-Fläche={area_km2:.2f} km²{area_text}"
    )
    return result


def prepare_routing_lags(gauge_routing: dict, velocity_m_s: float) -> dict:
    if velocity_m_s <= 0:
        raise ValueError("Routing-Geschwindigkeit muss größer als 0 sein.")
    travel_days = gauge_routing["travel_cost"] / velocity_m_s / 86400.0
    lag_float = np.clip(
        travel_days / 30.4375,
        0.0,
        float(CFG.MAX_ROUTING_LAG_MONTHS),
    )
    lag0 = np.floor(lag_float).astype(np.int16)
    frac1 = lag_float - lag0
    lag1 = lag0 + 1
    over = lag1 > CFG.MAX_ROUTING_LAG_MONTHS
    lag1[over] = CFG.MAX_ROUTING_LAG_MONTHS
    frac1[over] = 0.0
    return {
        "travel_days": travel_days,
        "lag_months": lag_float,
        "lag0": lag0,
        "lag1": lag1,
        "frac1": frac1,
        "max_lag": int(CFG.MAX_ROUTING_LAG_MONTHS),
        "velocity_m_s": float(velocity_m_s),
    }


def route_local_runoff_to_gauge(
    local_runoff_mm: np.ndarray,
    cell_area_m2: float,
    month_position: int,
    arrivals_m3: np.ndarray,
    gauge_routing: dict,
    lag_info: dict,
) -> None:
    values = local_runoff_mm[gauge_routing["rows"], gauge_routing["cols"]]
    volumes = np.nan_to_num(values, nan=0.0) * cell_area_m2 / 1000.0
    current = volumes * (1.0 - lag_info["frac1"])
    following = volumes * lag_info["frac1"]
    max_lag = lag_info["max_lag"]
    by0 = np.bincount(lag_info["lag0"], weights=current, minlength=max_lag + 1)
    by1 = np.bincount(lag_info["lag1"], weights=following, minlength=max_lag + 1)
    for lag in range(max_lag + 1):
        target = month_position + lag
        if target < len(arrivals_m3):
            arrivals_m3[target] += by0[lag] + by1[lag]


def compact_to_raster(
    values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    arr = np.full(shape, np.nan, dtype="float64")
    arr[rows, cols] = values
    return arr


def mask_to_geodataframe(mask: np.ndarray, profile: dict, name: str) -> gpd.GeoDataFrame:
    from rasterio.features import shapes as raster_shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    geoms = [
        shape(geom)
        for geom, value in raster_shapes(
            mask.astype("uint8"), mask=mask, transform=profile["transform"]
        )
        if int(value) == 1
    ]
    if not geoms:
        raise ValueError("Maske konnte nicht polygonisiert werden.")
    merged = unary_union(geoms)
    return gpd.GeoDataFrame({"name": [name]}, geometry=[merged], crs=profile["crs"])


# =============================================================================
# GEODATEN: EINZUGSGEBIET, MASKE, FELDKAPAZITÄT
# =============================================================================

def load_catchment(ref_crs) -> gpd.GeoDataFrame:
    catchment_path = find_first(data_dir() / CFG.CATCHMENT_DIRNAME, [".shp", ".gpkg", ".geojson"])
    print(f"[OK] Einzugsgebiet: {catchment_path}")

    gdf = gpd.read_file(catchment_path)
    if gdf.empty:
        raise ValueError("Einzugsgebiet ist leer.")

    if gdf.crs is None:
        warnings.warn("Einzugsgebiet hat kein CRS. Es wird angenommen, dass es im Referenz-CRS liegt.")
        gdf = gdf.set_crs(ref_crs)

    gdf = gdf.to_crs(ref_crs)
    gdf = gdf.dissolve()
    return gdf


def create_mask(catchment: gpd.GeoDataFrame, ref_profile: dict) -> np.ndarray:
    mask = geometry_mask(
        geometries=list(catchment.geometry),
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        invert=True,
    )
    return mask


def choose_numeric_column(gdf: gpd.GeoDataFrame, manual: Optional[str] = None) -> str:
    if manual is not None:
        if manual not in gdf.columns:
            raise ValueError(f"Spalte {manual!r} nicht gefunden. Vorhanden: {list(gdf.columns)}")
        return manual

    numeric_cols = [
        c for c in gdf.columns
        if c != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])
    ]

    if not numeric_cols:
        raise ValueError(
            "Keine numerische Spalte gefunden. Setze NFK_COLUMN manuell in der CONFIG."
        )

    # Heuristik: bevorzugt Spaltennamen mit nfk, fk, kap, awc, we, feld
    preferred_tokens = ["nfk", "fk", "kap", "awc", "we", "feld", "wasser"]
    for token in preferred_tokens:
        for col in numeric_cols:
            if token.lower() in col.lower():
                return col

    return numeric_cols[0]


def parse_fc_mm_from_bezeichner(text) -> float:
    """
    Wandelt die BEZEICHNER-Klassen der nutzbaren Feldkapazität in mm-Werte um.
    """
    if pd.isna(text):
        return np.nan

    s = str(text).lower()

    if "gewässer" in s or "gewaesser" in s:
        return 0.0
    if "siedlungs-kern" in s or "siedlungskern" in s:
        return 20.0
    if any(tok in s for tok in ["siedlung", "industrie", "verkehr"]):
        return 30.0
    if any(tok in s for tok in ["abbau", "aufschüttung", "aufschuettung"]):
        return 40.0

    nums = re.findall(r"\d+(?:[,.]\d+)?", s)
    nums = [float(n.replace(",", ".")) for n in nums]

    if len(nums) >= 2:
        return float(np.mean(nums[:2]))

    if len(nums) == 1:
        if ">" in s or "über" in s or "mehr" in s:
            return nums[0] + 40.0
        return nums[0]

    return np.nan


def load_field_capacity_raster(ref_profile: dict, mask: np.ndarray) -> np.ndarray:
    nfk_dir = data_dir() / CFG.NFK_DIRNAME
    fc_dir = data_dir() / CFG.FC_FALLBACK_DIRNAME

    if find_files(nfk_dir, [".shp"]):
        shp = find_first(nfk_dir, [".shp"])
        source_name = "nutzbare Feldkapazität"
    else:
        shp = find_first(fc_dir, [".shp"])
        source_name = "Feldkapazität"

    print(f"[OK] Bodenspeicherquelle ({source_name}): {shp}")

    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        warnings.warn("Feldkapazitäts-Shapefile hat kein CRS. Es wird Referenz-CRS angenommen.")
        gdf = gdf.set_crs(ref_profile["crs"])

    gdf = gdf.to_crs(ref_profile["crs"])

    print("\nSpalten im Feldkapazitäts-Shapefile:")
    print(list(gdf.columns))

    if "BEZEICHNER" in gdf.columns and CFG.NFK_COLUMN is None:
        print("[INFO] Beispiele aus BEZEICHNER:")
        print(gdf["BEZEICHNER"].dropna().astype(str).drop_duplicates().head(20).to_string(index=False))

        gdf["_FC_MM"] = gdf["BEZEICHNER"].apply(parse_fc_mm_from_bezeichner)
        valid_share = gdf["_FC_MM"].notna().mean()

        if valid_share > 0.5:
            value_col = "_FC_MM"
            print("[OK] Bodenspeicher aus BEZEICHNER-Klassen in mm umgerechnet.")
            print("[OK] Zuordnung BEZEICHNER -> FC_mm, Beispiele:")
            print(
                gdf[["BEZEICHNER", "_FC_MM"]]
                .dropna()
                .drop_duplicates()
                .head(20)
                .to_string(index=False)
            )
        else:
            warnings.warn(
                "BEZEICHNER konnte nicht zuverlässig in mm umgerechnet werden. "
                "Nutze numerische Spalte."
            )
            value_col = choose_numeric_column(gdf, CFG.NFK_COLUMN)
    else:
        value_col = choose_numeric_column(gdf, CFG.NFK_COLUMN)

    print(f"[OK] Verwendete Bodenspeicher-Spalte: {value_col}")

    shapes = [
        (geom, float(value) * CFG.FC_UNIT_FACTOR)
        for geom, value in zip(gdf.geometry, gdf[value_col])
        if geom is not None and np.isfinite(value)
    ]

    fc = rasterize(
        shapes=shapes,
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        fill=np.nan,
        dtype="float64",
    )

    fc = np.where(mask, fc, np.nan)

    missing_inside = mask & ~np.isfinite(fc)
    if missing_inside.any():
        warnings.warn(
            f"{missing_inside.sum()} Zellen im Einzugsgebiet haben keine FC/nFK. "
            f"Fülle mit DEFAULT_FC_MM={CFG.DEFAULT_FC_MM} mm."
        )
        fc[missing_inside] = CFG.DEFAULT_FC_MM

    return fc


# =============================================================================
# ZEITREIHEN: NIEDERSCHLAG, PET, QOBS
# =============================================================================

def find_time_dim(da) -> str:
    candidates = [d for d in da.dims if "time" in d.lower() or d.lower() in ["date", "valid_time"]]
    if not candidates:
        raise ValueError(f"Keine Zeitdimension gefunden in Variable mit Dimensionen {da.dims}")
    return candidates[0]


def choose_data_var(ds, manual: Optional[str] = None) -> str:
    if manual is not None:
        if manual not in ds.data_vars:
            raise ValueError(f"Variable {manual!r} nicht im NetCDF. Vorhanden: {list(ds.data_vars)}")
        return manual

    candidates = []
    for name, da in ds.data_vars.items():
        dims = [d.lower() for d in da.dims]
        if any("time" in d or d in ["date", "valid_time"] for d in dims) and da.ndim >= 2:
            candidates.append(name)

    if not candidates:
        candidates = list(ds.data_vars)

    # Heuristik: bevorzugt Niederschlagsnamen
    preferred_tokens = ["precip", "pr", "rr", "rain", "nied", "regnie"]
    for token in preferred_tokens:
        for name in candidates:
            if token in name.lower():
                return name

    return candidates[0]


def set_spatial_dims_and_crs(da, default_crs):
    import rioxarray  # noqa: F401

    dims_lower = {d.lower(): d for d in da.dims}

    x_dim = None
    y_dim = None

    for key in ["x", "lon", "longitude", "rlon"]:
        if key in dims_lower:
            x_dim = dims_lower[key]
            break

    for key in ["y", "lat", "latitude", "rlat"]:
        if key in dims_lower:
            y_dim = dims_lower[key]
            break

    if x_dim is None or y_dim is None:
        raise ValueError(
            f"Konnte räumliche Dimensionen nicht erkennen. Dimensionen: {da.dims}"
        )

    da = da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)

    if da.rio.crs is None:
        # Wenn lon/lat: EPSG:4326, sonst Referenz-CRS
        if x_dim.lower() in ["lon", "longitude"] and y_dim.lower() in ["lat", "latitude"]:
            da = da.rio.write_crs("EPSG:4326", inplace=False)
        else:
            da = da.rio.write_crs(default_crs, inplace=False)

    return da


def load_precip_monthly(catchment: gpd.GeoDataFrame, ref_crs) -> pd.Series:
    """
    Lädt Niederschlag aus HYRAS/NetCDF, bildet pro Tag den Mittelwert im Einzugsgebiet
    und aggregiert anschließend zu Monatssummen [mm/Monat].
    """
    import xarray as xr

    p_dir = data_dir() / CFG.PRECIP_DIRNAME
    nc_files = find_files(p_dir, [".nc", ".nc4", ".cdf"])

    if not nc_files:
        raise FileNotFoundError(f"Keine NetCDF-Dateien gefunden in: {p_dir}")

    print(f"[OK] Niederschlags-NetCDF-Dateien: {len(nc_files)}")
    for f in nc_files[:5]:
        print("     ", f)
    if len(nc_files) > 5:
        print("      ...")

    try:
        ds = xr.open_mfdataset(nc_files, combine="by_coords", data_vars="all")
    except Exception:
        warnings.warn("open_mfdataset fehlgeschlagen. Versuche Dateien einzeln zu öffnen und zu kombinieren.")
        datasets = [xr.open_dataset(f) for f in nc_files]
        var0 = choose_data_var(datasets[0], CFG.P_VAR)
        time_dim0 = find_time_dim(datasets[0][var0])
        ds = xr.concat(datasets, dim=time_dim0)

    var = choose_data_var(ds, CFG.P_VAR)
    da = ds[var]
    print(f"[OK] Niederschlagsvariable: {var}")
    print(f"     Dimensionen: {da.dims}")

    time_dim = find_time_dim(da)

    dims_lower = {d.lower(): d for d in da.dims}
    x_dim = None
    y_dim = None
    for key in ["x", "lon", "longitude", "rlon"]:
        if key in dims_lower:
            x_dim = dims_lower[key]
            break
    for key in ["y", "lat", "latitude", "rlat"]:
        if key in dims_lower:
            y_dim = dims_lower[key]
            break

    if x_dim is None or y_dim is None:
        raise ValueError(f"Konnte x/y-Dimensionen im Niederschlag nicht erkennen: {da.dims}")

    x = np.asarray(da[x_dim].values)
    y = np.asarray(da[y_dim].values)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Niederschlagskoordinaten sind nicht 1D.")

    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))

    transform = Affine(dx, 0.0, float(x[0]) - dx / 2.0,
                       0.0, dy, float(y[0]) - dy / 2.0)

    print("[INFO] Niederschlagsraster:")
    print(f"       x: {float(np.nanmin(x)):.3f} bis {float(np.nanmax(x)):.3f}, dx={dx:.3f}")
    print(f"       y: {float(np.nanmin(y)):.3f} bis {float(np.nanmax(y)):.3f}, dy={dy:.3f}")
    print(f"       Größe: {len(x)} x {len(y)}")

    crs_candidates = ["EPSG:25832", "EPSG:3035", "EPSG:31467", "EPSG:31468", "EPSG:4326"]

    tests = []
    for crs_text in crs_candidates:
        try:
            c = catchment.copy()
            if c.crs is None:
                c = c.set_crs(crs_text)
            else:
                c = c.to_crs(crs_text)
            c = c.dissolve()

            m = geometry_mask(
                geometries=list(c.geometry),
                out_shape=(len(y), len(x)),
                transform=transform,
                invert=True,
            )
            tests.append((int(m.sum()), crs_text, m))
        except Exception as e:
            print(f"[INFO] Niederschlags-CRS-Test fehlgeschlagen für {crs_text}: {e}")

    if not tests:
        raise ValueError("Kein CRS-Test für Niederschlag möglich.")

    tests.sort(key=lambda t: t[0], reverse=True)

    print("[INFO] Niederschlags-CRS-Test, Pixel im Einzugsgebiet:")
    for n, crs_text, _ in tests:
        print(f"       {crs_text}: {n}")

    overlap, chosen_crs, p_mask = tests[0]
    if overlap == 0:
        raise ValueError("Das Einzugsgebiet überlappt mit keinem Niederschlagsraster-Pixel.")

    print(f"[OK] Gewähltes Niederschlags-CRS: {chosen_crs}")
    print(f"[OK] Niederschlags-Pixel im Einzugsgebiet: {overlap}")

    mask_da = xr.DataArray(
        p_mask,
        dims=(y_dim, x_dim),
        coords={y_dim: da[y_dim], x_dim: da[x_dim]},
    )

    basin_mean = da.where(mask_da).mean(dim=(y_dim, x_dim), skipna=True)
    monthly = basin_mean.resample(**{time_dim: "MS"}).sum(skipna=True) * CFG.P_UNIT_FACTOR

    ts = monthly.to_series()
    ts.index = pd.to_datetime(ts.index)
    ts.name = "P_mm_month"
    ts = ts.loc[CFG.START:CFG.END]
    ts = ts[~ts.index.duplicated()].sort_index()

    print("[OK] Beispiel Niederschlags-Monatswerte:")
    print(ts.head())

    if ts.notna().sum() == 0 or float(ts.fillna(0).sum()) == 0.0:
        warnings.warn("Alle Niederschlagswerte sind 0 oder NaN. Bitte CRS/Variable/Einheit prüfen.")

    return ts


def parse_date_from_path(path: Path) -> Optional[pd.Timestamp]:
    """
    Erkennt Datumsangaben in Dateinamen oder Elternordnern:
    YYYYMMDD oder YYYYMM.
    Beispiel: grids_germany_daily_evapo_p_202001 -> 2020-01-01
    """
    text = " ".join([path.stem] + [p.name for p in path.parents[:4]])

    # YYYYMMDD
    m = re.search(r"(20\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])", text)
    if m:
        return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    # YYYYMM
    m = re.search(r"(20\d{2})(0[1-9]|1[0-2])", text)
    if m:
        return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-01")

    return None


def load_pet_monthly(ref_profile: dict, mask: np.ndarray) -> pd.Series:
    """
    Schnelle PET-Ladefunktion für viele ASC-Dateien.

    Wichtig:
    Die DWD-ASC-Dateien enthalten häufig kein CRS. Wenn das falsche CRS angenommen
    wird, überlappt das Einzugsgebiet nicht mit dem PET-Raster und der Mittelwert
    wird NaN. Diese Version testet deshalb mehrere plausible CRS-Kandidaten und
    nimmt das CRS mit der größten Überlappung.
    """
    pet_dir = data_dir() / CFG.PET_DIRNAME
    raster_files = find_files(pet_dir, [".asc", ".tif", ".tiff"])

    if not raster_files:
        raise FileNotFoundError(f"Keine ASC/TIF-Dateien gefunden in: {pet_dir}")

    cache_path = out_dir() / "tables" / "pet_monthly_cache_v7_primary_gauge.csv"
    if cache_path.exists():
        try:
            cached_df = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")
            cached = cached_df["PET_mm_month"]
            cached = cached.loc[CFG.START:CFG.END]
            if len(cached) > 0 and cached.notna().sum() > 0:
                print(f"[OK] PET aus Cache geladen: {cache_path}")
                return cached
            else:
                print("[WARNUNG] PET-Cache enthält keine gültigen Werte. Erzeuge PET neu.")
        except Exception:
            warnings.warn("PET-Cache konnte nicht gelesen werden. Erzeuge PET neu.")

    print(f"[OK] PET/evaporation-Rasterdateien gefunden: {len(raster_files)}")

    start = pd.Timestamp(CFG.START)
    end = pd.Timestamp(CFG.END)

    # Erst nur Dateien mit Datum und innerhalb des Modellzeitraums behalten.
    dated_files: list[tuple[pd.Timestamp, Path]] = []
    skipped_no_date = 0
    skipped_outside = 0

    for f in raster_files:
        dt = parse_date_from_path(f)
        if dt is None:
            skipped_no_date += 1
            continue

        if dt < start or dt > end:
            skipped_outside += 1
            continue

        dated_files.append((dt, f))

    if skipped_no_date:
        warnings.warn(f"{skipped_no_date} PET-Dateien ohne Datum im Namen/Pfad wurden ignoriert.")
    if skipped_outside:
        print(f"[OK] {skipped_outside} PET-Dateien außerhalb {CFG.START} bis {CFG.END} übersprungen.")

    if not dated_files:
        raise ValueError("Keine PET-Dateien im gewünschten Zeitraum gefunden.")

    print(f"[OK] PET-Dateien im Modellzeitraum: {len(dated_files)}")
    print("     Beispiel:", dated_files[0][1])

    # Einzugsgebiet laden
    catchment_path = find_first(data_dir() / CFG.CATCHMENT_DIRNAME, [".shp", ".gpkg", ".geojson"])
    catchment = gpd.read_file(catchment_path)

    # CRS-Kandidaten für DWD-/Deutschland-Raster.
    # 25832 = UTM32N, 3035 = ETRS89/LAEA Europe, 31467 = DHDN/GK3,
    # 31468 = DHDN/GK4, 4326 = WGS84 lon/lat.
    crs_candidates = [
        CFG.PET_ASC_CRS_FALLBACK,
        "EPSG:25832",
        "EPSG:3035",
        "EPSG:31467",
        "EPSG:31468",
        "EPSG:4326",
    ]
    # Duplikate entfernen, Reihenfolge behalten
    crs_candidates = list(dict.fromkeys([c for c in crs_candidates if c is not None]))

    def build_mask_for_crs(src, crs_text: str) -> tuple[np.ndarray, int]:
        c = catchment.copy()
        if c.crs is None:
            warnings.warn("Einzugsgebiet hat kein CRS. Es wird das Test-CRS angenommen.")
            c = c.set_crs(crs_text)
        else:
            c = c.to_crs(crs_text)
        c = c.dissolve()

        m = geometry_mask(
            geometries=list(c.geometry),
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        )
        return m, int(m.sum())

    # Für das erste PET-Raster das passende CRS bestimmen.
    first_file = dated_files[0][1]
    with rasterio.open(first_file) as src:
        print("[INFO] Erstes PET-Raster:")
        print(f"       Datei:  {first_file.name}")
        print(f"       CRS:    {src.crs}")
        print(f"       Bounds: {src.bounds}")
        print(f"       Größe:  {src.width} x {src.height}")
        print(f"       Nodata: {src.nodata}")

        if src.crs is not None:
            chosen_crs = src.crs
            first_mask, overlap = build_mask_for_crs(src, chosen_crs)
        else:
            tests = []
            for crs_text in crs_candidates:
                try:
                    m, n = build_mask_for_crs(src, crs_text)
                    tests.append((n, crs_text, m))
                except Exception as e:
                    print(f"[INFO] PET-CRS-Test fehlgeschlagen für {crs_text}: {e}")

            if not tests:
                raise ValueError("Kein PET-CRS-Kandidat konnte getestet werden.")

            tests.sort(key=lambda x: x[0], reverse=True)
            overlap, chosen_crs, first_mask = tests[0]

            print("[INFO] PET-CRS-Test, Pixel im Einzugsgebiet:")
            for n, crs_text, _ in tests:
                print(f"       {crs_text}: {n}")

        if overlap == 0:
            raise ValueError(
                "Das Einzugsgebiet überlappt mit keinem PET-Raster-Pixel. "
                "Wahrscheinlich ist das CRS der ASC-Dateien falsch. "
                "Bitte die ersten 6 Zeilen einer ASC-Datei prüfen."
            )

        print(f"[OK] Gewähltes PET-CRS: {chosen_crs}")
        print(f"[OK] PET-Pixel im Einzugsgebiet: {overlap}")

    # Masken-Cache für den Fall, dass nicht alle ASC-Dateien exakt dasselbe Grid haben.
    pet_mask_cache = {}

    def get_pet_mask(src):
        src_crs = src.crs if src.crs is not None else chosen_crs

        key = (
            src.width,
            src.height,
            tuple(src.transform),
            str(src_crs),
        )

        if key in pet_mask_cache:
            return pet_mask_cache[key]

        m, n = build_mask_for_crs(src, src_crs)
        if n == 0:
            warnings.warn(
                f"Keine PET-Überlappung für Raster {src.name}. "
                f"Verwendetes CRS: {src_crs}. Datei wird NaN liefern."
            )

        pet_mask_cache[key] = m
        return m

    # Tageswerte zu Monatssummen aggregieren.
    month_values: dict[pd.Timestamp, float] = {}
    counts_by_month: dict[pd.Timestamp, int] = {}
    nan_files = 0

    for i, (dt, f) in enumerate(dated_files, start=1):
        month = pd.Timestamp(dt.year, dt.month, 1)

        with rasterio.open(f) as src:
            arr = src.read(1).astype("float64")
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan

            m = get_pet_mask(src)
            val = float(np.nanmean(np.where(m, arr, np.nan))) * CFG.PET_UNIT_FACTOR

        if not np.isfinite(val):
            nan_files += 1
            if nan_files <= 5:
                warnings.warn(f"PET-Mittelwert ist NaN für {f}")
            continue

        month_values[month] = month_values.get(month, 0.0) + val
        counts_by_month[month] = counts_by_month.get(month, 0) + 1

        if i % 100 == 0 or i == len(dated_files):
            print(f"     PET gelesen: {i}/{len(dated_files)} Dateien", flush=True)

    if nan_files > 0:
        warnings.warn(
            f"{nan_files} PET-Dateien hatten NaN-Mittelwerte. "
            "Wenn das fast alle sind, stimmt weiterhin CRS/Nodata/Einheit nicht."
        )

    if not month_values:
        raise ValueError(
            "Es konnten keine gültigen PET-Monatswerte berechnet werden. "
            "Bitte PET-Raster-CRS und ASC-Header prüfen."
        )

    ts = pd.Series(month_values, name="PET_mm_month").sort_index()
    ts.index.name = "date"
    ts = ts.loc[CFG.START:CFG.END]

    counts = pd.Series(counts_by_month, name="n_pet_files").sort_index()
    pet_check = pd.concat([ts, counts], axis=1)
    pet_check.to_csv(cache_path, encoding="utf-8")

    print(f"[OK] PET-Monatswerte erzeugt und gecacht: {cache_path}")
    print("[OK] PET-Dateien pro Monat, erste Monate:")
    print(pet_check.head())

    return ts


def read_csv_auto(path: Path) -> pd.DataFrame:
    """
    Robust für ; oder , als Separator und deutsche Dezimal-Kommas.
    """
    for sep in [None, ";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1:
                # Komma-Dezimalzahlen in numerischen Textspalten umwandeln
                for c in df.columns:
                    if df[c].dtype == object:
                        converted = pd.to_numeric(
                            df[c].astype(str).str.replace(",", ".", regex=False),
                            errors="ignore"
                        )
                        df[c] = converted
                return df
        except Exception:
            pass

    raise ValueError(f"CSV konnte nicht gelesen werden: {path}")


def choose_date_col(df: pd.DataFrame, manual: Optional[str] = None) -> str:
    if manual is not None:
        return manual

    for col in df.columns:
        if any(tok in col.lower() for tok in ["date", "datum", "time", "zeit"]):
            return col

    # Fallback: erste Spalte, die sich gut als Datum parsen lässt
    best_col = None
    best_count = -1
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        count = parsed.notna().sum()
        if count > best_count:
            best_count = count
            best_col = col

    if best_col is None or best_count == 0:
        raise ValueError("Keine Datumsspalte in Qobs-CSV erkannt. Setze QOBS_DATE_COLUMN.")
    return best_col


def choose_value_col(df: pd.DataFrame, date_col: str, manual: Optional[str] = None) -> str:
    if manual is not None:
        return manual

    candidates = [c for c in df.columns if c != date_col]

    preferred_tokens = ["q", "durchfluss", "abfluss", "discharge", "m3", "m³"]
    for token in preferred_tokens:
        for col in candidates:
            if token in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
                return col

    numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("Keine numerische Qobs-Spalte erkannt. Setze QOBS_VALUE_COLUMN.")
    return numeric_cols[0]



def extract_q_mean_from_min_mean_max(series: pd.Series) -> pd.Series:
    """
    Extrahiert den Mittelwert aus Strings wie:
    "1,03 - 1,12 - 1,25"
    oder
    "1.03 - 1.12 - 1.25"

    Falls nur eine Zahl vorkommt, wird diese verwendet.
    """
    out = []
    for value in series.astype(str):
        nums = re.findall(r"[-+]?\d+(?:[,.]\d+)?", value)
        nums = [float(x.replace(",", ".")) for x in nums]

        if len(nums) >= 3:
            out.append(nums[1])  # Min - Mittel - Max
        elif len(nums) == 2:
            out.append(float(np.mean(nums)))
        elif len(nums) == 1:
            out.append(nums[0])
        else:
            out.append(np.nan)

    return pd.Series(out, index=series.index, dtype="float64")


def load_qobs_monthly(q_dirname: Optional[str] = None) -> pd.Series:
    q_dir = data_dir() / (q_dirname or CFG.QOBS_DIRNAME)
    csv_files = find_files(q_dir, [".csv", ".txt"])

    if not csv_files:
        raise FileNotFoundError(f"Keine Qobs-CSV/TXT-Dateien gefunden in: {q_dir}")

    print(f"[OK] Beobachtete Durchflussdateien: {len(csv_files)}")
    for f in csv_files:
        print("     ", f)

    all_q = []

    for csv_file in csv_files:
        df = read_csv_auto(csv_file)
        print(f"\nSpalten in Qobs-Datei {csv_file.name}:")
        print(list(df.columns))

        date_col = choose_date_col(df, CFG.QOBS_DATE_COLUMN)

        if CFG.QOBS_VALUE_COLUMN is not None:
            value_col = CFG.QOBS_VALUE_COLUMN
        else:
            candidates = [c for c in df.columns if c != date_col]
            if not candidates:
                warnings.warn(f"Keine Wertespalte in {csv_file} gefunden. Datei wird übersprungen.")
                continue

            value_col = None
            for c in candidates:
                cname = c.lower()
                if any(tok in cname for tok in ["durchfluss", "abfluss", "discharge", "q", "m3", "m³"]):
                    value_col = c
                    break

            if value_col is None:
                value_col = candidates[0]

        print(f"[OK] Qobs-Datumsspalte: {date_col}")
        print(f"[OK] Qobs-Rohwertspalte: {value_col}")

        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        raw_values = df[value_col]

        values = pd.to_numeric(
            raw_values.astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

        if values.notna().sum() < max(3, 0.5 * len(raw_values.dropna())):
            print("[OK] Qobs-Spalte enthält offenbar 'Min - Mittel - Max'. Nutze den mittleren Wert.")
            values = extract_q_mean_from_min_mean_max(raw_values)

        q_part = pd.Series(values.values, index=dates, name="Qobs_m3s")
        q_part = q_part.dropna().sort_index()
        all_q.append(q_part)

    if not all_q:
        raise ValueError("Keine gültigen Qobs-Werte gelesen.")

    q = pd.concat(all_q).sort_index()
    q = q[~q.index.duplicated(keep="first")]
    q = q.loc[CFG.START:CFG.END]

    if q.empty:
        raise ValueError("Qobs ist nach dem Einlesen leer.")

    print("[OK] Qobs-Zeitraum:")
    print(f"     {q.index.min().date()} bis {q.index.max().date()}, {len(q)} Tageswerte")
    print("[OK] Beispiel Qobs-Tageswerte:")
    print(q.head())

    q_monthly = q.resample("MS").mean()
    q_monthly.name = "Qobs_m3s"

    print("[OK] Beispiel Qobs-Monatswerte:")
    print(q_monthly.head())

    return q_monthly


# =============================================================================
# MODELL, KALIBRIERUNG UND METRIKEN
# =============================================================================

def nse(obs: pd.Series | np.ndarray, sim: pd.Series | np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[m]
    sim = sim[m]
    if len(obs) < 3:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((obs - sim) ** 2) / denom


def kge(obs: pd.Series | np.ndarray, sim: pd.Series | np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[m]
    sim = sim[m]
    if len(obs) < 3:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs) if np.std(obs) != 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def pbias(obs: pd.Series | np.ndarray, sim: pd.Series | np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[m]
    sim = sim[m]
    if len(obs) == 0 or np.sum(obs) == 0:
        return np.nan
    return 100.0 * np.sum(sim - obs) / np.sum(obs)


def run_monthly_model(
    p_monthly: pd.Series,
    pet_monthly: pd.Series,
    fc: np.ndarray,
    mask: np.ndarray,
    ref_profile: dict,
    params: dict[str, float],
    slope_quickflow_factor: np.ndarray,
    landuse: dict[str, np.ndarray],
    gauge_routings: dict[str, dict],
    routing_velocity_m_s: Optional[float],
    primary_key: str,
    collect_maps: bool = True,
) -> tuple[pd.DataFrame, Optional[dict[str, np.ndarray]]]:
    """Raster-Wasserbilanz und D8-Routing zu einem oder mehreren Pegeln."""
    alpha_fast = params["alpha_fast"]
    beta_perc = params["beta_perc"]
    k_base = params["k_base"]
    init_soil_frac = params["init_soil_frac"]

    idx = p_monthly.index.intersection(pet_monthly.index).sort_values()
    fc = np.where(mask, fc, np.nan)
    soil = np.where(mask, fc * init_soil_frac, np.nan)
    groundwater = np.where(mask, 0.0, np.nan)
    cell_area_m2 = abs(ref_profile["transform"].a * ref_profile["transform"].e)

    alpha_grid = np.clip(
        alpha_fast
        * slope_quickflow_factor
        * landuse["quickflow_factor"],
        0.0,
        0.98,
    )
    beta_grid = np.clip(
        beta_perc * landuse["percolation_factor"],
        0.0,
        0.95,
    )
    alpha_grid = np.where(mask, alpha_grid, np.nan)
    beta_grid = np.where(mask, beta_grid, np.nan)

    lag_infos: dict[str, dict] = {}
    arrivals: dict[str, np.ndarray] = {}
    if CFG.ROUTING_ENABLED and routing_velocity_m_s is not None:
        for key, routing in gauge_routings.items():
            lag_infos[key] = prepare_routing_lags(routing, routing_velocity_m_s)
            arrivals[key] = np.zeros(
                len(idx) + CFG.MAX_ROUTING_LAG_MONTHS + 1,
                dtype="float64",
            )

    rows_out = []
    maps = None
    if collect_maps:
        maps = {
            "runoff_sum_mm": np.zeros_like(fc, dtype="float64"),
            "quick_runoff_sum_mm": np.zeros_like(fc, dtype="float64"),
            "baseflow_sum_mm": np.zeros_like(fc, dtype="float64"),
            "recharge_sum_mm": np.zeros_like(fc, dtype="float64"),
            "aet_sum_mm": np.zeros_like(fc, dtype="float64"),
            "soil_final_mm": np.zeros_like(fc, dtype="float64"),
            "alpha_fast_grid": alpha_grid.copy(),
            "beta_perc_grid": beta_grid.copy(),
        }

    for month_position, date in enumerate(idx):
        p = float(p_monthly.loc[date])
        pet = float(pet_monthly.loc[date])

        water = soil + p
        pet_grid = pet * landuse["pet_factor"]
        aet = np.minimum(pet_grid, water)
        water_after_et = water - aet

        soil_filled = np.minimum(water_after_et, fc)
        excess = np.maximum(water_after_et - fc, 0.0)
        quick_runoff = alpha_grid * excess

        perc_from_soil = beta_grid * soil_filled
        soil_new = np.maximum(soil_filled - perc_from_soil, 0.0)
        recharge = (1.0 - alpha_grid) * excess + perc_from_soil

        groundwater = groundwater + recharge
        baseflow = k_base * groundwater
        groundwater = np.maximum(groundwater - baseflow, 0.0)
        local_runoff = quick_runoff + baseflow

        aet = np.where(mask, aet, np.nan)
        recharge = np.where(mask, recharge, np.nan)
        quick_runoff = np.where(mask, quick_runoff, np.nan)
        baseflow = np.where(mask, baseflow, np.nan)
        local_runoff = np.where(mask, local_runoff, np.nan)
        soil = np.where(mask, soil_new, np.nan)

        if arrivals:
            for key, routing in gauge_routings.items():
                route_local_runoff_to_gauge(
                    local_runoff_mm=local_runoff,
                    cell_area_m2=cell_area_m2,
                    month_position=month_position,
                    arrivals_m3=arrivals[key],
                    gauge_routing=routing,
                    lag_info=lag_infos[key],
                )

        primary_mask = gauge_routings[primary_key]["upstream_mask"]
        rows_out.append({
            "date": date,
            "P_mm_month": p,
            "PET_mm_month": pet,
            "AET_mean_model_mm": float(np.nanmean(aet[mask])),
            "AET_mean_primary_mm": float(np.nanmean(aet[primary_mask])),
            "Recharge_mean_model_mm": float(np.nanmean(recharge[mask])),
            "Recharge_mean_primary_mm": float(np.nanmean(recharge[primary_mask])),
            "Runoff_generated_mean_model_mm": float(np.nanmean(local_runoff[mask])),
            "Runoff_generated_mean_primary_mm": float(np.nanmean(local_runoff[primary_mask])),
            "Quick_runoff_mean_primary_mm": float(np.nanmean(quick_runoff[primary_mask])),
            "Baseflow_mean_primary_mm": float(np.nanmean(baseflow[primary_mask])),
            "Soil_mean_primary_mm": float(np.nanmean(soil[primary_mask])),
            "Groundwater_mean_primary_mm": float(np.nanmean(groundwater[primary_mask])),
        })

        if maps is not None:
            maps["runoff_sum_mm"] += np.nan_to_num(local_runoff)
            maps["quick_runoff_sum_mm"] += np.nan_to_num(quick_runoff)
            maps["baseflow_sum_mm"] += np.nan_to_num(baseflow)
            maps["recharge_sum_mm"] += np.nan_to_num(recharge)
            maps["aet_sum_mm"] += np.nan_to_num(aet)

    for i, row in enumerate(rows_out):
        date = row["date"]
        days = int(pd.Period(date, freq="M").days_in_month)
        for key, arr in arrivals.items():
            q = float(arr[i]) / (days * 86400.0)
            row[f"Qsim_{key}_m3s"] = q
        row["Qsim_m3s"] = row.get(f"Qsim_{primary_key}_m3s", np.nan)

    if maps is not None:
        maps["soil_final_mm"] = soil
        for key in maps:
            maps[key] = np.where(mask, maps[key], np.nan)

    df = pd.DataFrame(rows_out).set_index("date")
    df.attrs["routing_velocity_m_s"] = routing_velocity_m_s
    for key, arr in arrivals.items():
        df.attrs[f"routing_tail_m3_{key}"] = float(np.sum(arr[len(idx):]))
    return df, maps


def make_param_dict(x: Iterable[float]) -> dict[str, float]:
    return {
        "alpha_fast": float(x[0]),
        "beta_perc": float(x[1]),
        "k_base": float(x[2]),
        "init_soil_frac": float(x[3]),
    }


def run_lumped_monthly_model(
    p_monthly: pd.Series,
    pet_monthly: pd.Series,
    fc_mean_mm: float,
    basin_area_m2: float,
    params: dict[str, float],
) -> pd.DataFrame:
    alpha_fast = params["alpha_fast"]
    beta_perc = params["beta_perc"]
    k_base = params["k_base"]
    init_soil_frac = params["init_soil_frac"]
    idx = p_monthly.index.intersection(pet_monthly.index).sort_values()
    soil = fc_mean_mm * init_soil_frac
    groundwater = 0.0
    rows = []
    for date in idx:
        p = float(p_monthly.loc[date])
        pet = float(pet_monthly.loc[date])
        days = int(pd.Period(date, freq="M").days_in_month)
        water = soil + p
        aet = min(pet, water)
        after = water - aet
        soil_filled = min(after, fc_mean_mm)
        excess = max(after - fc_mean_mm, 0.0)
        quick = alpha_fast * excess
        perc = beta_perc * soil_filled
        soil = max(soil_filled - perc, 0.0)
        recharge = (1.0 - alpha_fast) * excess + perc
        groundwater += recharge
        base = k_base * groundwater
        groundwater = max(groundwater - base, 0.0)
        runoff = quick + base
        q = runoff * basin_area_m2 / 1000.0 / (days * 86400.0)
        rows.append({"date": date, "Qsim_m3s": q})
    return pd.DataFrame(rows).set_index("date")


def calibrate(
    p_monthly: pd.Series,
    pet_monthly: pd.Series,
    qobs: pd.Series,
    fc: np.ndarray,
    calibration_mask: np.ndarray,
    ref_profile: dict,
) -> dict[str, float]:
    try:
        from scipy.optimize import differential_evolution
    except Exception:
        warnings.warn("scipy nicht verfügbar; Default-Parameter werden verwendet.")
        return make_param_dict([0.25, 0.05, 0.20, 0.50])

    cell_area = abs(ref_profile["transform"].a * ref_profile["transform"].e)
    basin_area = float(calibration_mask.sum() * cell_area)
    fc_mean = float(np.nanmean(fc[calibration_mask]))
    cstart, cend = pd.Timestamp(CFG.CALIB_START), pd.Timestamp(CFG.CALIB_END)

    print("\n[INFO] Hydrologische Kalibrierung für primäres Pegel-Einzugsgebiet")
    print(f"       Fläche: {basin_area/1e6:.2f} km², mittlere nFK: {fc_mean:.1f} mm")

    def objective(x):
        sim = run_lumped_monthly_model(
            p_monthly, pet_monthly, fc_mean, basin_area, make_param_dict(x)
        )
        joined = pd.concat([qobs.rename("Qobs_m3s"), sim["Qsim_m3s"]], axis=1).dropna()
        joined = joined.loc[cstart:cend]
        score = nse(joined["Qobs_m3s"], joined["Qsim_m3s"])
        return -score if np.isfinite(score) else 9999.0

    bounds = [
        (0.001, 0.95),
        (0.0001, 0.60),
        (0.001, 0.95),
        (0.01, 1.00),
    ]
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=CFG.RANDOM_SEED,
        maxiter=80,
        popsize=10,
        polish=True,
        updating="immediate",
        workers=1,
        tol=0.0005,
    )
    params = make_param_dict(result.x)
    print("[OK] Hydrologische Parameter:")
    for key, value in params.items():
        print(f"     {key}: {value:.4f}")
    print(f"     lumped NSE: {-result.fun:.3f}")
    return params


def calibrate_routing_velocity(
    p_monthly: pd.Series,
    pet_monthly: pd.Series,
    qobs: pd.Series,
    fc: np.ndarray,
    mask: np.ndarray,
    ref_profile: dict,
    params: dict[str, float],
    slope_quickflow_factor: np.ndarray,
    landuse: dict[str, np.ndarray],
    gauge_routings: dict[str, dict],
    primary_key: str,
) -> tuple[float, pd.DataFrame]:
    candidates = (
        CFG.ROUTING_VELOCITY_CANDIDATES
        if CFG.CALIBRATE_ROUTING_VELOCITY
        else (CFG.ROUTING_VELOCITY_M_S,)
    )
    cstart, cend = pd.Timestamp(CFG.CALIB_START), pd.Timestamp(CFG.CALIB_END)
    records = []
    best_velocity = float(CFG.ROUTING_VELOCITY_M_S)
    best_score = -np.inf
    print("\n[INFO] Kalibriere D8-Routinggeschwindigkeit am primären Pegel:")
    for velocity in candidates:
        sim, _ = run_monthly_model(
            p_monthly, pet_monthly, fc, mask, ref_profile, params,
            slope_quickflow_factor, landuse, gauge_routings,
            float(velocity), primary_key, collect_maps=False,
        )
        joined = pd.concat([qobs.rename("Qobs_m3s"), sim["Qsim_m3s"]], axis=1).dropna()
        joined = joined.loc[cstart:cend]
        nn = nse(joined["Qobs_m3s"], joined["Qsim_m3s"])
        kk = kge(joined["Qobs_m3s"], joined["Qsim_m3s"])
        pp = pbias(joined["Qobs_m3s"], joined["Qsim_m3s"])
        records.append({
            "routing_velocity_m_s": float(velocity),
            "NSE_calibration": nn,
            "KGE_calibration": kk,
            "PBIAS_calibration_percent": pp,
        })
        print(f"     v={velocity:.3f} m/s -> NSE={nn:.3f}, KGE={kk:.3f}, PBIAS={pp:.1f} %")
        if np.isfinite(nn) and nn > best_score:
            best_score, best_velocity = float(nn), float(velocity)
    print(f"[OK] Beste Routinggeschwindigkeit: {best_velocity:.3f} m/s")
    return best_velocity, pd.DataFrame(records).sort_values("NSE_calibration", ascending=False)


# =============================================================================
# OUTPUT: PLOTS, TABELLEN, KARTEN
# =============================================================================

def plot_q_timeseries(results: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(11, 5))
    if "Qobs_m3s" in results.columns:
        plt.plot(results.index, results["Qobs_m3s"], label="Qobs")
    plt.plot(results.index, results["Qsim_m3s"], label="Qsim")
    plt.ylabel("Q [m³/s]")
    plt.xlabel("Monat")
    plt.title("Beobachteter und simulierter Monatsabfluss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_map(arr: np.ndarray, title: str, path: Path, label: str = "mm") -> None:
    plt.figure(figsize=(8, 6))
    im = plt.imshow(arr)
    plt.colorbar(im, label=label)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def summarize_performance(results: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "calibration": (CFG.CALIB_START, CFG.CALIB_END),
        "validation": (CFG.VALID_START, CFG.VALID_END),
        "full": (CFG.START, CFG.END),
    }

    rows = []
    for name, (start, end) in periods.items():
        part = results.loc[start:end].dropna(subset=["Qobs_m3s", "Qsim_m3s"])
        rows.append({
            "period": name,
            "start": start,
            "end": end,
            "n_months": len(part),
            "NSE": nse(part["Qobs_m3s"], part["Qsim_m3s"]),
            "KGE": kge(part["Qobs_m3s"], part["Qsim_m3s"]),
            "PBIAS_percent": pbias(part["Qobs_m3s"], part["Qsim_m3s"]),
            "Qobs_mean_m3s": part["Qobs_m3s"].mean(),
            "Qsim_mean_m3s": part["Qsim_m3s"].mean(),
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main_v7_legacy() -> None:
    """Nicht ausgeführter V7-Referenzworkflow; V8 startet über main_v8()."""
    OUT = out_dir()
    print("============================================================")
    print("HydroMod v7: Multi-Pegel, D8-Routing, Schwalm und LBM-DE")
    print("============================================================")
    print(f"Projektordner: {CFG.BASE_DIR}")
    print(f"Datenordner:   {data_dir()}")
    print(f"Output:        {OUT}")

    # 1. Projektgebiet und DGM
    dgm_path = find_first(data_dir() / CFG.DGM_DIRNAME, [".tif", ".tiff"])
    dgm_full, profile_full = read_raster(dgm_path)
    catchment = load_catchment(profile_full["crs"])
    mask_full = create_mask(catchment, profile_full)
    print(f"[OK] Projektgebiet: {int(mask_full.sum())} Rasterzellen")

    dgm, mask, profile, crop_offset = crop_raster_to_mask(
        dgm_full, mask_full, profile_full, padding=CFG.CROP_PADDING_CELLS
    )
    del dgm_full, mask_full
    print(f"[OK] Crop: {profile['width']} x {profile['height']} Zellen; Offset={crop_offset}")

    # 2. Boden, Schwalm, LBM-DE und Hangneigung
    fc = load_field_capacity_raster(profile, mask)
    river_mask, river_gdf = load_river_mask(profile, mask)
    landuse = load_landuse_rasters(profile, mask, catchment)

    dgm = fill_missing_dem_nearest(dgm, mask)
    dgm_burned, dgm_conditioned = condition_dem_priority_flood(
        dgm, mask, river_mask, profile
    )
    slope_fraction = calculate_slope_fraction(dgm_conditioned, mask, profile)
    slope_percent = slope_fraction * 100.0
    slope_factor = make_slope_quickflow_factor(
        slope_fraction, mask, CFG.SLOPE_QUICKFLOW_WEIGHT
    )
    print(f"[OK] Mittlere Hangneigung: {np.nanmean(slope_percent):.2f} %")

    # 3. Globales D8-Netz der gesamten Modelldomäne
    network = build_d8_flow_network(
        dgm_conditioned, mask, profile, river_mask
    )

    # 4. Pegel laden, snappen und jeweilige Upstream-Gebiete ableiten
    gauges = load_gauge_points(profile["crs"])
    gauge_routings: dict[str, dict] = {}
    for _, gauge_row in gauges.iterrows():
        routing = build_gauge_routing(
            gauge_row, network, mask, profile, river_mask
        )
        gauge_routings[routing["key"]] = routing

    primary = next(r for r in gauge_routings.values() if r["is_primary"])
    primary_key = primary["key"]
    primary_mask = primary["upstream_mask"]
    primary_area = primary["area_km2"]
    area_diff = 100.0 * (
        primary_area - CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2
    ) / CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2
    print(
        f"[INFO] Heidelbach: D8-Fläche={primary_area:.2f} km²; "
        f"offiziell={CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2:.2f} km²; "
        f"Differenz={area_diff:+.1f} %"
    )
    if abs(area_diff) > 10.0:
        warnings.warn(
            "Das abgeleitete Heidelbach-Einzugsgebiet weicht um mehr als 10 % "
            "von der offiziellen Fläche ab. DGM, Stream Burning, Pegel-Snap und "
            "Gebietsgrenze müssen in QGIS geprüft werden."
        )

    primary_catchment = mask_to_geodataframe(
        primary_mask, profile, "Heidelbach_D8_upstream"
    )

    # 5. Forcing: für den Qobs-Vergleich standardmäßig das Upstream-Gebiet
    if CFG.FORCING_USE_PRIMARY_GAUGE_CATCHMENT:
        forcing_geometry = primary_catchment
        forcing_mask = primary_mask
        print("[INFO] P und PET werden über das Heidelbach-Upstream-Gebiet gemittelt.")
    else:
        forcing_geometry = catchment
        forcing_mask = mask
        print("[INFO] P und PET werden über das gesamte Projektgebiet gemittelt.")

    p_monthly = load_precip_monthly(forcing_geometry, profile["crs"])
    pet_monthly = load_pet_monthly(profile, forcing_mask)

    qobs_map: dict[str, str] = {str(k): str(v) for k, v in CFG.GAUGE_QOBS_FOLDERS}
    primary_qobs_dir = qobs_map.get(str(primary["station_id"]), CFG.QOBS_DIRNAME)
    qobs_primary = load_qobs_monthly(primary_qobs_dir)

    forcing = pd.concat(
        [p_monthly, pet_monthly, qobs_primary.rename("Qobs_m3s")], axis=1
    ).loc[CFG.START:CFG.END]
    forcing.to_csv(OUT / "tables" / "monthly_input_timeseries.csv", encoding="utf-8")

    p = forcing["P_mm_month"].dropna()
    pet = forcing["PET_mm_month"].dropna()
    qobs = forcing["Qobs_m3s"]
    common = p.index.intersection(pet.index)
    p, pet = p.loc[common], pet.loc[common]

    # 6. Kalibrierung am primären Pegelgebiet
    if CFG.USE_CALIBRATION:
        params = calibrate(p, pet, qobs, fc, primary_mask, profile)
    else:
        params = make_param_dict([0.25, 0.05, 0.20, 0.50])

    routing_velocity, routing_diag = calibrate_routing_velocity(
        p, pet, qobs, fc, mask, profile, params, slope_factor,
        landuse, gauge_routings, primary_key,
    )
    routing_diag.to_csv(
        OUT / "tables" / "routing_velocity_calibration.csv",
        index=False, encoding="utf-8",
    )

    # 7. Finale Simulation
    sim, maps = run_monthly_model(
        p, pet, fc, mask, profile, params, slope_factor, landuse,
        gauge_routings, routing_velocity, primary_key, collect_maps=True,
    )
    if maps is None:
        raise RuntimeError("Keine Kartenoutputs erzeugt.")

    results = pd.concat(
        [forcing, sim.drop(columns=["P_mm_month", "PET_mm_month"], errors="ignore")],
        axis=1,
    ).loc[CFG.START:CFG.END]

    # Optionale Qobs-Reihen weiterer Pegel ergänzen.
    for routing in gauge_routings.values():
        if routing["is_primary"]:
            continue
        folder = qobs_map.get(str(routing["station_id"]))
        if folder:
            try:
                qextra = load_qobs_monthly(folder)
                results[f"Qobs_{routing['key']}_m3s"] = qextra
            except Exception as exc:
                warnings.warn(f"Qobs für {routing['name']} konnte nicht geladen werden: {exc}")

    perf = summarize_performance(results)
    perf.to_csv(OUT / "tables" / "performance_metrics.csv", index=False, encoding="utf-8")
    results.to_csv(OUT / "tables" / "monthly_model_results.csv", encoding="utf-8")

    parameter_out = dict(params)
    parameter_out.update({
        "routing_velocity_m_s": routing_velocity,
        "stream_burn_depth_m": CFG.STREAM_BURN_DEPTH_M,
        "channel_speed_multiplier": CFG.CHANNEL_SPEED_MULTIPLIER,
        "primary_gauge_area_km2": primary_area,
        "primary_gauge_official_area_km2": CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2,
    })
    pd.DataFrame([parameter_out]).to_csv(
        OUT / "tables" / "calibrated_parameters.csv", index=False, encoding="utf-8"
    )

    # Pegeldiagnostik
    gauge_records = []
    for routing in gauge_routings.values():
        gauge_records.append({
            "gauge_key": routing["key"],
            "name": routing["name"],
            "station_id": routing["station_id"],
            "is_primary": routing["is_primary"],
            "original_x": routing.get("original_x", routing.get("official_point_x", routing.get("configured_point_x", np.nan))),
            "original_y": routing.get("original_y", routing.get("official_point_y", routing.get("configured_point_y", np.nan))),
            "snapped_x": routing["x"],
            "snapped_y": routing["y"],
            "snap_distance_m": routing["snap_distance_m"],
            "on_river_mask": routing["on_river_mask"],
            "upstream_area_km2": routing["area_km2"],
        })
    pd.DataFrame(gauge_records).to_csv(
        OUT / "tables" / "gauge_routing_diagnostics.csv",
        index=False, encoding="utf-8",
    )

    # 8. Rasteroutputs
    outputs = {
        "field_capacity_mm.tif": fc,
        "dgm_cropped_m.tif": np.where(mask, dgm, np.nan),
        "dgm_stream_burned_m.tif": dgm_burned,
        "conditioned_dgm_m.tif": dgm_conditioned,
        "slope_percent.tif": slope_percent,
        "slope_quickflow_factor.tif": slope_factor,
        "river_mask.tif": np.where(mask, river_mask.astype(float), np.nan),
        "sealed_percent_lbm.tif": landuse["sealed_percent"],
        "vegetation_percent_lbm.tif": landuse["vegetation_percent"],
        "landuse_quickflow_factor.tif": landuse["quickflow_factor"],
        "landuse_pet_factor.tif": landuse["pet_factor"],
        "landuse_percolation_factor.tif": landuse["percolation_factor"],
        "flow_accumulation_cells.tif": network["accumulation_raster"],
        "runoff_generated_sum_2021_2025_mm.tif": maps["runoff_sum_mm"],
        "quick_runoff_sum_2021_2025_mm.tif": maps["quick_runoff_sum_mm"],
        "baseflow_sum_2021_2025_mm.tif": maps["baseflow_sum_mm"],
        "recharge_sum_2021_2025_mm.tif": maps["recharge_sum_mm"],
        "aet_sum_2021_2025_mm.tif": maps["aet_sum_mm"],
        "soil_final_mm.tif": maps["soil_final_mm"],
        "alpha_fast_spatial.tif": maps["alpha_fast_grid"],
        "beta_perc_spatial.tif": maps["beta_perc_grid"],
        "primary_gauge_upstream_mask.tif": np.where(mask, primary_mask.astype(float), np.nan),
    }
    cell_area = abs(profile["transform"].a * profile["transform"].e)
    outputs["flow_accumulation_area_km2.tif"] = network["accumulation_raster"] * cell_area / 1e6

    primary_lags = prepare_routing_lags(primary, routing_velocity)
    outputs["primary_routing_travel_time_days.tif"] = compact_to_raster(
        primary_lags["travel_days"], primary["rows"], primary["cols"], mask.shape
    )
    outputs["primary_routing_lag_months.tif"] = compact_to_raster(
        primary_lags["lag_months"], primary["rows"], primary["cols"], mask.shape
    )

    for filename, array in outputs.items():
        write_geotiff(OUT / "maps" / filename, array, profile)

    # Vektoroutputs
    primary_path = OUT / "maps" / "primary_gauge_catchment.gpkg"
    if primary_path.exists():
        primary_path.unlink()
    primary_catchment.to_file(primary_path, layer="heidelbach_upstream", driver="GPKG")

    from shapely.geometry import Point
    snapped_records = []
    for routing in gauge_routings.values():
        snapped_records.append({
            "name": routing["name"],
            "station_id": routing["station_id"],
            "is_primary": routing["is_primary"],
            "snap_m": routing["snap_distance_m"],
            "area_km2": routing["area_km2"],
            "geometry": Point(routing["x"], routing["y"]),
        })
    snapped_gdf = gpd.GeoDataFrame(snapped_records, crs=profile["crs"])
    snapped_path = OUT / "maps" / "gauges_snapped.gpkg"
    if snapped_path.exists():
        snapped_path.unlink()
    snapped_gdf.to_file(snapped_path, layer="gauges_snapped", driver="GPKG")

    # 9. Plots
    plot_q_timeseries(results, OUT / "plots" / "qobs_qsim_monthly_routed.png")
    plot_map(
        outputs["flow_accumulation_area_km2.tif"],
        "D8-Fließakkumulation im Projektgebiet",
        OUT / "plots" / "map_flow_accumulation_km2.png",
        "beitragende Fläche [km²]",
    )
    plot_map(
        outputs["primary_routing_travel_time_days.tif"],
        "Reisezeit zum Pegel Heidelbach",
        OUT / "plots" / "map_primary_travel_time_days.png",
        "Tage",
    )
    plot_map(
        landuse["sealed_percent"],
        "Versiegelungsgrad aus LBM-DE2021",
        OUT / "plots" / "map_sealed_percent.png",
        "%",
    )

    print("\n[OK] Performance am primären Pegel:")
    print(perf)
    print("\n============================================================")
    print("[FERTIG]")
    print(f"Ergebnisse: {OUT}")
    print("Wichtig zur Kontrolle:")
    print(" - tables/gauge_routing_diagnostics.csv")
    print(" - maps/primary_gauge_catchment.gpkg")
    print(" - maps/gauges_snapped.gpkg")
    print(" - maps/river_mask.tif")
    print(" - maps/flow_accumulation_area_km2.tif")
    print(" - maps/sealed_percent_lbm.tif")
    print(" - plots/qobs_qsim_monthly_routed.png")
    print("============================================================")


# =============================================================================
# HYDROMOD V8: DIREKTE RASTERKALIBRIERUNG, BODENEINHEITEN UND
# RÄUMLICHE VALIDIERUNG
# =============================================================================

import shutil
import tarfile


PARAM_NAMES_V8 = (
    "alpha_fast",
    "beta_perc",
    "k_gw_fast",
    "k_gw_slow",
    "slow_recharge_frac",
    "init_soil_frac",
    "routing_velocity_m_s",
)

PARAM_BOUNDS_V8 = (
    (0.005, 0.85),    # alpha_fast
    (0.0001, 0.30),   # beta_perc
    (0.04, 0.65),     # k_gw_fast [1/Monat]
    (0.003, 0.12),    # k_gw_slow [1/Monat]
    (0.20, 0.95),     # Anteil Recharge -> langsamer GW-Speicher
    (0.15, 1.00),     # Anfangsfüllung Boden
    (0.20, 2.50),     # effektive Routinggeschwindigkeit [m/s]
)


def out_dir() -> Path:
    """V8-Ausgabe mit vier Ergebnisordnern."""
    p = CFG.BASE_DIR / "results" / "monthly_v8_1_revised"
    for sub in ("maps", "plots", "tables", "hausarbeit"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def make_param_dict_v8(x: Iterable[float]) -> dict[str, float]:
    values = list(x)
    if len(values) != len(PARAM_NAMES_V8):
        raise ValueError(
            f"V8 erwartet {len(PARAM_NAMES_V8)} Parameter, erhalten: {len(values)}"
        )
    return {name: float(value) for name, value in zip(PARAM_NAMES_V8, values)}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not m.any():
        return np.nan
    return float(np.sum(values[m] * weights[m]) / np.sum(weights[m]))


def _normalize_text(value) -> str:
    return str(value).strip().lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")


def _soil_parameterization(value) -> tuple[float, float, float, str]:
    """
    Heuristische Parameterisierung aus Bodenklassenbezeichnungen.

    Rückgabe:
      quickflow_factor, percolation_factor, slow_recharge_factor, rule

    Die Faktoren werden später auf ein Gebietsmittel von 1 normiert. Die nFK
    bleibt weiterhin der eigentliche maximale Bodenspeicher; Bodeneinheiten
    steuern hier nur Abflussaufteilung und Versickerung.
    """
    s = _normalize_text(value)

    if any(t in s for t in ("gewaesser", "wasserflaeche", "see", "fluss")):
        return 1.00, 0.05, 0.70, "water"
    if any(t in s for t in ("sied", "verkehr", "industrie", "bebau", "versiegel")):
        return 1.55, 0.35, 0.65, "sealed"
    if any(t in s for t in ("moor", "torf", "anmoor")):
        return 0.85, 0.65, 1.25, "peat"
    if any(t in s for t in ("ton", "pseudogley", "stau", "pelosol")):
        return 1.30, 0.58, 0.82, "clayey/stagnic"
    if any(t in s for t in ("gley", "aue", "grundwasser", "nass", "feucht")):
        return 1.12, 0.80, 1.15, "groundwater-influenced"
    if any(t in s for t in ("fels", "stein", "skelett", "rendzina", "ranker", "flachgruendig")):
        return 1.22, 0.76, 0.85, "shallow/skeletal"
    if any(t in s for t in ("sand", "podsol", "duene", "flug")):
        return 0.76, 1.38, 1.18, "sandy"
    if any(t in s for t in ("loess", "loess", "lehm", "schluff", "parabraun", "braunerde")):
        return 0.95, 1.08, 1.05, "loamy/silty"

    return 1.00, 1.00, 1.00, "neutral/unclassified"



def _choose_soil_class_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """Wählt eine fachlich interpretierbare Bodenklassenspalte.

    Zuerst werden typische Feldnamen geprüft. Falls diese fehlen, werden
    Textspalten anhand ihrer tatsächlichen Werte bewertet. Reine numerische IDs
    werden bewusst nicht automatisch hydrologisch interpretiert, weil dafür eine
    Legendentabelle nötig wäre.
    """
    preferred = _find_column(
        gdf.columns,
        None,
        [
            "BEZEICHNER", "BODENTYP", "BODENART", "BODENEINHEIT",
            "BODENEINH", "BODENGRUPPE", "BODENGR", "BODEN", "NAME",
            "KURZ", "KLASSE", "SYMBOL", "LEGENDE", "EINHEIT",
            "SUBSTRAT", "GENESE", "TYP", "CODE",
        ],
    )
    if preferred is not None:
        return preferred

    soil_tokens = (
        "braunerde", "parabraun", "pseudogley", "gley", "podsol",
        "rendzina", "ranker", "pelosol", "moor", "torf", "aue",
        "sand", "lehm", "schluff", "ton", "loess", "loess",
        "skelett", "fels", "stau", "grundwasser", "boden",
    )
    best_col: Optional[str] = None
    best_score = 0.0
    for col in gdf.columns:
        if col == "geometry":
            continue
        series = gdf[col].dropna()
        if series.empty:
            continue
        # Nur textartige oder überschaubar kategoriale Spalten prüfen.
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            continue
        values = series.astype(str).drop_duplicates().head(500)
        normalized = [_normalize_text(v) for v in values]
        hit_share = np.mean([
            any(token in value for token in soil_tokens)
            for value in normalized
        ]) if len(normalized) else 0.0
        name = _normalize_text(col)
        name_bonus = 0.25 if any(t in name for t in ("boden", "typ", "art", "klasse", "legend", "einheit")) else 0.0
        score = float(hit_share + name_bonus)
        if hit_share > 0 and score > best_score:
            best_score = score
            best_col = str(col)

    if best_col is not None:
        print(f"[OK] Bodenklassenspalte über Werte erkannt: {best_col}")
        return best_col

    print("[INFO] Spalten der Bodeneinheiten:")
    print(list(gdf.columns))
    for col in gdf.columns:
        if col == "geometry":
            continue
        vals = gdf[col].dropna().astype(str).drop_duplicates().head(8).tolist()
        if vals:
            print(f"       {col}: {vals}")
    return None


def load_soil_unit_factors(
    ref_profile: dict,
    mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Rasterisiert Bodeneinheiten und leitet räumliche Prozessfaktoren ab."""
    neutral = {
        "class_code": np.where(mask, 0.0, np.nan),
        "quickflow_factor": np.where(mask, 1.0, np.nan),
        "percolation_factor": np.where(mask, 1.0, np.nan),
        "slow_recharge_factor": np.where(mask, 1.0, np.nan),
    }

    soil_dir = data_dir() / CFG.SOIL_UNITS_DIRNAME
    try:
        path = find_first(soil_dir, [".shp", ".gpkg", ".geojson"])
        print(f"[OK] Bodeneinheiten: {path}")
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError("Bodeneinheiten sind leer.")
        if gdf.crs is None:
            warnings.warn("Bodeneinheiten haben kein CRS; Referenz-CRS wird angenommen.")
            gdf = gdf.set_crs(ref_profile["crs"])
        gdf = gdf.to_crs(ref_profile["crs"])
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

        if CFG.SOIL_CLASS_FIELD is not None:
            if CFG.SOIL_CLASS_FIELD not in gdf.columns:
                raise ValueError(
                    f"SOIL_CLASS_FIELD={CFG.SOIL_CLASS_FIELD!r} nicht vorhanden. "
                    f"Spalten: {list(gdf.columns)}"
                )
            class_col = CFG.SOIL_CLASS_FIELD
        else:
            class_col = _choose_soil_class_column(gdf)

        if class_col is None:
            warnings.warn(
                "Keine interpretierbare Bodenklassenspalte gefunden. "
                "Bodeneinheiten bleiben als neutrale Faktoren eingebunden."
            )
            return neutral, pd.DataFrame([{
                "soil_class": "unclassified",
                "rule": "neutral",
                "quickflow_factor_raw": 1.0,
                "percolation_factor_raw": 1.0,
                "slow_recharge_factor_raw": 1.0,
                "n_polygons": len(gdf),
            }])

        print(f"[OK] Bodenklassenspalte: {class_col}")
        gdf["_SOIL_TEXT"] = gdf[class_col].astype(str)
        factors = gdf["_SOIL_TEXT"].apply(_soil_parameterization)
        gdf["_QF"] = factors.apply(lambda x: x[0])
        gdf["_PERC"] = factors.apply(lambda x: x[1])
        gdf["_SLOW"] = factors.apply(lambda x: x[2])
        gdf["_RULE"] = factors.apply(lambda x: x[3])

        unique_classes = sorted(gdf["_SOIL_TEXT"].dropna().unique().tolist())
        class_codes = {value: i + 1 for i, value in enumerate(unique_classes)}
        gdf["_CLASS_CODE"] = gdf["_SOIL_TEXT"].map(class_codes).astype(float)

        def burn(column: str, fill: float) -> np.ndarray:
            shapes_values = [
                (geom, float(value))
                for geom, value in zip(gdf.geometry, gdf[column])
                if geom is not None and not geom.is_empty and np.isfinite(value)
            ]
            arr = rasterize(
                shapes=shapes_values,
                out_shape=(ref_profile["height"], ref_profile["width"]),
                transform=ref_profile["transform"],
                fill=float(fill),
                dtype="float32",
                all_touched=True,
            ).astype("float64")
            return np.where(mask, arr, np.nan)

        class_raster = burn("_CLASS_CODE", 0.0)
        qf_raw = burn("_QF", 1.0)
        perc_raw = burn("_PERC", 1.0)
        slow_raw = burn("_SLOW", 1.0)

        qf = _normalize_factor(qf_raw, mask, 0.45, 1.90)
        perc = _normalize_factor(perc_raw, mask, 0.30, 1.80)
        slow = _normalize_factor(slow_raw, mask, 0.55, 1.55)

        cell_area = abs(ref_profile["transform"].a * ref_profile["transform"].e)
        table_rows = []
        for text, group in gdf.groupby("_SOIL_TEXT", dropna=False):
            code = class_codes.get(str(text), 0)
            n_cells = int(np.sum(class_raster == code)) if code else 0
            table_rows.append({
                "soil_class": str(text),
                "class_code": code,
                "rule": str(group["_RULE"].iloc[0]),
                "quickflow_factor_raw": float(group["_QF"].iloc[0]),
                "percolation_factor_raw": float(group["_PERC"].iloc[0]),
                "slow_recharge_factor_raw": float(group["_SLOW"].iloc[0]),
                "n_polygons": int(len(group)),
                "raster_cells": n_cells,
                "raster_area_km2": n_cells * cell_area / 1e6,
            })

        print(
            f"[OK] Bodenfaktoren: Quickflow {np.nanmin(qf):.2f}–{np.nanmax(qf):.2f}, "
            f"Perkolation {np.nanmin(perc):.2f}–{np.nanmax(perc):.2f}"
        )
        return {
            "class_code": class_raster,
            "quickflow_factor": qf,
            "percolation_factor": perc,
            "slow_recharge_factor": slow,
        }, pd.DataFrame(table_rows)

    except Exception as exc:
        warnings.warn(
            f"Bodeneinheiten konnten nicht räumlich parametrisiert werden ({exc}). "
            "Neutrale Bodenfaktoren werden verwendet."
        )
        return neutral, pd.DataFrame([{
            "soil_class": "fallback",
            "rule": "neutral after read error",
            "quickflow_factor_raw": 1.0,
            "percolation_factor_raw": 1.0,
            "slow_recharge_factor_raw": 1.0,
            "n_polygons": 0,
        }])


def _derive_spatial_parameter_grids_v8(
    params: dict[str, float],
    mask: np.ndarray,
    slope_quickflow_factor: np.ndarray,
    landuse: dict[str, np.ndarray],
    soil_units: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_grid = np.clip(
        params["alpha_fast"]
        * slope_quickflow_factor
        * landuse["quickflow_factor"]
        * soil_units["quickflow_factor"],
        0.0,
        0.98,
    )
    beta_grid = np.clip(
        params["beta_perc"]
        * landuse["percolation_factor"]
        * soil_units["percolation_factor"],
        0.0,
        0.75,
    )
    slow_fraction_grid = np.clip(
        params["slow_recharge_frac"] * soil_units["slow_recharge_factor"],
        0.05,
        0.98,
    )
    return (
        np.where(mask, alpha_grid, np.nan),
        np.where(mask, beta_grid, np.nan),
        np.where(mask, slow_fraction_grid, np.nan),
    )


def _water_balance_step_v8(
    p: float,
    pet: float,
    fc: np.ndarray,
    soil: np.ndarray,
    gw_fast: np.ndarray,
    gw_slow: np.ndarray,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    slow_fraction_grid: np.ndarray,
    pet_factor: np.ndarray,
    params: dict[str, float],
) -> dict[str, np.ndarray]:
    storage_before = soil + gw_fast + gw_slow
    water = soil + p

    relative_wetness = np.clip(water / np.maximum(fc, 1.0), 0.0, 1.0)
    aet_demand = pet * pet_factor * np.power(
        relative_wetness, float(CFG.ET_STRESS_EXPONENT)
    )
    aet = np.minimum(aet_demand, water)
    water_after_et = np.maximum(water - aet, 0.0)

    soil_filled = np.minimum(water_after_et, fc)
    excess = np.maximum(water_after_et - fc, 0.0)
    quick_runoff = alpha_grid * excess

    perc_from_soil = beta_grid * soil_filled
    soil_new = np.maximum(soil_filled - perc_from_soil, 0.0)
    recharge = (1.0 - alpha_grid) * excess + perc_from_soil

    recharge_slow = recharge * slow_fraction_grid
    recharge_fast = recharge - recharge_slow

    gw_fast_new = gw_fast + recharge_fast
    gw_slow_new = gw_slow + recharge_slow
    baseflow_fast = params["k_gw_fast"] * gw_fast_new
    baseflow_slow = params["k_gw_slow"] * gw_slow_new
    gw_fast_new = np.maximum(gw_fast_new - baseflow_fast, 0.0)
    gw_slow_new = np.maximum(gw_slow_new - baseflow_slow, 0.0)

    baseflow = baseflow_fast + baseflow_slow
    local_runoff = quick_runoff + baseflow
    storage_after = soil_new + gw_fast_new + gw_slow_new
    residual = p - aet - local_runoff - (storage_after - storage_before)

    return {
        "soil": soil_new,
        "gw_fast": gw_fast_new,
        "gw_slow": gw_slow_new,
        "aet": aet,
        "excess": excess,
        "quick_runoff": quick_runoff,
        "percolation": perc_from_soil,
        "recharge": recharge,
        "baseflow_fast": baseflow_fast,
        "baseflow_slow": baseflow_slow,
        "baseflow": baseflow,
        "local_runoff": local_runoff,
        "water_balance_residual": residual,
    }


def _make_compact_primary_data(
    fc: np.ndarray,
    slope_factor: np.ndarray,
    landuse: dict[str, np.ndarray],
    soil_units: dict[str, np.ndarray],
    primary_routing: dict,
    ref_profile: dict,
) -> dict[str, np.ndarray | float]:
    rows = primary_routing["rows"]
    cols = primary_routing["cols"]
    return {
        "fc": fc[rows, cols].astype("float64"),
        "slope_q": slope_factor[rows, cols].astype("float64"),
        "land_q": landuse["quickflow_factor"][rows, cols].astype("float64"),
        "land_pet": landuse["pet_factor"][rows, cols].astype("float64"),
        "land_perc": landuse["percolation_factor"][rows, cols].astype("float64"),
        "soil_q": soil_units["quickflow_factor"][rows, cols].astype("float64"),
        "soil_perc": soil_units["percolation_factor"][rows, cols].astype("float64"),
        "soil_slow": soil_units["slow_recharge_factor"][rows, cols].astype("float64"),
        "travel_cost": primary_routing["travel_cost"].astype("float64"),
        "weights": np.ones(len(rows), dtype="float64"),
        "cell_area_m2": float(abs(ref_profile["transform"].a * ref_profile["transform"].e)),
    }


def _quantile_codes(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.nanmax(finite) <= np.nanmin(finite):
        return np.zeros(len(values), dtype=np.int16)
    edges = np.unique(np.nanquantile(finite, np.linspace(0.0, 1.0, n_bins + 1))[1:-1])
    if edges.size == 0:
        return np.zeros(len(values), dtype=np.int16)
    return np.digitize(values, edges, right=False).astype(np.int16)


def make_stratified_calibration_sample(
    compact: dict[str, np.ndarray | float],
    max_cells: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """
    Erzeugt eine gewichtete, räumlich geschichtete Stichprobe für die globale
    Parametersuche. Die anschließende Feinoptimierung läuft auf allen Zellen.
    """
    n = len(np.asarray(compact["fc"]))
    if n <= max_cells:
        print(f"[INFO] Kalibrierung nutzt alle {n} Pegelgebietszellen.")
        return compact

    features = (
        np.asarray(compact["fc"]),
        np.asarray(compact["slope_q"]),
        np.asarray(compact["land_q"]),
        np.asarray(compact["soil_perc"]),
        np.log1p(np.asarray(compact["travel_cost"])),
    )
    codes = np.zeros(n, dtype=np.int32)
    multiplier = 1
    for feature in features:
        qcode = _quantile_codes(feature, n_bins=4)
        codes += qcode.astype(np.int32) * multiplier
        multiplier *= 4

    rng = np.random.default_rng(seed)
    unique, counts = np.unique(codes, return_counts=True)
    selected_parts: list[np.ndarray] = []
    weights_parts: list[np.ndarray] = []

    for code, count in zip(unique, counts):
        idx = np.flatnonzero(codes == code)
        n_take = max(1, int(round(max_cells * count / n)))
        n_take = min(n_take, len(idx))
        chosen = rng.choice(idx, size=n_take, replace=False)
        selected_parts.append(chosen)
        weights_parts.append(np.full(n_take, len(idx) / n_take, dtype="float64"))

    selected = np.concatenate(selected_parts)
    weights = np.concatenate(weights_parts)
    if len(selected) > max_cells * 1.15:
        keep = rng.choice(len(selected), size=max_cells, replace=False)
        selected = selected[keep]
        weights = weights[keep]
        weights *= n / np.sum(weights)

    sampled: dict[str, np.ndarray | float] = {}
    for key, value in compact.items():
        if isinstance(value, np.ndarray):
            sampled[key] = value[selected]
        else:
            sampled[key] = value
    sampled["weights"] = weights

    print(
        f"[INFO] Globale Kalibrierung: {len(selected)} gewichtete Zellen "
        f"repräsentieren {n} Rasterzellen."
    )
    return sampled


def simulate_primary_compact_v8(
    p_monthly: pd.Series,
    pet_monthly: pd.Series,
    compact: dict[str, np.ndarray | float],
    params: dict[str, float],
) -> pd.DataFrame:
    """Schnelle, aber prozessidentische Simulation des primären Pegelgebiets."""
    idx = p_monthly.index.intersection(pet_monthly.index).sort_values()
    fc = np.asarray(compact["fc"], dtype="float64")
    weights = np.asarray(compact["weights"], dtype="float64")
    cell_area = float(compact["cell_area_m2"])

    alpha = np.clip(
        params["alpha_fast"]
        * np.asarray(compact["slope_q"])
        * np.asarray(compact["land_q"])
        * np.asarray(compact["soil_q"]),
        0.0, 0.98,
    )
    beta = np.clip(
        params["beta_perc"]
        * np.asarray(compact["land_perc"])
        * np.asarray(compact["soil_perc"]),
        0.0, 0.75,
    )
    slow_frac = np.clip(
        params["slow_recharge_frac"] * np.asarray(compact["soil_slow"]),
        0.05, 0.98,
    )
    pet_factor = np.asarray(compact["land_pet"])

    soil = fc * params["init_soil_frac"]
    gw_fast = np.zeros_like(fc)
    gw_slow = np.zeros_like(fc)

    velocity = params["routing_velocity_m_s"]
    travel_days = np.asarray(compact["travel_cost"]) / velocity / 86400.0
    lag_float = np.clip(
        travel_days / 30.4375,
        0.0,
        float(CFG.MAX_ROUTING_LAG_MONTHS),
    )
    lag0 = np.floor(lag_float).astype(np.int16)
    frac1 = lag_float - lag0
    lag1 = np.minimum(lag0 + 1, CFG.MAX_ROUTING_LAG_MONTHS).astype(np.int16)
    frac1[lag0 >= CFG.MAX_ROUTING_LAG_MONTHS] = 0.0

    arrivals = np.zeros(len(idx) + CFG.MAX_ROUTING_LAG_MONTHS + 1, dtype="float64")
    details = []

    # Das Kalenderjahr 2021 bleibt Warm-up. Zusätzliche Zyklen sind optional.
    warm_dates = [d for d in idx if d <= pd.Timestamp(CFG.WARMUP_END)]
    for _ in range(max(0, int(CFG.SPINUP_CYCLES) - 1)):
        for date in warm_dates:
            step = _water_balance_step_v8(
                float(p_monthly.loc[date]), float(pet_monthly.loc[date]),
                fc, soil, gw_fast, gw_slow, alpha, beta, slow_frac,
                pet_factor, params,
            )
            soil, gw_fast, gw_slow = step["soil"], step["gw_fast"], step["gw_slow"]

    for i, date in enumerate(idx):
        p = float(p_monthly.loc[date])
        pet = float(pet_monthly.loc[date])
        step = _water_balance_step_v8(
            p, pet, fc, soil, gw_fast, gw_slow, alpha, beta, slow_frac,
            pet_factor, params,
        )
        soil, gw_fast, gw_slow = step["soil"], step["gw_fast"], step["gw_slow"]

        volumes = np.nan_to_num(step["local_runoff"]) * weights * cell_area / 1000.0
        by0 = np.bincount(
            lag0,
            weights=volumes * (1.0 - frac1),
            minlength=CFG.MAX_ROUTING_LAG_MONTHS + 1,
        )
        by1 = np.bincount(
            lag1,
            weights=volumes * frac1,
            minlength=CFG.MAX_ROUTING_LAG_MONTHS + 1,
        )
        for lag in range(CFG.MAX_ROUTING_LAG_MONTHS + 1):
            target = i + lag
            if target < len(arrivals):
                arrivals[target] += by0[lag] + by1[lag]

        details.append({
            "date": date,
            "AET_mean_mm": _weighted_mean(step["aet"], weights),
            "Recharge_mean_mm": _weighted_mean(step["recharge"], weights),
            "Percolation_mean_mm": _weighted_mean(step["percolation"], weights),
            "Baseflow_fast_mean_mm": _weighted_mean(step["baseflow_fast"], weights),
            "Baseflow_slow_mean_mm": _weighted_mean(step["baseflow_slow"], weights),
            "Water_balance_residual_mm": _weighted_mean(step["water_balance_residual"], weights),
        })

    for i, row in enumerate(details):
        days = int(pd.Period(row["date"], freq="M").days_in_month)
        row["Qsim_m3s"] = arrivals[i] / (days * 86400.0)

    return pd.DataFrame(details).set_index("date")


def _composite_calibration_score(obs: np.ndarray, sim: np.ndarray) -> tuple[float, dict[str, float]]:
    nn = nse(obs, sim)
    kk = kge(obs, sim)
    pp = pbias(obs, sim)
    if not np.isfinite(nn) or not np.isfinite(kk) or not np.isfinite(pp):
        return 9999.0, {"NSE": nn, "KGE": kk, "PBIAS_percent": pp}
    score = (
        CFG.OBJECTIVE_WEIGHT_NSE * (1.0 - nn)
        + CFG.OBJECTIVE_WEIGHT_KGE * (1.0 - kk)
        + CFG.OBJECTIVE_WEIGHT_PBIAS * abs(pp) / 100.0
    )
    return float(score), {"NSE": float(nn), "KGE": float(kk), "PBIAS_percent": float(pp)}


def _evaluate_compact_parameters(
    x: Iterable[float],
    p_monthly: pd.Series,
    pet_monthly: pd.Series,
    qobs: pd.Series,
    compact: dict[str, np.ndarray | float],
) -> tuple[float, dict[str, float]]:
    params = make_param_dict_v8(x)
    sim = simulate_primary_compact_v8(p_monthly, pet_monthly, compact, params)
    joined = pd.concat([qobs.rename("Qobs_m3s"), sim["Qsim_m3s"]], axis=1).dropna()
    joined = joined.loc[CFG.CALIB_START:CFG.CALIB_END]
    if len(joined) < 6:
        return 9999.0, {"NSE": np.nan, "KGE": np.nan, "PBIAS_percent": np.nan}
    return _composite_calibration_score(
        joined["Qobs_m3s"].to_numpy(), joined["Qsim_m3s"].to_numpy()
    )


def calibrate_v8_full_spatial(
    p_monthly: pd.Series,
    pet_monthly: pd.Series,
    qobs: pd.Series,
    full_compact: dict[str, np.ndarray | float],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Zweistufige Kalibrierung:
    1. globale Suche auf räumlich geschichteter, flächengewichteter Stichprobe;
    2. direkte Feinoptimierung mit allen Rasterzellen des Pegelgebiets.
    """
    try:
        from scipy.optimize import differential_evolution
    except Exception:
        warnings.warn("scipy nicht verfügbar; konservative V8-Defaultparameter werden verwendet.")
        defaults = make_param_dict_v8([0.25, 0.03, 0.22, 0.035, 0.70, 0.70, 1.0])
        return defaults, pd.DataFrame(), pd.DataFrame([defaults])

    sampled = make_stratified_calibration_sample(
        full_compact,
        max_cells=int(CFG.CALIBRATION_SAMPLE_CELLS),
        seed=int(CFG.RANDOM_SEED),
    )

    generation_records: list[dict] = []

    def objective_sample(x):
        score, _ = _evaluate_compact_parameters(
            x, p_monthly, pet_monthly, qobs, sampled
        )
        return score

    def callback(xk, convergence):
        score, metrics = _evaluate_compact_parameters(
            xk, p_monthly, pet_monthly, qobs, sampled
        )
        generation_records.append({
            "stage": "global_sample",
            "generation": len(generation_records) + 1,
            "objective": score,
            "convergence": float(convergence),
            **metrics,
            **make_param_dict_v8(xk),
        })
        print(
            f"     Generation {len(generation_records):02d}: "
            f"Obj={score:.4f}, NSE={metrics['NSE']:.3f}, "
            f"KGE={metrics['KGE']:.3f}, PBIAS={metrics['PBIAS_percent']:.1f}%"
        )
        return False

    print("\n[INFO] V8 globale Kalibrierung mit räumlichen Faktoren und Routing")
    result = differential_evolution(
        objective_sample,
        bounds=PARAM_BOUNDS_V8,
        seed=CFG.RANDOM_SEED,
        maxiter=int(CFG.CALIBRATION_GLOBAL_MAXITER),
        popsize=int(CFG.CALIBRATION_GLOBAL_POPSIZE),
        polish=True,
        updating="immediate",
        workers=1,
        tol=0.001,
        callback=callback,
    )

    best_x = np.asarray(result.x, dtype="float64")
    best_score, best_metrics = _evaluate_compact_parameters(
        best_x, p_monthly, pet_monthly, qobs, full_compact
    )
    refinement_records = [{
        "stage": "full_grid_start",
        "pass": 0,
        "parameter": "start",
        "objective": best_score,
        **best_metrics,
        **make_param_dict_v8(best_x),
    }]

    print("\n[INFO] Direkte Feinoptimierung auf allen Zellen des Pegelgebiets")
    for pass_no, fraction in enumerate(CFG.FULL_GRID_REFINEMENT_FRACTIONS, start=1):
        improved_in_pass = False
        for j, name in enumerate(PARAM_NAMES_V8):
            lo, hi = PARAM_BOUNDS_V8[j]
            span = hi - lo
            local_best_x = best_x.copy()
            local_best_score = best_score
            local_best_metrics = best_metrics

            for direction in (-1.0, 1.0):
                trial = best_x.copy()
                trial[j] = np.clip(best_x[j] + direction * fraction * span, lo, hi)
                if np.isclose(trial[j], best_x[j]):
                    continue
                trial_score, trial_metrics = _evaluate_compact_parameters(
                    trial, p_monthly, pet_monthly, qobs, full_compact
                )
                refinement_records.append({
                    "stage": "full_grid_refinement",
                    "pass": pass_no,
                    "parameter": name,
                    "direction": int(direction),
                    "objective": trial_score,
                    **trial_metrics,
                    **make_param_dict_v8(trial),
                })
                if trial_score < local_best_score:
                    local_best_x = trial
                    local_best_score = trial_score
                    local_best_metrics = trial_metrics

            if local_best_score < best_score:
                best_x, best_score, best_metrics = (
                    local_best_x, local_best_score, local_best_metrics
                )
                improved_in_pass = True
                print(
                    f"     Pass {pass_no}, {name}: Obj={best_score:.4f}, "
                    f"NSE={best_metrics['NSE']:.3f}, KGE={best_metrics['KGE']:.3f}, "
                    f"PBIAS={best_metrics['PBIAS_percent']:.1f}%"
                )
        if not improved_in_pass:
            print(f"     Pass {pass_no}: keine weitere Verbesserung.")

    params = make_param_dict_v8(best_x)
    print("[OK] V8-Parameter nach Full-Grid-Refinement:")
    for key, value in params.items():
        print(f"     {key}: {value:.5f}")
    print(
        f"     Kalibrierung: NSE={best_metrics['NSE']:.3f}, "
        f"KGE={best_metrics['KGE']:.3f}, "
        f"PBIAS={best_metrics['PBIAS_percent']:.1f}%"
    )

    trace = pd.concat(
        [pd.DataFrame(generation_records), pd.DataFrame(refinement_records)],
        ignore_index=True,
        sort=False,
    )
    summary = pd.DataFrame([{
        "objective": best_score,
        **best_metrics,
        **params,
        "sample_cells": len(np.asarray(sampled["fc"])),
        "full_grid_cells": len(np.asarray(full_compact["fc"])),
    }])
    return params, trace, summary


def run_monthly_model_v8(
    p_monthly: pd.Series,
    pet_monthly: pd.Series,
    fc: np.ndarray,
    mask: np.ndarray,
    ref_profile: dict,
    params: dict[str, float],
    slope_quickflow_factor: np.ndarray,
    landuse: dict[str, np.ndarray],
    soil_units: dict[str, np.ndarray],
    gauge_routings: dict[str, dict],
    primary_key: str,
    collect_maps: bool = True,
) -> tuple[pd.DataFrame, Optional[dict[str, np.ndarray]]]:
    """Finales V8-Rastermodell mit zwei Grundwasserspeichern und D8-Routing."""
    idx = p_monthly.index.intersection(pet_monthly.index).sort_values()
    fc = np.where(mask, fc, np.nan)
    alpha_grid, beta_grid, slow_fraction_grid = _derive_spatial_parameter_grids_v8(
        params, mask, slope_quickflow_factor, landuse, soil_units
    )

    soil = np.where(mask, fc * params["init_soil_frac"], np.nan)
    gw_fast = np.where(mask, 0.0, np.nan)
    gw_slow = np.where(mask, 0.0, np.nan)
    cell_area_m2 = abs(ref_profile["transform"].a * ref_profile["transform"].e)

    lag_infos = {
        key: prepare_routing_lags(routing, params["routing_velocity_m_s"])
        for key, routing in gauge_routings.items()
    }
    arrivals = {
        key: np.zeros(len(idx) + CFG.MAX_ROUTING_LAG_MONTHS + 1, dtype="float64")
        for key in gauge_routings
    }

    maps = None
    if collect_maps:
        maps = {
            "runoff_sum_mm": np.zeros_like(fc),
            "quick_runoff_sum_mm": np.zeros_like(fc),
            "baseflow_sum_mm": np.zeros_like(fc),
            "baseflow_fast_sum_mm": np.zeros_like(fc),
            "baseflow_slow_sum_mm": np.zeros_like(fc),
            "percolation_sum_mm": np.zeros_like(fc),
            "recharge_sum_mm": np.zeros_like(fc),
            "aet_sum_mm": np.zeros_like(fc),
            "soil_final_mm": np.zeros_like(fc),
            "gw_fast_final_mm": np.zeros_like(fc),
            "gw_slow_final_mm": np.zeros_like(fc),
            "alpha_fast_grid": alpha_grid.copy(),
            "beta_perc_grid": beta_grid.copy(),
            "slow_recharge_fraction_grid": slow_fraction_grid.copy(),
            "water_balance_residual_sum_mm": np.zeros_like(fc),
        }

    warm_dates = [d for d in idx if d <= pd.Timestamp(CFG.WARMUP_END)]
    for _ in range(max(0, int(CFG.SPINUP_CYCLES) - 1)):
        for date in warm_dates:
            step = _water_balance_step_v8(
                float(p_monthly.loc[date]), float(pet_monthly.loc[date]),
                fc, soil, gw_fast, gw_slow, alpha_grid, beta_grid,
                slow_fraction_grid, landuse["pet_factor"], params,
            )
            soil, gw_fast, gw_slow = step["soil"], step["gw_fast"], step["gw_slow"]

    rows_out = []
    primary_mask = gauge_routings[primary_key]["upstream_mask"]

    for month_position, date in enumerate(idx):
        p = float(p_monthly.loc[date])
        pet = float(pet_monthly.loc[date])
        step = _water_balance_step_v8(
            p, pet, fc, soil, gw_fast, gw_slow, alpha_grid, beta_grid,
            slow_fraction_grid, landuse["pet_factor"], params,
        )
        soil, gw_fast, gw_slow = step["soil"], step["gw_fast"], step["gw_slow"]

        for key, routing in gauge_routings.items():
            route_local_runoff_to_gauge(
                local_runoff_mm=step["local_runoff"],
                cell_area_m2=cell_area_m2,
                month_position=month_position,
                arrivals_m3=arrivals[key],
                gauge_routing=routing,
                lag_info=lag_infos[key],
            )

        rows_out.append({
            "date": date,
            "P_mm_month": p,
            "PET_mm_month": pet,
            "AET_mean_model_mm": float(np.nanmean(step["aet"][mask])),
            "AET_mean_primary_mm": float(np.nanmean(step["aet"][primary_mask])),
            "Percolation_mean_primary_mm": float(np.nanmean(step["percolation"][primary_mask])),
            "Recharge_mean_model_mm": float(np.nanmean(step["recharge"][mask])),
            "Recharge_mean_primary_mm": float(np.nanmean(step["recharge"][primary_mask])),
            "Runoff_generated_mean_model_mm": float(np.nanmean(step["local_runoff"][mask])),
            "Runoff_generated_mean_primary_mm": float(np.nanmean(step["local_runoff"][primary_mask])),
            "Quick_runoff_mean_primary_mm": float(np.nanmean(step["quick_runoff"][primary_mask])),
            "Baseflow_fast_mean_primary_mm": float(np.nanmean(step["baseflow_fast"][primary_mask])),
            "Baseflow_slow_mean_primary_mm": float(np.nanmean(step["baseflow_slow"][primary_mask])),
            "Baseflow_mean_primary_mm": float(np.nanmean(step["baseflow"][primary_mask])),
            "Soil_mean_primary_mm": float(np.nanmean(soil[primary_mask])),
            "Groundwater_fast_mean_primary_mm": float(np.nanmean(gw_fast[primary_mask])),
            "Groundwater_slow_mean_primary_mm": float(np.nanmean(gw_slow[primary_mask])),
            "Water_balance_residual_primary_mm": float(np.nanmean(step["water_balance_residual"][primary_mask])),
        })

        if maps is not None:
            maps["runoff_sum_mm"] += np.nan_to_num(step["local_runoff"])
            maps["quick_runoff_sum_mm"] += np.nan_to_num(step["quick_runoff"])
            maps["baseflow_sum_mm"] += np.nan_to_num(step["baseflow"])
            maps["baseflow_fast_sum_mm"] += np.nan_to_num(step["baseflow_fast"])
            maps["baseflow_slow_sum_mm"] += np.nan_to_num(step["baseflow_slow"])
            maps["percolation_sum_mm"] += np.nan_to_num(step["percolation"])
            maps["recharge_sum_mm"] += np.nan_to_num(step["recharge"])
            maps["aet_sum_mm"] += np.nan_to_num(step["aet"])
            maps["water_balance_residual_sum_mm"] += np.nan_to_num(step["water_balance_residual"])

    for i, row in enumerate(rows_out):
        days = int(pd.Period(row["date"], freq="M").days_in_month)
        for key, arr in arrivals.items():
            row[f"Qsim_{key}_m3s"] = float(arr[i]) / (days * 86400.0)
        row["Qsim_m3s"] = row[f"Qsim_{primary_key}_m3s"]

    if maps is not None:
        maps["soil_final_mm"] = soil
        maps["gw_fast_final_mm"] = gw_fast
        maps["gw_slow_final_mm"] = gw_slow
        for key in maps:
            maps[key] = np.where(mask, maps[key], np.nan)

    df = pd.DataFrame(rows_out).set_index("date")
    for key, arr in arrivals.items():
        df.attrs[f"routing_tail_m3_{key}"] = float(np.sum(arr[len(idx):]))
    return df, maps


def _find_data_folder_fuzzy(configured_name: str, keywords: tuple[str, ...]) -> Optional[Path]:
    direct = data_dir() / configured_name
    if direct.exists():
        return direct
    candidates = [p for p in data_dir().iterdir() if p.is_dir()]
    for path in candidates:
        normalized = _normalize_text(path.name)
        if all(_normalize_text(k) in normalized for k in keywords):
            return path
    return None


def _reproject_array_to_ref(
    src_arr: np.ndarray,
    src_profile: dict,
    ref_profile: dict,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    dst = np.full(
        (ref_profile["height"], ref_profile["width"]),
        np.nan,
        dtype="float64",
    )
    reproject(
        source=src_arr.astype("float64"),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=np.nan,
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def _date_granularity_from_text(text: str) -> tuple[Optional[int], str]:
    m = re.search(r"(20\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])", text)
    if m:
        return int(m.group(1)), "daily"
    m = re.search(r"(20\d{2})(0[1-9]|1[0-2])", text)
    if m:
        return int(m.group(1)), "monthly"
    m = re.search(r"(20\d{2})", text)
    if m:
        return int(m.group(1)), "annual"
    return None, "unknown"


def _date_granularity(path: Path) -> tuple[Optional[int], str]:
    text = " ".join([path.stem] + [p.name for p in path.parents[:4]])
    return _date_granularity_from_text(text)


def _read_raster_from_tar_member(
    archive_path: Path,
    member: tarfile.TarInfo,
) -> tuple[np.ndarray, dict]:
    """Liest ASC/TIF direkt aus einem TAR-Archiv, ohne es dauerhaft zu entpacken."""
    with tarfile.open(archive_path, mode="r:*") as archive:
        fh = archive.extractfile(member)
        if fh is None:
            raise ValueError(f"Archivmitglied nicht lesbar: {member.name}")
        data = fh.read()
    with MemoryFile(data, filename=Path(member.name).name) as memfile:
        with memfile.open() as src:
            arr = src.read(1).astype("float64")
            profile = src.profile.copy()
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
    return arr, profile


def _safe_nanmean_stack(layers: list[np.ndarray]) -> np.ndarray:
    """Mittel ohne RuntimeWarning in Zellen, die in allen Layern NoData sind."""
    stack = np.stack(layers, axis=0)
    count = np.sum(np.isfinite(stack), axis=0)
    total = np.nansum(stack, axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype="float64")
    valid = count > 0
    out[valid] = total[valid] / count[valid]
    return out


def load_reference_annual_raster(
    configured_folder: str,
    keywords: tuple[str, ...],
    ref_profile: dict,
    mask: np.ndarray,
    unit_factor: float,
    product_name: str,
    src_crs_fallback: Optional[str] = None,
) -> tuple[Optional[np.ndarray], dict]:
    """
    Lädt TIF/ASC-Referenzdaten und erzeugt einen vergleichbaren Jahresmittelwert.

    Unterstützt zusätzlich TAR/TAR.GZ-Archive mit täglichen ASC/TIF-Dateien,
    wie die DWD-Dateien der realen Evapotranspiration.

    - Tages-/Monatsdateien: Jahressummen, anschließend Mittel über Jahre.
    - Jahresdateien oder undatierte Einzelraster: Mittel der Raster.
    """
    folder = _find_data_folder_fuzzy(configured_folder, keywords)
    metadata = {
        "product": product_name,
        "configured_folder": configured_folder,
        "resolved_folder": str(folder) if folder else "",
        "status": "not_found",
        "n_files": 0,
        "n_archives": 0,
        "aggregation": "",
        "unit_factor": unit_factor,
        "src_crs_fallback": src_crs_fallback or "",
    }
    if folder is None:
        warnings.warn(f"Referenzprodukt {product_name}: Ordner nicht gefunden.")
        return None, metadata

    regular_files = find_files(folder, [".tif", ".tiff", ".asc"])
    archive_files = sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.name.lower().endswith((".tar", ".tar.gz", ".tgz"))
    ])
    metadata["n_archives"] = len(archive_files)

    archive_members: list[tuple[Path, tarfile.TarInfo]] = []
    for archive_path in archive_files:
        try:
            with tarfile.open(archive_path, mode="r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    if member.name.lower().endswith((".asc", ".tif", ".tiff")):
                        archive_members.append((archive_path, member))
        except Exception as exc:
            warnings.warn(f"{product_name}: Archiv {archive_path.name} nicht lesbar ({exc}).")

    metadata["n_files"] = len(regular_files) + len(archive_members)
    if metadata["n_files"] == 0:
        metadata["status"] = "no_supported_rasters"
        warnings.warn(
            f"Referenzprodukt {product_name}: keine TIF/ASC-Dateien oder TAR-Archive in {folder}."
        )
        return None, metadata

    print(
        f"[OK] Referenz {product_name}: {len(regular_files)} freie Raster, "
        f"{len(archive_files)} Archive mit {len(archive_members)} Rastermitgliedern."
    )

    annual_sums: dict[int, np.ndarray] = {}
    annual_valid_counts: dict[int, np.ndarray] = {}
    annual_file_counts: dict[int, int] = {}
    annual_layers: list[np.ndarray] = []
    unknown_layers: list[np.ndarray] = []
    granularities: set[str] = set()
    processed = 0

    def process_layer(
        arr: np.ndarray,
        src_profile: dict,
        year: Optional[int],
        granularity: str,
        source_label: str,
    ) -> None:
        nonlocal processed
        try:
            src_crs = src_profile.get("crs")
            if src_crs is None:
                src_crs = src_crs_fallback or ref_profile["crs"]
                src_profile = src_profile.copy()
                src_profile["crs"] = src_crs
            arr = arr * float(unit_factor)
            projected = _reproject_array_to_ref(arr, src_profile, ref_profile)
        except Exception as exc:
            warnings.warn(f"{product_name}: {source_label} übersprungen ({exc}).")
            return

        granularities.add(granularity)
        if year is not None and granularity in ("daily", "monthly"):
            if year not in annual_sums:
                annual_sums[year] = np.zeros_like(projected)
                annual_valid_counts[year] = np.zeros_like(projected, dtype="uint16")
                annual_file_counts[year] = 0
            valid_projected = np.isfinite(projected)
            annual_sums[year][valid_projected] += projected[valid_projected]
            annual_valid_counts[year][valid_projected] += 1
            annual_file_counts[year] += 1
        elif granularity == "annual":
            annual_layers.append(projected)
        else:
            unknown_layers.append(projected)
        processed += 1
        if processed % 100 == 0:
            print(f"     {product_name}: {processed}/{metadata['n_files']} Raster verarbeitet")

    for path in regular_files:
        year, granularity = _date_granularity(path)
        try:
            arr, src_profile = read_raster(path)
        except Exception as exc:
            warnings.warn(f"{product_name}: {path.name} übersprungen ({exc}).")
            continue
        process_layer(arr, src_profile, year, granularity, path.name)

    for archive_path, member in archive_members:
        date_text = f"{archive_path} {member.name}"
        year, granularity = _date_granularity_from_text(date_text)
        try:
            arr, src_profile = _read_raster_from_tar_member(archive_path, member)
        except Exception as exc:
            warnings.warn(
                f"{product_name}: {archive_path.name}/{member.name} übersprungen ({exc})."
            )
            continue
        process_layer(
            arr, src_profile, year, granularity,
            f"{archive_path.name}/{member.name}",
        )

    layers_for_mean: list[np.ndarray] = []
    aggregation_parts = []
    if annual_sums:
        for year in sorted(annual_sums):
            annual_layer = np.full_like(annual_sums[year], np.nan)
            valid = annual_valid_counts[year] > 0
            annual_layer[valid] = annual_sums[year][valid]
            layers_for_mean.append(annual_layer)
        aggregation_parts.append(
            f"Jahressummen aus Tages/Monatsdaten ({len(annual_sums)} Jahre; "
            f"Dateien/Jahr: {annual_file_counts})"
        )
    if annual_layers:
        layers_for_mean.extend(annual_layers)
        aggregation_parts.append(f"Mittel aus {len(annual_layers)} Jahresrastern")
    if unknown_layers:
        layers_for_mean.extend(unknown_layers)
        aggregation_parts.append(f"Mittel aus {len(unknown_layers)} undatierten Rastern")

    if not layers_for_mean:
        metadata["status"] = "read_failed"
        return None, metadata

    reference = _safe_nanmean_stack(layers_for_mean)
    reference = np.where(mask, reference, np.nan)
    valid_reference = mask & np.isfinite(reference)
    metadata.update({
        "status": "ok",
        "aggregation": "; ".join(aggregation_parts),
        "granularities": ",".join(sorted(granularities)),
        "valid_cells": int(np.sum(valid_reference)),
        "mean_mm_per_year": float(np.mean(reference[valid_reference])) if valid_reference.any() else np.nan,
    })
    print(
        f"[OK] Referenz {product_name}: {metadata['aggregation']}; "
        f"Mittel={metadata['mean_mm_per_year']:.1f} mm/a"
    )
    return reference, metadata


def _spatial_metrics(
    model: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    valid = mask & np.isfinite(model) & np.isfinite(reference)
    if valid.sum() < 3:
        return {
            "n_cells": int(valid.sum()), "model_mean": np.nan,
            "reference_mean": np.nan, "bias_mm_a": np.nan,
            "PBIAS_percent": np.nan, "MAE_mm_a": np.nan,
            "RMSE_mm_a": np.nan, "correlation": np.nan,
        }
    m = model[valid]
    r = reference[valid]
    corr = np.corrcoef(r, m)[0, 1] if np.std(r) > 0 and np.std(m) > 0 else np.nan
    return {
        "n_cells": int(valid.sum()),
        "model_mean": float(np.mean(m)),
        "reference_mean": float(np.mean(r)),
        "bias_mm_a": float(np.mean(m - r)),
        "PBIAS_percent": float(100.0 * np.sum(m - r) / np.sum(r)) if np.sum(r) != 0 else np.nan,
        "MAE_mm_a": float(np.mean(np.abs(m - r))),
        "RMSE_mm_a": float(np.sqrt(np.mean((m - r) ** 2))),
        "correlation": float(corr) if np.isfinite(corr) else np.nan,
    }


def plot_spatial_scatter(
    model: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    title: str,
    path: Path,
) -> None:
    valid = mask & np.isfinite(model) & np.isfinite(reference)
    idx = np.flatnonzero(valid)
    if len(idx) < 3:
        return
    if len(idx) > 30000:
        rng = np.random.default_rng(CFG.RANDOM_SEED)
        idx = rng.choice(idx, size=30000, replace=False)
    m = model.ravel()[idx]
    r = reference.ravel()[idx]
    lo = float(min(np.nanmin(m), np.nanmin(r)))
    hi = float(max(np.nanmax(m), np.nanmax(r)))
    plt.figure(figsize=(6, 6))
    plt.scatter(r, m, s=4, alpha=0.25)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("Referenz [mm/a]")
    plt.ylabel("Modell [mm/a]")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def validate_spatial_products_v8(
    maps: dict[str, np.ndarray],
    ref_profile: dict,
    model_mask: np.ndarray,
    primary_mask: np.ndarray,
    n_years: int,
    out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    model_annual = {
        "recharge": maps["recharge_sum_mm"] / n_years,
        "percolation": maps["percolation_sum_mm"] / n_years,
        "aet": maps["aet_sum_mm"] / n_years,
    }

    ref_specs = (
        (
            "groundwater_recharge", CFG.GWN_REFERENCE_DIRNAME,
            ("grundwasser", "neubild"), CFG.GWN_REFERENCE_UNIT_FACTOR, None,
        ),
        (
            "seepage", CFG.SEEPAGE_REFERENCE_DIRNAME,
            ("sickerwasser",), CFG.SEEPAGE_REFERENCE_UNIT_FACTOR, None,
        ),
        (
            "actual_et", CFG.AET_REFERENCE_DIRNAME,
            ("evapotransp",), CFG.AET_REFERENCE_UNIT_FACTOR,
            CFG.AET_REFERENCE_CRS_FALLBACK,
        ),
    )

    references: dict[str, np.ndarray] = {}
    metadata_rows = []
    for key, folder, keywords, factor, crs_fallback in ref_specs:
        arr, meta = load_reference_annual_raster(
            folder, keywords, ref_profile, model_mask, factor, key, crs_fallback
        )
        metadata_rows.append(meta)
        if arr is not None:
            references[key] = arr
            write_geotiff(out / "maps" / f"reference_{key}_annual_mm.tif", arr, ref_profile)

    comparisons = []
    if "groundwater_recharge" in references:
        comparisons.append(("recharge_vs_groundwater_recharge", model_annual["recharge"], references["groundwater_recharge"]))
    if "seepage" in references:
        comparisons.append(("percolation_vs_seepage", model_annual["percolation"], references["seepage"]))
        comparisons.append(("recharge_vs_seepage_secondary", model_annual["recharge"], references["seepage"]))
    if "actual_et" in references:
        comparisons.append(("aet_vs_actual_et", model_annual["aet"], references["actual_et"]))

    metric_rows = []
    extra_maps: dict[str, np.ndarray] = {}
    domains = (("project_model_domain", model_mask), ("heidelbach_upstream", primary_mask))
    for name, model_arr, ref_arr in comparisons:
        diff = np.where(model_mask, model_arr - ref_arr, np.nan)
        extra_maps[f"difference_{name}_mm_a"] = diff
        write_geotiff(out / "maps" / f"difference_{name}_mm_a.tif", diff, ref_profile)
        plot_spatial_scatter(
            model_arr, ref_arr, primary_mask,
            f"Räumlicher Vergleich: {name}",
            out / "plots" / f"scatter_{name}.png",
        )
        for domain_name, domain_mask in domains:
            metric_rows.append({
                "comparison": name,
                "domain": domain_name,
                **_spatial_metrics(model_arr, ref_arr, domain_mask),
            })

    return pd.DataFrame(metric_rows), pd.DataFrame(metadata_rows), extra_maps


def make_annual_water_balance(
    results: pd.DataFrame,
    primary_area_km2: float,
) -> pd.DataFrame:
    df = results.copy()
    days = np.asarray([pd.Period(d, freq="M").days_in_month for d in df.index])
    area_m2 = primary_area_km2 * 1e6
    df["Qobs_depth_mm"] = df["Qobs_m3s"] * days * 86400.0 / area_m2 * 1000.0
    df["Qsim_depth_mm"] = df["Qsim_m3s"] * days * 86400.0 / area_m2 * 1000.0
    df["year"] = df.index.year

    sum_cols = [
        "P_mm_month", "PET_mm_month", "AET_mean_primary_mm",
        "Percolation_mean_primary_mm", "Recharge_mean_primary_mm",
        "Runoff_generated_mean_primary_mm", "Qobs_depth_mm", "Qsim_depth_mm",
    ]
    annual = df.groupby("year")[sum_cols].sum(min_count=1).reset_index()
    annual["P_minus_AET_minus_Qsim_mm"] = (
        annual["P_mm_month"] - annual["AET_mean_primary_mm"] - annual["Qsim_depth_mm"]
    )
    return annual


def plot_annual_water_balance(annual: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(annual))
    width = 0.18
    plt.figure(figsize=(10, 5))
    plt.bar(x - 1.5 * width, annual["P_mm_month"], width, label="Niederschlag")
    plt.bar(x - 0.5 * width, annual["AET_mean_primary_mm"], width, label="Reale Evapotranspiration (modelliert)")
    plt.bar(x + 0.5 * width, annual["Qobs_depth_mm"], width, label="Qobs")
    plt.bar(x + 1.5 * width, annual["Qsim_depth_mm"], width, label="Qsim")
    plt.xticks(x, annual["year"].astype(str))
    plt.ylabel("Jahressumme [mm/a]")
    plt.xlabel("Jahr")
    plt.title("Jährliche Wasserbilanz am Pegel Heidelbach")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_q_timeseries_v8(results: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(12, 5.5))
    plt.plot(results.index, results["Qobs_m3s"], label="Qobs")
    plt.plot(results.index, results["Qsim_m3s"], label="Qsim")
    plt.axvspan(pd.Timestamp(CFG.CALIB_START), pd.Timestamp(CFG.CALIB_END), alpha=0.08, label="Kalibrierung")
    plt.axvspan(pd.Timestamp(CFG.VALID_START), pd.Timestamp(CFG.VALID_END), alpha=0.08, label="Validierung")
    plt.ylabel("Q [m³/s]")
    plt.xlabel("Monat")
    plt.title("Beobachteter und simulierter Monatsabfluss – HydroMod v8.1")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_hausarbeit_bundle(
    out: Path,
    performance: pd.DataFrame,
    params: dict[str, float],
    spatial_metrics: pd.DataFrame,
    primary_area: float,
) -> None:
    haus = out / "hausarbeit"
    selections = {
        out / "plots" / "qobs_qsim_monthly_v8_1.png": "Abb_01_Qobs_Qsim.png",
        out / "plots" / "annual_water_balance.png": "Abb_02_Jahreswasserbilanz.png",
        out / "plots" / "map_flow_accumulation_km2.png": "Abb_03_Fliessakkumulation.png",
        out / "plots" / "map_recharge_annual_mean.png": "Abb_04_Recharge_Modell_mm_a.png",
        out / "plots" / "map_aet_annual_mean.png": "Abb_05_AET_Modell_mm_a.png",
        out / "plots" / "scatter_recharge_vs_groundwater_recharge.png": "Abb_06_Recharge_Referenzvergleich.png",
        out / "plots" / "scatter_aet_vs_actual_et.png": "Abb_07_AET_Referenzvergleich.png",
        out / "tables" / "performance_metrics.csv": "Tab_01_Performance.csv",
        out / "tables" / "calibrated_parameters.csv": "Tab_02_Parameter.csv",
        out / "tables" / "annual_water_balance.csv": "Tab_03_Jahreswasserbilanz.csv",
        out / "tables" / "spatial_validation_metrics.csv": "Tab_04_Raeumliche_Validierung.csv",
        out / "tables" / "gauge_routing_diagnostics.csv": "Tab_05_Pegeldiagnostik.csv",
        out / "tables" / "soil_class_parameterization.csv": "Tab_06_Bodenparameterisierung.csv",
    }
    for source, target_name in selections.items():
        if source.exists():
            shutil.copy2(source, haus / target_name)

    cal = performance.loc[performance["period"] == "calibration"].iloc[0]
    val = performance.loc[performance["period"] == "validation"].iloc[0]
    readme = f"""HydroMod v8.1 – Auswahl zentraler Ergebnisse für die Hausarbeit

Primärer Pegel: Heidelbach (42880550)
D8-Pegeleinzugsgebiet: {primary_area:.3f} km²
Kalibrierung: {CFG.CALIB_START} bis {CFG.CALIB_END}
Validierung: {CFG.VALID_START} bis {CFG.VALID_END}

Kalibrierung:
  NSE = {cal['NSE']:.3f}
  KGE = {cal['KGE']:.3f}
  PBIAS = {cal['PBIAS_percent']:.2f} %

Validierung:
  NSE = {val['NSE']:.3f}
  KGE = {val['KGE']:.3f}
  PBIAS = {val['PBIAS_percent']:.2f} %

Wichtige methodische Hinweise:
- Qobs ist Durchfluss in m³/s, nicht nur Pegelstand/Wasserstand.
- 2021 dient als Warm-up; 2022–2023 werden kalibriert; 2024–2025 zeitlich unabhängig validiert.
- Die globale Parametersuche nutzt eine geschichtete Rasterstichprobe. Die Feinoptimierung und finale Simulation nutzen alle Zellen des Heidelbach-Einzugsgebiets.
- Bodeneinheiten steuern räumliche Faktoren für Schnellabfluss, Perkolation und die Aufteilung auf schnelle/langsame Grundwasserspeicher. Die nFK bleibt der maximale Bodenspeicher.
- GWN, Sickerwasser und reale ET dienen als zusätzliche räumliche Plausibilisierung. Abweichende Referenzzeiträume und Produktdefinitionen müssen in der Diskussion genannt werden.
- Niederschlag und PET liegen im Modell weiterhin als monatliche Gebietsmittel vor und werden über räumliche Faktoren verteilt. Dies bleibt eine zentrale Einschränkung.

Kalibrierte Parameter:
{json.dumps(params, indent=2, ensure_ascii=False)}

Räumliche Vergleiche verfügbar: {len(spatial_metrics)} Tabellenzeilen.
"""
    (haus / "README_Hausarbeit.txt").write_text(readme, encoding="utf-8")


def main_v8() -> None:
    OUT = out_dir()
    print("============================================================")
    print("HydroMod v8.1: Full-Grid-Kalibrierung, Boden, D8 und Validierung")
    print("============================================================")
    print(f"Projektordner: {CFG.BASE_DIR}")
    print(f"Datenordner:   {data_dir()}")
    print(f"Output:        {OUT}")

    # 1. Projektgebiet und DGM
    dgm_path = find_first(data_dir() / CFG.DGM_DIRNAME, [".tif", ".tiff"])
    dgm_full, profile_full = read_raster(dgm_path)
    catchment = load_catchment(profile_full["crs"])
    mask_full = create_mask(catchment, profile_full)
    dgm, mask, profile, crop_offset = crop_raster_to_mask(
        dgm_full, mask_full, profile_full, padding=CFG.CROP_PADDING_CELLS
    )
    del dgm_full, mask_full
    print(f"[OK] Modellraster: {profile['width']} x {profile['height']}; Offset={crop_offset}")

    # 2. Räumliche statische Daten
    fc = load_field_capacity_raster(profile, mask)
    river_mask, river_gdf = load_river_mask(profile, mask)
    landuse = load_landuse_rasters(profile, mask, catchment)
    soil_units, soil_table = load_soil_unit_factors(profile, mask)
    soil_table.to_csv(
        OUT / "tables" / "soil_class_parameterization.csv",
        index=False, encoding="utf-8",
    )

    dgm = fill_missing_dem_nearest(dgm, mask)
    dgm_burned, dgm_conditioned = condition_dem_priority_flood(
        dgm, mask, river_mask, profile
    )
    slope_fraction = calculate_slope_fraction(dgm_conditioned, mask, profile)
    slope_percent = slope_fraction * 100.0
    slope_factor = make_slope_quickflow_factor(
        slope_fraction, mask, CFG.SLOPE_QUICKFLOW_WEIGHT
    )
    network = build_d8_flow_network(dgm_conditioned, mask, profile, river_mask)

    # 3. Pegel und Pegeleinzugsgebiete
    gauges = load_gauge_points(profile["crs"])
    gauge_routings: dict[str, dict] = {}
    for _, gauge_row in gauges.iterrows():
        routing = build_gauge_routing(gauge_row, network, mask, profile, river_mask)
        gauge_routings[routing["key"]] = routing
    primary = next(r for r in gauge_routings.values() if r["is_primary"])
    primary_key = primary["key"]
    primary_mask = primary["upstream_mask"]
    primary_area = primary["area_km2"]
    area_diff = 100.0 * (
        primary_area - CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2
    ) / CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2
    print(
        f"[INFO] Heidelbach-Fläche: Modell={primary_area:.3f} km², "
        f"offiziell={CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2:.3f} km², "
        f"Abweichung={area_diff:+.2f}%"
    )
    if abs(area_diff) > 10.0:
        warnings.warn("Heidelbach-Pegeleinzugsgebiet weicht um mehr als 10 % ab.")

    primary_catchment = mask_to_geodataframe(
        primary_mask, profile, "Heidelbach_D8_upstream"
    )

    # 4. Zeitreihenforcing und beobachteter Durchfluss
    forcing_geometry = primary_catchment if CFG.FORCING_USE_PRIMARY_GAUGE_CATCHMENT else catchment
    forcing_mask = primary_mask if CFG.FORCING_USE_PRIMARY_GAUGE_CATCHMENT else mask
    p_monthly = load_precip_monthly(forcing_geometry, profile["crs"])
    pet_monthly = load_pet_monthly(profile, forcing_mask)
    qobs_map: dict[str, str] = {str(k): str(v) for k, v in CFG.GAUGE_QOBS_FOLDERS}
    primary_qobs_dir = qobs_map.get(str(primary["station_id"]), CFG.QOBS_DIRNAME)
    qobs_primary = load_qobs_monthly(primary_qobs_dir)

    forcing = pd.concat([
        p_monthly, pet_monthly, qobs_primary.rename("Qobs_m3s")
    ], axis=1).loc[CFG.START:CFG.END]
    forcing.to_csv(OUT / "tables" / "monthly_input_timeseries.csv", encoding="utf-8")
    p = forcing["P_mm_month"].dropna()
    pet = forcing["PET_mm_month"].dropna()
    qobs = forcing["Qobs_m3s"]
    common = p.index.intersection(pet.index)
    p, pet = p.loc[common], pet.loc[common]

    # 5. V8-Kalibrierung – gleiche Prozessstruktur wie im finalen Modell
    full_compact = _make_compact_primary_data(
        fc, slope_factor, landuse, soil_units, primary, profile
    )
    if CFG.USE_CALIBRATION:
        params, calibration_trace, calibration_summary = calibrate_v8_full_spatial(
            p, pet, qobs, full_compact
        )
    else:
        params = make_param_dict_v8([0.25, 0.03, 0.22, 0.035, 0.70, 0.70, 1.0])
        calibration_trace = pd.DataFrame()
        calibration_summary = pd.DataFrame([params])

    calibration_trace.to_csv(
        OUT / "tables" / "calibration_trace.csv", index=False, encoding="utf-8"
    )
    calibration_summary.to_csv(
        OUT / "tables" / "calibration_summary.csv", index=False, encoding="utf-8"
    )

    # 6. Finale räumliche Simulation
    sim, maps = run_monthly_model_v8(
        p, pet, fc, mask, profile, params, slope_factor, landuse, soil_units,
        gauge_routings, primary_key, collect_maps=True,
    )
    if maps is None:
        raise RuntimeError("V8 hat keine Kartenoutputs erzeugt.")
    results = pd.concat([
        forcing,
        sim.drop(columns=["P_mm_month", "PET_mm_month"], errors="ignore"),
    ], axis=1).loc[CFG.START:CFG.END]

    # Weitere Pegel bleiben optional; ohne Qobs sind sie Routing-Kontrollpunkte.
    for routing in gauge_routings.values():
        if routing["is_primary"]:
            continue
        folder = qobs_map.get(str(routing["station_id"]))
        if folder:
            try:
                results[f"Qobs_{routing['key']}_m3s"] = load_qobs_monthly(folder)
            except Exception as exc:
                warnings.warn(f"Qobs für {routing['name']} konnte nicht geladen werden: {exc}")

    perf = summarize_performance(results)
    perf.to_csv(OUT / "tables" / "performance_metrics.csv", index=False, encoding="utf-8")
    results.to_csv(OUT / "tables" / "monthly_model_results.csv", encoding="utf-8")

    parameter_out = {
        **params,
        "et_stress_exponent": CFG.ET_STRESS_EXPONENT,
        "stream_burn_depth_m": CFG.STREAM_BURN_DEPTH_M,
        "channel_speed_multiplier": CFG.CHANNEL_SPEED_MULTIPLIER,
        "primary_gauge_area_km2": primary_area,
        "primary_gauge_official_area_km2": CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2,
        "calibration_objective_nse_weight": CFG.OBJECTIVE_WEIGHT_NSE,
        "calibration_objective_kge_weight": CFG.OBJECTIVE_WEIGHT_KGE,
        "calibration_objective_pbias_weight": CFG.OBJECTIVE_WEIGHT_PBIAS,
    }
    pd.DataFrame([parameter_out]).to_csv(
        OUT / "tables" / "calibrated_parameters.csv", index=False, encoding="utf-8"
    )

    # 7. Pegeldiagnostik und Jahreswasserbilanz
    gauge_records = []
    for routing in gauge_routings.values():
        gauge_records.append({
            "gauge_key": routing["key"], "name": routing["name"],
            "station_id": routing["station_id"], "is_primary": routing["is_primary"],
            "original_x": routing.get("original_x", routing.get("official_point_x", routing.get("configured_point_x", np.nan))), "original_y": routing.get("original_y", routing.get("official_point_y", routing.get("configured_point_y", np.nan))),
            "snapped_x": routing["x"], "snapped_y": routing["y"],
            "snap_distance_m": routing["snap_distance_m"],
            "on_river_mask": routing["on_river_mask"],
            "upstream_area_km2": routing["area_km2"],
        })
    pd.DataFrame(gauge_records).to_csv(
        OUT / "tables" / "gauge_routing_diagnostics.csv", index=False, encoding="utf-8"
    )

    annual = make_annual_water_balance(results, primary_area)
    annual.to_csv(OUT / "tables" / "annual_water_balance.csv", index=False, encoding="utf-8")

    # 8. Rasteroutputs
    n_years = len(set(results.index.year))
    outputs = {
        "field_capacity_mm.tif": fc,
        "soil_unit_class_code.tif": soil_units["class_code"],
        "soil_quickflow_factor.tif": soil_units["quickflow_factor"],
        "soil_percolation_factor.tif": soil_units["percolation_factor"],
        "soil_slow_recharge_factor.tif": soil_units["slow_recharge_factor"],
        "dgm_cropped_m.tif": np.where(mask, dgm, np.nan),
        "dgm_stream_burned_m.tif": dgm_burned,
        "conditioned_dgm_m.tif": dgm_conditioned,
        "slope_percent.tif": slope_percent,
        "slope_quickflow_factor.tif": slope_factor,
        "river_mask.tif": np.where(mask, river_mask.astype(float), np.nan),
        "sealed_percent_lbm.tif": landuse["sealed_percent"],
        "vegetation_percent_lbm.tif": landuse["vegetation_percent"],
        "flow_accumulation_cells.tif": network["accumulation_raster"],
        "runoff_generated_sum_2021_2025_mm.tif": maps["runoff_sum_mm"],
        "quick_runoff_sum_2021_2025_mm.tif": maps["quick_runoff_sum_mm"],
        "baseflow_sum_2021_2025_mm.tif": maps["baseflow_sum_mm"],
        "baseflow_fast_sum_2021_2025_mm.tif": maps["baseflow_fast_sum_mm"],
        "baseflow_slow_sum_2021_2025_mm.tif": maps["baseflow_slow_sum_mm"],
        "percolation_sum_2021_2025_mm.tif": maps["percolation_sum_mm"],
        "recharge_sum_2021_2025_mm.tif": maps["recharge_sum_mm"],
        "aet_sum_2021_2025_mm.tif": maps["aet_sum_mm"],
        "runoff_annual_mean_mm_a.tif": maps["runoff_sum_mm"] / n_years,
        "percolation_annual_mean_mm_a.tif": maps["percolation_sum_mm"] / n_years,
        "recharge_annual_mean_mm_a.tif": maps["recharge_sum_mm"] / n_years,
        "aet_annual_mean_mm_a.tif": maps["aet_sum_mm"] / n_years,
        "soil_final_mm.tif": maps["soil_final_mm"],
        "groundwater_fast_final_mm.tif": maps["gw_fast_final_mm"],
        "groundwater_slow_final_mm.tif": maps["gw_slow_final_mm"],
        "alpha_fast_spatial.tif": maps["alpha_fast_grid"],
        "beta_perc_spatial.tif": maps["beta_perc_grid"],
        "slow_recharge_fraction_spatial.tif": maps["slow_recharge_fraction_grid"],
        "water_balance_residual_sum_mm.tif": maps["water_balance_residual_sum_mm"],
        "primary_gauge_upstream_mask.tif": np.where(mask, primary_mask.astype(float), np.nan),
    }
    cell_area = abs(profile["transform"].a * profile["transform"].e)
    outputs["flow_accumulation_area_km2.tif"] = network["accumulation_raster"] * cell_area / 1e6
    primary_lags = prepare_routing_lags(primary, params["routing_velocity_m_s"])
    outputs["primary_routing_travel_time_days.tif"] = compact_to_raster(
        primary_lags["travel_days"], primary["rows"], primary["cols"], mask.shape
    )
    outputs["primary_routing_lag_months.tif"] = compact_to_raster(
        primary_lags["lag_months"], primary["rows"], primary["cols"], mask.shape
    )
    for filename, array in outputs.items():
        write_geotiff(OUT / "maps" / filename, array, profile)

    # 9. Räumliche Validierung gegen vorhandene Referenzprodukte
    spatial_metrics, ref_metadata, comparison_maps = validate_spatial_products_v8(
        maps, profile, mask, primary_mask, n_years, OUT
    )
    spatial_metrics.to_csv(
        OUT / "tables" / "spatial_validation_metrics.csv",
        index=False, encoding="utf-8",
    )
    ref_metadata.to_csv(
        OUT / "tables" / "reference_product_metadata.csv",
        index=False, encoding="utf-8",
    )

    # 10. Vektoroutputs
    primary_path = OUT / "maps" / "primary_gauge_catchment.gpkg"
    if primary_path.exists():
        primary_path.unlink()
    primary_catchment.to_file(primary_path, layer="heidelbach_upstream", driver="GPKG")

    from shapely.geometry import Point
    snapped_records = [{
        "name": r["name"], "station_id": r["station_id"],
        "is_primary": r["is_primary"], "snap_m": r["snap_distance_m"],
        "area_km2": r["area_km2"], "geometry": Point(r["x"], r["y"]),
    } for r in gauge_routings.values()]
    snapped_gdf = gpd.GeoDataFrame(snapped_records, crs=profile["crs"])
    snapped_path = OUT / "maps" / "gauges_snapped.gpkg"
    if snapped_path.exists():
        snapped_path.unlink()
    snapped_gdf.to_file(snapped_path, layer="gauges_snapped", driver="GPKG")

    # 11. Plots
    plot_q_timeseries_v8(results, OUT / "plots" / "qobs_qsim_monthly_v8_1.png")
    plot_annual_water_balance(annual, OUT / "plots" / "annual_water_balance.png")
    plot_map(
        outputs["flow_accumulation_area_km2.tif"],
        "D8-Fließakkumulation im Projektgebiet",
        OUT / "plots" / "map_flow_accumulation_km2.png",
        "beitragende Fläche [km²]",
    )
    plot_map(
        outputs["recharge_annual_mean_mm_a.tif"],
        "Mittlere simulierte Grundwasserneubildung 2021–2025",
        OUT / "plots" / "map_recharge_annual_mean.png",
        "mm/a",
    )
    plot_map(
        outputs["aet_annual_mean_mm_a.tif"],
        "Mittlere simulierte reale Evapotranspiration 2021–2025",
        OUT / "plots" / "map_aet_annual_mean.png",
        "mm/a",
    )
    plot_map(
        soil_units["percolation_factor"],
        "Bodeneinheiten: räumlicher Perkolationsfaktor",
        OUT / "plots" / "map_soil_percolation_factor.png",
        "Faktor [-]",
    )

    # 12. Auswahl für die Hausarbeit
    write_hausarbeit_bundle(OUT, perf, params, spatial_metrics, primary_area)

    print("\n[OK] Performance am Pegel Heidelbach:")
    print(perf.to_string(index=False))
    if not spatial_metrics.empty:
        print("\n[OK] Räumliche Plausibilisierung:")
        print(spatial_metrics.to_string(index=False))
    print("\n============================================================")
    print("[FERTIG] HydroMod v8.1")
    print(f"Ergebnisse: {OUT}")
    print("Ordner:")
    print(" - maps       vollständige GIS-Ergebnisraster und Vektoren")
    print(" - plots      Diagramme und Kartenabbildungen")
    print(" - tables     vollständige Tabellen und Diagnostik")
    print(" - hausarbeit kuratierte Kernergebnisse für den Bericht")
    print("============================================================")

# =============================================================================
# HYDROMOD V9: TÄGLICHES, RÄUMLICH VERTEILTES FORCING UND TAGESROUTING
# =============================================================================

from dataclasses import dataclass as _daily_dataclass
from hashlib import sha256 as _sha256
from rasterio.warp import transform as _warp_transform


@_daily_dataclass
class DailyConfig:
    """Konfiguration der realistischeren täglichen V9.1-Version."""

    START: str = "2021-01-01"
    END: str = "2025-12-31"
    WARMUP_END: str = "2021-12-31"
    CALIB_START: str = "2022-01-01"
    CALIB_END: str = "2023-12-31"
    VALID_START: str = "2024-01-01"
    VALID_END: str = "2025-12-31"

    PRECIP_CRS_FALLBACK: str = "EPSG:3035"
    PET_CRS_FALLBACK: str = "EPSG:31467"
    PRECIP_UNIT_FACTOR: float = 1.0
    PET_UNIT_FACTOR: float = 0.1

    # Tagesprozesse
    ET_STRESS_EXPONENT: float = 0.75
    PERCOLATION_WETNESS_EXPONENT: float = 2.0
    IMPERVIOUS_RUNOFF_COEFF: float = 0.90
    MAX_ROUTING_LAG_DAYS: int = 60

    # Das Warm-up-Jahr wird mehrfach wiederholt, damit langsame Speicher nicht
    # künstlich leer in die Kalibrierung starten.
    SPINUP_CYCLES: int = 3

    # Kalibrierung
    USE_CALIBRATION: bool = True
    CALIBRATION_SAMPLE_CELLS: int = 10000
    CALIBRATION_MAXITER: int = 25
    CALIBRATION_POPSIZE: int = 6
    CALIBRATION_TOL: float = 0.001
    RANDOM_SEED: int = 42
    LOCAL_REFINEMENT_FRACTION: float = 0.04

    # Mehrzielkalibrierung. Abfluss bleibt dominant, die räumliche
    # Grundwasserneubildung verhindert aber kompensierende, unphysikalische
    # Parameterkombinationen.
    OBJECTIVE_WEIGHT_NSE: float = 0.20
    OBJECTIVE_WEIGHT_KGE: float = 0.20
    OBJECTIVE_WEIGHT_PBIAS: float = 0.07
    OBJECTIVE_WEIGHT_RECHARGE_MEAN: float = 0.03
    OBJECTIVE_WEIGHT_RECHARGE_CORRELATION: float = 0.05

    # V9.4: zusätzliche Ziele für drei verschachtelte Pegel am selben Fluss.
    # Alle Zielgrößen werden ausschließlich im Kalibrierungszeitraum berechnet.
    OBJECTIVE_WEIGHT_LOG_NSE: float = 0.15
    OBJECTIVE_WEIGHT_LOWFLOW_PBIAS: float = 0.10
    OBJECTIVE_WEIGHT_SUMMER_PBIAS: float = 0.08
    OBJECTIVE_WEIGHT_INCREMENTAL_PBIAS: float = 0.12
    LOWFLOW_QUANTILE: float = 0.20
    INCREMENTAL_PBIAS_CLIP_PERCENT: float = 150.0

    # Räumliche Niederschlagskorrektur für die zwei zusätzlichen Flächen.
    # Die Faktoren bleiben eng begrenzt und sind keine separaten Modelle.
    USE_NESTED_REACH_PRECIP_FACTORS: bool = True

    RECHARGE_REFERENCE_ENABLED: bool = True
    RECHARGE_REFERENCE_TOLERANCE_FRACTION: float = 0.15
    MIN_RECHARGE_REFERENCE_CELLS: int = 30

    # Fehlende Einzelpixel werden nur dann räumlich gemittelt, wenn wenigstens
    # ein gültiger Wert am betreffenden Tag vorhanden ist.
    FILL_MISSING_FORCING_WITH_SPATIAL_MEAN: bool = True

    # PET wird bilinear auf Modellzellzentren übertragen. Niederschlag bleibt
    # nearest-neighbour, um Starkregenmaxima nicht künstlich zu glätten.
    PET_BILINEAR_INTERPOLATION: bool = True

    PROGRESS_EVERY_DAYS: int = 100


DCFG = DailyConfig()

PARAM_NAMES_V9 = (
    "alpha_fast",
    "excess_recharge_frac",
    "beta_perc_daily",
    "k_interflow_daily",
    "k_gw_fast_daily",
    "k_gw_slow_daily",
    "slow_recharge_frac",
    "init_soil_frac",
    "routing_velocity_m_s",
    "precip_factor_alsfeld_to_heidelbach",
    "precip_factor_heidelbach_to_roellshausen",
)

PARAM_BOUNDS_V9 = (
    (0.03, 0.80),       # direkter Sättigungs-Schnellabfluss
    (0.02, 0.55),       # Restüberschuss -> Recharge; Rest -> Interflow
    (0.00005, 0.030),   # tägliche, feuchteabhängige Bodenperkolation
    (0.01, 0.50),       # tägliche Entleerung des Zwischenabflussspeichers
    (0.003, 0.250),     # tägliche Entleerung schneller GW-Speicher
    (0.00005, 0.015),   # tägliche Entleerung langsamer GW-Speicher
    (0.25, 0.95),       # Recharge-Anteil in langsamen GW-Speicher
    (0.20, 1.00),       # anfängliche Bodenfüllung
    (0.05, 1.50),       # effektive Hang-/Routinggeschwindigkeit [m/s]
    (0.90, 1.15),       # P-Korrektur Alsfeld -> Heidelbach
    (0.85, 1.10),       # P-Korrektur Heidelbach -> Röllshausen
)


def daily_out_dir() -> Path:
    p = CFG.BASE_DIR / "results" / "daily_v9_1_realistic"
    for sub in ("maps", "plots", "tables", "cache", "hausarbeit"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p

def make_param_dict_v9(x: Iterable[float]) -> dict[str, float]:
    values = list(x)
    if len(values) != len(PARAM_NAMES_V9):
        raise ValueError(
            f"V9.1 erwartet {len(PARAM_NAMES_V9)} Parameter, erhalten: {len(values)}"
        )
    return {name: float(value) for name, value in zip(PARAM_NAMES_V9, values)}


def _cell_centers(profile: dict, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = xy(
        profile["transform"],
        rows.astype(int).tolist(),
        cols.astype(int).tolist(),
        offset="center",
    )
    return np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")


def _nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nächste Indizes für auf- oder absteigende 1D-Koordinaten."""
    axis = np.asarray(axis, dtype="float64")
    values = np.asarray(values, dtype="float64")
    if axis.ndim != 1 or len(axis) < 2:
        raise ValueError("Forcing-Koordinatenachse muss 1D mit mindestens zwei Werten sein.")

    ascending = bool(axis[-1] >= axis[0])
    work = axis if ascending else axis[::-1]
    pos = np.searchsorted(work, values, side="left")
    pos = np.clip(pos, 0, len(work) - 1)
    prev = np.clip(pos - 1, 0, len(work) - 1)
    choose_prev = np.abs(values - work[prev]) <= np.abs(values - work[pos])
    idx_work = np.where(choose_prev, prev, pos)
    idx = idx_work if ascending else (len(axis) - 1 - idx_work)

    half = 0.5 * float(np.nanmedian(np.abs(np.diff(axis))))
    valid = (values >= float(np.nanmin(axis)) - half) & (values <= float(np.nanmax(axis)) + half)
    return idx.astype(np.int32), valid


def _forcing_fill(values: np.ndarray, label: str, date: pd.Timestamp) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    missing = ~np.isfinite(values)
    if not missing.any():
        return np.maximum(values, 0.0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{label} enthält am {date.date()} ausschließlich NaN/NoData.")
    if not DCFG.FILL_MISSING_FORCING_WITH_SPATIAL_MEAN:
        raise ValueError(f"{label} enthält am {date.date()} {int(missing.sum())} fehlende Zellen.")
    fill = float(np.nanmean(finite))
    out = values.copy()
    out[missing] = fill
    return np.maximum(out, 0.0)


class NetCDFDailyPrecipProvider:
    """Liest HYRAS täglich und liefert Werte an beliebigen Modellzellzentren."""

    def __init__(self, folder: Path):
        import xarray as xr

        files = find_files(folder, [".nc", ".nc4", ".cdf"])
        if not files:
            raise FileNotFoundError(f"Keine Niederschlags-NetCDF-Dateien in {folder}")
        print(f"[OK] Tägliche Niederschlagsdateien: {len(files)}")

        try:
            self.ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal")
        except Exception:
            datasets = [xr.open_dataset(f) for f in files]
            var0 = choose_data_var(datasets[0], CFG.P_VAR)
            t0 = find_time_dim(datasets[0][var0])
            self.ds = xr.concat(datasets, dim=t0)

        self.var = choose_data_var(self.ds, CFG.P_VAR)
        self.da = self.ds[self.var]
        self.time_dim = find_time_dim(self.da)
        lower = {d.lower(): d for d in self.da.dims}
        self.x_dim = next((lower[k] for k in ("x", "lon", "longitude", "rlon") if k in lower), None)
        self.y_dim = next((lower[k] for k in ("y", "lat", "latitude", "rlat") if k in lower), None)
        if self.x_dim is None or self.y_dim is None:
            raise ValueError(f"Niederschlagsdimensionen nicht erkannt: {self.da.dims}")

        self.x = np.asarray(self.da[self.x_dim].values, dtype="float64")
        self.y = np.asarray(self.da[self.y_dim].values, dtype="float64")
        raw_dates = pd.to_datetime(np.asarray(self.da[self.time_dim].values))
        dates = pd.DatetimeIndex(raw_dates).normalize()
        self.date_to_pos: dict[pd.Timestamp, int] = {}
        for i, d in enumerate(dates):
            self.date_to_pos.setdefault(pd.Timestamp(d), int(i))
        self.available_dates = pd.DatetimeIndex(sorted(self.date_to_pos))

        crs = None
        try:
            import rioxarray  # noqa: F401
            crs = self.da.rio.crs
        except Exception:
            crs = None
        self.crs = str(crs) if crs is not None else DCFG.PRECIP_CRS_FALLBACK
        print(f"[OK] Niederschlag: Variable={self.var}, CRS={self.crs}, Tage={len(self.available_dates)}")

    def build_mapping(
        self,
        target_x: np.ndarray,
        target_y: np.ndarray,
        target_crs,
    ) -> dict:
        tx, ty = _warp_transform(target_crs, self.crs, target_x.tolist(), target_y.tolist())
        xi, valid_x = _nearest_axis_indices(self.x, np.asarray(tx))
        yi, valid_y = _nearest_axis_indices(self.y, np.asarray(ty))
        valid = valid_x & valid_y
        coverage = float(np.mean(valid)) if len(valid) else 0.0
        if coverage < 0.80:
            raise ValueError(
                f"Nur {coverage:.1%} der Modellzellen liegen im Niederschlagsraster. "
                f"Prüfe PRECIP_CRS_FALLBACK={self.crs}."
            )
        if coverage < 1.0:
            warnings.warn(f"{int((~valid).sum())} Zielzellen liegen außerhalb des Niederschlagsrasters.")
        y0, y1 = int(yi[valid].min()), int(yi[valid].max())
        x0, x1 = int(xi[valid].min()), int(xi[valid].max())
        return {
            "x_idx": xi,
            "y_idx": yi,
            "valid": valid,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
        }

    def read(self, date: pd.Timestamp, mapping: dict) -> np.ndarray:
        date = pd.Timestamp(date).normalize()
        if date not in self.date_to_pos:
            raise KeyError(f"Kein Niederschlag für {date.date()}")
        pos = self.date_to_pos[date]
        selected = self.da.isel({self.time_dim: pos}).transpose(self.y_dim, self.x_dim)
        sub = np.asarray(
            selected.isel({
                self.y_dim: slice(mapping["y0"], mapping["y1"] + 1),
                self.x_dim: slice(mapping["x0"], mapping["x1"] + 1),
            }).values,
            dtype="float64",
        )
        out = np.full(len(mapping["x_idx"]), np.nan, dtype="float64")
        valid = mapping["valid"]
        out[valid] = sub[
            mapping["y_idx"][valid] - mapping["y0"],
            mapping["x_idx"][valid] - mapping["x0"],
        ] * float(DCFG.PRECIP_UNIT_FACTOR)
        return out

    def close(self) -> None:
        try:
            self.ds.close()
        except Exception:
            pass


class RasterDailyPETProvider:
    """Liest tägliche DWD-PET-Raster und interpoliert auf Modellzellzentren."""

    def __init__(self, folder: Path):
        files = find_files(folder, [".asc", ".tif", ".tiff"])
        dated: dict[pd.Timestamp, Path] = {}
        for path in files:
            date = parse_date_from_path(path)
            if date is not None:
                dated.setdefault(pd.Timestamp(date).normalize(), path)
        if not dated:
            raise FileNotFoundError(f"Keine datierten PET-Raster in {folder}")
        self.date_to_file = dated
        self.available_dates = pd.DatetimeIndex(sorted(dated))
        self.first_file = dated[self.available_dates[0]]
        with rasterio.open(self.first_file) as src:
            self.width = src.width
            self.height = src.height
            self.transform = src.transform
            self.crs = str(src.crs) if src.crs is not None else DCFG.PET_CRS_FALLBACK
        method = "bilinear" if DCFG.PET_BILINEAR_INTERPOLATION else "nearest"
        print(
            f"[OK] PET: CRS={self.crs}, Tage={len(self.available_dates)}, "
            f"Interpolation={method}, erstes Raster={self.first_file.name}"
        )

    def build_mapping(
        self,
        target_x: np.ndarray,
        target_y: np.ndarray,
        target_crs,
    ) -> dict:
        tx, ty = _warp_transform(
            target_crs, self.crs, target_x.tolist(), target_y.tolist()
        )
        tx = np.asarray(tx, dtype="float64")
        ty = np.asarray(ty, dtype="float64")

        if not DCFG.PET_BILINEAR_INTERPOLATION:
            rr, cc = rowcol(self.transform, tx, ty)
            rr = np.asarray(rr, dtype=np.int32)
            cc = np.asarray(cc, dtype=np.int32)
            valid = (rr >= 0) & (rr < self.height) & (cc >= 0) & (cc < self.width)
            coverage = float(np.mean(valid)) if len(valid) else 0.0
            if coverage < 0.80:
                raise ValueError(
                    f"Nur {coverage:.1%} der Modellzellen liegen im PET-Raster. "
                    f"Prüfe PET_CRS_FALLBACK={self.crs}."
                )
            r0, r1 = int(rr[valid].min()), int(rr[valid].max())
            c0, c1 = int(cc[valid].min()), int(cc[valid].max())
            return {
                "method": "nearest", "row": rr, "col": cc, "valid": valid,
                "r0": r0, "r1": r1, "c0": c0, "c1": c1,
            }

        inv = ~self.transform
        col_corner = inv.a * tx + inv.b * ty + inv.c
        row_corner = inv.d * tx + inv.e * ty + inv.f

        # Rasterwerte liegen an Pixelzentren; daher 0,5 Pixel abziehen.
        col_f = col_corner - 0.5
        row_f = row_corner - 0.5
        c0_arr = np.floor(col_f).astype(np.int32)
        r0_arr = np.floor(row_f).astype(np.int32)
        c1_arr = c0_arr + 1
        r1_arr = r0_arr + 1
        wx = col_f - c0_arr
        wy = row_f - r0_arr

        valid = (
            (r0_arr >= 0) & (r1_arr < self.height)
            & (c0_arr >= 0) & (c1_arr < self.width)
        )
        coverage = float(np.mean(valid)) if len(valid) else 0.0
        if coverage < 0.80:
            raise ValueError(
                f"Nur {coverage:.1%} der Modellzellen liegen vollständig im PET-Raster. "
                f"Prüfe PET_CRS_FALLBACK={self.crs}."
            )
        if coverage < 1.0:
            warnings.warn(
                f"{int((~valid).sum())} Zielzellen liegen am/außerhalb des PET-Randbereichs."
            )

        rmin = int(r0_arr[valid].min())
        rmax = int(r1_arr[valid].max())
        cmin = int(c0_arr[valid].min())
        cmax = int(c1_arr[valid].max())
        return {
            "method": "bilinear",
            "r0_arr": r0_arr, "r1_arr": r1_arr,
            "c0_arr": c0_arr, "c1_arr": c1_arr,
            "wx": wx.astype("float64"), "wy": wy.astype("float64"),
            "valid": valid,
            "r0": rmin, "r1": rmax, "c0": cmin, "c1": cmax,
        }

    def read(self, date: pd.Timestamp, mapping: dict) -> np.ndarray:
        date = pd.Timestamp(date).normalize()
        path = self.date_to_file.get(date)
        if path is None:
            raise KeyError(f"Keine PET-Datei für {date.date()}")
        with rasterio.open(path) as src:
            src_crs = str(src.crs) if src.crs is not None else self.crs
            signature_ok = (
                src.width == self.width
                and src.height == self.height
                and tuple(src.transform) == tuple(self.transform)
                and src_crs == self.crs
            )
            if not signature_ok:
                raise ValueError(
                    f"PET-Rastergrid änderte sich bei {path}. "
                    "Alle täglichen PET-Dateien müssen dasselbe Grid besitzen."
                )
            window = Window(
                mapping["c0"], mapping["r0"],
                mapping["c1"] - mapping["c0"] + 1,
                mapping["r1"] - mapping["r0"] + 1,
            )
            sub = src.read(1, window=window).astype("float64")
            if src.nodata is not None:
                sub[sub == src.nodata] = np.nan

        valid = mapping["valid"]
        out = np.full(len(valid), np.nan, dtype="float64")

        if mapping["method"] == "nearest":
            out[valid] = sub[
                mapping["row"][valid] - mapping["r0"],
                mapping["col"][valid] - mapping["c0"],
            ]
        else:
            r0 = mapping["r0_arr"][valid] - mapping["r0"]
            r1 = mapping["r1_arr"][valid] - mapping["r0"]
            c0 = mapping["c0_arr"][valid] - mapping["c0"]
            c1 = mapping["c1_arr"][valid] - mapping["c0"]
            wx = mapping["wx"][valid]
            wy = mapping["wy"][valid]

            values = np.vstack((
                sub[r0, c0], sub[r0, c1], sub[r1, c0], sub[r1, c1],
            ))
            weights = np.vstack((
                (1.0 - wx) * (1.0 - wy),
                wx * (1.0 - wy),
                (1.0 - wx) * wy,
                wx * wy,
            ))
            finite = np.isfinite(values)
            weighted_sum = np.nansum(np.where(finite, values * weights, 0.0), axis=0)
            weight_sum = np.sum(np.where(finite, weights, 0.0), axis=0)
            interpolated = np.full(len(r0), np.nan, dtype="float64")
            good = weight_sum > 0
            interpolated[good] = weighted_sum[good] / weight_sum[good]
            out[valid] = interpolated

        return out * float(DCFG.PET_UNIT_FACTOR)

def load_qobs_daily(q_dirname: Optional[str] = None) -> pd.Series:
    """Liest beobachteten Tagesdurchfluss und behält die Tagesauflösung."""
    q_dir = data_dir() / (q_dirname or CFG.QOBS_DIRNAME)
    files = find_files(q_dir, [".csv", ".txt"])
    if not files:
        raise FileNotFoundError(f"Keine Qobs-Dateien in {q_dir}")
    parts: list[pd.Series] = []
    for path in files:
        df = read_csv_auto(path)
        date_col = choose_date_col(df, CFG.QOBS_DATE_COLUMN)
        if CFG.QOBS_VALUE_COLUMN is not None:
            value_col = CFG.QOBS_VALUE_COLUMN
        else:
            candidates = [c for c in df.columns if c != date_col]
            if not candidates:
                continue
            value_col = next(
                (c for c in candidates if any(
                    token in str(c).lower()
                    for token in ("durchfluss", "abfluss", "discharge", "m3", "m³")
                )),
                candidates[0],
            )
        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dt.normalize()
        raw = df[value_col]
        values = pd.to_numeric(
            raw.astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
        if values.notna().sum() < max(3, int(0.5 * raw.notna().sum())):
            values = extract_q_mean_from_min_mean_max(raw)
        series = pd.Series(values.to_numpy(), index=dates, name="Qobs_m3s").dropna()
        parts.append(series)
    if not parts:
        raise ValueError("Keine gültigen Qobs-Tageswerte gelesen.")
    q = pd.concat(parts).sort_index()
    q = q[~q.index.duplicated(keep="first")]
    q = q.loc[DCFG.START:DCFG.END]
    print(f"[OK] Qobs täglich: {q.index.min().date()} bis {q.index.max().date()}, {len(q)} Werte")
    return q


def _common_daily_dates(
    precip: NetCDFDailyPrecipProvider,
    pet: RasterDailyPETProvider,
) -> pd.DatetimeIndex:
    requested = pd.date_range(DCFG.START, DCFG.END, freq="D")
    common = requested.intersection(precip.available_dates).intersection(pet.available_dates)
    missing = requested.difference(common)
    if len(common) == 0:
        raise ValueError("Keine gemeinsamen täglichen Niederschlags- und PET-Daten.")
    if len(missing):
        warnings.warn(
            f"{len(missing)} Tage fehlen in Niederschlag oder PET und werden nicht simuliert. "
            f"Erster fehlender Tag: {missing[0].date()}"
        )
    print(f"[OK] Gemeinsame tägliche Forcings: {len(common)} Tage")
    return common


def _daily_spatial_grids(
    params: dict[str, float],
    slope_q: np.ndarray,
    land_q: np.ndarray,
    land_perc: np.ndarray,
    soil_q: np.ndarray,
    soil_perc: np.ndarray,
    soil_slow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.clip(
        params["alpha_fast"] * slope_q * land_q * soil_q,
        0.0, 0.98,
    )
    beta = np.clip(
        params["beta_perc_daily"] * land_perc * soil_perc,
        0.0, 0.25,
    )
    slow = np.clip(
        params["slow_recharge_frac"] * soil_slow,
        0.05, 0.98,
    )
    return alpha, beta, slow


def water_balance_step_daily_v9(
    p_grid: np.ndarray,
    pet_grid: np.ndarray,
    fc: np.ndarray,
    soil: np.ndarray,
    interflow_store: np.ndarray,
    gw_fast: np.ndarray,
    gw_slow: np.ndarray,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    slow_fraction_grid: np.ndarray,
    pet_factor: np.ndarray,
    impervious_fraction: np.ndarray,
    params: dict[str, float],
) -> dict[str, np.ndarray]:
    """Tägliche Wasserbilanz mit explizitem Zwischenabflussspeicher.

    Im Vorgängermodell wurde der gesamte nicht schnelle Sättigungsüberschuss
    direkt als Grundwasserneubildung verbucht. V9.1 trennt diesen Anteil in
    eine begrenzte Recharge-Komponente und einen verzögerten Zwischenabfluss.
    """
    storage_before = soil + interflow_store + gw_fast + gw_slow

    precipitation = np.maximum(p_grid, 0.0)
    imp_runoff = (
        precipitation
        * np.clip(impervious_fraction, 0.0, 1.0)
        * float(DCFG.IMPERVIOUS_RUNOFF_COEFF)
    )
    infiltrating_p = np.maximum(precipitation - imp_runoff, 0.0)
    water = soil + infiltrating_p

    relative_wetness = np.clip(water / np.maximum(fc, 1.0), 0.0, 1.0)
    aet_demand = (
        np.maximum(pet_grid, 0.0)
        * pet_factor
        * np.power(relative_wetness, float(DCFG.ET_STRESS_EXPONENT))
    )
    aet = np.minimum(aet_demand, water)
    water_after_et = np.maximum(water - aet, 0.0)

    soil_filled = np.minimum(water_after_et, fc)
    excess = np.maximum(water_after_et - fc, 0.0)

    # Direkte Sättigungsreaktion.
    saturation_quickflow = alpha_grid * excess
    remaining_excess = np.maximum(excess - saturation_quickflow, 0.0)

    # Nur ein kalibrierter Anteil des verbleibenden Überschusses gelangt direkt
    # ins Grundwasser. Der Rest wird als verzögerter Zwischenabfluss gespeichert.
    recharge_from_excess = (
        float(params["excess_recharge_frac"]) * remaining_excess
    )
    interflow_input = np.maximum(remaining_excess - recharge_from_excess, 0.0)

    # Perkolation nimmt nicht linear bei trockenen Böden zu. Sie wird bei hoher
    # relativer Bodenfüllung deutlich stärker.
    soil_wetness = np.clip(soil_filled / np.maximum(fc, 1.0), 0.0, 1.0)
    percolation = (
        beta_grid
        * soil_filled
        * np.power(soil_wetness, float(DCFG.PERCOLATION_WETNESS_EXPONENT))
    )
    percolation = np.minimum(percolation, soil_filled)
    soil_new = np.maximum(soil_filled - percolation, 0.0)

    interflow_store_new = interflow_store + interflow_input
    interflow = float(params["k_interflow_daily"]) * interflow_store_new
    interflow = np.minimum(interflow, interflow_store_new)
    interflow_store_new = np.maximum(interflow_store_new - interflow, 0.0)

    recharge = recharge_from_excess + percolation
    recharge_slow = recharge * slow_fraction_grid
    recharge_fast = recharge - recharge_slow
    gw_fast_new = gw_fast + recharge_fast
    gw_slow_new = gw_slow + recharge_slow

    baseflow_fast = params["k_gw_fast_daily"] * gw_fast_new
    baseflow_slow = params["k_gw_slow_daily"] * gw_slow_new
    baseflow_fast = np.minimum(baseflow_fast, gw_fast_new)
    baseflow_slow = np.minimum(baseflow_slow, gw_slow_new)
    gw_fast_new = np.maximum(gw_fast_new - baseflow_fast, 0.0)
    gw_slow_new = np.maximum(gw_slow_new - baseflow_slow, 0.0)

    quickflow = imp_runoff + saturation_quickflow
    baseflow = baseflow_fast + baseflow_slow
    local_runoff = quickflow + interflow + baseflow
    storage_after = soil_new + interflow_store_new + gw_fast_new + gw_slow_new
    residual = precipitation - aet - local_runoff - (storage_after - storage_before)

    return {
        "soil": soil_new,
        "interflow_store": interflow_store_new,
        "gw_fast": gw_fast_new,
        "gw_slow": gw_slow_new,
        "aet": aet,
        "impervious_runoff": imp_runoff,
        "excess": excess,
        "saturation_quickflow": saturation_quickflow,
        "quickflow": quickflow,
        "interflow_input": interflow_input,
        "interflow": interflow,
        "percolation": percolation,
        "recharge_from_excess": recharge_from_excess,
        "recharge": recharge,
        "baseflow_fast": baseflow_fast,
        "baseflow_slow": baseflow_slow,
        "baseflow": baseflow,
        "local_runoff": local_runoff,
        "water_balance_residual": residual,
    }

def prepare_daily_routing_lags(
    travel_cost: np.ndarray,
    velocity_m_s: float,
) -> dict[str, np.ndarray | int | float]:
    if velocity_m_s <= 0:
        raise ValueError("Routinggeschwindigkeit muss > 0 sein.")
    travel_days = np.asarray(travel_cost, dtype="float64") / velocity_m_s / 86400.0
    lag_float = np.clip(travel_days, 0.0, float(DCFG.MAX_ROUTING_LAG_DAYS))
    lag0 = np.floor(lag_float).astype(np.int16)
    frac1 = lag_float - lag0
    lag1 = np.minimum(lag0 + 1, DCFG.MAX_ROUTING_LAG_DAYS).astype(np.int16)
    frac1[lag0 >= DCFG.MAX_ROUTING_LAG_DAYS] = 0.0
    return {
        "travel_days": travel_days,
        "lag_days": lag_float,
        "lag0": lag0,
        "lag1": lag1,
        "frac1": frac1,
        "max_lag": int(DCFG.MAX_ROUTING_LAG_DAYS),
        "velocity_m_s": float(velocity_m_s),
    }


def _route_compact_daily(
    local_runoff_mm: np.ndarray,
    selected_ids: np.ndarray,
    weights: np.ndarray,
    cell_area_m2: float,
    day_position: int,
    arrivals_m3: np.ndarray,
    lag_info: dict,
) -> None:
    runoff = np.nan_to_num(local_runoff_mm[selected_ids], nan=0.0)
    volumes = runoff * weights * cell_area_m2 / 1000.0
    current = volumes * (1.0 - lag_info["frac1"])
    following = volumes * lag_info["frac1"]
    max_lag = int(lag_info["max_lag"])
    by0 = np.bincount(lag_info["lag0"], weights=current, minlength=max_lag + 1)
    by1 = np.bincount(lag_info["lag1"], weights=following, minlength=max_lag + 1)
    for lag in range(max_lag + 1):
        target = day_position + lag
        if target < len(arrivals_m3):
            arrivals_m3[target] += by0[lag] + by1[lag]


def _stratified_sample_indices_v9(
    features: tuple[np.ndarray, ...],
    max_cells: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(features[0])
    if n <= max_cells:
        return np.arange(n, dtype=np.int32), np.ones(n, dtype="float64")
    codes = np.zeros(n, dtype=np.int32)
    multiplier = 1
    for feature in features:
        codes += _quantile_codes(np.asarray(feature), n_bins=4).astype(np.int32) * multiplier
        multiplier *= 4
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    unique, counts = np.unique(codes, return_counts=True)
    for code, count in zip(unique, counts):
        candidates = np.flatnonzero(codes == code)
        take = max(1, int(round(max_cells * count / n)))
        take = min(take, len(candidates))
        chosen = rng.choice(candidates, size=take, replace=False)
        selected_parts.append(chosen)
        weight_parts.append(np.full(take, len(candidates) / take, dtype="float64"))
    selected = np.concatenate(selected_parts)
    weights = np.concatenate(weight_parts)
    if len(selected) > max_cells:
        keep = rng.choice(len(selected), size=max_cells, replace=False)
        selected = selected[keep]
        weights = weights[keep]
        weights *= n / np.sum(weights)
    return selected.astype(np.int32), weights


def _build_calibration_forcing_cache(
    dates: pd.DatetimeIndex,
    precip: NetCDFDailyPrecipProvider,
    pet: RasterDailyPETProvider,
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_crs,
    signature_values: np.ndarray,
    out: Path,
) -> tuple[np.ndarray, np.ndarray]:
    digest = _sha256()
    digest.update(np.asarray(signature_values, dtype="int32").tobytes())
    digest.update(str(dates[0]).encode())
    digest.update(str(dates[-1]).encode())
    digest.update(str(len(dates)).encode())
    cache_path = out / "cache" / f"daily_forcing_sample_{digest.hexdigest()[:16]}.npz"
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            cached_dates = pd.to_datetime(cached["dates_ns"])
            if len(cached_dates) == len(dates) and np.array_equal(
                cached_dates.to_numpy(), dates.to_numpy()
            ):
                print(f"[OK] Kalibrierungsforcing aus Cache: {cache_path}")
                return cached["p_mm"].astype("float64"), cached["pet_mm"].astype("float64")
        except Exception:
            warnings.warn("Kalibrierungsforcing-Cache ist ungültig und wird neu erzeugt.")

    p_map = precip.build_mapping(target_x, target_y, target_crs)
    pet_map = pet.build_mapping(target_x, target_y, target_crs)
    p_arr = np.empty((len(dates), len(target_x)), dtype="float32")
    pet_arr = np.empty_like(p_arr)
    for i, date in enumerate(dates):
        p_arr[i] = _forcing_fill(precip.read(date, p_map), "Niederschlag", date)
        pet_arr[i] = _forcing_fill(pet.read(date, pet_map), "PET", date)
        if (i + 1) % DCFG.PROGRESS_EVERY_DAYS == 0 or i + 1 == len(dates):
            print(f"     Kalibrierungsforcing: {i+1}/{len(dates)} Tage")
    np.savez_compressed(
        cache_path,
        dates_ns=dates.to_numpy(dtype="datetime64[ns]"),
        p_mm=p_arr,
        pet_mm=pet_arr,
    )
    print(f"[OK] Kalibrierungsforcing gecacht: {cache_path}")
    return p_arr.astype("float64"), pet_arr.astype("float64")


def _weighted_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    valid = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    )
    if int(valid.sum()) < 3:
        return np.nan
    xv, yv, wv = x[valid], y[valid], weights[valid]
    wv = wv / np.sum(wv)
    mx = float(np.sum(wv * xv))
    my = float(np.sum(wv * yv))
    cov = float(np.sum(wv * (xv - mx) * (yv - my)))
    vx = float(np.sum(wv * (xv - mx) ** 2))
    vy = float(np.sum(wv * (yv - my) ** 2))
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / np.sqrt(vx * vy))



def _nse_log1p_v94(obs: np.ndarray, sim: np.ndarray) -> float:
    """NSE auf log1p(Q); betont Niedrig- und Mittelwasser."""
    obs = np.asarray(obs, dtype="float64")
    sim = np.asarray(sim, dtype="float64")
    valid = np.isfinite(obs) & np.isfinite(sim) & (obs >= 0.0) & (sim >= 0.0)
    if int(valid.sum()) < 3:
        return np.nan
    return nse(np.log1p(obs[valid]), np.log1p(sim[valid]))


def _safe_pbias_v94(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype="float64")
    sim = np.asarray(sim, dtype="float64")
    valid = np.isfinite(obs) & np.isfinite(sim)
    if int(valid.sum()) < 3:
        return np.nan
    denominator = float(np.sum(obs[valid]))
    if abs(denominator) <= 1e-12:
        return np.nan
    return float(100.0 * np.sum(sim[valid] - obs[valid]) / denominator)


def _nested_zone_codes_v94(
    gauge_routings: dict[str, dict],
    network: dict,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Ordnet jede kompakte Rasterzelle einer verschachtelten Teilfläche zu.

    Codes:
      -1 = außerhalb des größten Pegelgebiets
       0 = Quelle bis oberster Pegel
       1 = oberster bis mittlerer Pegel
       2 = mittlerer bis unterster Pegel

    Es bleiben ein Modell, ein Fluss und ein gemeinsamer Prozessparametersatz.
    Nur zwei eng begrenzte Niederschlagskorrekturfaktoren sind räumlich variabel.
    """
    ordered = sorted(gauge_routings.values(), key=lambda r: float(r["area_km2"]))
    if len(ordered) != 3:
        raise ValueError(
            "V9.4 erwartet genau drei verschachtelte Pegel für die Zonenbildung."
        )

    masks = [np.asarray(r["upstream_mask"], dtype=bool) for r in ordered]
    if np.any(masks[0] & ~masks[1]) or np.any(masks[1] & ~masks[2]):
        raise ValueError(
            "Die drei Pegelgebiete sind im D8-Netz nicht vollständig verschachtelt. "
            "Pegelreihenfolge und Snapping prüfen."
        )

    full = np.full(masks[2].shape, -1, dtype=np.int8)
    full[masks[0]] = 0
    full[masks[1] & ~masks[0]] = 1
    full[masks[2] & ~masks[1]] = 2
    compact = full[network["rows"], network["cols"]]

    cell_area_km2 = abs(
        float(network.get("cell_area_m2", 0.0))
    ) / 1e6 if "cell_area_m2" in network else np.nan
    rows = []
    labels = (
        f"Quelle → {ordered[0]['name']}",
        f"{ordered[0]['name']} → {ordered[1]['name']}",
        f"{ordered[1]['name']} → {ordered[2]['name']}",
    )
    for code, label in enumerate(labels):
        count = int(np.sum(compact == code))
        rows.append({
            "zone_code": code,
            "zone_name": label,
            "n_cells": count,
            "area_from_gauge_difference_km2": (
                float(ordered[0]["area_km2"])
                if code == 0 else
                float(ordered[code]["area_km2"] - ordered[code - 1]["area_km2"])
            ),
            "upstream_station": "source" if code == 0 else ordered[code - 1]["station_id"],
            "downstream_station": ordered[code]["station_id"],
        })
    return compact.astype(np.int8), pd.DataFrame(rows)


def _regional_precip_factors_v94(
    zone_codes: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    factors = np.ones(len(zone_codes), dtype="float64")
    if not DCFG.USE_NESTED_REACH_PRECIP_FACTORS:
        return factors
    factors[np.asarray(zone_codes) == 1] = float(
        params["precip_factor_alsfeld_to_heidelbach"]
    )
    factors[np.asarray(zone_codes) == 2] = float(
        params["precip_factor_heidelbach_to_roellshausen"]
    )
    return factors


def _incremental_reach_metrics_v94(
    dates: pd.DatetimeIndex,
    calibration_data: dict[str, dict[str, object]],
    obs_arrays: dict[str, np.ndarray],
    simulations: dict[str, dict[str, object]],
    masks: dict[str, np.ndarray],
) -> tuple[list[dict[str, float | str]], float]:
    """Volumenfehler der zusätzlichen Flächen zwischen benachbarten Pegeln."""
    ordered_keys = sorted(
        calibration_data,
        key=lambda k: float(calibration_data[k]["upstream_area_km2"]),
    )
    rows: list[dict[str, float | str]] = []
    penalties: list[float] = []
    for upper_key, lower_key in zip(ordered_keys[:-1], ordered_keys[1:]):
        common = masks[upper_key] & masks[lower_key]
        obs_inc = obs_arrays[lower_key][common] - obs_arrays[upper_key][common]
        sim_lower = simulations[lower_key]["qsim"].to_numpy(dtype="float64")[common]
        sim_upper = simulations[upper_key]["qsim"].to_numpy(dtype="float64")[common]
        sim_inc = sim_lower - sim_upper
        reach_pbias = _safe_pbias_v94(obs_inc, sim_inc)
        if np.isfinite(reach_pbias):
            clipped = min(
                abs(float(reach_pbias)),
                float(DCFG.INCREMENTAL_PBIAS_CLIP_PERCENT),
            )
            penalties.append(clipped / 100.0)
        rows.append({
            "upper_key": upper_key,
            "lower_key": lower_key,
            "reach_name": (
                f"{calibration_data[upper_key]['name']} → "
                f"{calibration_data[lower_key]['name']}"
            ),
            "incremental_pbias_percent": float(reach_pbias),
            "incremental_area_km2": float(
                calibration_data[lower_key]["upstream_area_km2"]
                - calibration_data[upper_key]["upstream_area_km2"]
            ),
            "n_days": int(np.sum(common)),
        })
    return rows, float(np.mean(penalties)) if penalties else np.nan

def _simulate_calibration_sample_v9(
    dates: pd.DatetimeIndex,
    p_forcing: np.ndarray,
    pet_forcing: np.ndarray,
    static: dict[str, np.ndarray | float],
    params: dict[str, float],
) -> dict[str, object]:
    fc = np.asarray(static["fc"], dtype="float64")
    weights = np.asarray(static["weights"], dtype="float64")
    alpha, beta, slow = _daily_spatial_grids(
        params,
        np.asarray(static["slope_q"]),
        np.asarray(static["land_q"]),
        np.asarray(static["land_perc"]),
        np.asarray(static["soil_q"]),
        np.asarray(static["soil_perc"]),
        np.asarray(static["soil_slow"]),
    )
    pet_factor = np.asarray(static["pet_factor"], dtype="float64")
    impervious = np.asarray(static["impervious_fraction"], dtype="float64")
    zone_codes = np.asarray(
        static.get("nested_zone_code", np.zeros(len(fc), dtype=np.int8)),
        dtype=np.int8,
    )
    regional_p_factor = _regional_precip_factors_v94(zone_codes, params)
    soil = fc * params["init_soil_frac"]
    interflow_store = np.zeros_like(fc)
    gw_fast = np.zeros_like(fc)
    gw_slow = np.zeros_like(fc)

    lag_info = prepare_daily_routing_lags(
        np.asarray(static["travel_cost"]), params["routing_velocity_m_s"]
    )
    arrivals = np.zeros(len(dates) + DCFG.MAX_ROUTING_LAG_DAYS + 1, dtype="float64")
    ids = np.arange(len(fc), dtype=np.int32)
    cell_area = float(static["cell_area_m2"])

    warm_positions = np.flatnonzero(dates <= pd.Timestamp(DCFG.WARMUP_END))
    for _ in range(max(0, int(DCFG.SPINUP_CYCLES) - 1)):
        for i in warm_positions:
            step = water_balance_step_daily_v9(
                p_forcing[i] * regional_p_factor, pet_forcing[i], fc, soil, interflow_store,
                gw_fast, gw_slow, alpha, beta, slow, pet_factor, impervious, params,
            )
            soil = step["soil"]
            interflow_store = step["interflow_store"]
            gw_fast, gw_slow = step["gw_fast"], step["gw_slow"]

    calibration_days = (
        (dates >= pd.Timestamp(DCFG.CALIB_START))
        & (dates <= pd.Timestamp(DCFG.CALIB_END))
    )
    recharge_sum_cells = np.zeros_like(fc)
    recharge_days = 0

    for i in range(len(dates)):
        step = water_balance_step_daily_v9(
            p_forcing[i] * regional_p_factor, pet_forcing[i], fc, soil, interflow_store,
            gw_fast, gw_slow, alpha, beta, slow, pet_factor, impervious, params,
        )
        soil = step["soil"]
        interflow_store = step["interflow_store"]
        gw_fast, gw_slow = step["gw_fast"], step["gw_slow"]
        _route_compact_daily(
            step["local_runoff"], ids, weights, cell_area, i, arrivals, lag_info
        )
        if calibration_days[i]:
            recharge_sum_cells += step["recharge"]
            recharge_days += 1

    qsim = pd.Series(
        arrivals[:len(dates)] / 86400.0, index=dates, name="Qsim_m3s"
    )
    if recharge_days > 0:
        recharge_cells_mm_a = recharge_sum_cells * 365.2425 / recharge_days
        recharge_mean_mm_a = _weighted_mean(recharge_cells_mm_a, weights)
    else:
        recharge_cells_mm_a = np.full_like(fc, np.nan)
        recharge_mean_mm_a = np.nan

    reference = np.asarray(
        static.get("recharge_reference_mm_a", np.full_like(fc, np.nan)),
        dtype="float64",
    )
    reference_mean = _weighted_mean(reference, weights)
    recharge_corr = _weighted_correlation(
        recharge_cells_mm_a, reference, weights
    )
    return {
        "qsim": qsim,
        "recharge_cells_mm_a": recharge_cells_mm_a,
        "recharge_mean_mm_a": recharge_mean_mm_a,
        "recharge_reference_mean_mm_a": reference_mean,
        "recharge_correlation": recharge_corr,
    }


def _calibration_score_v9(
    obs: np.ndarray,
    sim: np.ndarray,
    recharge_model_mean_mm_a: float = np.nan,
    recharge_reference_mean_mm_a: float = np.nan,
    recharge_correlation: float = np.nan,
) -> tuple[float, dict[str, float]]:
    nn = nse(obs, sim)
    kk = kge(obs, sim)
    pp = pbias(obs, sim)
    if not np.isfinite(nn) or not np.isfinite(kk) or not np.isfinite(pp):
        return 9999.0, {
            "NSE": nn, "KGE": kk, "PBIAS_percent": pp,
            "Recharge_model_mm_a": recharge_model_mean_mm_a,
            "Recharge_reference_mm_a": recharge_reference_mean_mm_a,
            "Recharge_PBIAS_percent": np.nan,
            "Recharge_correlation": recharge_correlation,
        }

    terms = [
        (DCFG.OBJECTIVE_WEIGHT_NSE, 1.0 - nn),
        (DCFG.OBJECTIVE_WEIGHT_KGE, 1.0 - kk),
        (DCFG.OBJECTIVE_WEIGHT_PBIAS, abs(pp) / 100.0),
    ]

    recharge_pbias = np.nan
    recharge_mean_penalty = np.nan
    if (
        DCFG.RECHARGE_REFERENCE_ENABLED
        and np.isfinite(recharge_model_mean_mm_a)
        and np.isfinite(recharge_reference_mean_mm_a)
        and recharge_reference_mean_mm_a > 0
    ):
        recharge_pbias = (
            100.0
            * (recharge_model_mean_mm_a - recharge_reference_mean_mm_a)
            / recharge_reference_mean_mm_a
        )
        relative_error = abs(recharge_pbias) / 100.0
        recharge_mean_penalty = max(
            0.0,
            relative_error - float(DCFG.RECHARGE_REFERENCE_TOLERANCE_FRACTION),
        )
        terms.append((
            DCFG.OBJECTIVE_WEIGHT_RECHARGE_MEAN,
            recharge_mean_penalty,
        ))

    if DCFG.RECHARGE_REFERENCE_ENABLED and np.isfinite(recharge_correlation):
        correlation_penalty = 1.0 - np.clip(recharge_correlation, -1.0, 1.0)
        terms.append((
            DCFG.OBJECTIVE_WEIGHT_RECHARGE_CORRELATION,
            correlation_penalty,
        ))

    weight_sum = sum(weight for weight, _ in terms if weight > 0)
    score = sum(weight * value for weight, value in terms if weight > 0) / weight_sum
    return float(score), {
        "NSE": float(nn),
        "KGE": float(kk),
        "PBIAS_percent": float(pp),
        "Recharge_model_mm_a": float(recharge_model_mean_mm_a),
        "Recharge_reference_mm_a": float(recharge_reference_mean_mm_a),
        "Recharge_PBIAS_percent": float(recharge_pbias),
        "Recharge_mean_penalty": float(recharge_mean_penalty),
        "Recharge_correlation": float(recharge_correlation),
    }

def calibrate_daily_v9(
    dates: pd.DatetimeIndex,
    p_forcing: np.ndarray,
    pet_forcing: np.ndarray,
    qobs: pd.Series,
    static: dict[str, np.ndarray | float],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    try:
        from scipy.optimize import differential_evolution
    except Exception:
        warnings.warn("scipy nicht verfügbar; realistische Tages-Defaultparameter werden verwendet.")
        defaults = make_param_dict_v9([
            0.30, 0.20, 0.0010, 0.12, 0.06, 0.0010, 0.65, 0.65, 0.50, 1.00, 1.00
        ])
        return defaults, pd.DataFrame(), pd.DataFrame([defaults])

    obs_aligned = qobs.reindex(dates).to_numpy(dtype="float64")
    calib_mask = (
        (dates >= pd.Timestamp(DCFG.CALIB_START))
        & (dates <= pd.Timestamp(DCFG.CALIB_END))
        & np.isfinite(obs_aligned)
    )
    if int(calib_mask.sum()) < 30:
        raise ValueError("Zu wenige Qobs-Werte im täglichen Kalibrierungszeitraum.")

    records: list[dict] = []

    def evaluate(x) -> tuple[float, dict[str, float]]:
        params = make_param_dict_v9(x)
        simulation = _simulate_calibration_sample_v9(
            dates, p_forcing, pet_forcing, static, params
        )
        qsim = simulation["qsim"].to_numpy(dtype="float64")
        return _calibration_score_v9(
            obs_aligned[calib_mask],
            qsim[calib_mask],
            float(simulation["recharge_mean_mm_a"]),
            float(simulation["recharge_reference_mean_mm_a"]),
            float(simulation["recharge_correlation"]),
        )

    def objective(x):
        score, _ = evaluate(x)
        return score

    def callback(xk, convergence):
        score, metrics = evaluate(xk)
        row = {
            "stage": "global_sample",
            "generation": len(records) + 1,
            "objective": score,
            "convergence": float(convergence),
            **metrics,
            **make_param_dict_v9(xk),
        }
        records.append(row)
        print(
            f"     Generation {len(records):02d}: Obj={score:.4f}, "
            f"NSE={metrics['NSE']:.3f}, KGE={metrics['KGE']:.3f}, "
            f"Q-PBIAS={metrics['PBIAS_percent']:.1f}%, "
            f"GWN={metrics['Recharge_model_mm_a']:.1f} mm/a, "
            f"GWN-PBIAS={metrics['Recharge_PBIAS_percent']:.1f}%, "
            f"r_GWN={metrics['Recharge_correlation']:.3f}"
        )
        return False

    print("\n[INFO] V9.1 Mehrzielkalibrierung: Tagesabfluss + Grundwasserneubildung")
    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS_V9,
        seed=DCFG.RANDOM_SEED,
        maxiter=DCFG.CALIBRATION_MAXITER,
        popsize=DCFG.CALIBRATION_POPSIZE,
        polish=False,
        updating="immediate",
        workers=1,
        tol=DCFG.CALIBRATION_TOL,
        callback=callback,
    )
    best = np.asarray(result.x, dtype="float64")
    best_score, best_metrics = evaluate(best)

    local_records: list[dict] = []
    fraction = float(DCFG.LOCAL_REFINEMENT_FRACTION)
    for j, name in enumerate(PARAM_NAMES_V9):
        lo, hi = PARAM_BOUNDS_V9[j]
        for direction in (-1.0, 1.0):
            trial = best.copy()
            trial[j] = np.clip(best[j] + direction * fraction * (hi - lo), lo, hi)
            if np.isclose(trial[j], best[j]):
                continue
            score, metrics = evaluate(trial)
            local_records.append({
                "stage": "local_refinement",
                "parameter": name,
                "direction": int(direction),
                "objective": score,
                **metrics,
                **make_param_dict_v9(trial),
            })
            if score < best_score:
                best, best_score, best_metrics = trial, score, metrics
                print(
                    f"     Verbesserung {name}: Obj={best_score:.4f}, "
                    f"NSE={best_metrics['NSE']:.3f}, "
                    f"GWN-PBIAS={best_metrics['Recharge_PBIAS_percent']:.1f}%, "
                    f"r_GWN={best_metrics['Recharge_correlation']:.3f}"
                )

    params = make_param_dict_v9(best)
    print("[OK] Kalibrierte tägliche V9.1-Parameter:")
    for key, value in params.items():
        print(f"     {key}: {value:.6f}")
    print(
        f"     Kalibrierung: NSE={best_metrics['NSE']:.3f}, "
        f"KGE={best_metrics['KGE']:.3f}, "
        f"Q-PBIAS={best_metrics['PBIAS_percent']:.1f}%, "
        f"GWN={best_metrics['Recharge_model_mm_a']:.1f} mm/a, "
        f"GWN-PBIAS={best_metrics['Recharge_PBIAS_percent']:.1f}%, "
        f"r_GWN={best_metrics['Recharge_correlation']:.3f}"
    )
    trace = pd.concat(
        [pd.DataFrame(records), pd.DataFrame(local_records)],
        ignore_index=True,
        sort=False,
    )
    summary = pd.DataFrame([{"objective": best_score, **best_metrics, **params}])
    return params, trace, summary

def run_daily_model_v9(
    dates: pd.DatetimeIndex,
    precip: NetCDFDailyPrecipProvider,
    pet: RasterDailyPETProvider,
    profile: dict,
    network: dict,
    fc_raster: np.ndarray,
    slope_factor_raster: np.ndarray,
    landuse: dict[str, np.ndarray],
    soil_units: dict[str, np.ndarray],
    gauge_routings: dict[str, dict],
    primary_key: str,
    params: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Vollständige tägliche V9.1-Simulation auf allen Modellzellen."""
    rows = network["rows"]
    cols = network["cols"]
    n = len(rows)
    x, y = _cell_centers(profile, rows, cols)
    p_map = precip.build_mapping(x, y, profile["crs"])
    pet_map = pet.build_mapping(x, y, profile["crs"])

    fc = fc_raster[rows, cols].astype("float64")
    slope_q = slope_factor_raster[rows, cols].astype("float64")
    land_q = landuse["quickflow_factor"][rows, cols].astype("float64")
    land_pet = landuse["pet_factor"][rows, cols].astype("float64")
    land_perc = landuse["percolation_factor"][rows, cols].astype("float64")
    impervious = np.clip(
        landuse["sealed_percent"][rows, cols] / 100.0, 0.0, 1.0
    )
    soil_q = soil_units["quickflow_factor"][rows, cols].astype("float64")
    soil_perc = soil_units["percolation_factor"][rows, cols].astype("float64")
    soil_slow = soil_units["slow_recharge_factor"][rows, cols].astype("float64")
    alpha, beta, slow = _daily_spatial_grids(
        params, slope_q, land_q, land_perc, soil_q, soil_perc, soil_slow
    )
    nested_zone_codes_compact, _ = _nested_zone_codes_v94(gauge_routings, network)
    regional_p_factor = _regional_precip_factors_v94(
        nested_zone_codes_compact, params
    )

    soil = fc * params["init_soil_frac"]
    interflow_store = np.zeros(n, dtype="float64")
    gw_fast = np.zeros(n, dtype="float64")
    gw_slow = np.zeros(n, dtype="float64")
    cell_area = float(abs(profile["transform"].a * profile["transform"].e))

    # Konsistenter Spin-up wie in der Kalibrierung.
    warm_dates = dates[dates <= pd.Timestamp(DCFG.WARMUP_END)]
    for cycle in range(max(0, int(DCFG.SPINUP_CYCLES) - 1)):
        for j, date in enumerate(warm_dates):
            p_grid_raw = _forcing_fill(precip.read(date, p_map), "Niederschlag", date)
            p_grid = p_grid_raw * regional_p_factor
            pet_grid = _forcing_fill(pet.read(date, pet_map), "PET", date)
            step = water_balance_step_daily_v9(
                p_grid, pet_grid, fc, soil, interflow_store, gw_fast, gw_slow,
                alpha, beta, slow, land_pet, impervious, params,
            )
            soil = step["soil"]
            interflow_store = step["interflow_store"]
            gw_fast, gw_slow = step["gw_fast"], step["gw_slow"]
        print(
            f"     Spin-up-Zyklus {cycle + 1}/{max(0, int(DCFG.SPINUP_CYCLES) - 1)} "
            f"abgeschlossen ({len(warm_dates)} Tage)"
        )

    lag_infos = {
        key: prepare_daily_routing_lags(
            routing["travel_cost"], params["routing_velocity_m_s"]
        )
        for key, routing in gauge_routings.items()
    }
    arrivals = {
        key: np.zeros(
            len(dates) + DCFG.MAX_ROUTING_LAG_DAYS + 1, dtype="float64"
        )
        for key in gauge_routings
    }

    map_names = (
        "precip_sum_mm", "pet_sum_mm", "aet_sum_mm",
        "impervious_runoff_sum_mm", "saturation_quickflow_sum_mm",
        "quickflow_sum_mm", "interflow_input_sum_mm", "interflow_sum_mm",
        "baseflow_sum_mm", "baseflow_fast_sum_mm", "baseflow_slow_sum_mm",
        "percolation_sum_mm", "recharge_from_excess_sum_mm",
        "recharge_sum_mm", "runoff_sum_mm", "water_balance_residual_sum_mm",
    )
    sums = {name: np.zeros(n, dtype="float64") for name in map_names}
    primary_ids = gauge_routings[primary_key]["upstream_ids"].astype(np.int32)
    rows_out: list[dict] = []

    for i, date in enumerate(dates):
        p_grid_raw = _forcing_fill(precip.read(date, p_map), "Niederschlag", date)
        p_grid = p_grid_raw * regional_p_factor
        pet_grid = _forcing_fill(pet.read(date, pet_map), "PET", date)
        step = water_balance_step_daily_v9(
            p_grid, pet_grid, fc, soil, interflow_store, gw_fast, gw_slow,
            alpha, beta, slow, land_pet, impervious, params,
        )
        soil = step["soil"]
        interflow_store = step["interflow_store"]
        gw_fast, gw_slow = step["gw_fast"], step["gw_slow"]

        for key, routing in gauge_routings.items():
            ids = routing["upstream_ids"].astype(np.int32)
            _route_compact_daily(
                step["local_runoff"], ids, np.ones(len(ids)), cell_area,
                i, arrivals[key], lag_infos[key],
            )

        sums["precip_sum_mm"] += p_grid
        sums["pet_sum_mm"] += pet_grid
        sums["aet_sum_mm"] += step["aet"]
        sums["impervious_runoff_sum_mm"] += step["impervious_runoff"]
        sums["saturation_quickflow_sum_mm"] += step["saturation_quickflow"]
        sums["quickflow_sum_mm"] += step["quickflow"]
        sums["interflow_input_sum_mm"] += step["interflow_input"]
        sums["interflow_sum_mm"] += step["interflow"]
        sums["baseflow_sum_mm"] += step["baseflow"]
        sums["baseflow_fast_sum_mm"] += step["baseflow_fast"]
        sums["baseflow_slow_sum_mm"] += step["baseflow_slow"]
        sums["percolation_sum_mm"] += step["percolation"]
        sums["recharge_from_excess_sum_mm"] += step["recharge_from_excess"]
        sums["recharge_sum_mm"] += step["recharge"]
        sums["runoff_sum_mm"] += step["local_runoff"]
        sums["water_balance_residual_sum_mm"] += step["water_balance_residual"]

        rows_out.append({
            "date": date,
            "P_mean_primary_mm": float(np.mean(p_grid[primary_ids])),
            "PET_mean_primary_mm": float(np.mean(pet_grid[primary_ids])),
            "AET_mean_primary_mm": float(np.mean(step["aet"][primary_ids])),
            "Impervious_runoff_mean_primary_mm": float(np.mean(step["impervious_runoff"][primary_ids])),
            "Saturation_quickflow_mean_primary_mm": float(np.mean(step["saturation_quickflow"][primary_ids])),
            "Interflow_input_mean_primary_mm": float(np.mean(step["interflow_input"][primary_ids])),
            "Interflow_mean_primary_mm": float(np.mean(step["interflow"][primary_ids])),
            "Percolation_mean_primary_mm": float(np.mean(step["percolation"][primary_ids])),
            "Recharge_from_excess_mean_primary_mm": float(np.mean(step["recharge_from_excess"][primary_ids])),
            "Recharge_mean_primary_mm": float(np.mean(step["recharge"][primary_ids])),
            "Quickflow_mean_primary_mm": float(np.mean(step["quickflow"][primary_ids])),
            "Baseflow_fast_mean_primary_mm": float(np.mean(step["baseflow_fast"][primary_ids])),
            "Baseflow_slow_mean_primary_mm": float(np.mean(step["baseflow_slow"][primary_ids])),
            "Runoff_generated_mean_primary_mm": float(np.mean(step["local_runoff"][primary_ids])),
            "Soil_mean_primary_mm": float(np.mean(soil[primary_ids])),
            "Interflow_store_mean_primary_mm": float(np.mean(interflow_store[primary_ids])),
            "Groundwater_fast_mean_primary_mm": float(np.mean(gw_fast[primary_ids])),
            "Groundwater_slow_mean_primary_mm": float(np.mean(gw_slow[primary_ids])),
            "Water_balance_residual_primary_mm": float(np.mean(step["water_balance_residual"][primary_ids])),
        })
        if (i + 1) % DCFG.PROGRESS_EVERY_DAYS == 0 or i + 1 == len(dates):
            print(f"     Finale Tagesimulation V9.4: {i+1}/{len(dates)} Tage")

    for i, row in enumerate(rows_out):
        for key, arr in arrivals.items():
            row[f"Qsim_{key}_m3s"] = float(arr[i]) / 86400.0
        row["Qsim_m3s"] = row[f"Qsim_{primary_key}_m3s"]

    sums["soil_final_mm"] = soil
    sums["interflow_store_final_mm"] = interflow_store
    sums["gw_fast_final_mm"] = gw_fast
    sums["gw_slow_final_mm"] = gw_slow
    sums["alpha_fast_grid"] = alpha
    sums["beta_perc_daily_grid"] = beta
    sums["slow_recharge_fraction_grid"] = slow
    sums["impervious_fraction"] = impervious
    sums["nested_zone_code"] = nested_zone_codes_compact.astype("float64")
    sums["regional_precip_factor"] = regional_p_factor
    return pd.DataFrame(rows_out).set_index("date"), sums

def _compact_raster(values: np.ndarray, network: dict, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, np.nan, dtype="float64")
    out[network["rows"], network["cols"]] = np.asarray(values, dtype="float64")
    return out


def summarize_daily_performance_v9(results: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "calibration": (DCFG.CALIB_START, DCFG.CALIB_END),
        "validation": (DCFG.VALID_START, DCFG.VALID_END),
        "full": (DCFG.START, DCFG.END),
    }
    rows: list[dict] = []
    for name, (start, end) in periods.items():
        part = results.loc[start:end].dropna(subset=["Qobs_m3s", "Qsim_m3s"])
        rows.append({
            "period": name,
            "start": start,
            "end": end,
            "n_days": len(part),
            "NSE": nse(part["Qobs_m3s"], part["Qsim_m3s"]),
            "KGE": kge(part["Qobs_m3s"], part["Qsim_m3s"]),
            "PBIAS_percent": pbias(part["Qobs_m3s"], part["Qsim_m3s"]),
            "Qobs_mean_m3s": part["Qobs_m3s"].mean(),
            "Qsim_mean_m3s": part["Qsim_m3s"].mean(),
        })
    return pd.DataFrame(rows)


def make_annual_water_balance_daily_v9(
    results: pd.DataFrame,
    primary_area_km2: float,
) -> pd.DataFrame:
    df = results.copy()
    area_m2 = primary_area_km2 * 1e6
    df["Qobs_depth_mm"] = df["Qobs_m3s"] * 86400.0 / area_m2 * 1000.0
    df["Qsim_depth_mm"] = df["Qsim_m3s"] * 86400.0 / area_m2 * 1000.0
    df["year"] = df.index.year
    sum_cols = [
        "P_mean_primary_mm", "PET_mean_primary_mm", "AET_mean_primary_mm",
        "Impervious_runoff_mean_primary_mm",
        "Saturation_quickflow_mean_primary_mm",
        "Interflow_mean_primary_mm", "Percolation_mean_primary_mm",
        "Recharge_from_excess_mean_primary_mm", "Recharge_mean_primary_mm",
        "Runoff_generated_mean_primary_mm", "Qobs_depth_mm", "Qsim_depth_mm",
    ]
    available = [column for column in sum_cols if column in df.columns]
    annual = df.groupby("year")[available].sum(min_count=1).reset_index()
    annual["P_minus_AET_minus_Qsim_mm"] = (
        annual["P_mean_primary_mm"]
        - annual["AET_mean_primary_mm"]
        - annual["Qsim_depth_mm"]
    )
    return annual

def plot_q_timeseries_daily_v9(results: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(12, 5.5))
    plt.plot(results.index, results["Qobs_m3s"], label="Qobs", linewidth=1.0)
    plt.plot(results.index, results["Qsim_m3s"], label="Qsim", linewidth=1.0)
    plt.axvspan(pd.Timestamp(DCFG.CALIB_START), pd.Timestamp(DCFG.CALIB_END), alpha=0.08, label="Kalibrierung")
    plt.axvspan(pd.Timestamp(DCFG.VALID_START), pd.Timestamp(DCFG.VALID_END), alpha=0.08, label="Validierung")
    plt.ylabel("Q [m³/s]")
    plt.xlabel("Tag")
    plt.title("Beobachteter und simulierter Tagesabfluss – HydroMod v9")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_annual_water_balance_daily_v9(annual: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(annual))
    width = 0.18
    plt.figure(figsize=(10, 5))
    plt.bar(x - 1.5 * width, annual["P_mean_primary_mm"], width, label="Niederschlag")
    plt.bar(x - 0.5 * width, annual["AET_mean_primary_mm"], width, label="AET Modell")
    plt.bar(x + 0.5 * width, annual["Qobs_depth_mm"], width, label="Qobs")
    plt.bar(x + 1.5 * width, annual["Qsim_depth_mm"], width, label="Qsim")
    plt.xticks(x, annual["year"].astype(str))
    plt.ylabel("Jahressumme [mm/a]")
    plt.xlabel("Jahr")
    plt.title("Jährliche Wasserbilanz – tägliches HydroMod v9")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_daily_hausarbeit_bundle_v9(
    out: Path,
    performance: pd.DataFrame,
    params: dict[str, float],
    primary_area: float,
) -> None:
    haus = out / "hausarbeit"
    selections = {
        out / "plots" / "qobs_qsim_daily_v9.png": "Abb_01_Qobs_Qsim_Taeglich.png",
        out / "plots" / "annual_water_balance_daily_v9.png": "Abb_02_Jahreswasserbilanz.png",
        out / "plots" / "map_precip_annual_mean.png": "Abb_03_Niederschlag_Raeumlich.png",
        out / "plots" / "map_recharge_annual_mean.png": "Abb_04_Recharge.png",
        out / "plots" / "map_interflow_annual_mean.png": "Abb_05_Zwischenabfluss.png",
        out / "plots" / "map_aet_annual_mean.png": "Abb_06_AET.png",
        out / "tables" / "daily_performance_metrics.csv": "Tab_01_Performance_Taeglich.csv",
        out / "tables" / "calibrated_parameters_daily_v9.csv": "Tab_02_Parameter.csv",
        out / "tables" / "annual_water_balance_daily_v9.csv": "Tab_03_Jahreswasserbilanz.csv",
        out / "tables" / "spatial_validation_metrics.csv": "Tab_04_Raeumliche_Validierung.csv",
    }
    for source, target in selections.items():
        if source.exists():
            shutil.copy2(source, haus / target)
    readme = f"""HydroMod v9.1 – tägliches, räumlich verteiltes Modell

Primärer Pegel: Heidelbach (42880550)
D8-Pegeleinzugsgebiet: {primary_area:.3f} km²
Warm-up: {DCFG.START} bis {DCFG.WARMUP_END}; {DCFG.SPINUP_CYCLES} Zyklen
Kalibrierung: {DCFG.CALIB_START} bis {DCFG.CALIB_END}
Validierung: {DCFG.VALID_START} bis {DCFG.VALID_END}

V9.1 verwendet räumliche tägliche HYRAS-Niederschlagswerte und bilinear
interpolierte tägliche PET. Der nicht direkt abfließende Sättigungsüberschuss
wird nicht mehr vollständig als Recharge verbucht, sondern zwischen
Grundwasserneubildung und einem verzögerten Zwischenabflussspeicher aufgeteilt.
Die Kalibrierung berücksichtigt Tagesabfluss, mittlere Grundwasserneubildung
und deren räumliche Korrelation.

Kalibrierte Parameter:
{json.dumps(params, indent=2, ensure_ascii=False)}

Performance:
{performance.to_string(index=False)}

Einschränkungen:
- kein Schnee-/Temperaturmodul,
- Bodeneinheiten bleiben neutral, solange nur BN_ID ohne Legende vorliegt,
- die Recharge-Referenz kann einen anderen Bezugszeitraum besitzen,
- D8-Reisezeitrouting ist kein hydraulisches Flussmodell.
"""
    (haus / "README_Hausarbeit_Daily_V9_1.txt").write_text(readme, encoding="utf-8")

def main_v9_daily() -> None:
    out = daily_out_dir()
    print("============================================================")
    print("HydroMod v9.1: Interflow, Mehrzielkalibrierung, räumliche Tagesforcings")
    print("============================================================")
    print(f"Projektordner: {CFG.BASE_DIR}")
    print(f"Datenordner:   {data_dir()}")
    print(f"Output:        {out}")

    # 1. Räumliche Modelldomäne
    dgm_path = find_first(data_dir() / CFG.DGM_DIRNAME, [".tif", ".tiff"])
    dgm_full, profile_full = read_raster(dgm_path)
    catchment = load_catchment(profile_full["crs"])
    mask_full = create_mask(catchment, profile_full)
    dgm, mask, profile, crop_offset = crop_raster_to_mask(
        dgm_full, mask_full, profile_full, padding=CFG.CROP_PADDING_CELLS
    )
    del dgm_full, mask_full
    print(f"[OK] Modellraster: {profile['width']} x {profile['height']}; Offset={crop_offset}")

    # 2. Statische räumliche Daten
    fc = load_field_capacity_raster(profile, mask)
    river_mask, _ = load_river_mask(profile, mask)
    landuse = load_landuse_rasters(profile, mask, catchment)
    soil_units, soil_table = load_soil_unit_factors(profile, mask)
    soil_table.to_csv(
        out / "tables" / "soil_class_parameterization.csv",
        index=False, encoding="utf-8",
    )

    dgm = fill_missing_dem_nearest(dgm, mask)
    _, dgm_conditioned = condition_dem_priority_flood(
        dgm, mask, river_mask, profile
    )
    slope_fraction = calculate_slope_fraction(dgm_conditioned, mask, profile)
    slope_factor = make_slope_quickflow_factor(
        slope_fraction, mask, CFG.SLOPE_QUICKFLOW_WEIGHT
    )
    network = build_d8_flow_network(
        dgm_conditioned, mask, profile, river_mask
    )

    # 3. Pegel und D8-Pegeleinzugsgebiet
    gauges = load_gauge_points(profile["crs"])
    gauge_routings: dict[str, dict] = {}
    for _, gauge_row in gauges.iterrows():
        routing = build_gauge_routing(
            gauge_row, network, mask, profile, river_mask
        )
        gauge_routings[routing["key"]] = routing
    primary = next(r for r in gauge_routings.values() if r["is_primary"])
    primary_key = primary["key"]
    primary_mask = primary["upstream_mask"]
    primary_area = primary["area_km2"]
    area_diff = 100.0 * (
        primary_area - CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2
    ) / CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2
    print(
        f"[INFO] Heidelbach-Fläche: Modell={primary_area:.3f} km², "
        f"offiziell={CFG.PRIMARY_GAUGE_OFFICIAL_AREA_KM2:.3f} km², "
        f"Abweichung={area_diff:+.2f}%"
    )

    # 4. Tagesforcings und Qobs
    precip = NetCDFDailyPrecipProvider(data_dir() / CFG.PRECIP_DIRNAME)
    try:
        pet = RasterDailyPETProvider(data_dir() / CFG.PET_DIRNAME)
        dates = _common_daily_dates(precip, pet)
        qobs = load_qobs_daily(CFG.QOBS_DIRNAME)

        # Groundwater-Recharge-Referenz wird bereits in der Kalibrierung genutzt.
        gwn_reference = None
        gwn_metadata = {
            "product": "groundwater_recharge",
            "status": "disabled",
        }
        if DCFG.RECHARGE_REFERENCE_ENABLED:
            gwn_reference, gwn_metadata = load_reference_annual_raster(
                CFG.GWN_REFERENCE_DIRNAME,
                ("grundwasser", "neubild"),
                profile,
                mask,
                CFG.GWN_REFERENCE_UNIT_FACTOR,
                "groundwater_recharge_calibration",
                None,
            )
            if gwn_reference is None:
                warnings.warn(
                    "Recharge-Referenz nicht verfügbar. Kalibrierung läuft nur mit Qobs."
                )
            else:
                valid_gwn = primary_mask & np.isfinite(gwn_reference)
                print(
                    f"[OK] Recharge-Kalibrierungsziel Heidelbach: "
                    f"{float(np.nanmean(gwn_reference[valid_gwn])):.1f} mm/a, "
                    f"{int(valid_gwn.sum())} Zellen"
                )
        pd.DataFrame([gwn_metadata]).to_csv(
            out / "tables" / "calibration_recharge_reference_metadata.csv",
            index=False, encoding="utf-8",
        )

        # 5. Räumlich geschichtete Kalibrierungsstichprobe
        primary_ids = primary["upstream_ids"].astype(np.int32)
        pr = primary["rows"]
        pc = primary["cols"]
        fc_primary = fc[pr, pc].astype("float64")
        slope_primary = slope_factor[pr, pc].astype("float64")
        land_q_primary = landuse["quickflow_factor"][pr, pc].astype("float64")
        land_perc_primary = landuse["percolation_factor"][pr, pc].astype("float64")
        travel_primary = primary["travel_cost"].astype("float64")
        selected, weights = _stratified_sample_indices_v9(
            (
                fc_primary,
                slope_primary,
                land_q_primary,
                land_perc_primary,
                np.log1p(travel_primary),
            ),
            DCFG.CALIBRATION_SAMPLE_CELLS,
            DCFG.RANDOM_SEED,
        )
        sample_rows = pr[selected]
        sample_cols = pc[selected]
        sx, sy = _cell_centers(profile, sample_rows, sample_cols)
        p_sample, pet_sample = _build_calibration_forcing_cache(
            dates, precip, pet, sx, sy, profile["crs"], primary_ids[selected], out
        )
        if gwn_reference is not None:
            recharge_reference_sample = gwn_reference[sample_rows, sample_cols]
            if int(np.isfinite(recharge_reference_sample).sum()) < int(
                DCFG.MIN_RECHARGE_REFERENCE_CELLS
            ):
                warnings.warn(
                    "Zu wenige gültige Recharge-Referenzzellen in der Stichprobe; "
                    "Recharge-Terme werden für diesen Lauf deaktiviert."
                )
                recharge_reference_sample = np.full(len(selected), np.nan)
        else:
            recharge_reference_sample = np.full(len(selected), np.nan)

        static_sample: dict[str, np.ndarray | float] = {
            "fc": fc_primary[selected],
            "slope_q": slope_primary[selected],
            "land_q": land_q_primary[selected],
            "pet_factor": landuse["pet_factor"][pr, pc][selected],
            "land_perc": land_perc_primary[selected],
            "impervious_fraction": np.clip(
                landuse["sealed_percent"][pr, pc][selected] / 100.0, 0.0, 1.0
            ),
            "soil_q": soil_units["quickflow_factor"][pr, pc][selected],
            "soil_perc": soil_units["percolation_factor"][pr, pc][selected],
            "soil_slow": soil_units["slow_recharge_factor"][pr, pc][selected],
            "travel_cost": travel_primary[selected],
            "weights": weights,
            "cell_area_m2": float(abs(profile["transform"].a * profile["transform"].e)),
            "recharge_reference_mm_a": recharge_reference_sample,
        }
        print(
            f"[INFO] Tageskalibrierung: {len(selected)} gewichtete Zellen "
            f"repräsentieren {len(primary_ids)} Pegelgebietszellen."
        )

        if DCFG.USE_CALIBRATION:
            params, trace, calibration_summary = calibrate_daily_v9(
                dates, p_sample, pet_sample, qobs, static_sample
            )
        else:
            params = make_param_dict_v9([
                0.30, 0.20, 0.0010, 0.12, 0.06, 0.0010, 0.65, 0.65, 0.50, 1.00, 1.00
            ])
            trace = pd.DataFrame()
            calibration_summary = pd.DataFrame([params])

        trace.to_csv(
            out / "tables" / "calibration_trace_daily_v9.csv",
            index=False, encoding="utf-8",
        )
        calibration_summary.to_csv(
            out / "tables" / "calibration_summary_daily_v9.csv",
            index=False, encoding="utf-8",
        )
        parameter_output = {
            "model_version": "9.1_realistic",
            **params,
            "spinup_cycles": DCFG.SPINUP_CYCLES,
            "percolation_wetness_exponent": DCFG.PERCOLATION_WETNESS_EXPONENT,
            "objective_weight_nse": DCFG.OBJECTIVE_WEIGHT_NSE,
            "objective_weight_kge": DCFG.OBJECTIVE_WEIGHT_KGE,
            "objective_weight_pbias": DCFG.OBJECTIVE_WEIGHT_PBIAS,
            "objective_weight_log_nse": DCFG.OBJECTIVE_WEIGHT_LOG_NSE,
            "objective_weight_lowflow_pbias": DCFG.OBJECTIVE_WEIGHT_LOWFLOW_PBIAS,
            "objective_weight_summer_pbias": DCFG.OBJECTIVE_WEIGHT_SUMMER_PBIAS,
            "objective_weight_incremental_pbias": DCFG.OBJECTIVE_WEIGHT_INCREMENTAL_PBIAS,
            "objective_weight_recharge_mean": DCFG.OBJECTIVE_WEIGHT_RECHARGE_MEAN,
            "objective_weight_recharge_correlation": DCFG.OBJECTIVE_WEIGHT_RECHARGE_CORRELATION,
            "recharge_reference_tolerance_fraction": DCFG.RECHARGE_REFERENCE_TOLERANCE_FRACTION,
        }
        pd.DataFrame([parameter_output]).to_csv(
            out / "tables" / "calibrated_parameters_daily_v9.csv",
            index=False, encoding="utf-8",
        )

        # 6. Finale tägliche Simulation auf allen Modellzellen
        sim, compact_maps = run_daily_model_v9(
            dates, precip, pet, profile, network, fc, slope_factor,
            landuse, soil_units, gauge_routings, primary_key, params,
        )
        results = sim.copy()
        results["Qobs_m3s"] = qobs.reindex(results.index)
        results.to_csv(
            out / "tables" / "daily_model_results.csv", encoding="utf-8"
        )

        performance = summarize_daily_performance_v9(results)
        performance.to_csv(
            out / "tables" / "daily_performance_metrics.csv",
            index=False, encoding="utf-8",
        )
        annual = make_annual_water_balance_daily_v9(results, primary_area)
        annual.to_csv(
            out / "tables" / "annual_water_balance_daily_v9.csv",
            index=False, encoding="utf-8",
        )

        # 7. Kartenoutputs
        n_years = len(set(dates.year))
        map_arrays = {
            name: _compact_raster(values, network, mask.shape)
            for name, values in compact_maps.items()
        }
        outputs = {
            "field_capacity_mm.tif": fc,
            "dgm_conditioned_m.tif": dgm_conditioned,
            "slope_percent.tif": slope_fraction * 100.0,
            "river_mask.tif": np.where(mask, river_mask.astype(float), np.nan),
            "primary_gauge_upstream_mask.tif": np.where(mask, primary_mask.astype(float), np.nan),
            "flow_accumulation_cells.tif": network["accumulation_raster"],
            "precip_sum_2021_2025_mm.tif": map_arrays["precip_sum_mm"],
            "pet_sum_2021_2025_mm.tif": map_arrays["pet_sum_mm"],
            "aet_sum_2021_2025_mm.tif": map_arrays["aet_sum_mm"],
            "impervious_runoff_sum_2021_2025_mm.tif": map_arrays["impervious_runoff_sum_mm"],
            "saturation_quickflow_sum_2021_2025_mm.tif": map_arrays["saturation_quickflow_sum_mm"],
            "interflow_sum_2021_2025_mm.tif": map_arrays["interflow_sum_mm"],
            "quickflow_sum_2021_2025_mm.tif": map_arrays["quickflow_sum_mm"],
            "baseflow_sum_2021_2025_mm.tif": map_arrays["baseflow_sum_mm"],
            "percolation_sum_2021_2025_mm.tif": map_arrays["percolation_sum_mm"],
            "recharge_from_excess_sum_2021_2025_mm.tif": map_arrays["recharge_from_excess_sum_mm"],
            "recharge_sum_2021_2025_mm.tif": map_arrays["recharge_sum_mm"],
            "runoff_sum_2021_2025_mm.tif": map_arrays["runoff_sum_mm"],
            "precip_annual_mean_mm_a.tif": map_arrays["precip_sum_mm"] / n_years,
            "pet_annual_mean_mm_a.tif": map_arrays["pet_sum_mm"] / n_years,
            "aet_annual_mean_mm_a.tif": map_arrays["aet_sum_mm"] / n_years,
            "interflow_annual_mean_mm_a.tif": map_arrays["interflow_sum_mm"] / n_years,
            "percolation_annual_mean_mm_a.tif": map_arrays["percolation_sum_mm"] / n_years,
            "recharge_annual_mean_mm_a.tif": map_arrays["recharge_sum_mm"] / n_years,
            "runoff_annual_mean_mm_a.tif": map_arrays["runoff_sum_mm"] / n_years,
            "soil_final_mm.tif": map_arrays["soil_final_mm"],
            "interflow_store_final_mm.tif": map_arrays["interflow_store_final_mm"],
            "groundwater_fast_final_mm.tif": map_arrays["gw_fast_final_mm"],
            "groundwater_slow_final_mm.tif": map_arrays["gw_slow_final_mm"],
            "alpha_fast_spatial.tif": map_arrays["alpha_fast_grid"],
            "beta_perc_daily_spatial.tif": map_arrays["beta_perc_daily_grid"],
            "slow_recharge_fraction_spatial.tif": map_arrays["slow_recharge_fraction_grid"],
            "impervious_fraction.tif": map_arrays["impervious_fraction"],
            "nested_reach_zone_code.tif": map_arrays["nested_zone_code"],
            "regional_precipitation_factor.tif": map_arrays["regional_precip_factor"],
            "water_balance_residual_sum_mm.tif": map_arrays["water_balance_residual_sum_mm"],
        }
        cell_area = abs(profile["transform"].a * profile["transform"].e)
        outputs["flow_accumulation_area_km2.tif"] = (
            network["accumulation_raster"] * cell_area / 1e6
        )
        lag_primary = prepare_daily_routing_lags(
            primary["travel_cost"], params["routing_velocity_m_s"]
        )
        outputs["primary_routing_travel_time_days.tif"] = compact_to_raster(
            lag_primary["travel_days"], primary["rows"], primary["cols"], mask.shape
        )
        outputs["primary_routing_lag_days.tif"] = compact_to_raster(
            lag_primary["lag_days"], primary["rows"], primary["cols"], mask.shape
        )
        for filename, array in outputs.items():
            write_geotiff(out / "maps" / filename, array, profile)

        primary_catchment = mask_to_geodataframe(
            primary_mask, profile, "Heidelbach_D8_upstream"
        )
        primary_path = out / "maps" / "primary_gauge_catchment.gpkg"
        if primary_path.exists():
            primary_path.unlink()
        primary_catchment.to_file(
            primary_path, layer="heidelbach_upstream", driver="GPKG"
        )

        # 8. Räumliche Referenzvergleiche
        validation_maps = {
            "recharge_sum_mm": map_arrays["recharge_sum_mm"],
            "percolation_sum_mm": map_arrays["percolation_sum_mm"],
            "aet_sum_mm": map_arrays["aet_sum_mm"],
        }
        spatial_metrics, ref_metadata, _ = validate_spatial_products_v8(
            validation_maps, profile, mask, primary_mask, n_years, out
        )
        spatial_metrics.to_csv(
            out / "tables" / "spatial_validation_metrics.csv",
            index=False, encoding="utf-8",
        )
        ref_metadata.to_csv(
            out / "tables" / "reference_product_metadata.csv",
            index=False, encoding="utf-8",
        )

        # 9. Plots
        plot_q_timeseries_daily_v9(
            results, out / "plots" / "qobs_qsim_daily_v9.png"
        )
        plot_annual_water_balance_daily_v9(
            annual, out / "plots" / "annual_water_balance_daily_v9.png"
        )
        plot_map(
            outputs["precip_annual_mean_mm_a.tif"],
            "Mittlerer räumlicher Niederschlag 2021–2025",
            out / "plots" / "map_precip_annual_mean.png", "mm/a",
        )
        plot_map(
            outputs["recharge_annual_mean_mm_a.tif"],
            "Mittlere simulierte Recharge V9.1",
            out / "plots" / "map_recharge_annual_mean.png", "mm/a",
        )
        plot_map(
            outputs["interflow_annual_mean_mm_a.tif"],
            "Mittlerer simulierter Zwischenabfluss V9.1",
            out / "plots" / "map_interflow_annual_mean.png", "mm/a",
        )
        plot_map(
            outputs["aet_annual_mean_mm_a.tif"],
            "Mittlere simulierte AET V9.1",
            out / "plots" / "map_aet_annual_mean.png", "mm/a",
        )

        write_daily_hausarbeit_bundle_v9(
            out, performance, params, primary_area
        )

        print("\n[OK] Tägliche Performance am Pegel Heidelbach:")
        print(performance.to_string(index=False))
        if not spatial_metrics.empty:
            print("\n[OK] Räumliche Plausibilisierung V9.1:")
            print(spatial_metrics.to_string(index=False))
        print("\n============================================================")
        print("[FERTIG] HydroMod v9.1 realistisch und räumlich verteilt")
        print(f"Ergebnisse: {out}")
        print("Wichtige Dateien:")
        print(" - tables/calibrated_parameters_daily_v9.csv")
        print(" - tables/calibration_summary_daily_v9.csv")
        print(" - tables/daily_performance_metrics.csv")
        print(" - tables/spatial_validation_metrics.csv")
        print(" - maps/recharge_annual_mean_mm_a.tif")
        print(" - maps/interflow_annual_mean_mm_a.tif")
        print("============================================================")
    finally:
        precip.close()

# =============================================================================
# HYDROMOD V9.3: FLÄCHENBEWUSSTES MULTI-PEGEL-MODELL
# =============================================================================

@dataclass(frozen=True)
class DailyGaugeSpec:
    """Pegelpunkt, amtliche Stammdaten, Qobs-Dateien und Kalibrierungsgewicht."""

    station_id: str
    name: str
    point_path: Optional[str]
    qobs_paths: tuple[str, ...]
    is_primary: bool = False
    calibration_weight: float = 1.0
    official_area_km2: Optional[float] = None
    official_x: Optional[float] = None
    official_y: Optional[float] = None
    official_crs: str = "EPSG:31467"
    required: bool = True


DAILY_GAUGE_SPECS: tuple[DailyGaugeSpec, ...] = (
    DailyGaugeSpec(
        station_id="42880550",
        name="Heidelbach",
        point_path="gauge_points/42880550_Heidelbach.gpkg",
        qobs_paths=("Heidelbach durchfluss_5csv",),
        is_primary=True,
        calibration_weight=1.0,
        official_area_km2=162.19,
        official_x=3518821.0,
        official_y=5629547.0,
    ),
    DailyGaugeSpec(
        station_id="42880800",
        name="Röllshausen",
        point_path="gauge_points/42880800_Roellshausen.gpkg",
        qobs_paths=(
            r"C:\Users\E\Downloads\42880800_Röllshausen_graph_table 2021.csv",
            r"C:\Users\E\Downloads\42880800_Röllshausen_graph_table 2022.csv",
            r"C:\Users\E\Downloads\42880800_Röllshausen_graph_table 2023.csv",
            r"C:\Users\E\Downloads\42880800_Röllshausen_graph_table 2024.csv",
            r"C:\Users\E\Downloads\42880800_Röllshausen_graph_table_2025.csv",
        ),
        calibration_weight=1.0,
        official_area_km2=250.0,
        official_x=3520260.0,
        official_y=5635495.0,
    ),
    DailyGaugeSpec(
        station_id="42880458",
        name="Alsfeld",
        point_path="gauge_points/42880458_Alsfeld.gpkg",
        qobs_paths=(
            r"C:\Users\E\Downloads\42880458_Alsfeld_graph_table (2)_2021.csv",
            r"C:\Users\E\Downloads\42880458_Alsfeld_graph_table (1)_2022.csv",
            r"C:\Users\E\Downloads\42880458_Alsfeld_graph_table_2023.csv",
            r"C:\Users\E\Downloads\42880458_Alsfeld_graph_table_2024.csv",
            r"C:\Users\E\Downloads\42880458_Alsfeld_graph_table_2025.csv",
        ),
        calibration_weight=1.0,
        official_area_km2=131.4,
        official_x=3520196.0,
        official_y=5624741.0,
    ),
)

# V9.4-Pegel- und Snapping-Sicherheit.
MULTIGAUGE_SAMPLE_CELLS_PER_GAUGE: int = 5000
MULTIGAUGE_REQUIRE_ALL_GAUGES: bool = True
V93_USE_OFFICIAL_GAUGE_COORDINATES: bool = True
V93_GAUGE_SNAP_MAX_RADIUS_M: float = 1500.0
V93_GAUGE_AREA_TOLERANCE_FRACTION: float = 0.15
V93_STRICT_GAUGE_AREA_CHECK: bool = True
V93_AREA_ERROR_WEIGHT: float = 5.0
V93_DISTANCE_WEIGHT: float = 0.25
V93_NON_RIVER_PENALTY: float = 0.20
V93_RIVER_MASK_BONUS: float = 0.75
V93_LARGE_SNAP_WARNING_M: float = 500.0
V93_GAUGE_COORDINATE_WARNING_M: float = 50.0
V93_SNAP_DIAGNOSTIC_TOP_N: int = 20
V93_PRECHECK_ONLY: bool = False
V93_WRITE_MEAN_DISCHARGE_RASTER: bool = True
V93_QOBS_RUNOFF_DEPTH_MIN_MM_A: float = 20.0
V93_QOBS_RUNOFF_DEPTH_MAX_MM_A: float = 1500.0


def daily_out_dir() -> Path:
    """Eigener Ergebnisordner, damit V9.1-Ergebnisse nicht überschrieben werden."""
    p = CFG.BASE_DIR / "results" / "daily_v9_4_nested_gauge_diagnostic_calibration"
    for sub in ("maps", "plots", "tables", "cache", "hausarbeit"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def _resolve_external_or_project_path(raw_path: str | Path) -> Path:
    """
    Löst absolute, skript-relative und projekt-relative Eingaben robust auf.

    Die Funktion funktioniert sowohl als normale Python-Datei als auch bei
    direkter Ausführung des vollständigen Codes in Jupyter, wo ``__file__``
    nicht definiert ist.
    """
    raw = Path(raw_path).expanduser()

    # Absolute Pfade unverändert behandeln. Das ist für die konfigurierten
    # Windows-Pfade der Pegeldateien der Normalfall.
    if raw.is_absolute():
        return raw

    search_roots: list[Path] = []

    # Aktuelles Arbeitsverzeichnis: besonders wichtig für Jupyter.
    try:
        search_roots.append(Path.cwd().resolve())
    except OSError:
        search_roots.append(Path.cwd())

    # Verzeichnis der Python-Datei nur verwenden, wenn __file__ existiert.
    file_value = globals().get("__file__")
    if file_value:
        try:
            search_roots.append(Path(file_value).expanduser().resolve().parent)
        except (TypeError, OSError, RuntimeError):
            pass

    # Projektpfade des Modells ergänzen. Fehler einzelner Hilfsfunktionen
    # dürfen die Pfadauflösung nicht abbrechen.
    for root_getter in (
        lambda: data_dir(),
        lambda: CFG.BASE_DIR,
    ):
        try:
            value = root_getter()
            if value is not None:
                search_roots.append(Path(value).expanduser().resolve())
        except (AttributeError, NameError, OSError, RuntimeError, TypeError):
            pass

    # Reihenfolge erhalten und doppelte Suchwurzeln entfernen.
    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in search_roots:
        key = str(root).casefold()
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)

    # Direkte relative Kandidaten zuerst prüfen.
    direct_candidates = [raw] + [root / raw for root in unique_roots]
    for candidate in direct_candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue

    # Danach nur anhand des Dateinamens in den bekannten Projektwurzeln
    # suchen. Das erhält das bisherige Verhalten bei verschobenen Dateien.
    if raw.name:
        for root in unique_roots:
            try:
                if not root.exists():
                    continue
                matches = sorted(root.rglob(raw.name))
                if matches:
                    return matches[0]
            except (OSError, PermissionError):
                continue

    # Verständlicher Rückfallpfad; die aufrufende Funktion erzeugt bei einer
    # fehlenden Datei anschließend die konkrete Warnung oder Fehlermeldung.
    return raw


def _official_point_for_spec(spec: DailyGaugeSpec, ref_crs):
    if spec.official_x is None or spec.official_y is None:
        return None
    official = gpd.GeoSeries(
        [Point(float(spec.official_x), float(spec.official_y))],
        crs=spec.official_crs,
    ).to_crs(ref_crs)
    return official.iloc[0]


def _read_single_gauge_point(spec: DailyGaugeSpec, ref_crs) -> Optional[gpd.GeoDataFrame]:
    """Lädt einen Punkt, vergleicht ihn mit amtlichen Koordinaten und checkpointet WAL."""
    official_point = _official_point_for_spec(spec, ref_crs)
    configured_point = None
    source_path = ""
    point_path = _resolve_external_or_project_path(spec.point_path) if spec.point_path else None

    if point_path is not None and point_path.exists():
        try:
            layer = _choose_point_layer(point_path, None)
            gdf = gpd.read_file(point_path, layer=layer)
            if gdf.crs is None:
                warnings.warn(f"{point_path.name} hat kein CRS; EPSG:25832 wird angenommen.")
                gdf = gdf.set_crs("EPSG:25832")
            gdf = gdf.to_crs(ref_crs)
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
            if not gdf.empty:
                if len(gdf) > 1:
                    warnings.warn(
                        f"{point_path.name} enthält {len(gdf)} Objekte; "
                        f"für {spec.name} wird das erste verwendet."
                    )
                geometry = gdf.geometry.iloc[0]
                if geometry.geom_type not in ("Point", "MultiPoint"):
                    geometry = geometry.centroid
                if geometry.geom_type == "MultiPoint":
                    geometry = geometry.centroid
                configured_point = geometry
                source_path = str(point_path)
        except Exception as exc:
            warnings.warn(
                f"{spec.name}: Pegeldatei {point_path} konnte nicht sicher gelesen werden "
                f"({exc}). Amtliche Koordinate wird verwendet."
            )
    elif spec.point_path:
        warnings.warn(
            f"{spec.name}: konfigurierte Pegeldatei nicht gefunden: {point_path}. "
            "Amtliche Koordinate wird verwendet."
        )

    if configured_point is None and official_point is None:
        message = f"Für {spec.name} ist weder ein gültiger Punkt noch eine amtliche Koordinate verfügbar."
        if spec.required and MULTIGAUGE_REQUIRE_ALL_GAUGES:
            raise FileNotFoundError(message)
        warnings.warn(message)
        return None

    configured_to_official = np.nan
    if configured_point is not None and official_point is not None:
        configured_to_official = float(configured_point.distance(official_point))
        if configured_to_official > V93_GAUGE_COORDINATE_WARNING_M:
            warnings.warn(
                f"{spec.name}: konfigurierte Punktgeometrie liegt "
                f"{configured_to_official:.1f} m von der amtlichen Koordinate entfernt."
            )

    if V93_USE_OFFICIAL_GAUGE_COORDINATES and official_point is not None:
        selected_point = official_point
        point_source = "official_coordinate"
    elif configured_point is not None:
        selected_point = configured_point
        point_source = "configured_gpkg"
    else:
        selected_point = official_point
        point_source = "official_coordinate_fallback"

    # Optional: WAL/SHM durch Schreiben einer bereinigten Einpunktdatei im
    # Ergebnisordner vermeiden. Das Original wird nicht verändert.
    record = gpd.GeoDataFrame(
        [{
            "gauge_name": spec.name,
            "station_id": spec.station_id,
            "gauge_key": _safe_slug(f"{spec.station_id}_{spec.name}"),
            "is_primary": bool(spec.is_primary),
            "calibration_weight": float(spec.calibration_weight),
            "official_area_km2": (
                float(spec.official_area_km2)
                if spec.official_area_km2 is not None else np.nan
            ),
            "source_path": source_path or "embedded_official_coordinate",
            "point_source": point_source,
            "configured_point_x": (
                float(configured_point.x) if configured_point is not None else np.nan
            ),
            "configured_point_y": (
                float(configured_point.y) if configured_point is not None else np.nan
            ),
            "official_point_x": (
                float(official_point.x) if official_point is not None else np.nan
            ),
            "official_point_y": (
                float(official_point.y) if official_point is not None else np.nan
            ),
            "configured_to_official_distance_m": configured_to_official,
            "geometry": selected_point,
        }],
        geometry="geometry",
        crs=ref_crs,
    )
    return record


def load_gauge_points(ref_crs) -> gpd.GeoDataFrame:
    """Lädt Heidelbach, Röllshausen und Alsfeld aus getrennten GeoPackages."""
    parts: list[gpd.GeoDataFrame] = []
    for spec in DAILY_GAUGE_SPECS:
        point = _read_single_gauge_point(spec, ref_crs)
        if point is not None:
            parts.append(point)
    if not parts:
        raise ValueError("Keiner der konfigurierten Pegelpunkte konnte geladen werden.")

    gauges = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs=ref_crs
    )
    if gauges["station_id"].duplicated().any():
        duplicates = gauges.loc[gauges["station_id"].duplicated(), "station_id"].tolist()
        raise ValueError(f"Doppelte Stations-IDs in der Pegelkonfiguration: {duplicates}")
    if int(gauges["is_primary"].sum()) != 1:
        raise ValueError("Genau ein Pegel muss als is_primary=True markiert sein.")

    print("[OK] Konfigurierte Tagespegel:")
    print(
        gauges[["gauge_name", "station_id", "is_primary", "point_source", "configured_to_official_distance_m"]]
        .to_string(index=False)
    )
    return gauges


def _collect_qobs_files(paths: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    missing: list[str] = []
    for raw in paths:
        path = _resolve_external_or_project_path(raw)
        if path.is_dir():
            found = find_files(path, [".csv", ".txt"])
            if not found:
                missing.append(str(path))
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            missing.append(str(path))
    unique = list(dict.fromkeys(files))
    if missing:
        raise FileNotFoundError(
            "Folgende Qobs-Eingaben wurden nicht gefunden:\n - " + "\n - ".join(missing)
        )
    if not unique:
        raise FileNotFoundError("Keine Qobs-CSV/TXT-Dateien gefunden.")
    return unique


def _parse_qobs_value_column(df: pd.DataFrame, date_col: str) -> tuple[str, pd.Series]:
    candidates = [c for c in df.columns if c != date_col]
    if not candidates:
        raise ValueError("Qobs-Datei besitzt neben dem Datum keine Wertespalte.")

    preferred_tokens = (
        "durchfluss", "abfluss", "discharge", "m3/s", "m³/s", "m3", "m³", "q"
    )
    scored: list[tuple[float, str, pd.Series]] = []
    for col in candidates:
        raw = df[col]
        numeric = pd.to_numeric(
            raw.astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
        extracted = extract_q_mean_from_min_mean_max(raw)
        values = numeric if numeric.notna().sum() >= extracted.notna().sum() else extracted
        token_score = sum(token in str(col).lower() for token in preferred_tokens)
        finite_share = float(values.notna().mean()) if len(values) else 0.0
        # Durchflussbegriffe dominieren; danach gewinnt die am besten lesbare Spalte.
        score = 10.0 * token_score + finite_share
        scored.append((score, str(col), values.astype("float64")))
    scored.sort(key=lambda item: item[0], reverse=True)
    _, column, values = scored[0]
    return column, values


def load_qobs_daily_for_spec(
    spec: DailyGaugeSpec,
) -> tuple[pd.Series, pd.DataFrame]:
    """Liest alle Jahresdateien eines Pegels und dokumentiert die Datenabdeckung."""
    files = _collect_qobs_files(spec.qobs_paths)
    parts: list[pd.Series] = []
    diagnostics: list[dict] = []

    for path in files:
        df = read_csv_auto(path)
        date_col = choose_date_col(df, CFG.QOBS_DATE_COLUMN)
        if CFG.QOBS_VALUE_COLUMN is not None:
            if CFG.QOBS_VALUE_COLUMN not in df.columns:
                raise ValueError(
                    f"{path.name}: QOBS_VALUE_COLUMN={CFG.QOBS_VALUE_COLUMN!r} fehlt."
                )
            value_col = CFG.QOBS_VALUE_COLUMN
            raw = df[value_col]
            values = pd.to_numeric(
                raw.astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
            if values.notna().sum() < max(3, int(0.5 * raw.notna().sum())):
                values = extract_q_mean_from_min_mean_max(raw)
        else:
            value_col, values = _parse_qobs_value_column(df, date_col)

        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dt.normalize()
        series = pd.Series(values.to_numpy(), index=dates, name="Qobs_m3s")
        series = series[np.isfinite(series.to_numpy(dtype="float64"))]
        series = series[series.index.notna()].sort_index()
        # Negative Durchflüsse sind fachlich ungültig.
        negative = int((series < 0).sum())
        if negative:
            warnings.warn(f"{path.name}: {negative} negative Durchflusswerte werden entfernt.")
            series = series[series >= 0]
        parts.append(series)
        diagnostics.append({
            "station_id": spec.station_id,
            "gauge_name": spec.name,
            "source_file": str(path),
            "date_column": date_col,
            "value_column": value_col,
            "n_rows": len(df),
            "n_valid": len(series),
            "start": series.index.min() if len(series) else pd.NaT,
            "end": series.index.max() if len(series) else pd.NaT,
        })

    if not parts:
        raise ValueError(f"Keine gültigen Qobs-Werte für {spec.name} gelesen.")
    qobs = pd.concat(parts).sort_index()
    duplicate_count = int(qobs.index.duplicated(keep="first").sum())
    if duplicate_count:
        warnings.warn(
            f"{spec.name}: {duplicate_count} doppelte Tageswerte; erster Wert bleibt erhalten."
        )
    qobs = qobs[~qobs.index.duplicated(keep="first")]
    qobs = qobs.loc[DCFG.START:DCFG.END]
    if qobs.empty:
        raise ValueError(f"Qobs {spec.name} ist im Modellzeitraum leer.")
    print(
        f"[OK] Qobs {spec.name} ({spec.station_id}): "
        f"{qobs.index.min().date()} bis {qobs.index.max().date()}, {len(qobs)} Werte"
    )
    if spec.official_area_km2 is not None and spec.official_area_km2 > 0:
        annual_runoff_depth = (
            float(qobs.mean()) * 86400.0 * 365.2425
            / (float(spec.official_area_km2) * 1e6) * 1000.0
        )
        print(
            f"     Qobs-Plausibilität: mittlere Abflusshöhe="
            f"{annual_runoff_depth:.1f} mm/a"
        )
        if not (
            V93_QOBS_RUNOFF_DEPTH_MIN_MM_A
            <= annual_runoff_depth
            <= V93_QOBS_RUNOFF_DEPTH_MAX_MM_A
        ):
            raise ValueError(
                f"{spec.name}: Qobs ergibt {annual_runoff_depth:.1f} mm/a und ist "
                "für die konfigurierte Einzugsgebietsfläche unplausibel. "
                "Qobs-Spalte, Einheit und Pegelfläche prüfen."
            )
    return qobs, pd.DataFrame(diagnostics)


def _multigauge_calibration_score_v9(
    gauge_metrics: dict[str, dict[str, float]],
    gauge_weights: dict[str, float],
    recharge_model_mean_mm_a: float,
    recharge_reference_mean_mm_a: float,
    recharge_correlation: float,
    incremental_pbias_penalty: float = np.nan,
    incremental_rows: Optional[list[dict[str, float | str]]] = None,
) -> tuple[float, dict[str, float]]:
    keys = list(gauge_metrics)
    weights = np.asarray(
        [max(gauge_weights.get(k, 1.0), 0.0) for k in keys],
        dtype="float64",
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = np.ones(len(keys), dtype="float64")
    weights /= weights.sum()

    required = ("NSE", "KGE", "PBIAS_percent", "logNSE", "lowflow_PBIAS_percent", "summer_PBIAS_percent")
    arrays = {
        name: np.asarray([gauge_metrics[k][name] for k in keys], dtype="float64")
        for name in required
    }
    if not np.isfinite(arrays["NSE"]).all() or not np.isfinite(arrays["KGE"]).all() or not np.isfinite(arrays["PBIAS_percent"]).all():
        return 9999.0, {}

    mean_nse = float(np.sum(weights * arrays["NSE"]))
    mean_kge = float(np.sum(weights * arrays["KGE"]))
    mean_abs_pbias = float(np.sum(weights * np.abs(arrays["PBIAS_percent"])))
    mean_signed_pbias = float(np.sum(weights * arrays["PBIAS_percent"]))

    def weighted_finite(values: np.ndarray, absolute: bool = False) -> float:
        valid = np.isfinite(values)
        if not valid.any():
            return np.nan
        w = weights[valid]
        w = w / np.sum(w)
        v = np.abs(values[valid]) if absolute else values[valid]
        return float(np.sum(w * v))

    mean_log_nse = weighted_finite(arrays["logNSE"])
    mean_abs_lowflow_pbias = weighted_finite(arrays["lowflow_PBIAS_percent"], absolute=True)
    mean_abs_summer_pbias = weighted_finite(arrays["summer_PBIAS_percent"], absolute=True)

    terms: list[tuple[float, float]] = [
        (DCFG.OBJECTIVE_WEIGHT_NSE, 1.0 - mean_nse),
        (DCFG.OBJECTIVE_WEIGHT_KGE, 1.0 - mean_kge),
        (DCFG.OBJECTIVE_WEIGHT_PBIAS, mean_abs_pbias / 100.0),
    ]
    if np.isfinite(mean_log_nse):
        terms.append((DCFG.OBJECTIVE_WEIGHT_LOG_NSE, 1.0 - mean_log_nse))
    if np.isfinite(mean_abs_lowflow_pbias):
        terms.append((DCFG.OBJECTIVE_WEIGHT_LOWFLOW_PBIAS, mean_abs_lowflow_pbias / 100.0))
    if np.isfinite(mean_abs_summer_pbias):
        terms.append((DCFG.OBJECTIVE_WEIGHT_SUMMER_PBIAS, mean_abs_summer_pbias / 100.0))
    if np.isfinite(incremental_pbias_penalty):
        terms.append((DCFG.OBJECTIVE_WEIGHT_INCREMENTAL_PBIAS, incremental_pbias_penalty))

    recharge_pbias = np.nan
    recharge_mean_penalty = np.nan
    if (
        DCFG.RECHARGE_REFERENCE_ENABLED
        and np.isfinite(recharge_model_mean_mm_a)
        and np.isfinite(recharge_reference_mean_mm_a)
        and recharge_reference_mean_mm_a > 0
    ):
        recharge_pbias = 100.0 * (
            recharge_model_mean_mm_a - recharge_reference_mean_mm_a
        ) / recharge_reference_mean_mm_a
        recharge_mean_penalty = max(
            0.0,
            abs(recharge_pbias) / 100.0
            - float(DCFG.RECHARGE_REFERENCE_TOLERANCE_FRACTION),
        )
        terms.append((DCFG.OBJECTIVE_WEIGHT_RECHARGE_MEAN, recharge_mean_penalty))

    if DCFG.RECHARGE_REFERENCE_ENABLED and np.isfinite(recharge_correlation):
        terms.append((
            DCFG.OBJECTIVE_WEIGHT_RECHARGE_CORRELATION,
            1.0 - np.clip(recharge_correlation, -1.0, 1.0),
        ))

    active_weight = sum(weight for weight, _ in terms if weight > 0)
    score = sum(weight * term for weight, term in terms if weight > 0) / active_weight
    flat: dict[str, float] = {
        "NSE_mean": mean_nse,
        "KGE_mean": mean_kge,
        "logNSE_mean": float(mean_log_nse),
        "PBIAS_abs_mean_percent": mean_abs_pbias,
        "PBIAS_signed_mean_percent": mean_signed_pbias,
        "Lowflow_PBIAS_abs_mean_percent": float(mean_abs_lowflow_pbias),
        "Summer_PBIAS_abs_mean_percent": float(mean_abs_summer_pbias),
        "Incremental_PBIAS_penalty": float(incremental_pbias_penalty),
        "Recharge_model_mm_a": float(recharge_model_mean_mm_a),
        "Recharge_reference_mm_a": float(recharge_reference_mean_mm_a),
        "Recharge_PBIAS_percent": float(recharge_pbias),
        "Recharge_mean_penalty": float(recharge_mean_penalty),
        "Recharge_correlation": float(recharge_correlation),
    }
    for key, metrics in gauge_metrics.items():
        prefix = _safe_slug(key)
        flat[f"NSE_{prefix}"] = float(metrics["NSE"])
        flat[f"KGE_{prefix}"] = float(metrics["KGE"])
        flat[f"logNSE_{prefix}"] = float(metrics["logNSE"])
        flat[f"PBIAS_{prefix}_percent"] = float(metrics["PBIAS_percent"])
        flat[f"Lowflow_PBIAS_{prefix}_percent"] = float(metrics["lowflow_PBIAS_percent"])
        flat[f"Summer_PBIAS_{prefix}_percent"] = float(metrics["summer_PBIAS_percent"])
        flat[f"n_days_{prefix}"] = float(metrics["n_days"])
    for row in incremental_rows or []:
        prefix = _safe_slug(str(row["reach_name"]))
        flat[f"Incremental_PBIAS_{prefix}_percent"] = float(row["incremental_pbias_percent"])
    return float(score), flat


def _report_parameter_bound_proximity_v93(params: dict[str, float]) -> None:
    """Warnt, wenn ein Optimum wahrscheinlich durch eine Parametergrenze begrenzt ist."""
    for name, (lo, hi) in zip(PARAM_NAMES_V9, PARAM_BOUNDS_V9):
        value = float(params[name])
        span = max(float(hi - lo), 1e-12)
        relative = (value - lo) / span
        if relative <= 0.01:
            warnings.warn(
                f"Parameter {name}={value:.6g} liegt an der unteren Kalibrierungsgrenze "
                f"({lo:.6g}). Prozessgleichung und Grenze fachlich prüfen."
            )
        elif relative >= 0.99:
            warnings.warn(
                f"Parameter {name}={value:.6g} liegt an der oberen Kalibrierungsgrenze "
                f"({hi:.6g}). Der Parameter kompensiert möglicherweise Struktur- oder "
                "Routingfehler."
            )


def _parameter_logic_penalty_v93(params: dict[str, float]) -> float:
    """Harte hydrologische Reihenfolgen; 0 bedeutet zulässig."""
    if params["k_gw_fast_daily"] <= params["k_gw_slow_daily"]:
        return 1000.0
    if params["k_interflow_daily"] <= params["k_gw_slow_daily"]:
        return 1000.0
    if not (0.0 < params["alpha_fast"] < 1.0):
        return 1000.0
    if not (0.0 < params["excess_recharge_frac"] < 1.0):
        return 1000.0
    if not (0.0 < params["slow_recharge_frac"] < 1.0):
        return 1000.0
    if not (0.0 < params["init_soil_frac"] <= 1.0):
        return 1000.0
    return 0.0


def calibrate_daily_v9_multigauge(
    dates: pd.DatetimeIndex,
    calibration_data: dict[str, dict[str, object]],
    primary_key: str,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Gemeinsame Parametersuche mit gleichzeitiger Bewertung aller Pegel."""
    try:
        from scipy.optimize import differential_evolution
    except Exception:
        warnings.warn("scipy nicht verfügbar; V9.4-Defaultparameter werden verwendet.")
        defaults = make_param_dict_v9([
            0.30, 0.20, 0.0010, 0.12, 0.06, 0.0010, 0.65, 0.65, 0.50, 1.00, 1.00
        ])
        return defaults, pd.DataFrame(), pd.DataFrame([defaults])

    obs_arrays: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    gauge_weights: dict[str, float] = {}
    for key, payload in calibration_data.items():
        qobs = payload["qobs"]
        if not isinstance(qobs, pd.Series):
            raise TypeError(f"Kalibrierungs-Qobs für {key} ist keine pandas Series.")
        obs = qobs.reindex(dates).to_numpy(dtype="float64")
        mask = (
            (dates >= pd.Timestamp(DCFG.CALIB_START))
            & (dates <= pd.Timestamp(DCFG.CALIB_END))
            & np.isfinite(obs)
        )
        if int(mask.sum()) < 30:
            raise ValueError(f"Zu wenige Qobs-Kalibrierungstage für {key}: {int(mask.sum())}")
        obs_arrays[key] = obs
        masks[key] = mask
        gauge_weights[key] = float(payload.get("calibration_weight", 1.0))

    records: list[dict] = []

    def evaluate(x) -> tuple[float, dict[str, float]]:
        params = make_param_dict_v9(x)
        logic_penalty = _parameter_logic_penalty_v93(params)
        if logic_penalty > 0:
            return logic_penalty, {"parameter_logic_penalty": logic_penalty}
        gauge_metrics: dict[str, dict[str, float]] = {}
        simulations: dict[str, dict[str, object]] = {}
        for key, payload in calibration_data.items():
            simulation = _simulate_calibration_sample_v9(
                dates,
                np.asarray(payload["p_forcing"], dtype="float64"),
                np.asarray(payload["pet_forcing"], dtype="float64"),
                payload["static"],
                params,
            )
            simulations[key] = simulation
            sim = simulation["qsim"].to_numpy(dtype="float64")
            mask = masks[key]
            obs = obs_arrays[key]
            obs_cal = obs[mask]
            sim_cal = sim[mask]
            nn = nse(obs_cal, sim_cal)
            kk = kge(obs_cal, sim_cal)
            pp = pbias(obs_cal, sim_cal)
            log_nn = _nse_log1p_v94(obs_cal, sim_cal)

            q20 = float(np.nanquantile(obs_cal, DCFG.LOWFLOW_QUANTILE))
            low = obs_cal <= q20
            low_pbias = _safe_pbias_v94(obs_cal[low], sim_cal[low])

            calibration_dates = dates[mask]
            summer = np.isin(calibration_dates.month, [6, 7, 8])
            summer_pbias = _safe_pbias_v94(obs_cal[summer], sim_cal[summer])

            gauge_metrics[key] = {
                "NSE": nn,
                "KGE": kk,
                "PBIAS_percent": pp,
                "logNSE": log_nn,
                "lowflow_PBIAS_percent": low_pbias,
                "summer_PBIAS_percent": summer_pbias,
                "n_days": int(mask.sum()),
            }

        incremental_rows, incremental_penalty = _incremental_reach_metrics_v94(
            dates,
            calibration_data,
            obs_arrays,
            simulations,
            masks,
        )
        primary_sim = simulations[primary_key]
        return _multigauge_calibration_score_v9(
            gauge_metrics,
            gauge_weights,
            float(primary_sim["recharge_mean_mm_a"]),
            float(primary_sim["recharge_reference_mean_mm_a"]),
            float(primary_sim["recharge_correlation"]),
            incremental_pbias_penalty=incremental_penalty,
            incremental_rows=incremental_rows,
        )

    def objective(x):
        score, _ = evaluate(x)
        return score

    def callback(xk, convergence):
        score, metrics = evaluate(xk)
        row = {
            "stage": "global_multigauge",
            "generation": len(records) + 1,
            "objective": score,
            "convergence": float(convergence),
            **metrics,
            **make_param_dict_v9(xk),
        }
        records.append(row)
        gauge_text = []
        for key, payload in calibration_data.items():
            prefix = _safe_slug(key)
            name = str(payload["name"])
            gauge_text.append(
                f"{name}:NSE={metrics.get(f'NSE_{prefix}', np.nan):.3f},"
                f"KGE={metrics.get(f'KGE_{prefix}', np.nan):.3f},"
                f"PB={metrics.get(f'PBIAS_{prefix}_percent', np.nan):.1f}%"
            )
        print(
            f"     Generation {len(records):02d}: Obj={score:.4f}, "
            f"NSĒ={metrics.get('NSE_mean', np.nan):.3f}, "
            f"KGĒ={metrics.get('KGE_mean', np.nan):.3f}, "
            f"|PB|̄={metrics.get('PBIAS_abs_mean_percent', np.nan):.1f}%, "
            f"logNSĒ={metrics.get('logNSE_mean', np.nan):.3f}, "
            f"LowQ-PB={metrics.get('Lowflow_PBIAS_abs_mean_percent', np.nan):.1f}%, "
            f"Reach-Pen={metrics.get('Incremental_PBIAS_penalty', np.nan):.3f}, "
            f"GWN-PBIAS={metrics.get('Recharge_PBIAS_percent', np.nan):.1f}%, "
            f"r_GWN={metrics.get('Recharge_correlation', np.nan):.3f}"
        )
        print("       " + " | ".join(gauge_text))
        return False

    print("\n[INFO] V9.4 verschachtelte Multi-Pegel-Kalibrierung: Heidelbach + Röllshausen + Alsfeld")
    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS_V9,
        seed=DCFG.RANDOM_SEED,
        maxiter=DCFG.CALIBRATION_MAXITER,
        popsize=DCFG.CALIBRATION_POPSIZE,
        polish=False,
        updating="immediate",
        workers=1,
        tol=DCFG.CALIBRATION_TOL,
        callback=callback,
    )
    best = np.asarray(result.x, dtype="float64")
    best_score, best_metrics = evaluate(best)

    local_records: list[dict] = []
    fraction = float(DCFG.LOCAL_REFINEMENT_FRACTION)
    for j, name in enumerate(PARAM_NAMES_V9):
        lo, hi = PARAM_BOUNDS_V9[j]
        for direction in (-1.0, 1.0):
            trial = best.copy()
            trial[j] = np.clip(best[j] + direction * fraction * (hi - lo), lo, hi)
            if np.isclose(trial[j], best[j]):
                continue
            score, metrics = evaluate(trial)
            local_records.append({
                "stage": "local_multigauge",
                "parameter": name,
                "direction": int(direction),
                "objective": score,
                **metrics,
                **make_param_dict_v9(trial),
            })
            if score < best_score:
                best, best_score, best_metrics = trial, score, metrics
                print(
                    f"     Verbesserung {name}: Obj={best_score:.4f}, "
                    f"NSĒ={best_metrics['NSE_mean']:.3f}, "
                    f"KGĒ={best_metrics['KGE_mean']:.3f}"
                )

    params = make_param_dict_v9(best)
    print("[OK] Kalibrierte tägliche V9.4-Multi-Pegel-Parameter:")
    for key, value in params.items():
        print(f"     {key}: {value:.6f}")
    print(
        f"     Pegelmittel: NSE={best_metrics['NSE_mean']:.3f}, "
        f"KGE={best_metrics['KGE_mean']:.3f}, "
        f"mittlerer |PBIAS|={best_metrics['PBIAS_abs_mean_percent']:.1f}%"
    )
    _report_parameter_bound_proximity_v93(params)

    trace = pd.concat(
        [pd.DataFrame(records), pd.DataFrame(local_records)],
        ignore_index=True,
        sort=False,
    )
    summary = pd.DataFrame([{"objective": best_score, **best_metrics, **params}])
    return params, trace, summary


def summarize_daily_performance_multigauge(
    results: pd.DataFrame,
    gauge_routings: dict[str, dict],
) -> pd.DataFrame:
    periods = {
        "calibration": (DCFG.CALIB_START, DCFG.CALIB_END),
        "validation": (DCFG.VALID_START, DCFG.VALID_END),
        "full": (DCFG.START, DCFG.END),
    }
    rows: list[dict] = []
    for key, routing in gauge_routings.items():
        obs_col = f"Qobs_{key}_m3s"
        sim_col = f"Qsim_{key}_m3s"
        for period, (start, end) in periods.items():
            part = results.loc[start:end].dropna(subset=[obs_col, sim_col])
            rows.append({
                "gauge_name": routing["name"],
                "station_id": routing["station_id"],
                "gauge_key": key,
                "period": period,
                "start": start,
                "end": end,
                "n_days": len(part),
                "NSE": nse(part[obs_col], part[sim_col]),
                "KGE": kge(part[obs_col], part[sim_col]),
                "PBIAS_percent": pbias(part[obs_col], part[sim_col]),
                "Qobs_mean_m3s": part[obs_col].mean(),
                "Qsim_mean_m3s": part[sim_col].mean(),
                "upstream_area_km2": routing["area_km2"],
            })
    return pd.DataFrame(rows)


def plot_q_timeseries_daily_for_gauge(
    results: pd.DataFrame,
    routing: dict,
    path: Path,
) -> None:
    key = routing["key"]
    obs_col = f"Qobs_{key}_m3s"
    sim_col = f"Qsim_{key}_m3s"
    plt.figure(figsize=(12, 5.5))
    plt.plot(results.index, results[obs_col], label="Qobs", linewidth=0.8)
    plt.plot(results.index, results[sim_col], label="Qsim", linewidth=0.8)
    plt.axvspan(
        pd.Timestamp(DCFG.CALIB_START), pd.Timestamp(DCFG.CALIB_END),
        alpha=0.08, label="Kalibrierung",
    )
    plt.axvspan(
        pd.Timestamp(DCFG.VALID_START), pd.Timestamp(DCFG.VALID_END),
        alpha=0.08, label="Validierung",
    )
    plt.ylabel("Q [m³/s]")
    plt.xlabel("Tag")
    plt.title(f"Qobs und Qsim – {routing['name']} ({routing['station_id']})")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _write_multigauge_vector_and_diagnostics(
    out: Path,
    gauge_routings: dict[str, dict],
    profile: dict,
) -> None:
    records: list[dict] = []
    snapped: list[dict] = []
    candidates: list[dict] = []
    for key, routing in gauge_routings.items():
        official = routing.get("official_area_km2", np.nan)
        records.append({
            "gauge_key": key,
            "name": routing["name"],
            "station_id": routing["station_id"],
            "is_primary": routing["is_primary"],
            "point_source": routing.get("point_source", "unknown"),
            "configured_point_x": routing.get("configured_point_x", np.nan),
            "configured_point_y": routing.get("configured_point_y", np.nan),
            "official_point_x": routing.get("official_point_x", np.nan),
            "official_point_y": routing.get("official_point_y", np.nan),
            "configured_to_official_distance_m": routing.get(
                "configured_to_official_distance_m", np.nan
            ),
            "routing_origin_x": routing.get("original_x", routing.get("official_point_x", routing.get("configured_point_x", np.nan))),
            "routing_origin_y": routing.get("original_y", routing.get("official_point_y", routing.get("configured_point_y", np.nan))),
            "snapped_x": routing["x"],
            "snapped_y": routing["y"],
            "snap_distance_m": routing["snap_distance_m"],
            "snap_method": routing.get("snap_method", "unknown"),
            "on_river_mask": routing["on_river_mask"],
            "accumulation_cells": routing["accumulation_cells"],
            "accumulation_area_km2": routing.get("accumulation_area_km2", np.nan),
            "upstream_area_km2": routing["area_km2"],
            "official_area_km2": official,
            "area_difference_percent": routing.get("area_difference_percent", np.nan),
            "within_area_tolerance": routing.get("within_area_tolerance", False),
            "topology_area_difference_km2": routing.get(
                "topology_area_difference_km2", np.nan
            ),
        })
        snapped.append({
            "name": routing["name"],
            "station_id": routing["station_id"],
            "is_primary": routing["is_primary"],
            "snap_m": routing["snap_distance_m"],
            "area_km2": routing["area_km2"],
            "official_km2": official,
            "area_diff_pct": routing.get("area_difference_percent", np.nan),
            "snap_method": routing.get("snap_method", "unknown"),
            "geometry": Point(routing["x"], routing["y"]),
        })
        for rank, item in enumerate(routing.get("candidate_diagnostics", []), start=1):
            candidates.append({
                "station_id": routing["station_id"],
                "name": routing["name"],
                "rank_by_area": rank,
                "candidate_x": item["x"],
                "candidate_y": item["y"],
                "distance_m": item["distance_m"],
                "on_river_mask": item["on_river_mask"],
                "accumulation_cells": item["accumulation_cells"],
                "candidate_area_km2": item["area_km2"],
                "official_area_km2": official,
                "area_relative_error": item["area_relative_error"],
            })

        catchment_gdf = mask_to_geodataframe(
            routing["upstream_mask"], profile, f"{routing['name']}_D8_upstream"
        )
        catchment_path = out / "maps" / f"catchment_{routing['station_id']}.gpkg"
        if catchment_path.exists():
            catchment_path.unlink()
        catchment_gdf.to_file(
            catchment_path,
            layer=_safe_slug(f"{routing['station_id']}_{routing['name']}")[:60],
            driver="GPKG",
        )

    pd.DataFrame(records).to_csv(
        out / "tables" / "gauge_routing_diagnostics.csv",
        index=False,
        encoding="utf-8",
    )
    if candidates:
        pd.DataFrame(candidates).to_csv(
            out / "tables" / "gauge_snap_candidates_top20.csv",
            index=False,
            encoding="utf-8",
        )
    snapped_gdf = gpd.GeoDataFrame(snapped, geometry="geometry", crs=profile["crs"])
    snapped_path = out / "maps" / "gauges_snapped.gpkg"
    if snapped_path.exists():
        snapped_path.unlink()
    snapped_gdf.to_file(snapped_path, layer="gauges_snapped", driver="GPKG")


def write_daily_hausarbeit_bundle_multigauge(
    out: Path,
    performance: pd.DataFrame,
    params: dict[str, float],
    gauge_routings: dict[str, dict],
) -> None:
    haus = out / "hausarbeit"
    for key, routing in gauge_routings.items():
        source = out / "plots" / f"qobs_qsim_daily_{routing['station_id']}.png"
        if source.exists():
            shutil.copy2(source, haus / f"Abb_Qobs_Qsim_{routing['station_id']}.png")
    for source, target in {
        out / "plots" / "annual_water_balance_daily_v9.png": "Abb_Jahreswasserbilanz_Heidelbach.png",
        out / "plots" / "map_recharge_annual_mean.png": "Abb_Recharge.png",
        out / "plots" / "map_interflow_annual_mean.png": "Abb_Zwischenabfluss.png",
        out / "tables" / "daily_performance_metrics_multigauge.csv": "Tab_Performance_MultiPegel.csv",
        out / "tables" / "calibrated_parameters_daily_v9.csv": "Tab_Parameter.csv",
        out / "tables" / "gauge_routing_diagnostics.csv": "Tab_Pegeldiagnostik.csv",
    }.items():
        if source.exists():
            shutil.copy2(source, haus / target)

    gauge_lines = "\n".join(
        f"- {r['name']} ({r['station_id']}): D8-Fläche {r['area_km2']:.3f} km²"
        for r in gauge_routings.values()
    )
    readme = f"""HydroMod v9.4 – verschachtelte tägliche Multi-Pegel-Kalibrierung

Verwendete Pegel:
{gauge_lines}

Warm-up: {DCFG.START} bis {DCFG.WARMUP_END}; {DCFG.SPINUP_CYCLES} Zyklen
Kalibrierung: {DCFG.CALIB_START} bis {DCFG.CALIB_END}
Validierung: {DCFG.VALID_START} bis {DCFG.VALID_END}

Alle Pegel werden mit demselben globalen Parametersatz simuliert. Jeder Pegel
besitzt ein eigenes D8-Einzugsgebiet, eine eigene Reisezeitverteilung und eine
eigene Qobs-Reihe. Das Pegel-Snapping nutzt amtliche Koordinaten und amtliche
Einzugsgebietsflächen. Unplausible Zuordnungen stoppen den Lauf vor der
Kalibrierung. Die räumliche Recharge-Referenz wird weiterhin für das primäre
Heidelbachgebiet verwendet. Zusätzlich wird der langjährige mittlere Abfluss
an jeder Flussrasterzelle als river_mean_discharge_m3s.tif ausgegeben.

Kalibrierte Parameter:
{json.dumps(params, indent=2, ensure_ascii=False)}

Performance:
{performance.to_string(index=False)}
"""
    (haus / "README_Hausarbeit_Daily_V9_4_NestedGauge.txt").write_text(
        readme, encoding="utf-8"
    )


def compute_mean_discharge_raster_v93(
    runoff_sum_mm: np.ndarray,
    n_days: int,
    network: dict,
    profile: dict,
    river_only: bool = True,
) -> np.ndarray:
    """Langjähriges mittleres Q an jeder D8-Zelle, unabhängig von Pegeln.

    Reisezeit verändert den langjährigen Mittelwert nicht. Deshalb kann der
    mittlere lokale Abfluss volumenbasiert über das D8-Netz akkumuliert werden.
    Das Ergebnis ist Q in m³/s an jeder Flussrasterzelle.
    """
    if n_days <= 0:
        raise ValueError("n_days muss positiv sein.")
    local_mean_mm_day = np.asarray(runoff_sum_mm, dtype="float64") / float(n_days)
    cell_area_m2 = abs(float(profile["transform"].a * profile["transform"].e))
    volume_m3_day = local_mean_mm_day / 1000.0 * cell_area_m2
    accumulated = volume_m3_day.copy()
    downstream = network["downstream"]
    for cid in network["order"]:
        parent = int(downstream[int(cid)])
        if parent >= 0:
            accumulated[parent] += accumulated[int(cid)]
    q_m3s = accumulated / 86400.0
    raster = _compact_raster(q_m3s, network, (profile["height"], profile["width"]))
    if river_only:
        river = np.zeros((profile["height"], profile["width"]), dtype=bool)
        river[network["rows"], network["cols"]] = network["river_compact"]
        raster[~river] = np.nan
    return raster


def main_v9_daily() -> None:
    """HydroMod V9.4 mit verschachtelter Drei-Pegel-Kalibrierung und räumlichen Forcing-Korrekturen."""
    out = daily_out_dir()
    print("============================================================")
    print("HydroMod v9.4: verschachtelte Pegel, Niedrigwasser- und Zwischengebiets-Kalibrierung")
    print("============================================================")
    print(f"Projektordner: {CFG.BASE_DIR}")
    print(f"Datenordner:   {data_dir()}")
    print(f"Output:        {out}")

    # 1. Räumliche Modelldomäne
    dgm_path = find_first(data_dir() / CFG.DGM_DIRNAME, [".tif", ".tiff"])
    dgm_full, profile_full = read_raster(dgm_path)
    catchment = load_catchment(profile_full["crs"])
    mask_full = create_mask(catchment, profile_full)
    dgm, mask, profile, crop_offset = crop_raster_to_mask(
        dgm_full, mask_full, profile_full, padding=CFG.CROP_PADDING_CELLS
    )
    del dgm_full, mask_full
    print(f"[OK] Modellraster: {profile['width']} x {profile['height']}; Offset={crop_offset}")

    # 2. Statische Daten und D8-Netz
    fc = load_field_capacity_raster(profile, mask)
    river_mask, _ = load_river_mask(profile, mask)
    landuse = load_landuse_rasters(profile, mask, catchment)
    soil_units, soil_table = load_soil_unit_factors(profile, mask)
    soil_table.to_csv(
        out / "tables" / "soil_class_parameterization.csv",
        index=False, encoding="utf-8",
    )
    dgm = fill_missing_dem_nearest(dgm, mask)
    _, dgm_conditioned = condition_dem_priority_flood(dgm, mask, river_mask, profile)
    slope_fraction = calculate_slope_fraction(dgm_conditioned, mask, profile)
    slope_factor = make_slope_quickflow_factor(
        slope_fraction, mask, CFG.SLOPE_QUICKFLOW_WEIGHT
    )
    network = build_d8_flow_network(dgm_conditioned, mask, profile, river_mask)

    # 3. Alle Pegelpunkte snappen und individuelle Einzugsgebiete ableiten
    gauges = load_gauge_points(profile["crs"])
    gauge_routings: dict[str, dict] = {}
    for _, gauge_row in gauges.iterrows():
        routing = build_gauge_routing(gauge_row, network, mask, profile, river_mask)
        routing["calibration_weight"] = float(gauge_row["calibration_weight"])
        routing["official_area_km2"] = float(gauge_row["official_area_km2"])
        routing["source_path"] = str(gauge_row["source_path"])
        gauge_routings[routing["key"]] = routing
    primary = next(r for r in gauge_routings.values() if r["is_primary"])
    primary_key = primary["key"]
    primary_mask = primary["upstream_mask"]
    primary_area = primary["area_km2"]

    nested_zone_codes_compact, nested_zone_table = _nested_zone_codes_v94(
        gauge_routings, network
    )
    nested_zone_table.to_csv(
        out / "tables" / "nested_reach_zone_definitions.csv",
        index=False,
        encoding="utf-8",
    )
    print("[OK] Verschachtelte Teilflächen für V9.4:")
    print(nested_zone_table.to_string(index=False))

    for routing in gauge_routings.values():
        official = routing.get("official_area_km2", np.nan)
        if np.isfinite(official) and official > 0:
            difference = 100.0 * (routing["area_km2"] - official) / official
            print(
                f"[INFO] {routing['name']}-Fläche: Modell={routing['area_km2']:.3f} km², "
                f"offiziell={official:.3f} km², Abweichung={difference:+.2f}%"
            )
        else:
            print(
                f"[INFO] {routing['name']}-Fläche: Modell={routing['area_km2']:.3f} km²; "
                "keine offizielle Vergleichsfläche konfiguriert."
            )

    # Diagnose wird bereits vor dem teuren Forcing-Cache und der Kalibrierung
    # geschrieben. Ein falscher Pegel kann so nicht unbemerkt optimiert werden.
    _write_multigauge_vector_and_diagnostics(out, gauge_routings, profile)
    if V93_PRECHECK_ONLY:
        print("[FERTIG] V9.3-Pegelvorprüfung ohne Kalibrierung.")
        print(f"Diagnose: {out / 'tables' / 'gauge_routing_diagnostics.csv'}")
        return

    spec_by_station = {spec.station_id: spec for spec in DAILY_GAUGE_SPECS}

    # 4. Tagesforcings und Qobs aller Pegel
    precip = NetCDFDailyPrecipProvider(data_dir() / CFG.PRECIP_DIRNAME)
    try:
        pet = RasterDailyPETProvider(data_dir() / CFG.PET_DIRNAME)
        dates = _common_daily_dates(precip, pet)
        qobs_by_key: dict[str, pd.Series] = {}
        qobs_diagnostics: list[pd.DataFrame] = []
        for key, routing in gauge_routings.items():
            spec = spec_by_station[routing["station_id"]]
            qobs, diagnostic = load_qobs_daily_for_spec(spec)
            qobs_by_key[key] = qobs
            qobs_diagnostics.append(diagnostic)
        pd.concat(qobs_diagnostics, ignore_index=True).to_csv(
            out / "tables" / "qobs_source_diagnostics.csv",
            index=False, encoding="utf-8",
        )

        # Recharge-Referenz bleibt ein unabhängiges räumliches Ziel.
        gwn_reference = None
        gwn_metadata = {"product": "groundwater_recharge", "status": "disabled"}
        if DCFG.RECHARGE_REFERENCE_ENABLED:
            gwn_reference, gwn_metadata = load_reference_annual_raster(
                CFG.GWN_REFERENCE_DIRNAME,
                ("grundwasser", "neubild"),
                profile,
                mask,
                CFG.GWN_REFERENCE_UNIT_FACTOR,
                "groundwater_recharge_calibration",
                None,
            )
            if gwn_reference is None:
                warnings.warn("Recharge-Referenz fehlt; Kalibrierung nutzt nur die drei Pegel.")
            else:
                valid_gwn = primary_mask & np.isfinite(gwn_reference)
                print(
                    f"[OK] Recharge-Kalibrierungsziel Heidelbach: "
                    f"{float(np.nanmean(gwn_reference[valid_gwn])):.1f} mm/a, "
                    f"{int(valid_gwn.sum())} Zellen"
                )
        pd.DataFrame([gwn_metadata]).to_csv(
            out / "tables" / "calibration_recharge_reference_metadata.csv",
            index=False, encoding="utf-8",
        )

        # 5. Eigene geschichtete Rasterstichprobe und Forcingmatrix je Pegel.
        calibration_data: dict[str, dict[str, object]] = {}
        for key, routing in gauge_routings.items():
            ids = routing["upstream_ids"].astype(np.int32)
            rr = routing["rows"]
            cc = routing["cols"]
            fc_g = fc[rr, cc].astype("float64")
            slope_g = slope_factor[rr, cc].astype("float64")
            land_q_g = landuse["quickflow_factor"][rr, cc].astype("float64")
            land_perc_g = landuse["percolation_factor"][rr, cc].astype("float64")
            travel_g = routing["travel_cost"].astype("float64")
            selected, weights = _stratified_sample_indices_v9(
                (
                    fc_g,
                    slope_g,
                    land_q_g,
                    land_perc_g,
                    np.log1p(travel_g),
                ),
                min(MULTIGAUGE_SAMPLE_CELLS_PER_GAUGE, len(ids)),
                DCFG.RANDOM_SEED + int(str(routing["station_id"])[-3:]),
            )
            sample_rows = rr[selected]
            sample_cols = cc[selected]
            sx, sy = _cell_centers(profile, sample_rows, sample_cols)
            p_sample, pet_sample = _build_calibration_forcing_cache(
                dates,
                precip,
                pet,
                sx,
                sy,
                profile["crs"],
                ids[selected],
                out,
            )
            if key == primary_key and gwn_reference is not None:
                recharge_reference_sample = gwn_reference[sample_rows, sample_cols]
                if int(np.isfinite(recharge_reference_sample).sum()) < int(
                    DCFG.MIN_RECHARGE_REFERENCE_CELLS
                ):
                    recharge_reference_sample = np.full(len(selected), np.nan)
            else:
                recharge_reference_sample = np.full(len(selected), np.nan)

            static: dict[str, np.ndarray | float] = {
                "fc": fc_g[selected],
                "slope_q": slope_g[selected],
                "land_q": land_q_g[selected],
                "pet_factor": landuse["pet_factor"][rr, cc][selected],
                "land_perc": land_perc_g[selected],
                "impervious_fraction": np.clip(
                    landuse["sealed_percent"][rr, cc][selected] / 100.0, 0.0, 1.0
                ),
                "soil_q": soil_units["quickflow_factor"][rr, cc][selected],
                "soil_perc": soil_units["percolation_factor"][rr, cc][selected],
                "soil_slow": soil_units["slow_recharge_factor"][rr, cc][selected],
                "travel_cost": travel_g[selected],
                "weights": weights,
                "cell_area_m2": float(abs(profile["transform"].a * profile["transform"].e)),
                "recharge_reference_mm_a": recharge_reference_sample,
                "nested_zone_code": nested_zone_codes_compact[ids[selected]],
            }
            calibration_data[key] = {
                "name": routing["name"],
                "station_id": routing["station_id"],
                "qobs": qobs_by_key[key],
                "p_forcing": p_sample,
                "pet_forcing": pet_sample,
                "static": static,
                "calibration_weight": routing["calibration_weight"],
                "upstream_area_km2": float(routing["area_km2"]),
            }
            print(
                f"[INFO] Kalibrierungsstichprobe {routing['name']}: "
                f"{len(selected)} Zellen repräsentieren {len(ids)} Zellen."
            )

        if DCFG.USE_CALIBRATION:
            params, trace, calibration_summary = calibrate_daily_v9_multigauge(
                dates, calibration_data, primary_key
            )
        else:
            params = make_param_dict_v9([
                0.30, 0.20, 0.0010, 0.12, 0.06, 0.0010, 0.65, 0.65, 0.50, 1.00, 1.00
            ])
            trace = pd.DataFrame()
            calibration_summary = pd.DataFrame([params])

        trace.to_csv(
            out / "tables" / "calibration_trace_daily_v9_multigauge.csv",
            index=False, encoding="utf-8",
        )
        calibration_summary.to_csv(
            out / "tables" / "calibration_summary_daily_v9_multigauge.csv",
            index=False, encoding="utf-8",
        )
        pd.DataFrame([{
            "model_version": "9.4_nested_gauge_diagnostic_calibration",
            **params,
            "spinup_cycles": DCFG.SPINUP_CYCLES,
            "n_gauges": len(gauge_routings),
            "sample_cells_per_gauge": MULTIGAUGE_SAMPLE_CELLS_PER_GAUGE,
            "gauge_weights": json.dumps({
    str(routing["station_id"]): float(routing["calibration_weight"])
    for routing in gauge_routings.values()
}),
            "objective_weight_nse": DCFG.OBJECTIVE_WEIGHT_NSE,
            "objective_weight_kge": DCFG.OBJECTIVE_WEIGHT_KGE,
            "objective_weight_pbias": DCFG.OBJECTIVE_WEIGHT_PBIAS,
            "objective_weight_recharge_mean": DCFG.OBJECTIVE_WEIGHT_RECHARGE_MEAN,
            "objective_weight_recharge_correlation": DCFG.OBJECTIVE_WEIGHT_RECHARGE_CORRELATION,
        }]).to_csv(
            out / "tables" / "calibrated_parameters_daily_v9.csv",
            index=False, encoding="utf-8",
        )

        # 6. Finale Simulation routet gleichzeitig zu allen Pegeln.
        sim, compact_maps = run_daily_model_v9(
            dates, precip, pet, profile, network, fc, slope_factor,
            landuse, soil_units, gauge_routings, primary_key, params,
        )
        results = sim.copy()
        for key, qobs in qobs_by_key.items():
            results[f"Qobs_{key}_m3s"] = qobs.reindex(results.index)
        results["Qobs_m3s"] = results[f"Qobs_{primary_key}_m3s"]
        results["Qsim_m3s"] = results[f"Qsim_{primary_key}_m3s"]
        results.to_csv(out / "tables" / "daily_model_results.csv", encoding="utf-8")

        performance = summarize_daily_performance_multigauge(results, gauge_routings)
        v94_diag_rows: list[dict] = []
        ordered_routings = sorted(
            gauge_routings.values(), key=lambda r: float(r["area_km2"])
        )
        for routing in ordered_routings:
            key = routing["key"]
            obs_col = f"Qobs_{key}_m3s"
            sim_col = f"Qsim_{key}_m3s"
            for period_name, (period_start, period_end) in {
                "calibration": (DCFG.CALIB_START, DCFG.CALIB_END),
                "validation": (DCFG.VALID_START, DCFG.VALID_END),
                "full": (DCFG.START, DCFG.END),
            }.items():
                part = results.loc[period_start:period_end, [obs_col, sim_col]].dropna()
                if part.empty:
                    continue
                obs_v = part[obs_col].to_numpy(dtype="float64")
                sim_v = part[sim_col].to_numpy(dtype="float64")
                q20 = float(np.nanquantile(obs_v, DCFG.LOWFLOW_QUANTILE))
                low = obs_v <= q20
                summer = np.isin(part.index.month, [6, 7, 8])
                v94_diag_rows.append({
                    "type": "gauge",
                    "location": routing["name"],
                    "period": period_name,
                    "NSE": nse(obs_v, sim_v),
                    "KGE": kge(obs_v, sim_v),
                    "logNSE": _nse_log1p_v94(obs_v, sim_v),
                    "PBIAS_percent": _safe_pbias_v94(obs_v, sim_v),
                    "lowflow_PBIAS_percent": _safe_pbias_v94(obs_v[low], sim_v[low]),
                    "summer_PBIAS_percent": _safe_pbias_v94(obs_v[summer], sim_v[summer]),
                    "n_days": len(part),
                })
        for upper, lower in zip(ordered_routings[:-1], ordered_routings[1:]):
            for period_name, (period_start, period_end) in {
                "calibration": (DCFG.CALIB_START, DCFG.CALIB_END),
                "validation": (DCFG.VALID_START, DCFG.VALID_END),
                "full": (DCFG.START, DCFG.END),
            }.items():
                cols_needed = [
                    f"Qobs_{upper['key']}_m3s", f"Qobs_{lower['key']}_m3s",
                    f"Qsim_{upper['key']}_m3s", f"Qsim_{lower['key']}_m3s",
                ]
                part = results.loc[period_start:period_end, cols_needed].dropna()
                if part.empty:
                    continue
                obs_inc = (
                    part[f"Qobs_{lower['key']}_m3s"]
                    - part[f"Qobs_{upper['key']}_m3s"]
                ).to_numpy(dtype="float64")
                sim_inc = (
                    part[f"Qsim_{lower['key']}_m3s"]
                    - part[f"Qsim_{upper['key']}_m3s"]
                ).to_numpy(dtype="float64")
                v94_diag_rows.append({
                    "type": "incremental_reach",
                    "location": f"{upper['name']} → {lower['name']}",
                    "period": period_name,
                    "NSE": nse(obs_inc, sim_inc),
                    "KGE": kge(obs_inc, sim_inc),
                    "logNSE": np.nan,
                    "PBIAS_percent": _safe_pbias_v94(obs_inc, sim_inc),
                    "lowflow_PBIAS_percent": np.nan,
                    "summer_PBIAS_percent": _safe_pbias_v94(
                        obs_inc[np.isin(part.index.month, [6, 7, 8])],
                        sim_inc[np.isin(part.index.month, [6, 7, 8])],
                    ),
                    "n_days": len(part),
                    "incremental_area_km2": float(
                        lower["area_km2"] - upper["area_km2"]
                    ),
                })
        pd.DataFrame(v94_diag_rows).to_csv(
            out / "tables" / "v94_lowflow_season_reach_diagnostics.csv",
            index=False,
            encoding="utf-8",
        )
        performance.to_csv(
            out / "tables" / "daily_performance_metrics_multigauge.csv",
            index=False, encoding="utf-8",
        )
        primary_performance = performance[
            performance["gauge_key"] == primary_key
        ].copy()
        primary_performance.to_csv(
            out / "tables" / "daily_performance_metrics.csv",
            index=False, encoding="utf-8",
        )
        annual = make_annual_water_balance_daily_v9(results, primary_area)
        annual.to_csv(
            out / "tables" / "annual_water_balance_daily_v9.csv",
            index=False, encoding="utf-8",
        )

        # 7. Karten für Wasserbilanz und für jedes Pegelgebiet.
        n_years = len(set(dates.year))
        map_arrays = {
            name: _compact_raster(values, network, mask.shape)
            for name, values in compact_maps.items()
        }
        outputs = {
            "field_capacity_mm.tif": fc,
            "dgm_conditioned_m.tif": dgm_conditioned,
            "slope_percent.tif": slope_fraction * 100.0,
            "river_mask.tif": np.where(mask, river_mask.astype(float), np.nan),
            "flow_accumulation_cells.tif": network["accumulation_raster"],
            "precip_sum_2021_2025_mm.tif": map_arrays["precip_sum_mm"],
            "pet_sum_2021_2025_mm.tif": map_arrays["pet_sum_mm"],
            "aet_sum_2021_2025_mm.tif": map_arrays["aet_sum_mm"],
            "interflow_sum_2021_2025_mm.tif": map_arrays["interflow_sum_mm"],
            "quickflow_sum_2021_2025_mm.tif": map_arrays["quickflow_sum_mm"],
            "baseflow_sum_2021_2025_mm.tif": map_arrays["baseflow_sum_mm"],
            "percolation_sum_2021_2025_mm.tif": map_arrays["percolation_sum_mm"],
            "recharge_sum_2021_2025_mm.tif": map_arrays["recharge_sum_mm"],
            "runoff_sum_2021_2025_mm.tif": map_arrays["runoff_sum_mm"],
            "precip_annual_mean_mm_a.tif": map_arrays["precip_sum_mm"] / n_years,
            "pet_annual_mean_mm_a.tif": map_arrays["pet_sum_mm"] / n_years,
            "aet_annual_mean_mm_a.tif": map_arrays["aet_sum_mm"] / n_years,
            "interflow_annual_mean_mm_a.tif": map_arrays["interflow_sum_mm"] / n_years,
            "percolation_annual_mean_mm_a.tif": map_arrays["percolation_sum_mm"] / n_years,
            "recharge_annual_mean_mm_a.tif": map_arrays["recharge_sum_mm"] / n_years,
            "runoff_annual_mean_mm_a.tif": map_arrays["runoff_sum_mm"] / n_years,
            "soil_final_mm.tif": map_arrays["soil_final_mm"],
            "interflow_store_final_mm.tif": map_arrays["interflow_store_final_mm"],
            "groundwater_fast_final_mm.tif": map_arrays["gw_fast_final_mm"],
            "groundwater_slow_final_mm.tif": map_arrays["gw_slow_final_mm"],
            "alpha_fast_spatial.tif": map_arrays["alpha_fast_grid"],
            "beta_perc_daily_spatial.tif": map_arrays["beta_perc_daily_grid"],
            "slow_recharge_fraction_spatial.tif": map_arrays["slow_recharge_fraction_grid"],
            "impervious_fraction.tif": map_arrays["impervious_fraction"],
            "water_balance_residual_sum_mm.tif": map_arrays["water_balance_residual_sum_mm"],
        }
        cell_area = abs(profile["transform"].a * profile["transform"].e)
        outputs["flow_accumulation_area_km2.tif"] = (
            network["accumulation_raster"] * cell_area / 1e6
        )
        for key, routing in gauge_routings.items():
            station = routing["station_id"]
            outputs[f"gauge_{station}_upstream_mask.tif"] = np.where(
                mask, routing["upstream_mask"].astype(float), np.nan
            )
            lag = prepare_daily_routing_lags(
                routing["travel_cost"], params["routing_velocity_m_s"]
            )
            outputs[f"gauge_{station}_routing_travel_time_days.tif"] = compact_to_raster(
                lag["travel_days"], routing["rows"], routing["cols"], mask.shape
            )
            outputs[f"gauge_{station}_routing_lag_days.tif"] = compact_to_raster(
                lag["lag_days"], routing["rows"], routing["cols"], mask.shape
            )
        if V93_WRITE_MEAN_DISCHARGE_RASTER:
            outputs["river_mean_discharge_m3s.tif"] = compute_mean_discharge_raster_v93(
                compact_maps["runoff_sum_mm"], len(dates), network, profile, river_only=True
            )
        for filename, array in outputs.items():
            write_geotiff(out / "maps" / filename, array, profile)

        _write_multigauge_vector_and_diagnostics(out, gauge_routings, profile)

        # 8. Räumliche Referenzvergleiche weiterhin auf Projekt- und Heidelbachgebiet.
        validation_maps = {
            "recharge_sum_mm": map_arrays["recharge_sum_mm"],
            "percolation_sum_mm": map_arrays["percolation_sum_mm"],
            "aet_sum_mm": map_arrays["aet_sum_mm"],
        }
        spatial_metrics, ref_metadata, _ = validate_spatial_products_v8(
            validation_maps, profile, mask, primary_mask, n_years, out
        )
        spatial_metrics.to_csv(
            out / "tables" / "spatial_validation_metrics.csv",
            index=False, encoding="utf-8",
        )
        ref_metadata.to_csv(
            out / "tables" / "reference_product_metadata.csv",
            index=False, encoding="utf-8",
        )

        # 9. Eigener Hydrograph je Pegel.
        for routing in gauge_routings.values():
            plot_q_timeseries_daily_for_gauge(
                results,
                routing,
                out / "plots" / f"qobs_qsim_daily_{routing['station_id']}.png",
            )
        plot_annual_water_balance_daily_v9(
            annual, out / "plots" / "annual_water_balance_daily_v9.png"
        )
        plot_map(
            outputs["precip_annual_mean_mm_a.tif"],
            "Mittlerer räumlicher Niederschlag 2021–2025",
            out / "plots" / "map_precip_annual_mean.png", "mm/a",
        )
        plot_map(
            outputs["recharge_annual_mean_mm_a.tif"],
            "Mittlere simulierte Recharge V9.4",
            out / "plots" / "map_recharge_annual_mean.png", "mm/a",
        )
        plot_map(
            outputs["interflow_annual_mean_mm_a.tif"],
            "Mittlerer simulierter Zwischenabfluss V9.4",
            out / "plots" / "map_interflow_annual_mean.png", "mm/a",
        )
        plot_map(
            outputs["aet_annual_mean_mm_a.tif"],
            "Mittlere simulierte AET V9.4",
            out / "plots" / "map_aet_annual_mean.png", "mm/a",
        )

        write_daily_hausarbeit_bundle_multigauge(
            out, performance, params, gauge_routings
        )

        print("\n[OK] Tägliche Performance an allen Pegeln:")
        print(performance.to_string(index=False))
        print("\n============================================================")
        print("[FERTIG] HydroMod v9.4 Nested-Gauge Diagnostic Calibration")
        print(f"Ergebnisse: {out}")
        print("Wichtige Dateien:")
        print(" - tables/daily_performance_metrics_multigauge.csv")
        print(" - tables/qobs_source_diagnostics.csv")
        print(" - tables/gauge_routing_diagnostics.csv")
        print(" - tables/gauge_snap_candidates_top20.csv")
        print(" - tables/calibration_trace_daily_v9_multigauge.csv")
        print(" - tables/v94_lowflow_season_reach_diagnostics.csv")
        print(" - tables/nested_reach_zone_definitions.csv")
        print(" - maps/regional_precipitation_factor.tif")
        print(" - maps/river_mean_discharge_m3s.tif")
        print(" - maps/catchment_42880550.gpkg")
        print(" - maps/catchment_42880800.gpkg")
        print(" - maps/catchment_42880458.gpkg")
        print("============================================================")
    finally:
        precip.close()

if __name__ == "__main__":
    # Fachlicher V9.4-Multi-Pegel-Endlauf
    DCFG.USE_CALIBRATION = True
    MULTIGAUGE_SAMPLE_CELLS_PER_GAUGE = 5000
    DCFG.CALIBRATION_MAXITER = 55
    DCFG.CALIBRATION_POPSIZE = 8
    DCFG.CALIBRATION_TOL = 0.001
    DCFG.LOCAL_REFINEMENT_FRACTION = 0.04
    DCFG.SPINUP_CYCLES = 8
    DCFG.RANDOM_SEED = 42
    main_v9_daily()
