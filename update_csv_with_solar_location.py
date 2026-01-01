import math
import pandas as pd
import pytz
from pysolar.solar import get_altitude, get_azimuth

# -----------------------------
# Station metadata (Metas site)
# -----------------------------
SITE_LAT = 37.0916
SITE_LON = -2.3636
SITE_TZ = "Etc/GMT-1"   # UTC+1 (IANA tz format)

TIME_COL = "timestamp"

# --------------------------------------
# Compute cyclic + solar-geometry values
# --------------------------------------
def compute_time_features(t, lat=SITE_LAT, lon=SITE_LON, tz=SITE_TZ):
    t = pd.to_datetime(t)

    # Localize if naive
    if t.tzinfo is None:
        t = pytz.timezone(tz).localize(t)

    hour = t.hour + t.minute / 60.0
    day_of_year = t.timetuple().tm_yday

    # Cyclic encodings
    tod_sin = math.sin(2 * math.pi * hour / 24)
    tod_cos = math.cos(2 * math.pi * hour / 24)

    doy_sin = math.sin(2 * math.pi * day_of_year / 365)
    doy_cos = math.cos(2 * math.pi * day_of_year / 365)

    # Solar position
    altitude = get_altitude(lat, lon, t)

    # Avoid nighttime negative altitude -> stabilize model
    if altitude < 0:
        altitude = 0.0

    zenith = 90.0 - altitude
    azimuth = get_azimuth(lat, lon, t)

    return pd.Series({
        "tod_sin": tod_sin,
        "tod_cos": tod_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "solar_zenith": zenith,
        "solar_azimuth": azimuth,
    })


if __name__ == "__main__":

    INPUT_CSV = "dataset_full_1M.csv"
    OUTPUT_CSV = "dataset_full_1M_solar.csv"

    print("\nLoading dataset...")
    df = pd.read_csv(INPUT_CSV)

    if TIME_COL not in df.columns:
        raise ValueError(f"Timestamp column '{TIME_COL}' not found in CSV")

    print("Computing solar-geometry + cyclic time features...")
    feats = df[TIME_COL].apply(compute_time_features)

    print("Appending new feature columns...")
    df = pd.concat([df, feats], axis=1)

    print(f"Saving updated dataset → {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nDone ✔")
    print("Added columns:")
    for c in feats.columns:
        print(" •", c)
