from __future__ import annotations
import os
os.makedirs("icartt", exist_ok=True)
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import getpass
import requests
import pandas as pd
from zipfile import ZipFile
from datetime import datetime, timedelta
from IPython.display import display
from http.cookiejar import MozillaCookieJar



PathLike = Union[str, Path]


@dataclass(frozen=True)
class ICARTTInfo:
    path: Path
    header_length: int
    ffi: str


@dataclass(frozen=True)
class VariableDef:
    name: str
    unit: Optional[str] = None
    description: Optional[str] = None
    missing: Optional[float] = None  # per-variable missing, if known


class ICARTTReader:
    """
    General ICARTT/ICT reader.

    Goals:
      - Robustly read the data table (CSV-like) for typical ICT files.
      - Avoid file-specific assumptions (campaign/platform/column names).
      - Provide best-effort metadata parsing (especially for FFI=1001) but
        never let metadata parsing prevent data extraction.

    Notes:
      - Many airborne ICT files are FFI=1001 (1D time series), but other FFIs exist.
      - Header length is always the first token on the first line in the files you've shown.
    """

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self.info = self._read_info()

    # ----------------------------
    # Core: file format info
    # ----------------------------
    def _read_info(self) -> ICARTTInfo:
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            line1 = f.readline().strip()

        parts = [p.strip() for p in line1.split(",")]
        if len(parts) < 2:
            raise ValueError(f"Unexpected ICARTT first line format: {line1!r}")

        header_length = int(parts[0])
        ffi = parts[1]
        return ICARTTInfo(path=self.path, header_length=header_length, ffi=ffi)

    def read_header_lines(self) -> List[str]:
        """Return the raw header lines (including line 1)."""
        n = self.info.header_length
        lines: List[str] = []
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        return lines

    # ----------------------------
    # Minimal assumptions: table extraction
    # ----------------------------
    def read_table(
        self,
        *,
        na_values: Optional[List[Union[str, float, int]]] = None,
        strip_colnames: bool = True,
    ) -> pd.DataFrame:
        """
        Extract the data table.

        Strategy:
          - Most ICARTT files place the column header row at line `header_length`.
          - So we skip `header_length - 1` lines and let pandas treat the next line as header.

        This is general and does not depend on campaign/platform.
        """
        skiprows = max(self.info.header_length - 1, 0)

        # Many ICT files use -9999, -99999, etc., but we won't assume; allow caller to pass.
        # We'll also attempt to auto-detect common missing indicators from header if possible.
        if na_values is None:
            na_values = self._guess_missing_values()

        df = pd.read_csv(
            self.path,
            skiprows=skiprows,
            sep=",",
            encoding = "latin-1",
            encoding_errors = "ignore",
            engine="python",
            na_values=na_values,
        )

        if strip_colnames:
            df.columns = [str(c).strip() for c in df.columns]

        return df

    # ----------------------------
    # Best-effort metadata parsing
    # ----------------------------
    def read_metadata(self) -> Dict[str, str]:
        """
        Best-effort metadata extraction.

        Returns a dict of key metadata fields when the header matches common ICARTT layouts.
        If parsing fails, returns what it can without throwing.
        """
        lines = self.read_header_lines()
        meta: Dict[str, str] = {}

        # Common ICARTT: line indices below assume a conventional layout often used with FFI=1001.
        # We'll guard everything with length checks.
        def safe(i: int) -> str:
            return lines[i].strip() if 0 <= i < len(lines) else ""

        meta["path"] = str(self.path)
        meta["header_length"] = str(self.info.header_length)
        meta["ffi"] = self.info.ffi

        # These are common but not guaranteed. Keep them best-effort.
        meta["pi"] = safe(1)
        meta["organization"] = safe(2)
        meta["data_description"] = safe(3)
        meta["mission"] = safe(4)
        meta["volume_info"] = safe(5)
        meta["date_info"] = safe(6)
        meta["data_interval"] = safe(7)
        meta["independent_variable"] = safe(8)
        meta["seconds"] = safe(9)

        return {k: v for k, v in meta.items() if v}

    def read_variable_defs(self) -> List[VariableDef]:
        """
        Best-effort variable definitions, primarily for common FFI=1001 layout:
          - line 10: number of dependent variables
          - line 12+: variable definition lines (often "NAME, UNIT, DESCRIPTION...")

        If layout doesn't match, returns empty list.
        """
        lines = self.read_header_lines()

        # Attempt the common ICARTT/FFI=1001 positions
        # Line 10 (0-index 9) is often number of dependent variables.
        if len(lines) < 11:
            return []

        try:
            n_dep = int(lines[9].strip())
        except Exception:
            return []

        start = 12  # 0-index start of var definition block in common layout
        block = lines[start : start + n_dep]
        out: List[VariableDef] = []

        for ln in block:
            parts = [p.strip() for p in ln.split(",")]
            if not parts:
                continue
            name = parts[0]
            unit = parts[1] if len(parts) > 1 else None
            desc = ",".join(parts[2:]).strip() if len(parts) > 2 else None
            out.append(VariableDef(name=name, unit=unit or None, description=desc or None))

        # Attach per-variable missing if we can infer it (optional)
        miss_map = self._guess_per_variable_missing()
        if miss_map:
            out = [
                VariableDef(v.name, v.unit, v.description, miss_map.get(v.name))
                for v in out
            ]

        return out

    # ----------------------------
    # Missing-value handling
    # ----------------------------
    def _guess_missing_values(self) -> List[Union[str, float, int]]:
        """
        Heuristic: try to extract missing indicators from the header.
        Falls back to common sentinel values.

        Many ICT files have a line describing missing indicators (often around line 12),
        but formats vary. We keep this conservative.
        """
        lines = self.read_header_lines()
        candidates: List[Union[str, float, int]] = []

        # Scan header for something that looks like a missing indicator list: "-9999" etc.
        for ln in lines[: min(len(lines), 200)]:
            # find numeric sentinels that look like -9999, -99999, 9999 etc.
            for tok in ln.replace(",", " ").split():
                if tok.startswith(("-", "+")) and tok[1:].isdigit():
                    val = int(tok)
                    # common missing sentinels are large magnitude
                    if abs(val) >= 999:
                        candidates.append(val)

        # De-duplicate while preserving order
        seen = set()
        ordered = []
        for v in candidates:
            if v not in seen:
                seen.add(v)
                ordered.append(v)

        # Add very common defaults if we found nothing
        if not ordered:
            ordered = [-9999, -99999, -8888, 9999, 99999]

        return ordered

    def _guess_per_variable_missing(self) -> Dict[str, float]:
        """
        Some ICARTT headers specify per-variable missing indicators.
        This is not standardized across all producers; implement only as a best-effort hook.

        Returns {} if nothing reliable is found.
        """
        # For now, keep minimal: many files effectively use a single sentinel across columns.
        # You can extend this if you encounter a known pattern you want to support.
        return {}

    # ----------------------------
    # Exports
    # ----------------------------
    def to_csv(
        self,
        out: Optional[PathLike] = None,
        *,
        na_values: Optional[List[Union[str, float, int]]] = None,
        strip_colnames: bool = True,
    ) -> Path:
        df = self.read_table(na_values=na_values, strip_colnames=strip_colnames)
        out_path = Path(out) if out else self.path.with_suffix(".csv")
        df.to_csv(out_path, index=False)
        return out_path

    def to_parquet(
        self,
        out: Optional[PathLike] = None,
        *,
        na_values: Optional[List[Union[str, float, int]]] = None,
        strip_colnames: bool = True,
    ) -> Path:
        df = self.read_table(na_values=na_values, strip_colnames=strip_colnames)
        out_path = Path(out) if out else self.path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        return out_path

open("icartt/_init_.py", "w").close()




home = Path.home()
netrc_path = home / "_netrc"

username = input("Enter username: ")
password = getpass.getpass("Enter password: ")

content = f"machine urs.earthdata.nasa.gov login {username} password {password}"

with open(netrc_path, "w") as f:
    f.write(content)

os.chmod(netrc_path, 0o600)

print(f".netrc file created at {netrc_path}")

file_path = home / ".urs_cookies"
file_path.touch(exist_ok=True)
print(f"Created: {file_path}")

session = requests.Session()
session.auth = None  # requests will use your .netrc automatically

auth_url = "http://asdc.larc.nasa.gov/soot-api/Authenticate/user"




#campaigns
base_url = "https://asdc.larc.nasa.gov/soot-api/campaigns"
response = requests.get(base_url)
response.raise_for_status()
campaign_json = response.json()
campaign_table = pd.DataFrame(campaign_json)

with pd.option_context('display.max_rows', None, 'display.max_columns', None,'display.max_colwidth', 200):
    display(campaign_table[["projectacronym", "description", "projecttitles"]])

campaign = input("Choose Campaign: ").strip()

# robust index selection (handles 0 or multiple matches)
matches = campaign_table.index[
    campaign_table["projectacronym"].astype(str).str.strip().str.upper() == campaign.upper()
]

if len(matches) == 0:
    raise ValueError(f"Campaign '{campaign}' not found. Check spelling/case.")
if len(matches) > 1:
    raise ValueError(f"Campaign '{campaign}' matched multiple rows: {matches.tolist()}")

index = matches[0]

campaign_specification = campaign_table.loc[index, "projectacronym"]  # choose campaign name
url = f'{base_url}/years/{campaign_specification}'
response = requests.get(url)
response.raise_for_status()
years_for_campaign_json = response.json()
years_for_campaign_table = pd.DataFrame(years_for_campaign_json)


# --- YEARS for given campaign ---
with pd.option_context(
    "display.max_rows", None,
    "display.max_columns", None,
    "display.max_colwidth", 200
):
    display(years_for_campaign_table)

year = input("Choose Year: ").strip()

# FIX: avoid int(numpy_array); select a single matching index robustly
year_matches = years_for_campaign_table.index[
    years_for_campaign_table["year"].astype(str).str.strip() == year
]
if len(year_matches) == 0:
    raise ValueError(f"Year '{year}' not found.")
if len(year_matches) > 1:
    raise ValueError(f"Year '{year}' matched multiple rows: {year_matches.tolist()}")

year_index = year_matches[0]
year_specification = years_for_campaign_table.loc[year_index, "year"]  # safer than iloc


url = f"{base_url}/years/{campaign_specification}/{year_specification}"
response = requests.get(url)
response.raise_for_status()
platforms_for_year_json = response.json()
platforms_for_year_table = pd.DataFrame(platforms_for_year_json)


# --- PI for given platform for given year for given campaign ---
with pd.option_context(
    "display.max_rows", None,
    "display.max_columns", None,
    "display.max_colwidth", 200
):
    display(platforms_for_year_table)

platform = input("Choose Platform (name NOT platformtype - DO NOT CHOOSE SATELLITE): ").strip()

platform_matches = platforms_for_year_table.index[
    platforms_for_year_table["name"].astype(str).str.strip().str.upper() == platform.upper()
]
if len(platform_matches) == 0:
    raise ValueError(f"Platform '{platform}' not found.")
if len(platform_matches) > 1:
    raise ValueError(f"Platform '{platform}' matched multiple rows: {platform_matches.tolist()}")

platform_index = platform_matches[0]
platform_specification = platforms_for_year_table.loc[platform_index, "name"]


url = f"{base_url}/years/{campaign_specification}/{year_specification}/{platform_specification}"
response = requests.get(url)
response.raise_for_status()
pi_for_platform_json = response.json()
pi_for_platform_table = pd.DataFrame(pi_for_platform_json)


# --- FILE NAMES for given PI for given platform for given year for given campaign ---
with pd.option_context(
    "display.max_rows", None,
    "display.max_columns", None,
    "display.max_colwidth", None  # FIX: pandas no longer accepts -1; use None for unlimited
):
    display(pi_for_platform_table[["investigatorid", "firstname", "lastname"]])

pi = input("Choose PI Last Name (Copy Exactly): ").strip()

# FIX: avoid int(numpy_array); handle duplicates safely
pi_matches = pi_for_platform_table.index[
    pi_for_platform_table["lastname"].astype(str).str.strip().str.upper() == pi.upper()
]
if len(pi_matches) == 0:
    raise ValueError(f"PI last name '{pi}' not found.")
if len(pi_matches) > 1:
    # Show duplicates so you can disambiguate
    dupes = pi_for_platform_table.loc[pi_matches, ["investigatorid", "firstname", "lastname"]]
    raise ValueError(
        f"PI last name '{pi}' matched multiple rows. Please choose a unique PI.\n"
        f"{dupes.to_string(index=False)}"
    )

pi_index = pi_matches[0]
pi_specification = pi_for_platform_table.loc[pi_index, "lastname"]


url = f"{base_url}/years/{campaign_specification}/{year_specification}/{platform_specification}/{pi_specification}"
response = requests.get(url)
response.raise_for_status()
file_names_for_pi_json = response.json()
file_names_for_pi_table = pd.DataFrame(file_names_for_pi_json)
session = requests.Session()
session.auth = None  # requests will use your .netrc automatically


auth_url = "http://asdc.larc.nasa.gov/soot-api/Authenticate/user"
download_files_url = "https://asdc.larc.nasa.gov/soot-api/data_files/downloadFiles"

# Load the cookie file from where it was created
cookie_file_path = file_path  # Use the same path as created above

cookies = MozillaCookieJar((cookie_file_path))
# Only load if the file exists and has content (to avoid format errors on empty file)
if cookie_file_path.exists() and cookie_file_path.stat().st_size > 0:
    cookies.load(ignore_expires=True)

session = requests.Session()
session.cookies.update(cookies)

# Authorize ONCE (outside loop) so failures are obvious
auth_resp = session.get(auth_url, allow_redirects=True)
print("AUTH status:", auth_resp.status_code, "| final URL:", auth_resp.url)

if auth_resp.status_code != 200:
    print("AUTH body (first 300 chars):", auth_resp.text[:300])
    raise RuntimeError("ERROR: User not authorized. Check that .urs_cookies exists, is valid, and is being loaded correctly.")

# Save updated cookies after successful auth
jar = MozillaCookieJar(str(cookie_file_path))
for cookie in session.cookies:
    jar.set_cookie(cookie)
jar.save(ignore_discard=True, ignore_expires=True)

print("User authorized. Beginning downloads...")

# Download each file listed
for file in file_names_for_pi_table["filename"]:
    file = str(file).strip()
    zip_file_name = file.split(".ict")[0]  # base name for zip

    # Use params= for proper URL encoding of filenames
    resp = session.get(download_files_url, params={"filenames": file}, allow_redirects=True)

    if resp.status_code != 200:
        print(f"ERROR: Unable to download {file}. Response code {resp.status_code}")
        print("Final URL:", resp.url)
        print("Body (first 300 chars):", resp.text[:300])
        continue

    out_zip = f"{zip_file_name}.zip"
    with open(out_zip, "wb") as f:
        f.write(resp.content)

    # unzip the file you downloaded
    try:
        with ZipFile(os.path.join(os.getcwd(), out_zip), "r") as zObject:
            zObject.extractall()
    except Exception as e:
        print(f"ERROR: Failed to unzip {out_zip}: {e}")
        # keep the zip for inspection if unzip fails
        continue

    # delete the zip file, only keep the unzipped files
    os.remove(out_zip)

folder = os.getcwd()

file_paths = []
for entry in os.scandir(folder):
    if entry.is_file():
        if entry.path[-3:] == "ict":
            file_paths += [entry.path]

format = '%Y,%m,%d'

# Collect all dataframes first, then concat once (much faster than concat in loop)
dfs_to_combine = []
for file in file_paths:
    r = ICARTTReader(file)
    df = r.read_table()
    meta = r.read_metadata()
    vars_ = r.read_variable_defs()
    
    #find start date/time (only first 3 values)
    s = meta.get("date_info").split(',')
    s = ','.join(s[0:3])
    s = s.replace(" ", "")
    
    start_date = datetime.strptime(s, format)
    start_time = timedelta(seconds = int(meta.get("seconds")))
    start_datetime = start_date + start_time
    
    #find columns that have UTC seconds (Start_UTC, Seconds_UTC, etc.)
    time_columns = [col for col in df.columns if "UTC" in col.upper()]
    
    #create new column with full date listed
    for col in time_columns:
        new_col_name = col.replace("UTC", "Datetime")
        df[new_col_name] = start_datetime + pd.to_timedelta(df[col], unit = "s")
    
    #some data types have Time instead of UTC
    new_time_columns = []
    if len(time_columns) == 0:
        new_time_columns = [col for col in df.columns if "TIME" in col.upper()]
    
    for col in new_time_columns:
        column = col.title()
        new_col_name = column.replace("Time", "Datetime")
        df[new_col_name] = start_datetime + pd.to_timedelta(df[col], unit = "s")
   
    # Collect dataframe to combine later
    dfs_to_combine.append(df)

# Combine all dataframes at once (much faster than concatenating in loop)
combined_df = pd.concat(dfs_to_combine, ignore_index=True) if dfs_to_combine else pd.DataFrame()

out_path = Path(f"{folder}\\{campaign}_{year}_{platform}_{pi}.csv")
csv_path = combined_df.to_csv(out_path)

#remove ICT files, only csvs left
for file in file_paths:
    file_name = file.split("\\")[-1]
    os.remove(file_name)