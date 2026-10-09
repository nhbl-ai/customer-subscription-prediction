import io
from pathlib import Path
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ROOT = Path(__file__).resolve().parent.parent
DESTINATION = ROOT / "data" / "raw" / "bank-additional-full.csv"


def main():
    if DESTINATION.exists():
        print(f"Already exists; unchanged: {DESTINATION}")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        names = [name for name in outer.namelist() if name.endswith("bank-additional-full.csv")]
        if len(names) == 1:
            csv = outer.read(names[0])
        else:
            nested_names = [name for name in outer.namelist() if name.endswith("bank-additional.zip")]
            if len(nested_names) != 1:
                raise ValueError("Unexpected UCI archive layout; download and extract the CSV manually.")
            with zipfile.ZipFile(io.BytesIO(outer.read(nested_names[0]))) as inner:
                names = [name for name in inner.namelist() if name.endswith("bank-additional-full.csv")]
                if len(names) != 1:
                    raise ValueError("Expected exactly one bank-additional-full.csv.")
                csv = inner.read(names[0])
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open("xb") as stream:
        stream.write(csv)
    print(f"Saved: {DESTINATION}")


if __name__ == "__main__":
    main()
