"""Read the unchanged EUROCONTROL workbook with the standard library."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

EXP = Path(__file__).resolve().parents[1]
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
AIRPORTS = {'EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM'}


def main():
    path = EXP / 'source/Taxi-Out_Additional_Time.xlsx'
    with zipfile.ZipFile(path) as book:
        strings = [''.join(s.itertext()) for s in
                   ET.fromstring(book.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        rows = []
        for row in ET.fromstring(book.read('xl/worksheets/sheet4.xml')).findall('m:sheetData/m:row', NS):
            values = [None] * 14
            for cell in row:
                value = cell.find('m:v', NS)
                letters = ''.join(c for c in cell.get('r') if c.isalpha())
                col = 0
                for letter in letters:
                    col = col * 26 + ord(letter) - ord('A') + 1
                if col <= len(values):
                    values[col - 1] = (None if value is None else
                                      strings[int(value.text)] if cell.get('t') == 's' else value.text)
            rows.append(values)
        header = rows[0]
        records = [dict(zip(header, row)) for row in rows[1:] if len(row) >= 4 and
                   row[3] in AIRPORTS and row[0] in {'2025', '2026'}]
        terms = ' '.join(ET.fromstring(book.read('xl/drawings/drawing1.xml')).itertext())
    with (EXP / 'source/published-rows.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(records)
    (EXP / 'source/terms.txt').write_text(terms + '\n')
    record = dict(url='https://www.eurocontrol.int/performance/data/download/xls/Taxi-Out_Additional_Time.xlsx',
                  downloaded_utc=dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                  method_url='https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf',
                  workbook_changed=False, private_only=True,
                  export_rule='Not listed in release/MANIFEST.tsv; keep all exp-151 source files private.',
                  extracted_rows=len(records), source_sheet='DATA', units='minutes, converted to seconds')
    (EXP / 'source/manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
