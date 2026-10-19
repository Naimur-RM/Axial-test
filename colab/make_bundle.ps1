# Packs the code Colab needs into output/axialmb_v2/axialmb_v2_code.zip.
# Python's zipfile writes forward-slash entries, which Linux unzip expects.
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
python -c @"
import pathlib, zipfile
out = pathlib.Path('output/axialmb_v2/axialmb_v2_code.zip')
out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for top in ('axialmb', 'experiments'):
        for f in pathlib.Path(top).rglob('*.py'):
            z.write(f, f.as_posix())
    z.write('requirements-axialmb.txt')
print('wrote', out)
"@
Pop-Location
