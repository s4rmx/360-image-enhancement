from pathlib import Path

import numpy as np
import piexif
import pytest
from PIL import Image

from enhance.io import AspectRatioError, build_manifest, load_image, save_jpeg, validate_equirect


def test_exif_xmp_survive(tmp_path: Path):
    # Build a tiny JPEG with EXIF + XMP APP1
    arr = np.zeros((32, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = 120
    arr[:, :, 1] = 100
    arr[:, :, 2] = 80
    src = tmp_path / "src.jpg"
    exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"TestCam"}, "Exif": {}, "GPS": {}, "1st": {}})
    Image.fromarray(arr).save(src, quality=95, exif=exif)

    # Inject minimal XMP with GPano hint
    raw = src.read_bytes()
    xmp_payload = (
        b"http://ns.adobe.com/xap/1.0/\x00"
        b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"'
        b' GPano:ProjectionType="equirectangular"/>'
        b"</rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
    )
    seg = b"\xff\xe1" + (len(xmp_payload) + 2).to_bytes(2, "big") + xmp_payload
    # Insert after SOI
    src.write_bytes(raw[:2] + seg + raw[2:])

    img = load_image(src)
    dst = tmp_path / "out.jpg"
    save_jpeg(dst, img, source_path=src, quality=95)
    out = dst.read_bytes()
    assert b"GPano" in out or b"equirectangular" in out
    # EXIF Make should survive
    loaded = piexif.load(str(dst))
    assert loaded["0th"].get(piexif.ImageIFD.Make) in (b"TestCam", "TestCam")


def test_non_2_1_rejected(tmp_path: Path):
    bad = tmp_path / "ds"
    bad.mkdir()
    p = bad / "square.jpg"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(p)
    meta = validate_equirect(p)
    assert meta["ok_ratio"] is False
    with pytest.raises(AspectRatioError):
        build_manifest(tmp_path, tmp_path / "manifest.csv", stop_on_bad_ratio=True)
