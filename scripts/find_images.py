"""
find_images.py

Performs a cone search over SPHEREx image targets using the IRSA TAP service
and returns the filepaths for all contained images.
"""

from argparse import ArgumentParser
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy import units as u

ap = ArgumentParser()
ap.add_argument('ra', help='right ascension')
ap.add_argument('dec', help='declination')
ap.add_argument('radius', help='search radius [degrees]', type=float)
ap.add_argument('output', help='path to dump output file list')
ap.add_argument('--limit', type=int, default=None)
args = ap.parse_args()

# coord = SkyCoord(ra="06h33m45s", dec="+04d59m54s")
# result = Irsa.query_sia(pos=(coord, 5 * u.deg), collection='spherex_qr2')
coord = SkyCoord(ra=args.ra, dec=args.dec)
result = Irsa.query_sia(pos=(coord, args.radius * u.deg), collection='spherex_qr2', maxrec=args.limit)
urls = set(
    u.split("spherex/qr2/", maxsplit=1)[1]
    for u in result["access_url"]
)
print(f"{len(urls)} images found. writing to {args.output}")

with open(args.output, 'w') as f:
    f.write('\n'.join(urls))
