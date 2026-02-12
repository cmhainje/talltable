"""
find_images.py

Performs a cone search over SPHEREx image targets using the IRSA TAP service
and returns the filepaths for all contained images.
"""

from argparse import ArgumentParser
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord

ap = ArgumentParser()
ap.add_argument('ra', help='right ascension')
ap.add_argument('dec', help='declination')
ap.add_argument('radius', help='search radius [degrees]', type=float)
ap.add_argument('output', help='path to dump output file list')
ap.add_argument('--limit', type=int, default=None)
args = ap.parse_args()

coord = SkyCoord(ra=args.ra, dec=args.dec)
limit_str = f"TOP {args.limit}" if args.limit is not None else ""
# query = f"""
# SELECT {limit_str} a.uri as uri
# FROM spherex.obscore o
# JOIN spherex.observation obs ON o.obs_id = obs.observationid
# JOIN spherex.plane       p   ON obs.obsid = p.obsid
# JOIN spherex.artifact    a   ON p.planeid = a.planeid
# WHERE CONTAINS(
#     POINT(o.s_ra, o.s_dec),
#     CIRCLE({coord.ra.deg:.4f}, {coord.dec.deg:.4f}, {args.radius:.4f})
# )=1
# """

query = f"""
SELECT {limit_str} a.uri as uri
FROM spherex.obscore o
JOIN spherex.plane p ON o.obs_publisher_did = p.obs_publisher_did
JOIN spherex.artifact a ON p.planeid = a.planeid
WHERE CONTAINS(
    POINT(o.s_ra, o.s_dec),
    CIRCLE({coord.ra.deg:.4f}, {coord.dec.deg:.4f}, {args.radius:.4f})
)=1
"""

print(f"executing query:\n{query}")
result = Irsa.query_tap(query)
urls = set(
    u.split("spherex/qr2/", maxsplit=1)[1]
    for u in result.to_table()["uri"]
)
print(f"{len(urls)} images found. writing to {args.output}")

with open(args.output, 'w') as f:
    f.write('\n'.join(urls))
