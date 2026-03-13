import threading

from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from subprocess import run
from time import sleep


src_dir = Path(__file__).parent.parent / "docs"
dist_dir = Path(__file__).parent.parent / "docs-dist"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super(Handler, self).__init__(
            request, client_address, server, directory=dist_dir
        )

    def log_message(self, format, *args):
        pass  # silence request logs


def get_mod_times():
    times = {}
    for ext in ["md", "css"]:
        for p in src_dir.glob(f"*.{ext}"):
            times[p] = p.stat().st_mtime
    return times


def build():
    run(
        # [
        #     "/bin/sh",
        #     str(src_dir / "build.sh"),
        # ],
        [
            "python",
            str(src_dir / "build.py"),
        ],
        check=True,
    )



def start_server():
    server = HTTPServer(("", 8000), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("serving at http://localhost:8000/")
    return server


build()
mod_times = get_mod_times()
server = start_server()

while True:
    sleep(1)
    new_times = get_mod_times()
    if new_times != mod_times:
        print("file change detected. rebuilding...")
        mod_times = new_times
        build()

