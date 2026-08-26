#!/usr/bin/env python3
"""Static file server for local development.

Same job as `python3 -m http.server`, with one addition: it sends `no-store`. Safari
in particular caches the viewer's JS and wasm hard enough to keep serving a stale copy
after you have already fixed something, which makes changes look like they had no
effect. Use this while iterating on `viewers/`; plain `http.server` is fine otherwise.

    python3 tools/serve.py            # http://localhost:8000
    python3 tools/serve.py 8765       # pick a port
"""

import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"serving on http://localhost:{port}  (no-store)")
    print("press ctrl-c to stop")
    try:
        ThreadingHTTPServer(("", port), NoCacheHandler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
