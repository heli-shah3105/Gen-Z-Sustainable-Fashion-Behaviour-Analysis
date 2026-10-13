import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable aggressive caching for local development
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, format, *args):
        # Format clean console log
        sys.stdout.write(f"[{self.log_date_time_string()}] {format % args}\n")
        sys.stdout.flush()

def main():
    # Ensure working directory is the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Find an open port
    selected_port = None
    httpd = None
    for port in [8000, 8080, 8888, 5000, 3000]:
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
            selected_port = port
            break
        except OSError:
            continue

    if not httpd:
        print("Error: Could not bind to any standard port (8000, 8080, etc.).")
        sys.exit(1)

    url = f"http://localhost:{selected_port}/index.html"
    print("=" * 70)
    print(" GEN Z SUSTAINABLE FASHION BEHAVIOUR ANALYSIS - LOCAL HOST SERVER")
    print("=" * 70)
    print(f" Server running at : {url}")
    print(f" Serving directory : {script_dir}")
    print(" Press Ctrl+C in this console window to stop the server.")
    print("=" * 70)

    # Automatically open the browser
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()

if __name__ == '__main__':
    main()
