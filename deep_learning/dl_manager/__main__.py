import argparse

from . import run_cli_app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9011, help='Port for the DL manager to bind to.')
    parser.add_argument('--certfile', type=str, help='Certificate file for HTTPS', default='')
    parser.add_argument('--keyfile', type=str, help='Key file for HTTPS', default='')
    parser.add_argument('--script', type=str, help='Path to a JSON file describing a set of endpoints to be called.', default='')
    args = parser.parse_args()
    run_cli_app(args.keyfile, args.certfile, args.port, args.script)

if __name__ == '__main__':
    main()
