import argparse

from . import run_cli_app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9011, help='Port for the DL manager to bind to.')
    parser.add_argument('--certfile', type=str, help='Certificate file for HTTPS')
    parser.add_argument('--keyfile', type=str, help='Key file for HTTPS')
    args = parser.parse_args()
    run_cli_app(args.keyfile, args.certfile, args.port)

if __name__ == '__main__':
    main()
