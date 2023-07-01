import argparse

import fastapi

from . import run_cli_app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9011, help='Port for the DL manager to bind to.')
    parser.add_argument('--certfile', type=str, help='Certificate file for HTTPS', default='')
    parser.add_argument('--keyfile', type=str, help='Key file for HTTPS', default='')
    parser.add_argument('--script', type=str, help='Path to a JSON file describing a set of endpoints to be called.', default='')
    parser.add_argument('--invalidate-checkpoints', action='store_true', default=False, help='If in --script mode, invalidate checkpoints')
    args = parser.parse_args()
    try:
        run_cli_app(args.keyfile, args.certfile, args.port, args.script, args.invalidate_checkpoints)
    except fastapi.HTTPException as e:
        raise ValueError(e.detail)

if __name__ == '__main__':
    main()
