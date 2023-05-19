from .cli import main as _main

def run_cli_app(keyfile, certfile, port=9011, script=''):
    _main(port, keyfile, certfile, script)

