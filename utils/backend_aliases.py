# backend_aliases.py

import sys, os, inspect

def setup_backend_path():
    plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vdir = os.path.join(plugin_root, "utils", "backends")
    if vdir not in sys.path:
        sys.path.append(vdir)  # 경로 오염 방지: append
