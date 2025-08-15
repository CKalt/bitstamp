#\!/bin/bash
# Quick grep for interactive shell modules

echo "=== Files using interactive shell/CLI modules ==="
grep -l -E "(import cmd|cmd\.Cmd|import code|InteractiveConsole|import IPython|import prompt_toolkit|import click|@click\.|import fire|class.*Shell|import readline)" *.py 2>/dev/null | while read f; do
    echo "$f"
done
