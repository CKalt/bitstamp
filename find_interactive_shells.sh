#\!/bin/bash
# Find Python scripts that likely implement interactive shells or CLIs

echo "=== Searching for interactive shell/CLI implementations ==="
echo ""

# Search for various interactive shell patterns
grep -l -E "(^import cmd|from cmd import|class.*\(cmd\.Cmd\)|^import code|from code import|InteractiveConsole|InteractiveShell|^import IPython|from IPython|^import prompt_toolkit|from prompt_toolkit|^import click|from click import|@click\.|^import fire|from fire import|^import argparse|ArgumentParser|^import readline|from readline import|^import rlcompleter|^import pdb|from pdb import|^import bpython|^import ptpython|^import rich\.console|from rich\.console|^import questionary|from questionary import|^import inquirer|from inquirer import|^import curses|from curses import|^import urwid|from urwid import|class.*Shell|InteractiveMode|CommandInterface|CLI\(|REPL\()" *.py 2>/dev/null | while read file; do
    echo "📄 $file:"
    # Show what type of interactive module it uses
    grep -H -E "(^import cmd|from cmd import|class.*\(cmd\.Cmd\))" "$file" 2>/dev/null | head -1 && echo "   → Uses cmd module (command interpreter)"
    grep -H -E "(^import code|from code import|InteractiveConsole)" "$file" 2>/dev/null | head -1 && echo "   → Uses code module (interactive console)"
    grep -H -E "(^import IPython|from IPython)" "$file" 2>/dev/null | head -1 && echo "   → Uses IPython (enhanced interactive shell)"
    grep -H -E "(^import prompt_toolkit|from prompt_toolkit)" "$file" 2>/dev/null | head -1 && echo "   → Uses prompt_toolkit (advanced CLI)"
    grep -H -E "(^import click|from click import|@click\.)" "$file" 2>/dev/null | head -1 && echo "   → Uses click (command line interface creation)"
    grep -H -E "(^import fire|from fire import)" "$file" 2>/dev/null | head -1 && echo "   → Uses fire (automatic CLI generation)"
    grep -H -E "(^import readline|from readline import)" "$file" 2>/dev/null | head -1 && echo "   → Uses readline (line editing)"
    grep -H -E "(^import curses|from curses import)" "$file" 2>/dev/null | head -1 && echo "   → Uses curses (terminal UI)"
    grep -H -E "(^import rich\.console|from rich\.console)" "$file" 2>/dev/null | head -1 && echo "   → Uses rich (rich terminal output)"
    grep -H -E "(^import questionary|from questionary|^import inquirer|from inquirer)" "$file" 2>/dev/null | head -1 && echo "   → Uses questionary/inquirer (interactive prompts)"
    echo ""
done

# Also show a summary count
echo "=== Summary ==="
echo "Files using cmd module: $(grep -l "^import cmd\|from cmd import\|cmd\.Cmd" *.py 2>/dev/null | wc -l)"
echo "Files using click: $(grep -l "^import click\|from click import\|@click\." *.py 2>/dev/null | wc -l)"
echo "Files using argparse: $(grep -l "^import argparse\|ArgumentParser" *.py 2>/dev/null | wc -l)"
echo "Files with 'Shell' classes: $(grep -l "class.*Shell" *.py 2>/dev/null | wc -l)"
