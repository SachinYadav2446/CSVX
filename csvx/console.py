"""Rich console styling and display utilities for CSVX."""

from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

import sys

# Ensure UTF-8 output encoding on Windows if supported
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

csvx_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
    "metric": "bold cyan",
    "score.good": "bold green",
    "score.warning": "bold yellow",
    "score.poor": "bold red",
})

console = Console(theme=csvx_theme, safe_box=True)
error_console = Console(stderr=True, theme=csvx_theme, safe_box=True)


def print_header(title: str, subtitle: Optional[str] = None):
    """Print a clean styled section header."""
    header_text = Text(title.upper(), style="bold cyan")
    if subtitle:
        header_text.append(f"\n{subtitle}", style="dim")
    console.print()
    console.print(header_text)
    console.print("-" * max(40, len(title) + 10), style="cyan")


def print_panel(content: Any, title: Optional[str] = None, border_style: str = "cyan"):
    """Print content wrapped in a Rich Panel."""
    console.print(Panel(content, title=title, border_style=border_style, expand=False))


def create_table(columns: List[str], title: Optional[str] = None) -> Table:
    """Create a standardized Rich Table."""
    table = Table(
        title=title,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        title_justify="left",
    )
    for col in columns:
        table.add_column(col)
    return table


def format_number(val: Any) -> str:
    """Format numbers cleanly with comma separators or decimals."""
    if val is None or (isinstance(val, float) and (val != val)): # NaN check
        return "N/A"
    if isinstance(val, (int,)):
        return f"{val:,}"
    if isinstance(val, float):
        if val.is_integer():
            return f"{int(val):,}"
        return f"{val:,.2f}"
    return str(val)


def format_bytes(num_bytes: int) -> str:
    """Format byte sizes into human-readable strings."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def print_error(msg: str):
    """Print an error message to stderr."""
    error_console.print(f"[bold red]Error:[/bold red] {msg}")


def print_warning(msg: str):
    """Print a warning message."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {msg}")


def print_success(msg: str):
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {msg}")
