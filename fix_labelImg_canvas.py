"""
Fix LabelImg crash: drawLine/drawRect get float instead of int (PyQt5/Python 3.13).
Run once: python fix_labelImg_canvas.py
"""
import sys
from pathlib import Path

# Find canvas.py (in site-packages/libs/ when using labelImg)
canvas_path = None
for s in sys.path:
    candidate = Path(s) / "libs" / "canvas.py"
    if candidate.exists():
        canvas_path = candidate
        break
if not canvas_path:
    print("Could not find labelImg libs/canvas.py")
    sys.exit(1)

text = canvas_path.read_text(encoding="utf-8", errors="replace")

# Fix drawLine and drawRect: wrap numeric args in int()
replacements = [
    (
        "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())",
        "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))",
    ),
    (
        "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
        "p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))",
    ),
    (
        "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
        "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))",
    ),
]

changed = False
for old, new in replacements:
    if old in text and new not in text:
        text = text.replace(old, new)
        changed = True

if changed:
    canvas_path.write_text(text, encoding="utf-8")
    print(f"Patched: {canvas_path}")
    print("Run labelImg again: python -m labelImg")
else:
    print("No changes needed (maybe already patched or different version).")
    print("If labelImg still crashes, try installing the HumanSignal fork:")
    print("  pip uninstall labelImg")
    print("  pip install labelImg  (or: pip install git+https://github.com/HumanSignal/labelImg.git)")
