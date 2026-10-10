"""Contact sheets: many renders on one image, each with a few lines of caption.

    sheets.contact_sheet([(path, 'line 1', 'line 2'), ...], 'sheet.png', 'title')
    sheets.labelled_sheet([('row label', [(path, 'line 1'), ...]), ...], 'sheet.png', 'title')

An entry is (image path, caption line, ...) with up to three lines, or None for an
empty cell.  Images are scaled to fit a `thumb` x `thumb` cell, keeping their shape.
"""
from PIL import Image, ImageDraw, ImageFont

from .render import Lighting

BACKGROUND = tuple(int(255 * c ** (1 / Lighting().gamma)) for c in Lighting().background)
TEXT = (220, 222, 230)
DIM = (150, 154, 168)
LINES = ((TEXT, 14), ((190, 194, 206), 13), (DIM, 12))     # (colour, size) of caption lines 1-3
LINE_H = 19


def font(size):
    return ImageFont.load_default(size=size)


def fit(draw, text, typeface, width):
    """`text`, cut short with an ellipsis if it is wider than `width` pixels."""
    if draw.textlength(text, font=typeface) <= width:
        return text
    while text and draw.textlength(text + '...', font=typeface) > width:
        text = text[:-1]
    return text + '...'


def labelled_sheet(rows, path, title, thumb=360, key=None):
    """rows: [(row label or None, [entry, ...]), ...], each row one line of cells.
    `key` is an optional PIL image (a colour legend, say) shown under the title."""
    cols = max([len(entries) for _, entries in rows] + [1])
    n_lines = max([len(e) - 1 for _, entries in rows for e in entries if e] + [1])
    label_h, row_head = 6 + LINE_H * min(n_lines, len(LINES)), 26
    header = 40 + (key.height + 12 if key is not None else 0)
    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    width = max(cols * thumb, int(probe.textlength(title, font=font(18))) + 20,
                key.width + 20 if key is not None else 0)
    height = header + sum((row_head if label else 0) + thumb + label_h for label, _ in rows)
    sheet = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=TEXT, font=font(18))
    if key is not None:
        sheet.paste(key, (10, 40))
    y = header
    for label, entries in rows:
        if label:
            draw.text((8, y + 4), label, fill=TEXT, font=font(14))
            y += row_head
        for k, entry in enumerate(entries):
            if entry is None:
                continue
            image, *lines = entry
            x = k * thumb
            with Image.open(image) as im:
                im = im.convert('RGB')
                scale = thumb / max(im.size)
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
                sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
            for j, (line, (colour, size)) in enumerate(zip(lines, LINES)):
                draw.text((x + 6, y + thumb + 3 + LINE_H * j), fit(draw, str(line), font(size), thumb - 12),
                          fill=colour, font=font(size))
        y += thumb + label_h
    sheet.save(path)


def contact_sheet(entries, path, title, thumb=360, cols=4, key=None):
    """entries in reading order, `cols` to a row (see `labelled_sheet`)."""
    rows = [(None, entries[i:i + cols]) for i in range(0, len(entries), cols)]
    labelled_sheet(rows, path, title, thumb, key)
