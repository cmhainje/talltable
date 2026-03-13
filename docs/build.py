from bs4 import BeautifulSoup
from shutil import copy
from subprocess import run
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC.parent / "docs-dist"

OUT.mkdir(exist_ok=True)

# copy assets
for p in SRC.glob("*.css"):
    copy(p, OUT)


def wrap_sections(soup, root, tags, level_index=0):
    if level_index >= len(tags):
        return
    
    tag = tags[level_index]
    for heading in list(soup.find_all(tag)):
        if "title" in heading.get("class", []):
            continue

        details = root.new_tag("details", open=True)
        summary = root.new_tag("summary")
        
        heading.replace_with(details)
        summary.append(heading)
        details.append(summary)
        
        # Collect siblings until same or higher level heading
        same_or_higher = tags[:level_index + 1]
        sib = details.find_next_sibling()
        while sib and sib.name not in same_or_higher:
            nxt = sib.find_next_sibling()
            details.append(sib.extract())
            sib = nxt
        
        # Recurse into this details block for deeper headings
        wrap_sections(details, root, tags, level_index + 1)


# pandoc markdown files
for p in SRC.glob("*.md"):
    outpath = str(OUT / (p.stem + ".html"))
    _proc = run(
        [
            "pandoc",
            str(p),
            "-s",
            "--toc",
            "--template",
            str(SRC / "template.html"),
            "-o",
            outpath,
        ],
        text=True,
        capture_output=True,
    )
    if _proc.returncode != 0:
        raise RuntimeError(f"failed to build {p}: {_proc.stderr}")

    with open(outpath, "r") as f:
        soup = BeautifulSoup(f, "html.parser")

    wrap_sections(soup, soup, ["h1", "h2", "h3", "h4", "h4", "h5", "h6"])

    with open(outpath, "w") as f:
        f.write(str(soup))

    print(f"  built {outpath}")

