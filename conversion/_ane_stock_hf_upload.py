"""Upload a Neural Engine lane staging dir (from `export_ane_stock.py --stage <Repo>`) to HF.
Generalizes `_minicpm5_ane_hf_upload.py`.

Adds two subtrees next to the repo's existing GPU bundle:
  ios-ane-h18p/   metadata.json + <name>.h18p.aimodelc + tokenizer/   — AOT for iPhone 17-class (h18p), ANE
  ios-static/     metadata.json + <name>.aimodel      + tokenizer/   — the same stock static export before AOT
and inserts a card section (a markdown file you write at ship time, with the measured numbers and
the gate transcript path) above the `--anchor` heading if the section's first line is not there yet.

    coreai-models/.venv/bin/python conversion/_ane_stock_hf_upload.py \\
        --repo mlboydaisuke/<Repo> --stage coreai-models/exports/hf_stage/<Repo> \\
        --section conversion/artifacts/ane_card_<model>.md [--anchor "## Measured"] [--stage-only]
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

SUBTREES = ["ios-ane-h18p", "ios-static"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--stage", required=True, help="staging dir holding ios-ane-h18p/ and ios-static/")
    ap.add_argument("--section", default=None, help="markdown file to insert into the card (first line = marker)")
    ap.add_argument("--anchor", default="## Measured", help="insert the section above this heading")
    ap.add_argument("--stage-only", action="store_true", help="prepare README.md in the stage dir, upload nothing")
    ap.add_argument("--subtrees", default=",".join(SUBTREES))
    a = ap.parse_args()
    stage = Path(a.stage).resolve()
    subtrees = [s for s in a.subtrees.split(",") if s]
    for sub in subtrees:
        d = stage / sub
        assert (d / "metadata.json").exists() and (d / "tokenizer").is_dir(), f"stage incomplete: {d}"
    api = HfApi()
    who = api.whoami()["name"]
    assert a.repo.split("/")[0] == who, f"repo owner is not {who}: {a.repo} (other people's repos take a PR + user GO)"

    if a.section:
        section = Path(a.section).read_text()
        mark = section.splitlines()[0].strip()
        readme = Path(hf_hub_download(a.repo, "README.md"))
        card = readme.read_text()
        if mark in card:
            print("card: section already present")
        else:
            assert a.anchor in card, f"anchor {a.anchor!r} not in the card; insert by hand"
            card = card.replace(a.anchor, section.rstrip() + "\n\n" + a.anchor, 1)
            (stage / "README.md").write_text(card)
            print("card: section inserted ->", stage / "README.md")
    if a.stage_only:
        print("stage only; nothing uploaded")
        return
    for sub in subtrees:
        print("uploading", sub)
        api.upload_folder(repo_id=a.repo, folder_path=str(stage / sub), path_in_repo=sub,
                          commit_message=f"{sub}: stock static iOS export, AOT h18p neural-engine (device gate transcript in the zoo)")
    if (stage / "README.md").exists():
        api.upload_file(repo_id=a.repo, path_or_fileobj=str(stage / "README.md"), path_in_repo="README.md",
                        commit_message="card: Neural Engine bundle section")
    print("done")


if __name__ == "__main__":
    main()
