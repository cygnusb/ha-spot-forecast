import json
d = json.load(open("data.json"))
k = next(i for i, t in enumerate(d["t"]) if t >= "2026-04-01")
out = {"t": d["t"][k:], "actual": d["actual"][k:], "pred": {h: v[k:] for h, v in d["pred"].items()}, "p90": {h: v[k:] for h, v in d["p90"].items()}}
s = open("template.html").read().replace("/*DATA*/", json.dumps(out, separators=(",", ":")))
open("spot-entwuerfe.html", "w").write(s)
open("preview.html", "w").write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + s + '</body></html>')
