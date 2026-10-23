// Linear buckling of tragwerk3d models from a JSON file: α_cr per ULS combination and the governing modes.
// Usage: node web/parity_buckling.js models.json > out.json
const fs = require("fs");
const T = require("./bridge.js");
const items = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = items.map((item) => {
  try {
    const m = T.parseModel3d(item.model), r = T.buckling(m, item.options || {});
    return { ok: true, governing: r.governing, alpha: r.alpha, combos: r.combos.map((c) => ({ id: c.id, alpha: c.alpha })), warnings: r.warnings,
      modes: r.modes.map((v) => Array.from(v)) };
  } catch (e) {
    return { ok: false, error: `${e.constructor.name}: ${e.message}`, stack: e.stack };
  }
});
process.stdout.write(JSON.stringify(out));
