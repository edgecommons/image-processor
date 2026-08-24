/* The explorer. Hand written, no framework, no build step, no network.
 *
 * Everything the page shows is already in window.RESULTS, which data.js assigned before this file
 * ran. The three views are summary, gallery, and detail; the location hash is the whole of the
 * application state, so any view is a link somebody can send.
 */

(function () {
  "use strict";

  var DATA = window.RESULTS || { runs: [], models: [], entries: [], refusals: [] };

  var FILTER_KEYS = ["family", "model", "suite", "outcome", "run", "q"];
  var ANY = "";

  /* ---- indexes ------------------------------------------------------------------------- */

  function modelKeyOf(runId, key) {
    return runId + "\u0000" + key;
  }

  var MODELS = {};
  DATA.models.forEach(function (row) {
    MODELS[modelKeyOf(row.runId, row.key)] = row;
  });

  var RUNS = {};
  DATA.runs.forEach(function (run) {
    RUNS[run.runId] = run;
  });

  var ENTRIES = DATA.entries.slice().sort(function (a, b) {
    var left = modelOf(a);
    var right = modelOf(b);
    var suite = (left ? left.suite : "").localeCompare(right ? right.suite : "");
    if (suite) return suite;
    var model = a.modelKey.localeCompare(b.modelKey);
    if (model) return model;
    var run = a.runId.localeCompare(b.runId);
    if (run) return run;
    return a.image.name.localeCompare(b.image.name);
  });

  var BY_ID = {};
  ENTRIES.forEach(function (entry) {
    BY_ID[entry.id] = entry;
  });

  function modelOf(entry) {
    return MODELS[modelKeyOf(entry.runId, entry.modelKey)];
  }

  /* ---- state --------------------------------------------------------------------------- */

  var defaultRun = DATA.runs.length > 1 ? DATA.runs[0].runId : ANY;

  var state = {
    view: "summary",
    entryId: null,
    filters: { family: ANY, model: ANY, suite: ANY, outcome: ANY, run: defaultRun, q: "" }
  };

  function readHash() {
    var raw = String(location.hash || "").replace(/^#/, "");
    var split = raw.indexOf("?");
    var path = split < 0 ? raw : raw.slice(0, split);
    var query = split < 0 ? "" : raw.slice(split + 1);
    var params = new URLSearchParams(query);
    var parts = path.split("/").filter(Boolean);

    var filters = {};
    FILTER_KEYS.forEach(function (key) {
      filters[key] = params.has(key) ? params.get(key) : (key === "run" ? defaultRun : ANY);
    });

    var view = parts[0] || "summary";
    if (view === "entry" && parts[1] && BY_ID[parts[1]]) {
      return { view: "detail", entryId: parts[1], filters: filters };
    }
    if (view !== "gallery" && view !== "summary") view = "summary";
    return { view: view, entryId: null, filters: filters };
  }

  function hashFor(view, entryId, filters) {
    var params = new URLSearchParams();
    FILTER_KEYS.forEach(function (key) {
      var value = filters[key];
      var fallback = key === "run" ? defaultRun : ANY;
      if (value !== fallback && value !== undefined && value !== null) params.set(key, value);
    });
    var query = params.toString();
    var path = view === "detail" ? "/entry/" + entryId : "/" + view;
    return "#" + path + (query ? "?" + query : "");
  }

  function go(view, entryId) {
    var next = hashFor(view, entryId, state.filters);
    if (location.hash === next) render();
    else location.hash = next;
  }

  /* ---- filtering ----------------------------------------------------------------------- */

  function matches(entry) {
    var model = modelOf(entry);
    var filters = state.filters;
    if (filters.run && entry.runId !== filters.run) return false;
    if (filters.model && entry.modelKey !== filters.model) return false;
    if (model) {
      if (filters.family && model.family !== filters.family) return false;
      if (filters.suite && model.suite !== filters.suite) return false;
    }
    if (filters.outcome) {
      var outcome = entry.decision ? entry.decision.outcome : null;
      if (outcome !== filters.outcome) return false;
    }
    if (filters.q) {
      var needle = filters.q.toLowerCase();
      if (entry.image.name.toLowerCase().indexOf(needle) < 0) return false;
    }
    return true;
  }

  function visible() {
    return ENTRIES.filter(matches);
  }

  /* ---- helpers ------------------------------------------------------------------------- */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  function baseName(name) {
    var parts = String(name).split("/");
    return parts[parts.length - 1];
  }

  function outcomeClass(outcome) {
    if (outcome === "CLEAR") return "clear";
    if (outcome === "FAIL") return "fail";
    return "hold";
  }

  function shortProvider(name) {
    return String(name).replace("ExecutionProvider", "");
  }

  function quantile(sorted, fraction) {
    if (!sorted.length) return 0;
    var index = Math.min(sorted.length - 1, Math.floor(sorted.length * fraction));
    return sorted[index];
  }

  function millis(value) {
    return Number(value).toFixed(2) + " ms";
  }

  function option(select, value, label) {
    var node = el("option", null, label);
    node.value = value;
    select.appendChild(node);
    return node;
  }

  function unique(values) {
    var seen = {};
    var out = [];
    values.forEach(function (value) {
      if (value === undefined || value === null || value === "") return;
      if (seen[value]) return;
      seen[value] = true;
      out.push(value);
    });
    return out.sort();
  }

  function runLabel(runId) {
    var run = RUNS[runId];
    if (!run) return runId;
    return shortProvider(run.provider) + (run.gpu ? " on " + run.gpu : "");
  }

  function pixelated(entry) {
    return entry.image.w < 320 || entry.image.h < 320;
  }

  /* ---- summary ------------------------------------------------------------------------- */

  function statsFor(row) {
    var samples = [];
    var outcomes = { CLEAR: 0, HOLD: 0, FAIL: 0 };
    ENTRIES.forEach(function (entry) {
      if (entry.runId !== row.runId || entry.modelKey !== row.key) return;
      samples.push(entry.timings.sessionMs);
      var outcome = entry.decision ? entry.decision.outcome : "HOLD";
      if (outcomes[outcome] === undefined) outcomes[outcome] = 0;
      outcomes[outcome] += 1;
    });
    samples.sort(function (a, b) { return a - b; });
    return {
      count: samples.length,
      outcomes: outcomes,
      min: samples.length ? samples[0] : 0,
      p50: quantile(samples, 0.5),
      p95: quantile(samples, 0.95),
      max: samples.length ? samples[samples.length - 1] : 0
    };
  }

  function modelCard(row) {
    var stats = statsFor(row);
    var card = el("button", "card");
    card.type = "button";

    var head = el("div", "card-head");
    head.appendChild(el("h3", null, row.key));
    head.appendChild(el("span", "badge family", row.family));
    card.appendChild(head);

    var badges = el("div", "badges");
    badges.appendChild(el("span", "badge", row.suite + " / " + row.corpus));
    badges.appendChild(el("span", "badge", stats.count + " images"));
    badges.appendChild(el("span", "badge", row.card.providers.map(shortProvider).join(" + ")));
    if (row.card.gpu) badges.appendChild(el("span", "badge", row.card.gpu));
    ["CLEAR", "HOLD", "FAIL"].forEach(function (outcome) {
      var value = stats.outcomes[outcome] || 0;
      if (!value) return;
      badges.appendChild(el("span", "badge " + outcomeClass(outcome), outcome + " " + value));
    });
    card.appendChild(badges);

    var grid = el("div", "stats");
    [["min", stats.min], ["p50", stats.p50], ["p95", stats.p95], ["max", stats.max]].forEach(
      function (pair) {
        var cell = el("div", "stat");
        cell.appendChild(el("b", null, Number(pair[1]).toFixed(2)));
        cell.appendChild(el("span", null, pair[0] + " ms"));
        grid.appendChild(cell);
      }
    );
    card.appendChild(grid);

    card.addEventListener("click", function () {
      state.filters.model = row.key;
      state.filters.run = row.runId;
      state.filters.family = ANY;
      state.filters.suite = ANY;
      state.filters.outcome = ANY;
      state.filters.q = "";
      go("gallery", null);
    });
    return card;
  }

  function renderRunsStrip() {
    var panel = document.getElementById("runs-strip");
    var table = document.getElementById("runs-table");
    clear(table);
    var runIds = DATA.runs.map(function (run) { return run.runId; });
    if (runIds.length < 2) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;

    var keys = unique(DATA.models.map(function (row) { return row.key; }));
    var head = el("tr");
    head.appendChild(el("th", null, "Model"));
    runIds.forEach(function (runId) {
      head.appendChild(el("th", "num", runLabel(runId)));
    });
    if (runIds.length === 2) head.appendChild(el("th", "num", "ratio"));
    var thead = el("thead");
    thead.appendChild(head);
    table.appendChild(thead);

    var body = el("tbody");
    keys.forEach(function (key) {
      var row = el("tr");
      row.appendChild(el("td", null, key));
      var values = [];
      runIds.forEach(function (runId) {
        var model = MODELS[modelKeyOf(runId, key)];
        var value = model ? statsFor(model).p50 : null;
        values.push(value);
        row.appendChild(el("td", "num", value === null ? "-" : Number(value).toFixed(2)));
      });
      if (runIds.length === 2) {
        var ratio =
          values[0] && values[1] ? (values[0] / values[1]).toFixed(2) + "x" : "-";
        row.appendChild(el("td", "num", ratio));
      }
      body.appendChild(row);
    });
    table.appendChild(body);
  }

  function renderRefusals() {
    var panel = document.getElementById("refusals-panel");
    var table = document.getElementById("refusals-table");
    clear(table);
    var rows = (DATA.refusals || []).filter(function (row) {
      return !state.filters.run || row.runId === state.filters.run;
    });
    if (!rows.length) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;

    var head = el("tr");
    ["Fixture", "Bytes", "Required", "Observed", "Result"].forEach(function (label, index) {
      head.appendChild(el("th", index === 1 ? "num" : null, label));
    });
    var thead = el("thead");
    thead.appendChild(head);
    table.appendChild(thead);

    var body = el("tbody");
    rows.forEach(function (row) {
      var line = el("tr");
      line.appendChild(el("td", null, row.name));
      line.appendChild(el("td", "num", row.bytes));
      line.appendChild(el("td", null, row.expected || "decodes"));
      line.appendChild(el("td", null, row.observed || "decoded"));
      var verdict = el("td");
      var agreed = (row.observed || null) === (row.expected || null);
      verdict.appendChild(
        el("span", "badge " + (agreed ? "clear" : "fail"), agreed ? "as required" : "differs")
      );
      line.appendChild(verdict);
      body.appendChild(line);
    });
    table.appendChild(body);
  }

  function renderSummary() {
    var host = clear(document.getElementById("model-cards"));
    var rows = DATA.models.filter(function (row) {
      return !state.filters.run || row.runId === state.filters.run;
    });
    if (!rows.length) {
      host.appendChild(el("p", "empty", "This run holds no models."));
    }
    rows.forEach(function (row) {
      host.appendChild(modelCard(row));
    });
    renderRunsStrip();
    renderRefusals();
  }

  /* ---- gallery ------------------------------------------------------------------------- */

  function fillFacets() {
    var families = unique(DATA.models.map(function (row) { return row.family; }));
    var suites = unique(DATA.models.map(function (row) { return row.suite; }));
    var models = unique(DATA.models.map(function (row) { return row.key; }));
    var outcomes = unique(
      ENTRIES.map(function (entry) { return entry.decision ? entry.decision.outcome : null; })
    );

    fill("filter-family", families, "Any family", state.filters.family);
    fill("filter-model", models, "Any model", state.filters.model);
    fill("filter-suite", suites, "Any suite", state.filters.suite);
    fill("filter-outcome", outcomes, "Any outcome", state.filters.outcome);
    fill(
      "filter-run",
      DATA.runs.map(function (run) { return run.runId; }),
      "All runs",
      state.filters.run,
      runLabel
    );
    document.getElementById("filter-q").value = state.filters.q || "";
  }

  function fill(id, values, anyLabel, selected, labeller) {
    var select = clear(document.getElementById(id));
    option(select, ANY, anyLabel);
    values.forEach(function (value) {
      option(select, value, labeller ? labeller(value) : value);
    });
    select.value = selected || ANY;
    if (select.value !== (selected || ANY)) select.value = ANY;
  }

  function thumbCard(entry, position) {
    var model = modelOf(entry);
    var outcome = entry.decision ? entry.decision.outcome : "HOLD";
    var card = el("button", "thumb" + (pixelated(entry) ? " pixelated" : ""));
    card.type = "button";
    card.title = entry.image.name;

    var figure = el("figure");
    var img = document.createElement("img");
    img.src = entry.image.thumb;
    img.alt = entry.image.name + " through " + entry.modelKey;
    img.loading = "lazy";
    img.decoding = "async";
    figure.appendChild(img);
    card.appendChild(figure);

    var meta = el("div", "meta");
    meta.appendChild(el("span", "name", baseName(entry.image.name)));
    var row = el("div", "row");
    var left = el("span", "model", entry.modelKey);
    row.appendChild(left);
    var right = el("span", "verdict");
    right.appendChild(el("span", "dot " + outcomeClass(outcome)));
    right.appendChild(el("span", "ms", Number(entry.timings.sessionMs).toFixed(1) + " ms"));
    row.appendChild(right);
    meta.appendChild(row);
    if (model) meta.appendChild(el("span", "model", model.suite + " / " + model.family));
    card.appendChild(meta);

    card.addEventListener("click", function () {
      go("detail", entry.id);
    });
    return card;
  }

  function renderGallery() {
    fillFacets();
    var entries = visible();
    var grid = clear(document.getElementById("thumb-grid"));
    document.getElementById("match-count").textContent =
      entries.length + " of " + ENTRIES.length + " entries match";
    if (!entries.length) {
      grid.appendChild(el("p", "empty", "No entry matches these filters."));
      return;
    }
    entries.forEach(function (entry, index) {
      grid.appendChild(thumbCard(entry, index));
    });
  }

  /* ---- detail -------------------------------------------------------------------------- */

  function frame(caption, href, src, alt, isPixelated) {
    var figure = el("figure", "frame" + (isPixelated ? " pixelated" : ""));
    var head = el("figcaption");
    head.appendChild(el("span", null, caption));
    var link = el("a", null, "open full size");
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener";
    head.appendChild(link);
    figure.appendChild(head);

    var wrap = el("a");
    wrap.href = href;
    wrap.target = "_blank";
    wrap.rel = "noopener";
    var img = document.createElement("img");
    img.src = src;
    img.alt = alt;
    wrap.appendChild(img);
    figure.appendChild(wrap);
    return figure;
  }

  function topKTable(entry) {
    var panel = el("figure", "frame");
    var head = el("figcaption");
    head.appendChild(el("span", null, "Top classes"));
    head.appendChild(el("span", null, "no overlay for a ranking"));
    panel.appendChild(head);

    var scroller = el("div", "scroller");
    var table = el("table", "grid-table");
    var thead = el("thead");
    var headRow = el("tr");
    headRow.appendChild(el("th", "num", "#"));
    headRow.appendChild(el("th", null, "Label"));
    headRow.appendChild(el("th", "num", "Index"));
    headRow.appendChild(el("th", "num", "Score"));
    thead.appendChild(headRow);
    table.appendChild(thead);

    var body = el("tbody");
    var classes = (entry.resultBody.outputs || {}).classes || [];
    classes.forEach(function (item, index) {
      var row = el("tr");
      row.appendChild(el("td", "num", index + 1));
      row.appendChild(el("td", "wrap", item.label));
      row.appendChild(el("td", "num", item.index));
      row.appendChild(el("td", "num", Number(item.score).toFixed(6)));
      body.appendChild(row);
    });
    table.appendChild(body);
    scroller.appendChild(table);
    panel.appendChild(scroller);
    return panel;
  }

  function kv(pairs) {
    var list = el("dl", "kv");
    pairs.forEach(function (pair) {
      if (pair[1] === undefined || pair[1] === null || pair[1] === "") return;
      list.appendChild(el("dt", null, pair[0]));
      list.appendChild(el("dd", null, pair[1]));
    });
    return list;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  var TOKENS =
    /("(?:\\u[0-9a-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g;

  function highlight(value) {
    return escapeHtml(JSON.stringify(value, null, 2)).replace(TOKENS, function (match) {
      var cls = "n";
      if (match.charAt(0) === "\"") cls = /:$/.test(match) ? "k" : "s";
      else if (match === "true" || match === "false") cls = "b";
      else if (match === "null") cls = "z";
      return "<span class=\"" + cls + "\">" + match + "</span>";
    });
  }

  function copyText(text, button) {
    function done(ok) {
      var original = button.dataset.label || button.textContent;
      button.dataset.label = original;
      button.textContent = ok ? "Copied" : "Copy failed";
      window.setTimeout(function () {
        button.textContent = button.dataset.label;
      }, 1400);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { done(true); },
        function () { done(fallbackCopy(text)); }
      );
      return;
    }
    done(fallbackCopy(text));
  }

  function fallbackCopy(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "readonly");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (error) {
      ok = false;
    }
    document.body.removeChild(area);
    return ok;
  }

  function renderDetail() {
    var entry = BY_ID[state.entryId];
    var entries = visible();
    var index = entries.findIndex(function (item) { return item.id === entry.id; });
    if (index < 0) {
      entries = [entry];
      index = 0;
    }

    var model = modelOf(entry);
    var card = model ? model.card : {};
    document.getElementById("detail-position").textContent =
      index + 1 + " / " + entries.length;
    document.getElementById("detail-title").textContent = entry.image.name;
    document.getElementById("detail-subtitle").textContent =
      entry.modelKey +
      "  \u00b7  " +
      (model ? model.suite + " / " + model.corpus : "") +
      "  \u00b7  " +
      runLabel(entry.runId) +
      "  \u00b7  " +
      entry.image.w + " by " + entry.image.h + " px";

    var previous = document.getElementById("detail-prev");
    var next = document.getElementById("detail-next");
    previous.disabled = index <= 0;
    next.disabled = index >= entries.length - 1;
    previous.onclick = function () {
      if (index > 0) go("detail", entries[index - 1].id);
    };
    next.onclick = function () {
      if (index < entries.length - 1) go("detail", entries[index + 1].id);
    };

    var pair = clear(document.getElementById("detail-pair"));
    pair.appendChild(
      frame(
        "Input  \u00b7  " + entry.image.w + " x " + entry.image.h,
        entry.image.src,
        entry.image.src,
        entry.image.name,
        pixelated(entry)
      )
    );
    if (entry.image.overlay) {
      pair.appendChild(
        frame(
          "Overlay  \u00b7  what the model reported",
          entry.image.overlay,
          entry.image.overlay,
          "Regions " + entry.modelKey + " reported on " + entry.image.name,
          false
        )
      );
    } else {
      pair.appendChild(topKTable(entry));
    }

    var modelPanel = clear(document.getElementById("detail-model"));
    modelPanel.appendChild(el("h2", null, "Model"));
    modelPanel.appendChild(
      kv([
        ["id", card.modelId],
        ["version", card.version],
        ["digest", entry.resultBody.model ? entry.resultBody.model.digest : null],
        ["family", model ? model.family : null],
        ["providers", (card.providers || []).join(", ")],
        ["gpu", card.gpu || "none"],
        ["input", (card.inputName || "?") + " " + (card.inputShape || []).join(" x ")],
        ["dtype", card.inputDtype],
        ["preprocess", card.preprocessSummary],
        ["policy", card.providerPolicy],
        ["transform", card.transformVersion],
        ["signature", card.signed ? "signed as " + card.keyId : "unsigned"]
      ])
    );

    var decisionPanel = clear(document.getElementById("detail-decision"));
    decisionPanel.appendChild(el("h2", null, "Decision"));
    var decision = entry.decision || {};
    var badge = el(
      "span",
      "badge " + outcomeClass(decision.outcome),
      decision.outcome || "unknown"
    );
    decisionPanel.appendChild(badge);
    decisionPanel.appendChild(
      kv([
        ["pass", String(decision.pass)],
        ["confidence", decision.confidence === null ? "null" : decision.confidence],
        ["threshold", decision.threshold === null ? "null" : decision.threshold],
        ["rule", decision.rule]
      ])
    );

    var timingPanel = clear(document.getElementById("detail-timings"));
    timingPanel.appendChild(el("h2", null, "Timings"));
    timingPanel.appendChild(
      kv([
        ["session", millis(entry.timings.sessionMs)],
        ["wall total", millis(entry.timings.totalMs)]
      ])
    );
    timingPanel.appendChild(
      el(
        "p",
        "note",
        "The session time is the graph alone; the wall total covers reading the file, decoding, " +
          "preprocessing, the graph, postprocessing, and the decision rules. The result body " +
          "reports the two the builder measures and leaves the per-stage fields at zero."
      )
    );

    var panel = document.getElementById("detail-json-panel");
    panel.open = false;
    document.getElementById("detail-json-note").textContent =
      "The body of the app/inference/result message for this image.";
    document.getElementById("detail-json").innerHTML = highlight(entry.resultBody);
    var copy = document.getElementById("copy-json");
    copy.onclick = function () {
      copyText(JSON.stringify(entry.resultBody, null, 2), copy);
    };
  }

  /* ---- chrome -------------------------------------------------------------------------- */

  function renderChrome() {
    var picker = document.getElementById("run-picker");
    var select = document.getElementById("run-select");
    if (DATA.runs.length > 1) {
      picker.hidden = false;
      clear(select);
      option(select, ANY, "All runs");
      DATA.runs.forEach(function (run) {
        option(select, run.runId, runLabel(run.runId));
      });
      select.value = state.filters.run || ANY;
    } else {
      picker.hidden = true;
    }

    var counter = document.getElementById("entry-count");
    var shown =
      state.view === "summary"
        ? ENTRIES.filter(function (entry) {
            return !state.filters.run || entry.runId === state.filters.run;
          }).length
        : visible().length;
    counter.textContent =
      shown === ENTRIES.length
        ? ENTRIES.length + " entries"
        : shown + " of " + ENTRIES.length + " entries";

    Array.prototype.forEach.call(document.querySelectorAll("#tabs .tab"), function (tab) {
      var active = tab.dataset.view === state.view ||
        (state.view === "detail" && tab.dataset.view === "gallery");
      tab.setAttribute("aria-current", active ? "true" : "false");
    });
  }

  function render() {
    var parsed = readHash();
    state.view = parsed.view;
    state.entryId = parsed.entryId;
    state.filters = parsed.filters;

    document.getElementById("view-summary").hidden = state.view !== "summary";
    document.getElementById("view-gallery").hidden = state.view !== "gallery";
    document.getElementById("view-detail").hidden = state.view !== "detail";

    if (state.view === "summary") renderSummary();
    else if (state.view === "gallery") renderGallery();
    else renderDetail();

    renderChrome();
    window.scrollTo(0, 0);
  }

  /* ---- wiring -------------------------------------------------------------------------- */

  function bind() {
    Array.prototype.forEach.call(document.querySelectorAll("#tabs .tab"), function (tab) {
      tab.addEventListener("click", function () {
        go(tab.dataset.view, null);
      });
    });

    document.getElementById("run-select").addEventListener("change", function (event) {
      state.filters.run = event.target.value;
      go(state.view === "detail" ? "summary" : state.view, null);
    });

    [
      ["filter-family", "family"],
      ["filter-model", "model"],
      ["filter-suite", "suite"],
      ["filter-outcome", "outcome"],
      ["filter-run", "run"]
    ].forEach(function (pair) {
      document.getElementById(pair[0]).addEventListener("change", function (event) {
        state.filters[pair[1]] = event.target.value;
        go("gallery", null);
      });
    });

    var search = document.getElementById("filter-q");
    var timer = null;
    search.addEventListener("input", function (event) {
      var value = event.target.value;
      window.clearTimeout(timer);
      timer = window.setTimeout(function () {
        state.filters.q = value;
        var next = hashFor("gallery", null, state.filters);
        if (location.hash === next) render();
        else location.hash = next;
        document.getElementById("filter-q").focus();
      }, 180);
    });

    document.getElementById("filter-clear").addEventListener("click", function () {
      FILTER_KEYS.forEach(function (key) {
        state.filters[key] = key === "run" ? defaultRun : ANY;
      });
      go("gallery", null);
    });

    document.getElementById("detail-back").addEventListener("click", function () {
      go("gallery", null);
    });

    document.addEventListener("keydown", function (event) {
      var tag = event.target && event.target.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") {
        if (event.key === "Escape") event.target.blur();
        return;
      }
      if (state.view === "detail") {
        if (event.key === "ArrowLeft") {
          event.preventDefault();
          document.getElementById("detail-prev").click();
        } else if (event.key === "ArrowRight") {
          event.preventDefault();
          document.getElementById("detail-next").click();
        } else if (event.key === "Escape") {
          event.preventDefault();
          go("gallery", null);
        }
      }
    });

    window.addEventListener("hashchange", render);
  }

  bind();
  render();
})();