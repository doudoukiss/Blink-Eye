(function () {
  const catalogEndpoint = "/api/runtime/models";
  const startEndpoint = "/start";
  const storageKey = "blink.modelSelection.profileId";
  const panelId = "blink-model-selector";
  const state = {
    profiles: [],
    selectedProfileId: "",
    currentProfileId: "",
    defaultProfileId: "",
    currentLanguage: "zh",
    modelSelection: null,
    notice: "Loading models",
    expanded: false,
  };

  function assignStyles(node, styles) {
    Object.assign(node.style, styles);
    return node;
  }

  function safeArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function safeObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function text(value) {
    return String(value == null ? "" : value).trim();
  }

  function selectedProfile() {
    return state.profiles.find((profile) => profile.id === state.selectedProfileId) || null;
  }

  function preferredProfileId(payload) {
    const profiles = safeArray(payload.profiles);
    const stored = text(window.localStorage.getItem(storageKey));
    if (stored && profiles.some((profile) => profile.id === stored)) {
      return stored;
    }
    if (payload.current_profile_id) {
      return payload.current_profile_id;
    }
    if (payload.default_profile_id) {
      return payload.default_profile_id;
    }
    const firstAvailable = profiles.find((profile) => profile.available);
    return firstAvailable ? firstAvailable.id : text(profiles[0] && profiles[0].id);
  }

  function summarize(profile) {
    if (!profile) {
      return "No model selected";
    }
    const parts = [
      profile.provider,
      profile.runtime_tier,
      profile.latency_tier,
      profile.capability_tier,
    ].filter(Boolean);
    const languageFit = safeArray(profile.language_fit).join("/");
    if (languageFit) {
      parts.push(`fit ${languageFit}`);
    }
    return parts.join(" - ");
  }

  function reasonSummary(reasonCodes) {
    return safeArray(reasonCodes).slice(0, 3).join(", ");
  }

  function resultText(result) {
    if (!result) {
      return "Selection applies to the next browser session.";
    }
    const accepted = result.accepted ? "Accepted" : "Rejected";
    const applied = result.applied ? "applied" : "not applied";
    const reasons = reasonSummary(result.reason_codes);
    return reasons ? `${accepted}, ${applied}: ${reasons}` : `${accepted}, ${applied}`;
  }

  function compactModelText() {
    const profile = selectedProfile();
    if (!profile) {
      return "Model";
    }
    return `Model: ${profile.label}`;
  }

  function ensurePanel() {
    let panel = document.getElementById(panelId);
    if (panel) {
      return panel;
    }
    panel = document.createElement("section");
    panel.id = panelId;
    panel.setAttribute("aria-label", "Blink model selector");
    assignStyles(panel, {
      position: "fixed",
      top: "8px",
      left: "8px",
      zIndex: "30",
      boxSizing: "border-box",
      border: "1px solid rgba(17, 24, 39, 0.14)",
      borderRadius: "8px",
      background: "rgba(255, 255, 255, 0.94)",
      color: "#111827",
      boxShadow: "0 10px 24px rgba(15, 23, 42, 0.12)",
      fontFamily:
        'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      fontSize: "12px",
      lineHeight: "1.35",
      backdropFilter: "blur(8px)",
    });
    document.body.appendChild(panel);
    return panel;
  }

  function render() {
    if (!document.body) {
      return;
    }
    const panel = ensurePanel();
    panel.replaceChildren();
    assignStyles(panel, {
      width: state.expanded ? "min(342px, calc(100vw - 28px))" : "min(220px, calc(100vw - 112px))",
      padding: state.expanded ? "10px" : "0",
    });

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.textContent = state.expanded ? "Hide model" : compactModelText();
    toggle.setAttribute(
      "aria-label",
      state.expanded ? "Blink model selector hide" : "Blink model selector show",
    );
    assignStyles(toggle, {
      width: "100%",
      minHeight: "34px",
      border: "0",
      borderRadius: "7px",
      background: state.expanded ? "rgba(241, 245, 249, 0.96)" : "rgba(255, 255, 255, 0.96)",
      color: "#111827",
      cursor: "pointer",
      fontSize: "12px",
      fontWeight: "740",
      overflow: "hidden",
      padding: "7px 9px",
      textAlign: "left",
      textOverflow: "ellipsis",
      whiteSpace: "nowrap",
    });
    toggle.addEventListener("click", () => {
      state.expanded = !state.expanded;
      render();
    });
    panel.append(toggle);

    if (!state.expanded) {
      return;
    }

    const label = document.createElement("label");
    label.textContent = "Model";
    label.setAttribute("for", "blink-model-selector-input");
    assignStyles(label, {
      display: "block",
      color: "#374151",
      fontSize: "11px",
      fontWeight: "760",
      letterSpacing: "0",
      textTransform: "uppercase",
      marginBottom: "6px",
    });

    const select = document.createElement("select");
    select.id = "blink-model-selector-input";
    assignStyles(select, {
      width: "100%",
      boxSizing: "border-box",
      minHeight: "34px",
      borderRadius: "6px",
      border: "1px solid #CBD5E1",
      background: "#FFFFFF",
      color: "#111827",
      fontSize: "13px",
      fontWeight: "650",
      padding: "6px 8px",
    });

    state.profiles.forEach((profile) => {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = profile.available
        ? `${profile.label} (${profile.model})`
        : `${profile.label} (${profile.model}) - unavailable`;
      option.disabled = !profile.available;
      select.appendChild(option);
    });
    select.value = state.selectedProfileId;
    select.addEventListener("change", () => {
      state.selectedProfileId = select.value;
      window.localStorage.setItem(storageKey, state.selectedProfileId);
      state.modelSelection = null;
      render();
    });

    const selected = selectedProfile();
    const summary = document.createElement("div");
    summary.textContent = selected ? summarize(selected) : state.notice;
    assignStyles(summary, {
      marginTop: "7px",
      color: "#475569",
      overflowWrap: "anywhere",
    });

    const status = document.createElement("div");
    status.textContent = resultText(state.modelSelection);
    assignStyles(status, {
      marginTop: "6px",
      color: state.modelSelection && !state.modelSelection.accepted ? "#991B1B" : "#166534",
      overflowWrap: "anywhere",
    });

    panel.append(label, select, summary, status);
  }

  async function loadCatalog() {
    try {
      const response = await fetch(catalogEndpoint, { cache: "no-store" });
      if (!response.ok) {
        state.notice = "Model list unavailable";
        render();
        return;
      }
      const payload = await response.json();
      state.profiles = safeArray(payload.profiles);
      state.currentProfileId = text(payload.current_profile_id);
      state.defaultProfileId = text(payload.default_profile_id);
      state.currentLanguage = text(payload.current_language) || "zh";
      state.selectedProfileId = preferredProfileId(payload);
      if (state.selectedProfileId) {
        window.localStorage.setItem(storageKey, state.selectedProfileId);
      }
      state.notice = state.profiles.length ? "" : "No model profiles available";
      render();
    } catch (_error) {
      state.notice = "Model list unavailable";
      render();
    }
  }

  function isStartRequest(input, init) {
    const rawUrl = typeof input === "string" ? input : text(input && input.url);
    if (!rawUrl) {
      return false;
    }
    const method = text((init && init.method) || (input && input.method) || "GET").toUpperCase();
    const url = new URL(rawUrl, window.location.href);
    return method === "POST" && url.pathname === startEndpoint;
  }

  function withSelectedModel(input, init) {
    const profile = selectedProfile();
    if (!profile || !profile.available) {
      return { input, init };
    }
    const nextInit = Object.assign({}, init || {});
    const rawBody = nextInit.body;
    if (rawBody != null && typeof rawBody !== "string") {
      return { input, init };
    }
    let payload = {};
    if (rawBody) {
      try {
        payload = JSON.parse(rawBody);
      } catch (_error) {
        return { input, init };
      }
    }
    const requestBody = Object.assign({}, safeObject(payload.body), {
      model_profile_id: profile.id,
    });
    payload = Object.assign({}, safeObject(payload), { body: requestBody });
    const headers = new Headers(nextInit.headers || (input && input.headers) || {});
    if (!headers.has("content-type")) {
      headers.set("content-type", "application/json");
    }
    nextInit.headers = headers;
    nextInit.body = JSON.stringify(payload);
    return { input, init: nextInit };
  }

  function installStartInterceptor() {
    if (window.__blinkModelSelectorFetchInstalled) {
      return;
    }
    window.__blinkModelSelectorFetchInstalled = true;
    const originalFetch = window.fetch.bind(window);
    window.fetch = async function blinkModelSelectorFetch(input, init) {
      const startRequest = isStartRequest(input, init);
      const request = startRequest ? withSelectedModel(input, init) : { input, init };
      const response = await originalFetch(request.input, request.init);
      if (startRequest) {
        response
          .clone()
          .json()
          .then((payload) => {
            if (payload && payload.modelSelection) {
              state.modelSelection = payload.modelSelection;
              render();
            }
          })
          .catch(function () {});
      }
      return response;
    };
  }

  installStartInterceptor();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadCatalog, { once: true });
  } else {
    loadCatalog();
  }
})();
