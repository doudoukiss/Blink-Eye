(function () {
  if (window.__blinkOperatorWorkbenchInitialized) {
    return;
  }
  window.__blinkOperatorWorkbenchInitialized = true;

  const operatorEndpoint = "/api/runtime/operator";
  const behaviorControlsEndpoint = "/api/runtime/behavior-controls";
  const stylePresetsEndpoint = "/api/runtime/style-presets";
  const memoryPersonaPresetPreviewEndpoint =
    "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/preview";
  const memoryPersonaPresetApplyEndpoint =
    "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/apply";
  const memoryEndpoint = "/api/runtime/memory";
  const rolloutEndpoint = "/api/runtime/rollout";
  const rolloutEvidenceEndpoint = "/api/runtime/rollout/evidence";
  const episodeEvidenceEndpoint = "/api/runtime/evidence";
  const performancePreferencesEndpoint = "/api/runtime/performance-preferences";
  const performancePolicyProposalApplyEndpoint = (proposalId) =>
    `/api/runtime/performance-preferences/policy-proposals/${encodeURIComponent(
      proposalId,
    )}/apply`;
  const panelId = "blink-operator-workbench";
  const activeRefreshMs = 8000;
  const hiddenRefreshMs = 30000;
  const evidenceRefreshMs = 30000;
  const staticRefreshMs = 300000;
  const coreControlFields = [
    "response_depth",
    "directness",
    "warmth",
    "teaching_mode",
    "memory_use",
    "initiative_mode",
    "evidence_visibility",
    "correction_mode",
    "explanation_structure",
    "challenge_style",
    "voice_mode",
    "question_budget",
  ];
  const styleControlFields = [
    "humor_mode",
    "vividness_mode",
    "sophistication_mode",
    "character_presence",
    "story_mode",
  ];
  const controlFields = coreControlFields.concat(styleControlFields);
  const performancePreferenceDimensions = [
    "felt_heard",
    "state_clarity",
    "interruption_naturalness",
    "voice_pacing",
    "camera_honesty",
    "memory_usefulness",
    "persona_consistency",
    "enjoyment",
    "not_fake_human",
  ];
  const performancePreferenceFailureLabels = [
    "voice_pacing_too_long",
    "interruption_awkward",
    "backchannel_misread",
    "camera_claim_unclear",
    "false_camera_claim",
    "memory_callback_missing",
    "persona_too_human",
  ];
  const performancePreferenceImprovementLabels = [
    "felt_heard_clearer",
    "state_clarity_better",
    "voice_pacing_better",
    "camera_honesty_better",
    "memory_usefulness_better",
    "persona_consistency_better",
  ];
  const controlOptions = {
    response_depth: ["concise", "balanced", "deep"],
    directness: ["gentle", "balanced", "rigorous"],
    warmth: ["low", "medium", "high"],
    teaching_mode: ["auto", "direct", "clarify", "walkthrough", "socratic"],
    memory_use: ["minimal", "balanced", "continuity_rich"],
    initiative_mode: ["minimal", "balanced", "proactive"],
    evidence_visibility: ["hidden", "compact", "rich"],
    correction_mode: ["gentle", "precise", "rigorous"],
    explanation_structure: ["answer_first", "walkthrough", "socratic"],
    challenge_style: ["avoid", "gentle", "direct"],
    voice_mode: ["off", "concise", "balanced"],
    question_budget: ["low", "medium", "high"],
    humor_mode: ["off", "subtle", "witty", "playful"],
    vividness_mode: ["spare", "balanced", "vivid"],
    sophistication_mode: ["plain", "smart", "sophisticated"],
    character_presence: ["minimal", "balanced", "character_rich"],
    story_mode: ["off", "light", "recurring_motifs"],
  };
  const controlLabels = {
    response_depth: "Response depth",
    directness: "Directness",
    warmth: "Warmth",
    teaching_mode: "Teaching mode",
    memory_use: "Memory use",
    initiative_mode: "Initiative",
    evidence_visibility: "Evidence",
    correction_mode: "Correction",
    explanation_structure: "Structure",
    challenge_style: "Challenge style",
    voice_mode: "Voice mode",
    question_budget: "Question budget",
    humor_mode: "Humor",
    vividness_mode: "Vividness",
    sophistication_mode: "Sophistication",
    character_presence: "Presence",
    story_mode: "Story mode",
  };
  const sectionKeys = [
    "overview",
    "memory",
    "controls",
    "teaching",
    "voice",
    "practice",
    "adapters",
    "sim-to-real",
    "evidence",
    "performance-learning",
    "rollouts",
  ];
  const state = {
    collapsed: true,
    sectionCollapsed: {
      overview: false,
      memory: false,
      controls: false,
      teaching: false,
      voice: false,
      practice: true,
      adapters: true,
      "sim-to-real": true,
      evidence: false,
      "performance-learning": false,
      rollouts: false,
    },
    snapshot: fallbackSnapshot("operator_initializing"),
    controlDrafts: {},
    controlDraftDirty: false,
    controlUpdatePending: false,
    controlUpdateResult: null,
    stylePresets: fallbackStylePresets("style_presets_initializing"),
    stylePresetDraft: "",
    memoryPersonaPresetPreview: fallbackMemoryPersonaPreset("memory_persona_preset_initializing"),
    memoryPersonaPresetApplyPending: false,
    memoryPersonaPresetApplyResult: null,
    correctionMemoryId: null,
    correctionDrafts: {},
    memoryActionResults: {},
    latestMemoryActionResult: null,
    memoryPendingActions: {},
    rolloutEvidence: fallbackRolloutEvidence("rollout_evidence_initializing"),
    episodeEvidence: fallbackEpisodeEvidence("episode_evidence_initializing"),
    rolloutTrafficDrafts: {},
    rolloutActionResults: {},
    latestRolloutActionResult: null,
    rolloutPendingActions: {},
    performancePreferenceDraft: {
      winner: "b",
      ratings: Object.fromEntries(performancePreferenceDimensions.map((dimension) => [dimension, 3])),
      failure_labels: [],
      improvement_labels: [],
    },
    performancePreferenceSubmitPending: false,
    latestPerformancePreferenceResult: null,
    performancePolicyApplyPending: {},
    latestPerformancePolicyApplyResult: null,
    advancedOpen: false,
  };
  const refreshState = {
    inFlight: false,
    timerId: null,
    lastEvidenceRefreshAt: 0,
    lastStaticRefreshAt: 0,
  };
  const supportedMemoryActions = {
    pin: { endpoint: "pin", label: "Pin" },
    suppress: { endpoint: "suppress", label: "Suppress", confirm: true },
    correct: { endpoint: "correct", label: "Correct" },
    forget: { endpoint: "forget", label: "Forget", confirm: true },
    "mark-stale": { endpoint: "mark-stale", label: "Mark stale", confirm: true },
  };
  const supportedRolloutActions = {
    approve: { endpoint: "approve", label: "Approve" },
    activate: { endpoint: "activate", label: "Activate", needsTraffic: true },
    pause: { endpoint: "pause", label: "Pause" },
    resume: { endpoint: "resume", label: "Resume", needsTraffic: true },
    rollback: { endpoint: "rollback", label: "Rollback", confirm: true },
  };

  function assignStyles(node, styles) {
    Object.assign(node.style, styles);
    return node;
  }

  function unavailableSection(name, reason) {
    return {
      available: false,
      summary: `${name} unavailable.`,
      payload: {},
      reason_codes: [reason || `operator_${name}:unavailable`],
    };
  }

  function fallbackSnapshot(reason) {
    return {
      schema_version: 1,
      available: false,
      expression: unavailableSection("expression", reason),
      behavior_controls: unavailableSection("behavior_controls", reason),
      teaching_knowledge: unavailableSection("teaching_knowledge", reason),
      voice_metrics: unavailableSection("voice_metrics", reason),
      memory: unavailableSection("memory", reason),
      practice: unavailableSection("practice", reason),
      adapters: unavailableSection("adapters", reason),
      sim_to_real: unavailableSection("sim_to_real", reason),
      rollout_status: unavailableSection("rollout_status", reason),
      episode_evidence: unavailableSection("episode_evidence", reason),
      performance_learning: unavailableSection("performance_learning", reason),
      reason_codes: [reason],
    };
  }

  function fallbackEpisodeEvidence(reason) {
    return {
      schema_version: 1,
      available: false,
      summary: "Episode evidence unavailable.",
      episode_count: 0,
      source_counts: {},
      reason_code_counts: {},
      rows: [],
      reason_codes: [reason],
    };
  }

  function fallbackRolloutEvidence(reason) {
    return {
      schema_version: 1,
      available: false,
      live_episodes: [],
      practice_plans: [],
      benchmark_reports: [],
      reason_codes: [reason],
    };
  }

  function fallbackStylePresets(reason) {
    return {
      schema_version: 1,
      available: false,
      presets: [],
      default_preset_id: "",
      reason_codes: [reason],
    };
  }

  function fallbackMemoryPersonaPreset(reason) {
    return {
      schema_version: 1,
      available: false,
      accepted: false,
      applied: false,
      preset_id: "witty_sophisticated",
      preset_label: "Witty Sophisticated",
      import_id: "",
      seed_sha256: "",
      counts: {
        accepted_candidates: 0,
        rejected_entries: 0,
        applied_entries: 0,
        memory_written: 0,
        memory_noop: 0,
        behavior_controls_applied: 0,
        behavior_controls_noop: 0,
      },
      candidates: [],
      rejected_entries: [],
      applied_entries: [],
      behavior_control_result: null,
      reason_codes: [reason],
    };
  }

  function safeObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function safeArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function text(value, fallback) {
    const normalized = String(value || "").replace(/\s+/g, " ").trim();
    return normalized || fallback || "unavailable";
  }

  function numberText(value) {
    const number = Number(value);
    return Number.isFinite(number) ? String(number) : "0";
  }

  function decimalText(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(2) : "0.00";
  }

  function statusPill(available) {
    return available ? "available" : "unavailable";
  }

  function statusColor(available) {
    return available ? "#047857" : "#B45309";
  }

  function makeChip(label, available) {
    const chip = document.createElement("span");
    chip.textContent = label;
    assignStyles(chip, {
      display: "inline-flex",
      alignItems: "center",
      minHeight: "24px",
      padding: "3px 9px",
      borderRadius: "999px",
      border: `1px solid ${available ? "#A7F3D0" : "#FCD34D"}`,
      background: available ? "#ECFDF5" : "#FFFBEB",
      color: statusColor(available),
      fontSize: "12px",
      fontWeight: "700",
      letterSpacing: "0",
      whiteSpace: "nowrap",
    });
    return chip;
  }

  function addSubheading(parent, label) {
    const node = document.createElement("div");
    node.textContent = label;
    assignStyles(node, {
      marginTop: "12px",
      color: "#374151",
      fontSize: "12px",
      fontWeight: "760",
      textTransform: "uppercase",
      letterSpacing: "0",
    });
    parent.appendChild(node);
  }

  function renderMetricTiles(parent, tiles) {
    const grid = document.createElement("div");
    assignStyles(grid, {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(112px, 1fr))",
      gap: "8px",
      marginTop: "8px",
    });
    tiles.forEach((tile) => {
      const card = document.createElement("div");
      assignStyles(card, {
        minWidth: "0",
        padding: "9px 10px",
        borderRadius: "8px",
        border: "1px solid #E5E7EB",
        background: "#F9FAFB",
      });
      const value = document.createElement("div");
      value.textContent = tile.value;
      assignStyles(value, {
        color: "#111827",
        fontSize: "19px",
        fontWeight: "780",
        lineHeight: "1.1",
      });
      const label = document.createElement("div");
      label.textContent = tile.label;
      assignStyles(label, {
        marginTop: "4px",
        color: "#6B7280",
        fontSize: "11px",
        fontWeight: "650",
        lineHeight: "1.25",
      });
      card.append(value, label);
      grid.appendChild(card);
    });
    parent.appendChild(grid);
  }

  function sectionLabel(key) {
    const labels = {
      overview: "Overview",
      memory: "Memory",
      controls: "Controls",
      teaching: "Teaching",
      voice: "Voice",
      practice: "Practice",
      adapters: "Adapters",
      "sim-to-real": "Sim-to-real",
      evidence: "Evidence",
      "performance-learning": "Performance learning",
      rollouts: "Rollouts",
    };
    return labels[key] || key;
  }

  function operatorSection(snapshot, key) {
    const sections = {
      memory: "memory",
      controls: "behavior_controls",
      teaching: "teaching_knowledge",
      practice: "practice",
      adapters: "adapters",
      "sim-to-real": "sim_to_real",
      evidence: "episode_evidence",
      "performance-learning": "performance_learning",
      rollouts: "rollout_status",
    };
    return safeObject(snapshot[sections[key] || key]);
  }

  function setTextLine(parent, label, value) {
    const row = document.createElement("div");
    assignStyles(row, {
      display: "grid",
      gridTemplateColumns: "126px minmax(0, 1fr)",
      gap: "12px",
      alignItems: "start",
      marginTop: "8px",
    });
    const labelNode = document.createElement("span");
    labelNode.textContent = `${label}:`;
    assignStyles(labelNode, {
      color: "#6B7280",
      fontSize: "11px",
      fontWeight: "750",
      textTransform: "uppercase",
      letterSpacing: "0",
    });
    const valueNode = document.createElement("span");
    valueNode.textContent = text(value);
    assignStyles(valueNode, {
      color: "#111827",
      overflowWrap: "anywhere",
      minWidth: "0",
    });
    row.append(labelNode, valueNode);
    parent.appendChild(row);
  }

  function addCompactList(parent, items, renderItem, maxItems) {
    const visible = safeArray(items).slice(0, maxItems || 4);
    if (visible.length === 0) {
      setTextLine(parent, "items", "none");
      return;
    }
    visible.forEach((item) => renderItem(parent, safeObject(item)));
    if (safeArray(items).length > visible.length) {
      setTextLine(parent, "more", String(safeArray(items).length - visible.length));
    }
  }

  function statusText(record) {
    const status = record.status || "unknown";
    const currentness = record.currentness_status || null;
    return currentness ? `${status}/${currentness}` : status;
  }

  function confidenceText(value) {
    if (value === null || value === undefined || value === "") {
      return null;
    }
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(2) : String(value);
  }

  function memoryUseText(record) {
    if (record.used_in_current_turn) {
      return "used now";
    }
    if (record.last_used_reason) {
      return `last used: ${text(record.last_used_reason)}`;
    }
    if (record.last_used_at) {
      return "last used";
    }
    return "";
  }

  function behaviorEffectText(value) {
    const effects = safeArray(value).map((item) => text(item)).filter(Boolean);
    return effects.length ? effects.slice(0, 5).join(", ") : "none";
  }

  function normalizeMemoryAction(action) {
    return String(action || "").trim().replaceAll("_", "-");
  }

  function supportedRecordActions(record) {
    const seen = new Set();
    return safeArray(record.user_actions)
      .map(normalizeMemoryAction)
      .filter((action) => {
        if (!supportedMemoryActions[action] || seen.has(action)) {
          return false;
        }
        seen.add(action);
        return true;
      });
  }

  function memoryActionEndpoint(memoryId, action) {
    return `${memoryEndpoint}/${encodeURIComponent(memoryId)}/${supportedMemoryActions[action].endpoint}`;
  }

  function pendingMemoryActionKey(memoryId, action) {
    return `${memoryId}:${action}`;
  }

  function formatReasonCodes(reasonCodes) {
    const codes = safeArray(reasonCodes)
      .map((code) => String(code || "").trim())
      .filter(Boolean)
      .slice(0, 6);
    return codes.length ? codes.join(", ") : "none";
  }

  function safeMemoryActionResultText(result) {
    if (!result || typeof result !== "object") {
      return "";
    }
    const status = result.accepted ? "Accepted" : "Rejected";
    const action = normalizeMemoryAction(result.action) || "action";
    const applied = result.applied ? "applied" : "not applied";
    return `${status}: ${action} ${applied}; codes: ${formatReasonCodes(result.reason_codes)}`;
  }

  function renderMemoryActionResult(parent, memoryId) {
    const resultText = safeMemoryActionResultText(state.memoryActionResults[memoryId]);
    if (!resultText) {
      return;
    }
    const result = document.createElement("div");
    result.textContent = resultText;
    assignStyles(result, {
      color: state.memoryActionResults[memoryId].accepted ? "#047857" : "#B91C1C",
      marginTop: "7px",
      fontSize: "12px",
      fontWeight: "650",
      lineHeight: "1.35",
      overflowWrap: "anywhere",
    });
    parent.appendChild(result);
  }

  function normalizeRolloutAction(action) {
    return String(action || "").trim().replaceAll("_", "-");
  }

  function rolloutActionEndpoint(planId, action) {
    return `${rolloutEndpoint}/${encodeURIComponent(planId)}/${supportedRolloutActions[action].endpoint}`;
  }

  function pendingRolloutActionKey(planId, action) {
    return `${planId}:${action}`;
  }

  function defaultTrafficFraction(plan) {
    const current = Number(plan.traffic_fraction);
    if (Number.isFinite(current) && current > 0) {
      return current.toFixed(4);
    }
    return "0.0100";
  }

  function rolloutActionsForPlan(plan) {
    const state = String(plan.routing_state || "");
    if (state === "proposed") {
      return ["approve"];
    }
    if (state === "approved") {
      return ["activate", "rollback"];
    }
    if (["active_limited", "default_candidate", "default"].includes(state)) {
      return ["pause", "rollback"];
    }
    if (state === "paused") {
      return ["resume", "rollback"];
    }
    return [];
  }

  function rolloutPlanSummary(plan) {
    return [
      plan.adapter_family || "adapter",
      plan.candidate_backend_id || "candidate",
      plan.routing_state || "unknown",
      Number.isFinite(Number(plan.traffic_fraction))
        ? `traffic ${Number(plan.traffic_fraction).toFixed(4)}`
        : "",
      plan.scope_key || "",
    ]
      .filter(Boolean)
      .join(" | ");
  }

  function safeRolloutActionResultText(result) {
    if (!result || typeof result !== "object") {
      return "";
    }
    const status = result.accepted ? "Accepted" : "Rejected";
    const action = normalizeRolloutAction(result.action) || "action";
    const applied = result.applied ? "applied" : "not applied";
    const before = safeObject(result.before).routing_state || result.from_state || "unknown";
    const after = safeObject(result.after).routing_state || result.to_state || "unknown";
    return `${status}: ${action} ${applied}; ${before} -> ${after}; codes: ${formatReasonCodes(
      result.reason_codes,
    )}`;
  }

  function renderRolloutActionResult(parent, planId) {
    const resultText = safeRolloutActionResultText(state.rolloutActionResults[planId]);
    if (!resultText) {
      return;
    }
    const result = document.createElement("div");
    result.textContent = resultText;
    assignStyles(result, {
      color: state.rolloutActionResults[planId].accepted ? "#047857" : "#B91C1C",
      marginTop: "7px",
      fontSize: "12px",
      fontWeight: "650",
      lineHeight: "1.35",
      overflowWrap: "anywhere",
    });
    parent.appendChild(result);
  }

  async function postRolloutAction(planId, action, body) {
    try {
      const response = await fetch(rolloutActionEndpoint(planId, action), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      if (!response.ok) {
        throw new Error("rollout action request failed");
      }
      return await response.json();
    } catch (_error) {
      return {
        schema_version: 1,
        accepted: false,
        applied: false,
        action,
        plan_id: planId,
        from_state: "unavailable",
        to_state: "unavailable",
        traffic_fraction: 0,
        reason_codes: ["rollout_action_request_failed"],
      };
    }
  }

  async function runRolloutAction(plan, action, body) {
    const planId = String(plan.plan_id || "");
    const key = pendingRolloutActionKey(planId, action);
    if (!planId || state.rolloutPendingActions[key]) {
      return;
    }
    state.rolloutPendingActions[key] = true;
    render();
    const result = await postRolloutAction(planId, action, body);
    delete state.rolloutPendingActions[key];
    state.rolloutActionResults[planId] = result;
    state.latestRolloutActionResult = result;
    await refresh();
  }

  function requestRolloutAction(plan, action) {
    const planId = String(plan.plan_id || "");
    if (!planId || !supportedRolloutActions[action]) {
      return;
    }
    if (
      supportedRolloutActions[action].confirm &&
      !window.confirm(`Confirm ${supportedRolloutActions[action].label.toLowerCase()} for this rollout?`)
    ) {
      return;
    }
    const body = {};
    if (supportedRolloutActions[action].needsTraffic) {
      const draft = Number(state.rolloutTrafficDrafts[planId] || defaultTrafficFraction(plan));
      body.traffic_fraction = Number.isFinite(draft) ? draft : 0;
    }
    if (action === "rollback") {
      body.regression_codes = ["operator_requested_rollback"];
    }
    runRolloutAction(plan, action, body);
  }

  function renderRolloutControls(parent, plan) {
    const planId = String(plan.plan_id || "");
    const actions = rolloutActionsForPlan(plan);
    if (!planId || actions.length === 0) {
      return;
    }
    const row = document.createElement("div");
    assignStyles(row, {
      display: "flex",
      flexWrap: "wrap",
      gap: "7px",
      marginTop: "9px",
      alignItems: "center",
    });
    if (actions.some((action) => supportedRolloutActions[action].needsTraffic)) {
      const input = document.createElement("input");
      input.type = "number";
      input.min = "0";
      input.max = "1";
      input.step = "0.01";
      input.value = state.rolloutTrafficDrafts[planId] || defaultTrafficFraction(plan);
      input.setAttribute("aria-label", "Rollout traffic fraction");
      assignStyles(input, {
        width: "92px",
        border: "1px solid #D1D5DB",
        borderRadius: "6px",
        background: "#FFFFFF",
        color: "#111827",
        padding: "5px 7px",
        fontSize: "12px",
      });
      input.addEventListener("input", () => {
        state.rolloutTrafficDrafts[planId] = input.value;
      });
      row.appendChild(input);
    }
    actions.forEach((action) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = supportedRolloutActions[action].label;
      button.disabled = Boolean(state.rolloutPendingActions[pendingRolloutActionKey(planId, action)]);
      assignStyles(button, {
        border: "1px solid #D1D5DB",
        borderRadius: "6px",
        background: "#FFFFFF",
        color: "#111827",
        cursor: button.disabled ? "wait" : "pointer",
        padding: "5px 8px",
        fontSize: "12px",
        fontWeight: "700",
      });
      button.addEventListener("click", () => requestRolloutAction(plan, action));
      row.appendChild(button);
    });
    parent.appendChild(row);
  }

  function renderRolloutPlan(parent, plan) {
    const item = document.createElement("div");
    assignStyles(item, {
      marginTop: "10px",
      padding: "10px 11px",
      border: "1px solid #E5E7EB",
      borderRadius: "8px",
      background: "#FFFFFF",
    });
    const title = document.createElement("div");
    title.textContent = rolloutPlanSummary(plan);
    assignStyles(title, {
      color: "#111827",
      fontWeight: "700",
      fontSize: "14px",
      lineHeight: "1.35",
    });
    const meta = document.createElement("div");
    meta.textContent = [
      plan.promotion_state || "",
      plan.embodied_live ? "embodied live" : "",
      plan.expires_at ? `expires ${plan.expires_at}` : "",
    ]
      .filter(Boolean)
      .join(" | ");
    assignStyles(meta, {
      color: "#4B5563",
      marginTop: "5px",
      fontSize: "12px",
      overflowWrap: "anywhere",
    });
    item.append(title, meta);
    renderRolloutControls(item, plan);
    renderRolloutActionResult(item, String(plan.plan_id || ""));
    parent.appendChild(item);
  }

  function renderEvidenceItem(parent, label, value) {
    const item = document.createElement("div");
    assignStyles(item, {
      marginTop: "8px",
      padding: "8px 9px",
      border: "1px solid #E5E7EB",
      borderRadius: "8px",
      background: "#FFFFFF",
    });
    const labelNode = document.createElement("div");
    labelNode.textContent = label;
    assignStyles(labelNode, {
      color: "#111827",
      fontSize: "12px",
      fontWeight: "760",
      lineHeight: "1.25",
    });
    const valueNode = document.createElement("div");
    valueNode.textContent = value || "unavailable";
    assignStyles(valueNode, {
      color: "#4B5563",
      marginTop: "3px",
      fontSize: "12px",
      lineHeight: "1.35",
      overflowWrap: "anywhere",
    });
    item.append(labelNode, valueNode);
    parent.appendChild(item);
  }

  async function postMemoryAction(memoryId, action, body) {
    try {
      const response = await fetch(memoryActionEndpoint(memoryId, action), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      if (!response.ok) {
        throw new Error("memory action request failed");
      }
      return await response.json();
    } catch (_error) {
      return {
        schema_version: 1,
        accepted: false,
        applied: false,
        action,
        memory_id: memoryId,
        record_kind: null,
        replacement_memory_id: null,
        reason_codes: ["memory_action_request_failed"],
      };
    }
  }

  async function runMemoryAction(memoryId, action, body) {
    const key = pendingMemoryActionKey(memoryId, action);
    if (state.memoryPendingActions[key]) {
      return;
    }
    state.memoryPendingActions[key] = true;
    render();
    const result = await postMemoryAction(memoryId, action, body);
    delete state.memoryPendingActions[key];
    state.memoryActionResults[memoryId] = result;
    state.latestMemoryActionResult = result;
    if (action === "correct" && result && result.accepted) {
      delete state.correctionDrafts[memoryId];
      if (state.correctionMemoryId === memoryId) {
        state.correctionMemoryId = null;
      }
    }
    await refresh();
  }

  function requestMemoryAction(record, action) {
    const memoryId = String(record.memory_id || "");
    if (!memoryId || !supportedMemoryActions[action]) {
      return;
    }
    if (action === "correct") {
      state.correctionMemoryId = memoryId;
      state.correctionDrafts[memoryId] = state.correctionDrafts[memoryId] || "";
      render();
      return;
    }
    if (
      supportedMemoryActions[action].confirm &&
      !window.confirm(`Confirm ${supportedMemoryActions[action].label.toLowerCase()} for this memory?`)
    ) {
      return;
    }
    runMemoryAction(memoryId, action, {});
  }

  function addMemoryActionControls(parent, record) {
    const memoryId = String(record.memory_id || "");
    const actions = supportedRecordActions(record);
    if (!memoryId || actions.length === 0) {
      return;
    }
    const row = document.createElement("div");
    assignStyles(row, {
      display: "flex",
      flexWrap: "wrap",
      gap: "6px",
      marginTop: "9px",
    });
    actions.forEach((action) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = supportedMemoryActions[action].label;
      button.disabled = Boolean(state.memoryPendingActions[pendingMemoryActionKey(memoryId, action)]);
      assignStyles(button, {
        border: "1px solid #D1D5DB",
        borderRadius: "6px",
        background: "#FFFFFF",
        color: "#111827",
        cursor: button.disabled ? "wait" : "pointer",
        padding: "5px 8px",
        fontSize: "12px",
        fontWeight: "700",
      });
      button.addEventListener("click", () => requestMemoryAction(record, action));
      row.appendChild(button);
    });
    parent.appendChild(row);
  }

  function renderCorrectionForm(parent, record) {
    const memoryId = String(record.memory_id || "");
    if (!memoryId || state.correctionMemoryId !== memoryId) {
      return;
    }
    const form = document.createElement("form");
    form.setAttribute("data-memory-correction-form", "");
    assignStyles(form, {
      display: "grid",
      gap: "8px",
      marginTop: "9px",
      padding: "10px",
      border: "1px solid #E5E7EB",
      borderRadius: "8px",
      background: "#F9FAFB",
    });
    const input = document.createElement("input");
    input.required = true;
    input.type = "text";
    input.value = state.correctionDrafts[memoryId] || "";
    input.placeholder = "Replacement memory value";
    assignStyles(input, {
      border: "1px solid #D1D5DB",
      borderRadius: "6px",
      background: "#FFFFFF",
      color: "#111827",
      padding: "7px 8px",
      fontSize: "13px",
      width: "100%",
    });
    input.addEventListener("input", () => {
      state.correctionDrafts[memoryId] = input.value;
    });
    const controls = document.createElement("div");
    assignStyles(controls, {
      display: "flex",
      flexWrap: "wrap",
      gap: "7px",
    });
    const submit = document.createElement("button");
    submit.type = "submit";
    submit.textContent = "Submit correction";
    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "Close";
    [submit, close].forEach((button) => {
      assignStyles(button, {
        border: "1px solid #D1D5DB",
        borderRadius: "6px",
        background: "#FFFFFF",
        color: "#111827",
        cursor: "pointer",
        padding: "6px 8px",
        fontSize: "12px",
        fontWeight: "700",
      });
    });
    close.addEventListener("click", () => {
      state.correctionMemoryId = null;
      render();
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const replacementValue = input.value.trim();
      state.correctionDrafts[memoryId] = replacementValue;
      if (!replacementValue) {
        state.memoryActionResults[memoryId] = {
          schema_version: 1,
          accepted: false,
          applied: false,
          action: "correct",
          memory_id: memoryId,
          reason_codes: ["replacement_value_required"],
        };
        state.latestMemoryActionResult = state.memoryActionResults[memoryId];
        render();
        return;
      }
      runMemoryAction(memoryId, "correct", { replacement_value: replacementValue });
    });
    controls.append(submit, close);
    form.append(input, controls);
    parent.appendChild(form);
  }

  function renderRecord(parent, record) {
    const item = document.createElement("div");
    assignStyles(item, {
      marginTop: "10px",
      padding: "10px 11px",
      border: "1px solid #E5E7EB",
      borderRadius: "8px",
      background: "#FFFFFF",
    });
    const title = document.createElement("div");
    title.textContent = text(record.summary || record.title, "Memory");
    assignStyles(title, {
      color: "#111827",
      fontWeight: "700",
      fontSize: "14px",
      lineHeight: "1.35",
    });
    const meta = document.createElement("div");
    meta.textContent = [
      record.display_kind || "memory",
      statusText(record),
      confidenceText(record.confidence) ? `confidence ${confidenceText(record.confidence)}` : "",
      record.pinned ? "pinned" : "",
      memoryUseText(record),
    ]
      .filter(Boolean)
      .join(" | ");
    assignStyles(meta, {
      color: "#4B5563",
      marginTop: "5px",
      fontSize: "12px",
    });
    item.append(title, meta);
    if (record.safe_provenance_label) {
      const provenance = document.createElement("div");
      provenance.textContent = text(record.safe_provenance_label);
      assignStyles(provenance, {
        color: "#6B7280",
        marginTop: "4px",
        fontSize: "12px",
      });
      item.appendChild(provenance);
    }
    addMemoryActionControls(item, record);
    renderCorrectionForm(item, record);
    renderMemoryActionResult(item, String(record.memory_id || ""));
    parent.appendChild(item);
  }

  function renderUsedMemoryRef(parent, record) {
    const item = document.createElement("div");
    assignStyles(item, {
      marginTop: "8px",
      padding: "8px 10px",
      border: "1px solid #D1D5DB",
      borderRadius: "8px",
      background: "#F9FAFB",
    });
    const title = document.createElement("div");
    title.textContent = text(record.title, "Memory");
    assignStyles(title, {
      color: "#111827",
      fontWeight: "700",
      fontSize: "13px",
      lineHeight: "1.35",
    });
    const meta = document.createElement("div");
    meta.textContent = [
      record.display_kind || "memory",
      record.used_reason ? `reason ${text(record.used_reason)}` : "",
      record.behavior_effect ? `effect ${text(record.behavior_effect)}` : "",
      safeArray(record.effect_labels).length
        ? `effects ${safeArray(record.effect_labels).slice(0, 3).join(", ")}`
        : "",
      safeArray(record.linked_discourse_episode_ids).length
        ? `${safeArray(record.linked_discourse_episode_ids).length} discourse refs`
        : "",
    ]
      .filter(Boolean)
      .join(" | ");
    assignStyles(meta, {
      color: "#4B5563",
      marginTop: "4px",
      fontSize: "12px",
    });
    item.append(title, meta);
    parent.appendChild(item);
  }

  function renderDiscourseEpisodeRef(parent, record) {
    const item = document.createElement("div");
    assignStyles(item, {
      marginTop: "8px",
      padding: "8px 10px",
      border: "1px solid #D1FAE5",
      borderRadius: "8px",
      background: "#ECFDF5",
    });
    const title = document.createElement("div");
    title.textContent = safeArray(record.category_labels).slice(0, 2).join(" + ") || "discourse";
    assignStyles(title, {
      color: "#111827",
      fontWeight: "700",
      fontSize: "13px",
      lineHeight: "1.35",
    });
    const meta = document.createElement("div");
    meta.textContent = [
      safeArray(record.effect_labels).length
        ? `effects ${safeArray(record.effect_labels).slice(0, 3).join(", ")}`
        : "",
      record.confidence_bucket ? `confidence ${text(record.confidence_bucket)}` : "",
      record.discourse_episode_id ? text(record.discourse_episode_id) : "",
    ]
      .filter(Boolean)
      .join(" | ");
    assignStyles(meta, {
      color: "#4B5563",
      marginTop: "4px",
      fontSize: "12px",
    });
    item.append(title, meta);
    parent.appendChild(item);
  }

  function renderPersonaReference(parent, record) {
    const item = document.createElement("div");
    assignStyles(item, {
      marginTop: "8px",
      padding: "8px 10px",
      border: "1px solid #E5E7EB",
      borderRadius: "8px",
      background: record.applies ? "#F0FDF4" : "#FFFFFF",
    });
    const title = document.createElement("div");
    title.textContent = text(record.label || record.mode, "persona reference");
    assignStyles(title, {
      color: "#111827",
      fontWeight: "700",
      fontSize: "13px",
      lineHeight: "1.35",
    });
    const meta = document.createElement("div");
    meta.textContent = [
      record.mode || "reference",
      record.applies ? "active" : "standby",
      record.behavior_effect || "",
    ]
      .filter(Boolean)
      .join(" | ");
    assignStyles(meta, {
      color: "#4B5563",
      marginTop: "4px",
      fontSize: "12px",
    });
    item.append(title, meta);
    parent.appendChild(item);
  }

  function renderPersonaAnchor(parent, record) {
    const item = document.createElement("div");
    assignStyles(item, {
      marginTop: "8px",
      padding: "8px 10px",
      border: "1px solid #DBEAFE",
      borderRadius: "8px",
      background: "#EFF6FF",
    });
    const title = document.createElement("div");
    title.textContent = text(record.situation_key || record.anchor_id, "style anchor");
    assignStyles(title, {
      color: "#111827",
      fontWeight: "700",
      fontSize: "13px",
      lineHeight: "1.35",
    });
    const meta = document.createElement("div");
    meta.textContent = [
      record.stance_label || "",
      record.response_shape_label || "",
      record.negative_example_count !== undefined
        ? `${numberText(record.negative_example_count)} avoid rules`
        : "",
    ]
      .filter(Boolean)
      .join(" | ");
    assignStyles(meta, {
      color: "#4B5563",
      marginTop: "4px",
      fontSize: "12px",
    });
    item.append(title, meta);
    parent.appendChild(item);
  }

  function renderCountMap(parent, label, mapping) {
    const data = safeObject(mapping);
    const entries = Object.entries(data).slice(0, 4);
    if (entries.length === 0) {
      setTextLine(parent, label, "none");
      return;
    }
    setTextLine(
      parent,
      label,
      entries.map(([key, value]) => `${key}=${value}`).join(", "),
    );
  }

  function summarizeReasonCodes(reasonCodes) {
    const codes = safeArray(reasonCodes);
    if (codes.length === 0) {
      return "none";
    }
    const categories = Array.from(
      new Set(
        codes
          .map((code) => String(code || "").split(":")[0])
          .filter((category) => category.length > 0),
      ),
    ).slice(0, 5);
    return `${codes.length} codes${categories.length ? `; categories=${categories.join(", ")}` : ""}`;
  }

  function syncControlDrafts(profile, force) {
    const data = safeObject(profile);
    if (!force && state.controlDraftDirty) {
      return;
    }
    controlFields.forEach((field) => {
      const options = controlOptions[field] || [];
      const current = data[field];
      state.controlDrafts[field] = options.includes(current) ? current : options[0] || "";
    });
    state.controlDraftDirty = false;
  }

  function applyStylePresetDraft(presetId) {
    const presets = safeArray(safeObject(state.stylePresets).presets);
    const preset = safeObject(presets.find((item) => safeObject(item).preset_id === presetId));
    const updates = safeObject(preset.control_updates);
    controlFields.forEach((field) => {
      const value = updates[field];
      if ((controlOptions[field] || []).includes(value)) {
        state.controlDrafts[field] = value;
      }
    });
    state.stylePresetDraft = presetId;
    state.controlDraftDirty = true;
  }

  function safeBehaviorControlResultText(result) {
    if (!result || typeof result !== "object") {
      return "";
    }
    const status = result.accepted ? "Accepted" : "Rejected";
    const applied = result.applied ? "applied" : "not applied";
    return `${status}: behavior controls ${applied}; ${summarizeReasonCodes(result.reason_codes)}`;
  }

  function safeMemoryPersonaPresetResultText(result) {
    if (!result || typeof result !== "object") {
      return "";
    }
    const status = result.accepted ? "Accepted" : "Rejected";
    const applied = result.applied ? "applied" : "not applied";
    const counts = safeObject(result.counts);
    const written = Number(counts.memory_written || 0);
    const behavior = Number(counts.behavior_controls_applied || 0);
    return `${status}: story seed ${applied}; memory written ${written}; controls ${behavior}; ${summarizeReasonCodes(
      result.reason_codes,
    )}`;
  }

  async function postBehaviorControlsUpdate(payload) {
    try {
      const response = await fetch(behaviorControlsEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {}),
      });
      if (!response.ok) {
        throw new Error("behavior controls request failed");
      }
      return await response.json();
    } catch (_error) {
      return {
        schema_version: 1,
        accepted: false,
        applied: false,
        profile: null,
        compiled_effect_summary: "",
        rejected_fields: [],
        reason_codes: ["behavior_controls_update_request_failed"],
      };
    }
  }

  async function postPerformancePreferencePair(payload) {
    try {
      const response = await fetch(performancePreferencesEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {}),
      });
      if (!response.ok) {
        throw new Error("performance preference request failed");
      }
      return await response.json();
    } catch (_error) {
      return {
        schema_version: 3,
        accepted: false,
        applied: false,
        pair: null,
        policy_proposals: [],
        sanitizer: {
          accepted: false,
          blocked_keys: [],
          blocked_values: [],
          omitted_keys: [],
          reason_codes: ["performance_preference_request_failed"],
        },
        reason_codes: ["performance_preference_request_failed"],
      };
    }
  }

  async function postPerformancePolicyProposalApply(proposalId) {
    try {
      const response = await fetch(performancePolicyProposalApplyEndpoint(proposalId), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ operator_acknowledged: true }),
      });
      if (!response.ok) {
        throw new Error("performance policy proposal apply request failed");
      }
      return await response.json();
    } catch (_error) {
      return {
        schema_version: 3,
        accepted: false,
        applied: false,
        proposal_id: proposalId,
        proposal: null,
        behavior_control_result: null,
        reason_codes: ["performance_policy_proposal_apply_request_failed"],
      };
    }
  }

  function performancePreferenceCandidateFromEvidence(side) {
    const snapshot = safeObject(state.snapshot);
    const performanceLearning = safeObject(safeObject(snapshot.performance_learning).payload);
    const recentPairs = safeArray(performanceLearning.recent_pairs);
    const latestPair = safeObject(recentPairs[recentPairs.length - 1]);
    const previousCandidate = safeObject(
      side === "a" ? latestPair.candidate_b || latestPair.candidate_a : latestPair.candidate_a,
    );
    const evidencePayload = safeObject(safeObject(snapshot.episode_evidence).payload);
    const rows = safeArray(evidencePayload.rows);
    const row = safeObject(rows[side === "a" ? 1 : 0]);
    const fallbackId =
      row.episode_id || row.artifact_id || `${side === "a" ? "baseline" : "candidate"}-local`;
    const profile =
      row.profile ||
      previousCandidate.profile ||
      (snapshot.available ? "cross_profile" : "cross_profile");
    const language = row.language || previousCandidate.language || "mixed";
    const ttsRuntimeLabel = row.tts_runtime_label || previousCandidate.tts_runtime_label || "mixed";
    const summary =
      row.summary ||
      row.public_summary ||
      previousCandidate.public_summary ||
      (side === "a" ? "Baseline public-safe evidence." : "Current public-safe evidence.");
    return {
      candidate_id: previousCandidate.candidate_id || fallbackId,
      candidate_kind: side === "a" ? "baseline_trace" : "candidate_trace",
      profile,
      language,
      tts_runtime_label: ttsRuntimeLabel,
      candidate_label: side === "a" ? "Candidate A" : "Candidate B",
      episode_ids: row.episode_id ? [row.episode_id] : safeArray(previousCandidate.episode_ids),
      plan_ids: safeArray(previousCandidate.plan_ids),
      control_frame_ids: safeArray(previousCandidate.control_frame_ids),
      public_summary: summary,
      summary_hash: previousCandidate.summary_hash || row.summary_hash || "",
      segment_counts: safeObject(previousCandidate.segment_counts || row.segment_counts),
      metric_counts: safeObject(previousCandidate.metric_counts || row.metric_counts),
      policy_labels: safeArray(previousCandidate.policy_labels || row.policy_labels),
      camera_honesty_states: safeArray(
        previousCandidate.camera_honesty_states || row.camera_honesty_states,
      ),
      reason_codes: ["performance_learning:candidate_ref"],
    };
  }

  function buildPerformancePreferencePayload() {
    const draft = safeObject(state.performancePreferenceDraft);
    const ratings = {};
    performancePreferenceDimensions.forEach((dimension) => {
      const value = Number(safeObject(draft.ratings)[dimension] || 3);
      ratings[dimension] = Math.max(1, Math.min(5, Number.isFinite(value) ? value : 3));
    });
    const candidateA = performancePreferenceCandidateFromEvidence("a");
    const candidateB = performancePreferenceCandidateFromEvidence("b");
    const profile = candidateA.profile === candidateB.profile ? candidateA.profile : "cross_profile";
    const language = candidateA.language === candidateB.language ? candidateA.language : "mixed";
    const ttsRuntimeLabel =
      candidateA.tts_runtime_label === candidateB.tts_runtime_label
        ? candidateA.tts_runtime_label
        : "mixed";
    return {
      schema_version: 3,
      profile,
      language,
      tts_runtime_label: ttsRuntimeLabel,
      candidate_a: candidateA,
      candidate_b: candidateB,
      winner: ["a", "b", "same", "neither"].includes(draft.winner) ? draft.winner : "same",
      ratings,
      improvement_labels: safeArray(draft.improvement_labels),
      failure_labels: safeArray(draft.failure_labels),
      evidence_refs: [
        {
          evidence_kind: "operator_workbench",
          evidence_id: "performance_learning_pairwise_review",
          summary: "Local pairwise review from public-safe workbench evidence.",
          reason_codes: ["performance_learning:operator_rating"],
        },
      ],
      reason_codes: ["performance_learning:operator_submitted_pair"],
    };
  }

  async function runPerformancePreferenceSubmit() {
    if (state.performancePreferenceSubmitPending) {
      return;
    }
    state.performancePreferenceSubmitPending = true;
    render();
    const result = await postPerformancePreferencePair(buildPerformancePreferencePayload());
    state.performancePreferenceSubmitPending = false;
    state.latestPerformancePreferenceResult = result;
    await refresh();
  }

  async function applyPerformancePolicyProposal(proposalId) {
    if (!proposalId || state.performancePolicyApplyPending[proposalId]) {
      return;
    }
    if (!window.confirm("Apply this performance-learning policy proposal?")) {
      return;
    }
    state.performancePolicyApplyPending[proposalId] = true;
    render();
    const result = await postPerformancePolicyProposalApply(proposalId);
    state.performancePolicyApplyPending[proposalId] = false;
    state.latestPerformancePolicyApplyResult = result;
    await refresh();
  }

  async function postMemoryPersonaPresetApply(preview) {
    try {
      const response = await fetch(memoryPersonaPresetApplyEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved_report: preview || {} }),
      });
      if (!response.ok) {
        throw new Error("memory persona preset request failed");
      }
      return await response.json();
    } catch (_error) {
      return fallbackMemoryPersonaPreset("memory_persona_preset_apply_request_failed");
    }
  }

  async function requestMemoryPersonaPresetApply() {
    if (state.memoryPersonaPresetApplyPending) {
      return;
    }
    const preview = safeObject(state.memoryPersonaPresetPreview);
    if (!preview.accepted) {
      state.memoryPersonaPresetApplyResult = fallbackMemoryPersonaPreset(
        "memory_persona_preset_preview_not_accepted",
      );
      render();
      return;
    }
    if (!window.confirm("Apply the Witty Sophisticated memory/personality seed?")) {
      return;
    }
    state.memoryPersonaPresetApplyPending = true;
    render();
    const result = await postMemoryPersonaPresetApply(preview);
    state.memoryPersonaPresetApplyPending = false;
    state.memoryPersonaPresetApplyResult = result;
    await refresh();
  }

  async function runBehaviorControlsUpdate() {
    if (state.controlUpdatePending) {
      return;
    }
    const payload = {};
    controlFields.forEach((field) => {
      const value = state.controlDrafts[field];
      if ((controlOptions[field] || []).includes(value)) {
        payload[field] = value;
      }
    });
    state.controlUpdatePending = true;
    render();
    const result = await postBehaviorControlsUpdate(payload);
    state.controlUpdatePending = false;
    state.controlUpdateResult = result;
    if (result && result.accepted && result.profile) {
      syncControlDrafts(result.profile, true);
    }
    await refresh();
  }

  function renderOverview(body, snapshot) {
    const sections = [
      snapshot.expression,
      snapshot.behavior_controls,
      snapshot.teaching_knowledge,
      snapshot.voice_metrics,
      snapshot.memory,
      snapshot.practice,
      snapshot.adapters,
      snapshot.sim_to_real,
      snapshot.episode_evidence,
      snapshot.performance_learning,
      snapshot.rollout_status,
    ].map(safeObject);
    const availableCount = sections.filter((section) => section.available).length;
    setTextLine(body, "status", statusPill(snapshot.available));
    setTextLine(body, "sections", `${availableCount}/${sections.length} available`);
    const expression = safeObject(safeObject(snapshot.expression).payload);
    setTextLine(body, "identity", expression.identity_label || "Blink");
    setTextLine(body, "teaching", expression.teaching_mode_label || "unavailable");
    setTextLine(body, "modality", expression.modality || "unavailable");
    setTextLine(body, "initiative", expression.initiative_label || "unavailable");
    setTextLine(body, "evidence", expression.evidence_visibility_label || "unavailable");
    setTextLine(body, "correction", expression.correction_mode_label || "unavailable");
    setTextLine(body, "structure", expression.explanation_structure_label || "unavailable");
    setTextLine(body, "humor mode", expression.humor_mode_label || "unavailable");
    setTextLine(body, "vividness mode", expression.vividness_mode_label || "unavailable");
    setTextLine(body, "presence mode", expression.character_presence_label || "unavailable");
    setTextLine(body, "style", expression.style_summary || "unavailable");
    setTextLine(body, "safety clamp", expression.safety_clamped ? "active" : "inactive");
  }

  function renderMemory(body, section) {
    const payload = safeObject(section.payload);
    const performance = safeObject(payload.memory_persona_performance);
    const usedInCurrentReply = safeArray(
      payload.used_in_current_reply || performance.used_in_current_reply,
    );
    const behaviorEffects = safeArray(payload.behavior_effects || performance.behavior_effects);
    const personaReferences = safeArray(
      payload.persona_references || performance.persona_references,
    );
    const selectedPersonaAnchors = safeArray(
      payload.persona_anchor_refs_v3 || performance.persona_anchor_refs_v3,
    );
    const personaAnchorBank = safeObject(
      payload.persona_anchor_bank_v3 || performance.persona_anchor_bank_v3,
    );
    const personaAnchorCatalog = safeArray(personaAnchorBank.anchors);
    const continuityTrace = safeObject(
      payload.memory_continuity_trace || performance.memory_continuity_trace,
    );
    const continuityV3 = safeObject(continuityTrace.memory_continuity_v3);
    const discourseEpisodes = safeArray(continuityV3.selected_discourse_episodes);
    const memoryEffectLabels = safeArray(continuityV3.effect_labels);
    setTextLine(body, "summary", section.summary || payload.summary);
    setTextLine(body, "health", payload.health_summary || "Memory health unavailable.");
    setTextLine(body, "Used in this reply", String(usedInCurrentReply.length));
    setTextLine(body, "Behavior effect", behaviorEffectText(behaviorEffects));
    setTextLine(
      body,
      "Memory effects",
      memoryEffectLabels.length ? memoryEffectLabels.slice(0, 4).join(", ") : "none",
    );
    setTextLine(body, "Discourse episodes", String(discourseEpisodes.length));
    setTextLine(
      body,
      "persona refs",
      `${personaReferences.filter((record) => record && record.applies).length} active`,
    );
    setTextLine(
      body,
      "Style anchors",
      `${selectedPersonaAnchors.length} selected; ${
        personaAnchorBank.anchor_count || personaAnchorCatalog.length
      } catalog`,
    );
    setTextLine(body, "visible", String(safeArray(payload.records).length));
    renderCountMap(body, "hidden", payload.hidden_counts);
    if (safeMemoryActionResultText(state.latestMemoryActionResult)) {
      setTextLine(body, "last action", safeMemoryActionResultText(state.latestMemoryActionResult));
    }
    addCompactList(body, usedInCurrentReply, renderUsedMemoryRef, 5);
    addCompactList(
      body,
      personaReferences.filter((record) => record && record.applies),
      renderPersonaReference,
      4,
    );
    addSubheading(body, "Discourse episodes");
    addCompactList(body, discourseEpisodes, renderDiscourseEpisodeRef, 4);
    addSubheading(body, "Style anchors");
    addCompactList(
      body,
      selectedPersonaAnchors.length ? selectedPersonaAnchors : personaAnchorCatalog,
      renderPersonaAnchor,
      4,
    );
    addCompactList(body, payload.records, renderRecord, 5);
  }

  function renderControls(body, section) {
    const payload = safeObject(section.payload);
    const profile = safeObject(payload.profile);
    syncControlDrafts(profile, false);
    setTextLine(body, "summary", section.summary || payload.compiled_effect_summary);
    setTextLine(body, "depth", profile.response_depth);
    setTextLine(body, "directness", profile.directness);
    setTextLine(body, "warmth", profile.warmth);
    setTextLine(body, "teaching", profile.teaching_mode);
    setTextLine(body, "memory", profile.memory_use);
    setTextLine(body, "initiative", profile.initiative_mode);
    setTextLine(body, "evidence", profile.evidence_visibility);
    setTextLine(body, "correction", profile.correction_mode);
    setTextLine(body, "structure", profile.explanation_structure);
    setTextLine(body, "challenge", profile.challenge_style);
    setTextLine(body, "voice", profile.voice_mode);
    setTextLine(body, "questions", profile.question_budget);
    setTextLine(body, "humor", profile.humor_mode);
    setTextLine(body, "vividness", profile.vividness_mode);
    setTextLine(body, "sophistication", profile.sophistication_mode);
    setTextLine(body, "presence", profile.character_presence);
    setTextLine(body, "story", profile.story_mode);
    const expression = safeObject(safeObject(state.snapshot.expression).payload);
    setTextLine(body, "applied style", expression.style_summary || "unavailable");
    setTextLine(
      body,
      "style metrics",
      `humor ${decimalText(expression.humor_budget)} | play ${decimalText(
        expression.playfulness,
      )} | metaphor ${decimalText(expression.metaphor_density)}`,
    );
    renderBehaviorControlForm(body, section);
  }

  function safePerformancePreferenceResultText(result) {
    if (!result || typeof result !== "object") {
      return "";
    }
    const status = result.accepted ? "Accepted" : "Rejected";
    const proposals = safeArray(result.policy_proposals).length;
    return `${status}: preference pair ${
      result.applied ? "recorded" : "not recorded"
    }; proposals ${proposals}; ${summarizeReasonCodes(result.reason_codes)}`;
  }

  function safePerformancePolicyApplyResultText(result) {
    if (!result || typeof result !== "object") {
      return "";
    }
    const status = result.accepted ? "Accepted" : "Rejected";
    return `${status}: policy proposal ${
      result.applied ? "applied" : "not applied"
    }; ${summarizeReasonCodes(result.reason_codes)}`;
  }

  function renderPerformancePreferencePair(parent, pair) {
    const candidateA = safeObject(pair.candidate_a);
    const candidateB = safeObject(pair.candidate_b);
    renderEvidenceItem(
      parent,
      `${pair.winner || "same"} | ${pair.profile || "cross_profile"}`,
      [
        `A=${candidateA.candidate_id || "candidate-a"}`,
        `B=${candidateB.candidate_id || "candidate-b"}`,
        `labels=${safeArray(pair.failure_labels).concat(safeArray(pair.improvement_labels)).join(", ") || "none"}`,
      ].join("; "),
    );
  }

  function renderPerformancePolicyProposal(parent, proposal) {
    const proposalId = String(proposal.proposal_id || "");
    const item = document.createElement("div");
    assignStyles(item, {
      marginTop: "8px",
      padding: "8px 9px",
      border: "1px solid #E5E7EB",
      borderRadius: "8px",
      background: "#FFFFFF",
    });
    const title = document.createElement("div");
    title.textContent = proposal.target || "policy proposal";
    assignStyles(title, {
      color: "#111827",
      fontSize: "12px",
      fontWeight: "760",
      lineHeight: "1.25",
    });
    const value = document.createElement("div");
    value.textContent = [
      proposal.status || "proposed",
      proposal.summary || "",
      `pairs=${safeArray(proposal.source_pair_ids).length}`,
      `updates=${Object.keys(safeObject(proposal.behavior_control_updates)).join(", ") || "none"}`,
    ]
      .filter(Boolean)
      .join("; ");
    assignStyles(value, {
      color: "#4B5563",
      marginTop: "3px",
      fontSize: "12px",
      lineHeight: "1.35",
      overflowWrap: "anywhere",
    });
    item.append(title, value);
    const canApply = proposalId && proposal.status !== "applied";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = state.performancePolicyApplyPending[proposalId]
      ? "Applying..."
      : "Apply proposal";
    button.disabled = !canApply || state.performancePolicyApplyPending[proposalId];
    assignStyles(button, {
      marginTop: "8px",
      border: "1px solid #D1D5DB",
      borderRadius: "6px",
      background: "#FFFFFF",
      color: "#111827",
      cursor: button.disabled ? "wait" : "pointer",
      padding: "6px 9px",
      fontSize: "12px",
      fontWeight: "700",
    });
    button.addEventListener("click", () => applyPerformancePolicyProposal(proposalId));
    item.appendChild(button);
    parent.appendChild(item);
  }

  function renderPerformanceLearningForm(parent) {
    const form = document.createElement("form");
    form.setAttribute("data-performance-learning-form", "");
    assignStyles(form, {
      display: "grid",
      gap: "10px",
      marginTop: "12px",
      padding: "12px",
      border: "1px solid #E5E7EB",
      borderRadius: "8px",
      background: "#F9FAFB",
      maxWidth: "760px",
    });
    const title = document.createElement("div");
    title.textContent = "Pairwise rating";
    assignStyles(title, {
      color: "#111827",
      fontWeight: "700",
      fontSize: "14px",
    });
    form.appendChild(title);

    const winnerRow = document.createElement("label");
    assignStyles(winnerRow, {
      display: "grid",
      gridTemplateColumns: "132px minmax(0, 1fr)",
      gap: "10px",
      alignItems: "center",
    });
    const winnerLabel = document.createElement("span");
    winnerLabel.textContent = "Winner";
    winnerLabel.style.color = "#374151";
    const winnerSelect = document.createElement("select");
    winnerSelect.name = "winner";
    winnerSelect.disabled = state.performancePreferenceSubmitPending;
    ["a", "b", "same", "neither"].forEach((winner) => {
      const option = document.createElement("option");
      option.value = winner;
      option.textContent = winner;
      if (state.performancePreferenceDraft.winner === winner) {
        option.selected = true;
      }
      winnerSelect.appendChild(option);
    });
    assignStyles(winnerSelect, {
      border: "1px solid #D1D5DB",
      borderRadius: "6px",
      background: "#FFFFFF",
      color: "#111827",
      padding: "7px 8px",
      fontSize: "13px",
      width: "100%",
    });
    winnerSelect.addEventListener("change", () => {
      state.performancePreferenceDraft.winner = winnerSelect.value;
    });
    winnerRow.append(winnerLabel, winnerSelect);
    form.appendChild(winnerRow);

    addSubheading(form, "Ratings");
    performancePreferenceDimensions.forEach((dimension) => {
      const row = document.createElement("label");
      assignStyles(row, {
        display: "grid",
        gridTemplateColumns: "190px minmax(0, 1fr)",
        gap: "10px",
        alignItems: "center",
      });
      const label = document.createElement("span");
      label.textContent = dimension;
      label.style.color = "#374151";
      const select = document.createElement("select");
      select.name = dimension;
      select.disabled = state.performancePreferenceSubmitPending;
      [1, 2, 3, 4, 5].forEach((score) => {
        const option = document.createElement("option");
        option.value = String(score);
        option.textContent = String(score);
        if (Number(state.performancePreferenceDraft.ratings[dimension]) === score) {
          option.selected = true;
        }
        select.appendChild(option);
      });
      assignStyles(select, {
        border: "1px solid #D1D5DB",
        borderRadius: "6px",
        background: "#FFFFFF",
        color: "#111827",
        padding: "7px 8px",
        fontSize: "13px",
        width: "100%",
      });
      select.addEventListener("change", () => {
        state.performancePreferenceDraft.ratings[dimension] = Number(select.value);
      });
      row.append(label, select);
      form.appendChild(row);
    });

    function appendLabelCheckboxes(heading, labels, field) {
      addSubheading(form, heading);
      const grid = document.createElement("div");
      assignStyles(grid, {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: "6px",
      });
      labels.forEach((labelValue) => {
        const row = document.createElement("label");
        assignStyles(row, {
          display: "inline-flex",
          alignItems: "center",
          gap: "7px",
          color: "#374151",
          fontSize: "12px",
        });
        const box = document.createElement("input");
        box.type = "checkbox";
        box.value = labelValue;
        box.checked = safeArray(state.performancePreferenceDraft[field]).includes(labelValue);
        box.disabled = state.performancePreferenceSubmitPending;
        box.addEventListener("change", () => {
          const current = new Set(safeArray(state.performancePreferenceDraft[field]));
          if (box.checked) {
            current.add(labelValue);
          } else {
            current.delete(labelValue);
          }
          state.performancePreferenceDraft[field] = Array.from(current);
        });
        const textNode = document.createElement("span");
        textNode.textContent = labelValue;
        row.append(box, textNode);
        grid.appendChild(row);
      });
      form.appendChild(grid);
    }
    appendLabelCheckboxes("Failure labels", performancePreferenceFailureLabels, "failure_labels");
    appendLabelCheckboxes(
      "Improvement labels",
      performancePreferenceImprovementLabels,
      "improvement_labels",
    );

    const controls = document.createElement("div");
    assignStyles(controls, {
      display: "flex",
      flexWrap: "wrap",
      gap: "8px",
      marginTop: "4px",
    });
    const submit = document.createElement("button");
    submit.type = "submit";
    submit.textContent = state.performancePreferenceSubmitPending
      ? "Saving..."
      : "Save preference pair";
    submit.disabled = state.performancePreferenceSubmitPending;
    assignStyles(submit, {
      border: "1px solid #D1D5DB",
      borderRadius: "6px",
      background: "#FFFFFF",
      color: "#111827",
      cursor: submit.disabled ? "wait" : "pointer",
      padding: "7px 10px",
      fontSize: "12px",
      fontWeight: "700",
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      runPerformancePreferenceSubmit();
    });
    controls.appendChild(submit);
    form.appendChild(controls);
    const resultText = safePerformancePreferenceResultText(
      state.latestPerformancePreferenceResult,
    );
    if (resultText) {
      const result = document.createElement("div");
      result.textContent = resultText;
      assignStyles(result, {
        color: safeObject(state.latestPerformancePreferenceResult).accepted
          ? "#047857"
          : "#B91C1C",
        marginTop: "2px",
      });
      form.appendChild(result);
    }
    parent.appendChild(form);
  }

  function renderPerformanceLearning(body, section) {
    const payload = safeObject(section.payload);
    const recentPairs = safeArray(payload.recent_pairs);
    const proposals = safeArray(payload.policy_proposals);
    setTextLine(body, "summary", section.summary || payload.summary);
    renderMetricTiles(body, [
      { value: String(payload.pair_count || recentPairs.length || 0), label: "preference pairs" },
      { value: String(payload.proposal_count || proposals.length || 0), label: "policy proposals" },
      { value: String(safeArray(payload.dimensions).length || performancePreferenceDimensions.length), label: "rating dimensions" },
    ]);
    renderCountMap(body, "profiles", payload.profile_counts);
    setTextLine(body, "dimensions", (safeArray(payload.dimensions).length ? payload.dimensions : performancePreferenceDimensions).join(", "));
    renderPerformanceLearningForm(body);
    const applyResult = safePerformancePolicyApplyResultText(
      state.latestPerformancePolicyApplyResult,
    );
    if (applyResult) {
      setTextLine(body, "last apply", applyResult);
    }
    addSubheading(body, "Recent pairs");
    addCompactList(body, recentPairs, renderPerformancePreferencePair, 4);
    addSubheading(body, "Policy proposals");
    addCompactList(body, proposals, renderPerformancePolicyProposal, 4);
  }

  function renderTeachingKnowledge(body, section) {
    const payload = safeObject(section.payload);
    const current = safeObject(payload.current_decision);
    setTextLine(body, "summary", section.summary || payload.summary);
    setTextLine(body, "route", current.selection_kind || "unavailable");
    setTextLine(body, "mode", current.teaching_mode || "unavailable");
    setTextLine(body, "language", current.language || "unavailable");
    setTextLine(body, "tokens", String(current.estimated_tokens || 0));
    renderCountMap(body, "selected", payload.selected_item_counts);
    addCompactList(
      body,
      current.selected_items || [],
      (parent, item) => {
        renderEvidenceItem(
          parent,
          [item.item_kind, item.item_id].filter(Boolean).join(" | ") || "knowledge item",
          [
            item.title,
            item.source_label ? `source=${item.source_label}` : null,
            item.provenance_kind ? `kind=${item.provenance_kind}` : null,
            item.provenance_version ? `version=${item.provenance_version}` : null,
          ]
            .filter(Boolean)
            .join("; "),
        );
      },
      4,
    );
  }

  function renderBehaviorControlForm(parent, section) {
    const form = document.createElement("form");
    form.setAttribute("data-behavior-controls-form", "");
    assignStyles(form, {
      display: "grid",
      gap: "10px",
      marginTop: "12px",
      padding: "12px",
      border: "1px solid #E5E7EB",
      borderRadius: "8px",
      background: "#F9FAFB",
      maxWidth: "760px",
    });
    const formTitle = document.createElement("div");
    formTitle.textContent = "Edit response behavior";
    assignStyles(formTitle, {
      color: "#111827",
      fontWeight: "700",
      fontSize: "14px",
      marginBottom: "2px",
    });
    form.appendChild(formTitle);
    const presets = safeArray(safeObject(state.stylePresets).presets);
    if (presets.length > 0) {
      const presetRow = document.createElement("label");
      presetRow.setAttribute("data-style-preset-control", "");
      assignStyles(presetRow, {
        display: "grid",
        gridTemplateColumns: "132px minmax(0, 1fr)",
        gap: "10px",
        alignItems: "center",
      });
      const presetLabel = document.createElement("span");
      presetLabel.textContent = "Style preset";
      presetLabel.style.color = "#374151";
      const presetSelect = document.createElement("select");
      presetSelect.disabled = !section.available || state.controlUpdatePending;
      assignStyles(presetSelect, {
        border: "1px solid #D1D5DB",
        borderRadius: "6px",
        background: "#FFFFFF",
        color: "#111827",
        padding: "7px 8px",
        fontSize: "13px",
        width: "100%",
      });
      const emptyOption = document.createElement("option");
      emptyOption.value = "";
      emptyOption.textContent = "custom";
      presetSelect.appendChild(emptyOption);
      presets.forEach((preset) => {
        const item = safeObject(preset);
        const option = document.createElement("option");
        option.value = text(item.preset_id, "");
        option.textContent = `${text(item.label)}${item.recommended ? " (recommended)" : ""}`;
        if (state.stylePresetDraft === option.value) {
          option.selected = true;
        }
        presetSelect.appendChild(option);
      });
      presetSelect.addEventListener("change", () => {
        if (presetSelect.value) {
          applyStylePresetDraft(presetSelect.value);
        } else {
          state.stylePresetDraft = "";
        }
        render();
      });
      presetRow.append(presetLabel, presetSelect);
      form.appendChild(presetRow);
    }

    function appendControlRows(fields) {
      fields.forEach((field) => {
        const row = document.createElement("label");
        assignStyles(row, {
          display: "grid",
          gridTemplateColumns: "132px minmax(0, 1fr)",
          gap: "10px",
          alignItems: "center",
        });
        const label = document.createElement("span");
        label.textContent = controlLabels[field] || field;
        label.style.color = "#374151";
        const select = document.createElement("select");
        select.name = field;
        select.disabled = !section.available || state.controlUpdatePending;
        assignStyles(select, {
          border: "1px solid #D1D5DB",
          borderRadius: "6px",
          background: "#FFFFFF",
          color: "#111827",
          padding: "7px 8px",
          fontSize: "13px",
          width: "100%",
        });
        (controlOptions[field] || []).forEach((optionValue) => {
          const option = document.createElement("option");
          option.value = optionValue;
          option.textContent = optionValue.replaceAll("_", " ");
          if (state.controlDrafts[field] === optionValue) {
            option.selected = true;
          }
          select.appendChild(option);
        });
        select.addEventListener("change", () => {
          state.controlDrafts[field] = select.value;
          state.stylePresetDraft = "";
          state.controlDraftDirty = true;
        });
        row.append(label, select);
        form.appendChild(row);
      });
    }
    addSubheading(form, "Core behavior");
    appendControlRows(coreControlFields);
    addSubheading(form, "Style");
    appendControlRows(styleControlFields);

    const controls = document.createElement("div");
    assignStyles(controls, {
      display: "flex",
      flexWrap: "wrap",
      gap: "8px",
      marginTop: "4px",
    });
    const submit = document.createElement("button");
    submit.type = "submit";
    submit.textContent = state.controlUpdatePending ? "Saving..." : "Save controls";
    submit.disabled = !section.available || state.controlUpdatePending;
    const reset = document.createElement("button");
    reset.type = "button";
    reset.textContent = "Reset draft";
    reset.disabled = !section.available || state.controlUpdatePending;
    [submit, reset].forEach((button) => {
      assignStyles(button, {
        border: "1px solid #D1D5DB",
        borderRadius: "6px",
        background: "#FFFFFF",
        color: "#111827",
        cursor: button.disabled ? "wait" : "pointer",
        padding: "7px 10px",
        fontSize: "12px",
        fontWeight: "700",
      });
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      runBehaviorControlsUpdate();
    });
    reset.addEventListener("click", () => {
      syncControlDrafts(safeObject(safeObject(section.payload).profile), true);
      state.controlUpdateResult = null;
      render();
    });
    controls.append(submit, reset);
    form.appendChild(controls);
    const resultText = safeBehaviorControlResultText(state.controlUpdateResult);
    if (resultText) {
      const result = document.createElement("div");
      result.textContent = resultText;
      assignStyles(result, {
        color: state.controlUpdateResult.accepted ? "#047857" : "#B91C1C",
        marginTop: "2px",
      });
      form.appendChild(result);
    }
    parent.appendChild(form);
    renderMemoryPersonaPresetPreview(parent);
  }

  function renderMemoryPersonaPresetPreview(parent) {
    const preview = safeObject(state.memoryPersonaPresetPreview);
    const counts = safeObject(preview.counts);
    addSubheading(parent, "Character/story seed");
    setTextLine(parent, "preset", preview.preset_label || "Witty Sophisticated");
    setTextLine(parent, "preview", preview.accepted ? "accepted" : "rejected");
    setTextLine(parent, "candidates", String(counts.accepted_candidates || 0));
    setTextLine(parent, "rejected", String(counts.rejected_entries || 0));
    setTextLine(parent, "reason cats", summarizeReasonCodes(preview.reason_codes));
    addCompactList(
      parent,
      preview.candidates,
      (itemParent, candidate) => {
        renderEvidenceItem(
          itemParent,
          [candidate.kind, candidate.namespace].filter(Boolean).join(" | ") || "candidate",
          candidate.summary || formatReasonCodes(candidate.reason_codes),
        );
      },
      4,
    );
    const rejectedEntries = safeArray(preview.rejected_entries);
    if (rejectedEntries.length > 0) {
      addSubheading(parent, "Rejected seed entries");
      addCompactList(
        parent,
        rejectedEntries,
        (itemParent, entry) => {
          renderEvidenceItem(
            itemParent,
            [entry.path, entry.fatal ? "fatal" : ""].filter(Boolean).join(" | ") ||
              "rejected entry",
            formatReasonCodes(entry.reason_codes),
          );
        },
        3,
      );
    }
    const controls = document.createElement("div");
    assignStyles(controls, {
      display: "flex",
      flexWrap: "wrap",
      gap: "8px",
      marginTop: "8px",
    });
    const apply = document.createElement("button");
    apply.type = "button";
    apply.textContent = state.memoryPersonaPresetApplyPending ? "Applying..." : "Apply story seed";
    apply.disabled = !preview.accepted || state.memoryPersonaPresetApplyPending;
    assignStyles(apply, {
      border: "1px solid #D1D5DB",
      borderRadius: "6px",
      background: "#FFFFFF",
      color: "#111827",
      cursor: apply.disabled ? "wait" : "pointer",
      padding: "7px 10px",
      fontSize: "12px",
      fontWeight: "700",
    });
    apply.addEventListener("click", requestMemoryPersonaPresetApply);
    controls.appendChild(apply);
    parent.appendChild(controls);
    const resultText = safeMemoryPersonaPresetResultText(state.memoryPersonaPresetApplyResult);
    if (resultText) {
      const result = document.createElement("div");
      result.textContent = resultText;
      assignStyles(result, {
        color: safeObject(state.memoryPersonaPresetApplyResult).accepted ? "#047857" : "#B91C1C",
        marginTop: "6px",
      });
      parent.appendChild(result);
    }
    const applyResult = safeObject(state.memoryPersonaPresetApplyResult);
    const appliedEntries = safeArray(applyResult.applied_entries);
    if (appliedEntries.length > 0) {
      addSubheading(parent, "Applied seed entries");
      addCompactList(
        parent,
        appliedEntries,
        (itemParent, entry) => {
          renderEvidenceItem(
            itemParent,
            [entry.status, entry.kind, entry.namespace].filter(Boolean).join(" | ") ||
              "applied entry",
            formatReasonCodes(entry.reason_codes),
          );
        },
        4,
      );
    }
  }

  function renderVoice(body, snapshot) {
    const expression = safeObject(safeObject(snapshot.expression).payload);
    const policy = safeObject(expression.voice_policy);
    const actuation = safeObject(expression.voice_actuation_plan);
    const metricsSection = safeObject(snapshot.voice_metrics);
    const metrics = safeObject(metricsSection.payload);
    const inputHealth = safeObject(metrics.input_health);
    setTextLine(body, "expression", safeObject(snapshot.expression).summary);
    setTextLine(body, "style", expression.voice_style_summary || "unavailable");
    setTextLine(body, "backend", actuation.backend_label || "provider-neutral");
    setTextLine(body, "chunking", policy.chunking_mode || "unavailable");
    setTextLine(body, "active", safeArray(policy.active_hints).slice(0, 3).join(", ") || "none");
    setTextLine(
      body,
      "requested",
      safeArray(actuation.requested_hints).slice(0, 4).join(", ") || "none",
    );
    setTextLine(
      body,
      "applied",
      safeArray(actuation.applied_hints).slice(0, 4).join(", ") || "none",
    );
    setTextLine(
      body,
      "unsupported",
      safeArray(actuation.unsupported_hints).slice(0, 4).join(", ") ||
        safeArray(policy.unsupported_hints).slice(0, 3).join(", ") ||
        "none",
    );
    setTextLine(
      body,
      "no-op cats",
      summarizeReasonCodes(actuation.noop_reason_codes || policy.noop_reason_codes),
    );
    addSubheading(body, "Input / STT");
    setTextLine(body, "mic", inputHealth.microphone_state || "unavailable");
    setTextLine(body, "stt", inputHealth.stt_state || "unavailable");
    setTextLine(body, "stt wait", inputHealth.stt_waiting_too_long ? "too long" : "normal");
    setTextLine(body, "track", inputHealth.track_reason || "none");
    setTextLine(body, "input reasons", summarizeReasonCodes(inputHealth.reason_codes));
    renderMetricTiles(body, [
      { label: "audio frames", value: numberText(inputHealth.audio_frame_count) },
      { label: "speech starts", value: numberText(inputHealth.speech_start_count) },
      { label: "speech stops", value: numberText(inputHealth.speech_stop_count) },
      { label: "transcripts", value: numberText(inputHealth.transcription_count) },
      { label: "STT errors", value: numberText(inputHealth.stt_error_count) },
      {
        label: "audio age",
        value:
          inputHealth.last_audio_frame_age_ms === null ||
          inputHealth.last_audio_frame_age_ms === undefined
            ? "none"
            : `${numberText(inputHealth.last_audio_frame_age_ms)} ms`,
      },
      {
        label: "STT wait",
        value:
          inputHealth.stt_wait_age_ms === null || inputHealth.stt_wait_age_ms === undefined
            ? "none"
            : `${numberText(inputHealth.stt_wait_age_ms)} ms`,
      },
    ]);
    setTextLine(body, "metrics", metricsSection.summary || "Voice metrics unavailable.");
    addSubheading(body, "TTS metrics");
    renderMetricTiles(body, [
      { label: "responses", value: numberText(metrics.response_count) },
      { label: "spoken chunks", value: numberText(metrics.chunk_count) },
      { label: "avg chars", value: decimalText(metrics.average_chunk_chars) },
      { label: "max chars", value: numberText(metrics.max_chunk_chars) },
      { label: "interrupts", value: numberText(metrics.interruption_frame_count) },
      {
        label: "flush / discard",
        value: `${numberText(metrics.buffer_flush_count)} / ${numberText(
          metrics.buffer_discard_count,
        )}`,
      },
    ]);
    setTextLine(body, "responses", numberText(metrics.response_count));
    setTextLine(
      body,
      "activations",
      numberText(metrics.concise_chunking_activation_count),
    );
    setTextLine(body, "chunks", numberText(metrics.chunk_count));
    setTextLine(body, "max chars", numberText(metrics.max_chunk_chars));
    setTextLine(body, "avg chars", decimalText(metrics.average_chunk_chars));
    setTextLine(body, "interrupts", numberText(metrics.interruption_frame_count));
    setTextLine(body, "flushes", numberText(metrics.buffer_flush_count));
    setTextLine(body, "discards", numberText(metrics.buffer_discard_count));
    setTextLine(body, "last mode", metrics.last_chunking_mode || "unavailable");
    setTextLine(body, "last max", numberText(metrics.last_max_spoken_chunk_chars));
    setTextLine(body, "reason cats", summarizeReasonCodes(metrics.reason_codes));
  }

  function renderPractice(body, section) {
    const payload = safeObject(section.payload);
    setTextLine(body, "summary", section.summary);
    renderCountMap(body, "families", payload.scenario_family_counts);
    renderCountMap(body, "reasons", payload.reason_code_counts);
    addCompactList(
      body,
      payload.recent_targets,
      (parent, target) => {
        setTextLine(
          parent,
          "target",
          [target.scenario_family, target.execution_backend, target.selected_profile_id]
            .filter(Boolean)
            .join(" | ") || "practice target",
        );
      },
      3,
    );
  }

  function renderAdapters(body, section) {
    const payload = safeObject(section.payload);
    setTextLine(body, "summary", section.summary);
    renderCountMap(body, "states", payload.state_counts);
    renderCountMap(body, "families", payload.family_counts);
    addCompactList(
      body,
      payload.current_default_cards,
      (parent, card) => {
        setTextLine(
          parent,
          "default",
          [card.adapter_family, card.backend_id, card.promotion_state].filter(Boolean).join(" | "),
        );
      },
      3,
    );
    setTextLine(body, "pending", String(safeArray(payload.pending_or_blocked_decisions).length));
  }

  function renderSimToReal(body, section) {
    const payload = safeObject(section.payload);
    setTextLine(body, "summary", section.summary);
    renderCountMap(body, "readiness", payload.readiness_counts);
    renderCountMap(body, "states", payload.promotion_state_counts);
    addCompactList(
      body,
      payload.readiness_reports,
      (parent, report) => {
        setTextLine(
          parent,
          "report",
          [
            report.adapter_family,
            report.backend_id,
            report.promotion_state,
            report.governance_only ? "governance only" : "",
          ]
            .filter(Boolean)
            .join(" | "),
        );
      },
      3,
    );
  }

  function renderEpisodeEvidence(body, section) {
    const endpointPayload = safeObject(state.episodeEvidence);
    const payload = endpointPayload.available ? endpointPayload : safeObject(section.payload);
    setTextLine(body, "summary", payload.summary || section.summary);
    setTextLine(body, "rows", String(safeArray(payload.rows).length));
    renderCountMap(body, "sources", payload.source_counts);
    renderCountMap(body, "reasons", payload.reason_code_counts);
    addCompactList(
      body,
      payload.rows,
      (parent, row) => {
        const artifacts = safeArray(row.artifact_refs)
          .map((artifact) =>
            [artifact.artifact_kind, artifact.uri_kind, artifact.redacted_uri ? "redacted" : ""]
              .filter(Boolean)
              .join(":"),
          )
          .slice(0, 3)
          .join(", ");
        const links = safeArray(row.links)
          .map((link) => [link.link_kind, link.link_id].filter(Boolean).join(":"))
          .slice(0, 3)
          .join(", ");
        renderEvidenceItem(
          parent,
          [row.source, row.scenario_family, row.outcome_label].filter(Boolean).join(" | "),
          [
            row.summary,
            row.execution_backend ? `backend ${row.execution_backend}` : "",
            row.candidate_backend_id ? `candidate ${row.candidate_backend_id}` : "",
            row.scenario_count ? `${row.scenario_count} scenarios` : "",
            artifacts ? `artifacts ${artifacts}` : "",
            links ? `links ${links}` : "",
            formatReasonCodes(row.reason_code_categories || row.reason_codes),
          ]
            .filter(Boolean)
            .join("; "),
        );
      },
      6,
    );
  }

  function renderRollouts(body, section) {
    const payload = safeObject(section.payload);
    setTextLine(body, "summary", section.summary || payload.summary);
    setTextLine(body, "available", statusPill(section.available));
    setTextLine(body, "governance", payload.governance_only ? "governance only" : "unavailable");
    setTextLine(body, "live", payload.live_routing_active ? "active" : "inactive");
    setTextLine(body, "plans", String(safeArray(payload.plan_summaries).length));
    if (safeRolloutActionResultText(state.latestRolloutActionResult)) {
      setTextLine(body, "last action", safeRolloutActionResultText(state.latestRolloutActionResult));
    }
    addCompactList(body, payload.plan_summaries, renderRolloutPlan, 4);
    if (safeArray(payload.recent_decisions).length > 0) {
      addSubheading(body, "Recent decisions");
      addCompactList(
        body,
        payload.recent_decisions,
        (parent, decision) => {
          renderEvidenceItem(
            parent,
            [decision.action, decision.to_state, decision.accepted ? "accepted" : "rejected"]
              .filter(Boolean)
              .join(" | "),
            `traffic ${Number(decision.traffic_fraction || 0).toFixed(4)}; ${formatReasonCodes(
              decision.reason_codes,
            )}`,
          );
        },
        3,
      );
    }
    addSubheading(body, "Evidence");
    const evidence = safeObject(state.rolloutEvidence);
    setTextLine(body, "evidence", statusPill(evidence.available));
    addCompactList(
      body,
      evidence.live_episodes,
      (parent, episode) => {
        renderEvidenceItem(
          parent,
          ["live", episode.created_at].filter(Boolean).join(" | "),
          episode.assistant_summary || "live episode",
        );
      },
      3,
    );
    addCompactList(
      body,
      evidence.practice_plans,
      (parent, plan) => {
        renderEvidenceItem(
          parent,
          ["practice", plan.plan_id].filter(Boolean).join(" | "),
          `${plan.target_count || 0} targets; ${plan.summary || "practice plan"}`,
        );
      },
      3,
    );
    addCompactList(
      body,
      evidence.benchmark_reports,
      (parent, report) => {
        renderEvidenceItem(
          parent,
          ["benchmark", report.adapter_family, report.candidate_backend_id]
            .filter(Boolean)
            .join(" | "),
          [
            `${report.scenario_count || 0} scenarios`,
            report.benchmark_passed === true ? "passed" : "not passed",
            formatReasonCodes(report.blocked_reason_codes),
          ]
            .filter(Boolean)
            .join("; "),
        );
      },
      3,
    );
  }

  function renderSectionBody(key, body, snapshot) {
    const section = operatorSection(snapshot, key);
    if (key !== "overview" && key !== "voice") {
      setTextLine(body, "status", statusPill(section.available));
    }
    if (key === "overview") {
      renderOverview(body, snapshot);
    } else if (key === "memory") {
      renderMemory(body, section);
    } else if (key === "controls") {
      renderControls(body, section);
    } else if (key === "teaching") {
      renderTeachingKnowledge(body, section);
    } else if (key === "voice") {
      renderVoice(body, snapshot);
    } else if (key === "practice") {
      renderPractice(body, section);
    } else if (key === "adapters") {
      renderAdapters(body, section);
    } else if (key === "sim-to-real") {
      renderSimToReal(body, section);
    } else if (key === "evidence") {
      renderEpisodeEvidence(body, section);
    } else if (key === "performance-learning") {
      renderPerformanceLearning(body, section);
    } else if (key === "rollouts") {
      renderRollouts(body, section);
    }
  }

  function ensurePanel() {
    let panel = document.getElementById(panelId);
    if (panel) {
      return panel;
    }
    panel = document.createElement("aside");
    panel.id = panelId;
    panel.setAttribute("aria-label", "Blink operator workbench");
    assignStyles(panel, {
      position: "relative",
      zIndex: "1",
      color: "#111827",
      fontFamily:
        "Geist, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
      fontSize: "14px",
      lineHeight: "1.45",
      pointerEvents: "auto",
    });
    const root = document.getElementById("root");
    if (root && root.parentElement) {
      root.insertAdjacentElement("afterend", panel);
    } else {
      document.body.appendChild(panel);
    }
    return panel;
  }

  function renderAdvanced(parent, snapshot) {
    const details = document.createElement("details");
    details.open = state.advancedOpen;
    details.addEventListener("toggle", () => {
      state.advancedOpen = details.open;
    });
    assignStyles(details, {
      marginTop: "12px",
      padding: "10px 12px",
      border: "1px solid #E5E7EB",
      borderRadius: "8px",
      background: "#F9FAFB",
    });
    const summary = document.createElement("summary");
    summary.textContent = "Advanced";
    assignStyles(summary, {
      cursor: "pointer",
      color: "#374151",
      fontWeight: "700",
    });
    const body = document.createElement("div");
    const expression = safeObject(safeObject(snapshot.expression).payload);
    const voicePolicy = safeObject(expression.voice_policy);
    setTextLine(body, "operator", summarizeReasonCodes(snapshot.reason_codes));
    setTextLine(body, "expression", summarizeReasonCodes(safeObject(snapshot.expression).reason_codes));
    setTextLine(body, "memory", summarizeReasonCodes(safeObject(snapshot.memory).reason_codes));
    setTextLine(
      body,
      "controls",
      summarizeReasonCodes(safeObject(snapshot.behavior_controls).reason_codes),
    );
    setTextLine(
      body,
      "teaching",
      summarizeReasonCodes(safeObject(snapshot.teaching_knowledge).reason_codes),
    );
    setTextLine(
      body,
      "control update",
      summarizeReasonCodes(safeObject(state.controlUpdateResult).reason_codes),
    );
    setTextLine(
      body,
      "memory action",
      summarizeReasonCodes(safeObject(state.latestMemoryActionResult).reason_codes),
    );
    setTextLine(
      body,
      "rollout action",
      summarizeReasonCodes(safeObject(state.latestRolloutActionResult).reason_codes),
    );
    setTextLine(
      body,
      "rollout evidence",
      summarizeReasonCodes(safeObject(state.rolloutEvidence).reason_codes),
    );
    setTextLine(
      body,
      "episode evidence",
      summarizeReasonCodes(safeObject(state.episodeEvidence).reason_codes),
    );
    setTextLine(body, "voice", summarizeReasonCodes(safeObject(snapshot.voice_metrics).reason_codes));
    setTextLine(body, "voice no-ops", summarizeReasonCodes(voicePolicy.noop_reason_codes));
    setTextLine(body, "practice", summarizeReasonCodes(safeObject(snapshot.practice).reason_codes));
    setTextLine(body, "adapters", summarizeReasonCodes(safeObject(snapshot.adapters).reason_codes));
    setTextLine(body, "rollouts", summarizeReasonCodes(safeObject(snapshot.rollout_status).reason_codes));
    setTextLine(
      body,
      "evidence",
      summarizeReasonCodes(safeObject(snapshot.episode_evidence).reason_codes),
    );
    setTextLine(
      body,
      "performance learning",
      summarizeReasonCodes(safeObject(snapshot.performance_learning).reason_codes),
    );
    details.append(summary, body);
    parent.appendChild(details);
  }

  function render() {
    const snapshot = safeObject(state.snapshot);
    const panel = ensurePanel();
    panel.replaceChildren();
    assignStyles(
      panel,
      state.collapsed
        ? {
            position: "relative",
            width: "calc(100% - 32px)",
            maxWidth: "none",
            maxHeight: "none",
            overflow: "visible",
            margin: "16px",
            padding: "0",
            borderRadius: "8px",
            border: "1px solid #E5E7EB",
            background: "#FFFFFF",
            boxShadow: "0 4px 14px rgba(15, 23, 42, 0.06)",
            backdropFilter: "none",
          }
        : {
            position: "relative",
            width: "calc(100% - 32px)",
            maxWidth: "none",
            maxHeight: "none",
            overflow: "visible",
            margin: "16px",
            padding: "16px",
            borderRadius: "8px",
            border: "1px solid #E5E7EB",
            background: "#FFFFFF",
            boxShadow: "0 6px 20px rgba(15, 23, 42, 0.06)",
            backdropFilter: "none",
          },
    );

    const header = document.createElement("button");
    header.type = "button";
    header.setAttribute("aria-expanded", state.collapsed ? "false" : "true");
    assignStyles(header, {
      appearance: "none",
      border: "0",
      background: "transparent",
      color: "#111827",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      width: "100%",
      padding: state.collapsed ? "10px 12px" : "0",
      margin: state.collapsed ? "0" : "0 0 12px",
      cursor: "pointer",
      textAlign: "left",
      letterSpacing: "0",
    });
    const titleWrap = document.createElement("span");
    assignStyles(titleWrap, {
      display: "grid",
      gap: "2px",
    });
    const title = document.createElement("span");
    title.textContent = "Blink operator workbench";
    assignStyles(title, {
      fontSize: state.collapsed ? "14px" : "18px",
      fontWeight: "760",
      lineHeight: "1.2",
    });
    const subtitle = document.createElement("span");
    subtitle.textContent = "Runtime state, memory, controls, voice, and governance";
    assignStyles(subtitle, {
      color: "#6B7280",
      fontSize: "12px",
      fontWeight: "500",
      display: "inline",
    });
    titleWrap.append(title, subtitle);
    const headerActions = document.createElement("span");
    assignStyles(headerActions, {
      display: "inline-flex",
      alignItems: "center",
      gap: "8px",
      flexShrink: "0",
    });
    headerActions.appendChild(makeChip(statusPill(snapshot.available), snapshot.available));
    const hint = document.createElement("span");
    hint.textContent = state.collapsed ? "show" : "hide";
    assignStyles(hint, {
      color: "#4B5563",
      fontSize: "13px",
      fontWeight: "700",
      textTransform: "uppercase",
      letterSpacing: "0",
    });
    headerActions.appendChild(hint);
    header.append(titleWrap, headerActions);
    header.addEventListener("click", () => {
      state.collapsed = !state.collapsed;
      render();
      if (!state.collapsed) {
        refresh({ force: true, includeEvidence: true, includeStatic: true });
      }
    });
    panel.appendChild(header);

    if (state.collapsed) {
      return;
    }

    const sectionGrid = document.createElement("div");
    assignStyles(sectionGrid, {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
      gap: "12px",
    });

    sectionKeys.forEach((key) => {
      const wrapper = document.createElement("section");
      assignStyles(wrapper, {
        minWidth: "0",
        padding: "12px",
        border: "1px solid #E5E7EB",
        borderRadius: "10px",
        background: "#FFFFFF",
        boxShadow: "0 1px 2px rgba(15, 23, 42, 0.03)",
      });
      if (key === "controls" || key === "performance-learning") {
        wrapper.style.gridColumn = "span 2";
      }
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-expanded", state.sectionCollapsed[key] ? "false" : "true");
      assignStyles(button, {
        appearance: "none",
        border: "0",
        background: "transparent",
        color: "#111827",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        gap: "12px",
        width: "100%",
        padding: "0",
        cursor: "pointer",
        textAlign: "left",
      });
      const label = document.createElement("span");
      label.textContent = sectionLabel(key);
      assignStyles(label, {
        fontSize: "14px",
        fontWeight: "760",
        lineHeight: "1.25",
      });
      const summary = document.createElement("span");
      const section = operatorSection(snapshot, key);
      summary.textContent =
        key === "overview"
          ? statusPill(snapshot.available)
          : text(section.summary, statusPill(section.available));
      assignStyles(summary, {
        color: "#6B7280",
        fontSize: "12px",
        fontWeight: "600",
        maxWidth: "56%",
        textAlign: "right",
        overflowWrap: "anywhere",
      });
      button.append(label, summary);
      button.addEventListener("click", () => {
        state.sectionCollapsed[key] = !state.sectionCollapsed[key];
        render();
      });
      wrapper.appendChild(button);
      if (!state.sectionCollapsed[key]) {
        const body = document.createElement("div");
        assignStyles(body, {
          marginTop: "10px",
          paddingTop: "10px",
          borderTop: "1px solid #F3F4F6",
        });
        renderSectionBody(key, body, snapshot);
        wrapper.appendChild(body);
      }
      sectionGrid.appendChild(wrapper);
    });

    panel.appendChild(sectionGrid);
    renderAdvanced(panel, snapshot);
  }

  async function fetchJson(endpoint, fallback) {
    try {
      const response = await fetch(endpoint, { cache: "no-store" });
      if (!response.ok) {
        throw new Error("request failed");
      }
      return await response.json();
    } catch (_error) {
      return fallback;
    }
  }

  async function refresh(options = {}) {
    if (state.collapsed && options.force !== true) {
      render();
      return;
    }
    if (refreshState.inFlight) {
      return;
    }
    refreshState.inFlight = true;
    const now = Date.now();
    const force = options.force === true;
    const shouldRefreshEvidence =
      force ||
      options.includeEvidence === true ||
      now - refreshState.lastEvidenceRefreshAt >= evidenceRefreshMs;
    const shouldRefreshStatic =
      force ||
      options.includeStatic === true ||
      now - refreshState.lastStaticRefreshAt >= staticRefreshMs;

    try {
      const snapshotPromise = fetchJson(operatorEndpoint, fallbackSnapshot("operator_fetch_failed"));
      const evidencePromise = shouldRefreshEvidence
        ? fetchJson(
            rolloutEvidenceEndpoint,
            fallbackRolloutEvidence("rollout_evidence_fetch_failed"),
          )
        : Promise.resolve(null);
      const episodeEvidencePromise = shouldRefreshEvidence
        ? fetchJson(
            episodeEvidenceEndpoint,
            fallbackEpisodeEvidence("episode_evidence_fetch_failed"),
          )
        : Promise.resolve(null);
      const stylePresetsPromise = shouldRefreshStatic
        ? fetchJson(stylePresetsEndpoint, fallbackStylePresets("style_presets_fetch_failed"))
        : Promise.resolve(null);
      const memoryPersonaPresetPreviewPromise = shouldRefreshStatic
        ? fetchJson(
            memoryPersonaPresetPreviewEndpoint,
            fallbackMemoryPersonaPreset("memory_persona_preset_fetch_failed"),
          )
        : Promise.resolve(null);

      const [snapshot, evidence, episodeEvidence, stylePresets, memoryPersonaPresetPreview] =
        await Promise.all([
          snapshotPromise,
          evidencePromise,
          episodeEvidencePromise,
          stylePresetsPromise,
          memoryPersonaPresetPreviewPromise,
        ]);

      state.snapshot = snapshot;
      if (evidence) {
        state.rolloutEvidence = evidence;
      }
      if (episodeEvidence) {
        state.episodeEvidence = episodeEvidence;
      }
      if (stylePresets) {
        state.stylePresets = stylePresets;
      }
      if (memoryPersonaPresetPreview) {
        state.memoryPersonaPresetPreview = memoryPersonaPresetPreview;
      }
      if (shouldRefreshEvidence) {
        refreshState.lastEvidenceRefreshAt = now;
      }
      if (shouldRefreshStatic) {
        refreshState.lastStaticRefreshAt = now;
      }
      render();
    } finally {
      refreshState.inFlight = false;
    }
  }

  function scheduleRefresh() {
    if (refreshState.timerId) {
      window.clearTimeout(refreshState.timerId);
    }
    const delay = document.hidden ? hiddenRefreshMs : activeRefreshMs;
    refreshState.timerId = window.setTimeout(async () => {
      await refresh();
      scheduleRefresh();
    }, delay);
  }

  if (document.readyState === "loading") {
    document.addEventListener(
      "DOMContentLoaded",
      () => {
        render();
        scheduleRefresh();
      },
      { once: true },
    );
  } else {
    render();
    scheduleRefresh();
  }
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !state.collapsed) {
      refresh({ includeEvidence: true });
    }
    scheduleRefresh();
  });
})();
