(function () {
  const expressionEndpoint = "/api/runtime/expression";
  const memoryEndpoint = "/api/runtime/memory";
  const performanceStateEndpoint = "/api/runtime/performance-state";
  const performanceEventsEndpoint = "/api/runtime/performance-events";
  const actorStateEndpoint = "/api/runtime/actor-state";
  const actorEventsEndpoint = "/api/runtime/actor-events";
  const clientMediaEndpoint = "/api/runtime/client-media";
  const visibleTextLimit = 280;
  const actorEventLimit = 50;
  const panelId = "blink-expression-panel";
  const fullRefreshMinMs = 10000;
  const refreshState = {
    fullInFlight: false,
    performanceInFlight: false,
    lastFullRefreshAt: 0,
  };
  const state = {
    collapsed: true,
    expression: null,
    memory: null,
    actor: null,
    actorEvents: [],
    latestActorEventId: 0,
    performance: null,
    performanceEvents: [],
    latestPerformanceEventId: 0,
    liveStatus: {
      assistantSubtitle: "",
      cameraState: "unknown",
      connectionState: "waiting",
      lastDeviceError: "",
      lastFinalTranscript: "",
      lastPartialTranscript: "",
      partialTranscriptAvailable: false,
      activeListeningPhase: "idle",
      activeListeningStartedAt: 0,
      activeListeningDurationMs: 0,
      mediaReportInFlight: false,
      mediaReportKey: "",
      mediaReportQueued: null,
      microphoneState: "unknown",
      echoCancellationState: "unknown",
      noiseSuppressionState: "unknown",
      autoGainControlState: "unknown",
      modeHint: "",
      ttsState: "idle",
      transportState: "waiting",
    },
    correctionMemoryId: null,
    correctionDrafts: {},
    actionResults: {},
    latestActionResult: null,
    pendingActions: {},
  };
  const supportedMemoryActions = {
    pin: { endpoint: "pin", label: "Pin" },
    suppress: { endpoint: "suppress", label: "Suppress", confirm: true },
    correct: { endpoint: "correct", label: "Correct" },
    forget: { endpoint: "forget", label: "Forget", confirm: true },
    "mark-stale": { endpoint: "mark-stale", label: "Mark stale" },
  };
  const modeLabels = {
    waiting: "waiting",
    connected: "connected",
    listening: "listening",
    heard: "heard",
    thinking: "thinking",
    looking: "looking",
    speaking: "speaking",
    interrupted: "interrupted",
    error: "degraded",
  };
  const actorSurfaceLabels = {
    en: {
      heard: "Heard",
      saying: "Blink is saying",
      looking: "Looking",
      memoryPersona: "Used memory/persona",
      interruption: "Interruption",
      debugTimeline: "Debug timeline",
      actorTitle: "Blink actor",
    },
    zh: {
      heard: "听到",
      saying: "Blink 正在说",
      looking: "正在看",
      memoryPersona: "使用的记忆/风格",
      interruption: "打断",
      debugTimeline: "调试时间线",
      actorTitle: "Blink actor",
    },
  };
  const unsafeActorEventKeyFragments = [
    "prompt",
    "message",
    "transcript",
    "audio",
    "image",
    "sdp",
    "ice",
    "token",
    "secret",
    "candidate",
    "credential",
    "raw",
  ];

  function assignStyles(node, styles) {
    Object.assign(node.style, styles);
    return node;
  }

  function safeObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function safeArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function text(value) {
    return String(value == null ? "" : value).replace(/\s+/g, " ").trim();
  }

  function visibleText(value, limit) {
    const source = safeObject(value);
    const data = safeObject(source.data);
    const raw =
      typeof value === "string"
        ? value
        : source.text ||
          source.transcript ||
          source.message ||
          source.content ||
          data.text ||
          data.transcript ||
          "";
    return text(raw).slice(0, limit || visibleTextLimit);
  }

  function transcriptText(payload) {
    return visibleText(payload, visibleTextLimit);
  }

  function transcriptIsFinal(payload) {
    const source = safeObject(payload);
    const data = safeObject(source.data);
    if (typeof source.final === "boolean") {
      return source.final;
    }
    if (typeof data.final === "boolean") {
      return data.final;
    }
    return true;
  }

  function activeListeningDurationMs() {
    const startedAt = Number(state.liveStatus.activeListeningStartedAt || 0);
    if (!startedAt) {
      return Number(state.liveStatus.activeListeningDurationMs || 0);
    }
    return Math.max(0, Date.now() - startedAt);
  }

  function appendVisibleText(current, chunk, limit) {
    const nextChunk = visibleText(chunk, limit || visibleTextLimit);
    if (!nextChunk) {
      return current || "";
    }
    const combined = `${current || ""}${nextChunk}`.replace(/\s+/g, " ").trim();
    const maxLength = limit || visibleTextLimit;
    return combined.length <= maxLength ? combined : combined.slice(-maxLength);
  }

  function appendTranscriptText(current, payload, limit) {
    const nextChunk = transcriptText(payload);
    if (!nextChunk) {
      return current || "";
    }
    const currentText = text(current);
    const maxLength = limit || visibleTextLimit;
    if (!currentText || nextChunk.startsWith(currentText)) {
      return nextChunk.length <= maxLength ? nextChunk : nextChunk.slice(-maxLength);
    }
    if (currentText.endsWith(nextChunk)) {
      return currentText.length <= maxLength ? currentText : currentText.slice(-maxLength);
    }
    const combined = `${currentText} ${nextChunk}`.replace(/\s+/g, " ").trim();
    return combined.length <= maxLength ? combined : combined.slice(-maxLength);
  }

  function modeText(mode) {
    return modeLabels[mode] || modeLabels.waiting;
  }

  function actorSurfaceV2Enabled() {
    const config = globalThis.BlinkRuntimeConfig || {};
    return config.actor_surface_v2_enabled !== false;
  }

  function actorLocale(actor) {
    const language = text(safeObject(actor).language).toLowerCase();
    return language.startsWith("zh") ? "zh" : "en";
  }

  function actorLabel(actor, key) {
    const labels = actorSurfaceLabels[actorLocale(actor)] || actorSurfaceLabels.en;
    return labels[key] || actorSurfaceLabels.en[key] || key;
  }

  function profileBadge(actor, performance) {
    const actorObject = safeObject(actor);
    const tts = safeObject(actorObject.tts);
    const profile = text(actorObject.profile || performance.profile || "manual");
    const language = text(actorObject.language || "unknown").toUpperCase();
    const ttsLabel = text(tts.label || performance.tts || performance.tts_backend || "unknown");
    const vision = safeObject(actorObject.vision);
    const visionLabel = text(vision.backend || "none");
    if (profile === "browser-zh-melo") {
      return "ZH · Melo · Moondream";
    }
    if (profile === "browser-en-kokoro") {
      return "EN · Kokoro · Moondream";
    }
    return `${language || "runtime"} · ${ttsLabel} · ${visionLabel}`;
  }

  function actorMode(actor, performance) {
    const actorObject = safeObject(actor);
    return text(actorObject.mode) || statusMode(performance);
  }

  function formatAgeMs(value) {
    const ageMs = Number(value);
    if (!Number.isFinite(ageMs) || ageMs < 0) {
      return "age pending";
    }
    return ageMs >= 1000 ? `${Math.round(ageMs / 1000)}s` : `${Math.round(ageMs)}ms`;
  }

  function hintValues(items) {
    return safeArray(items)
      .map((item) => {
        if (typeof item === "string") {
          return text(item);
        }
        const object = safeObject(item);
        return text(object.value || object.label || object.kind || object.display_kind);
      })
      .filter(Boolean)
      .slice(0, 3);
  }

  function hintListText(items, fallback) {
    const values = hintValues(items);
    return values.length ? values.join("; ") : fallback;
  }

  function countText(value, label) {
    const count = Number(value || 0);
    return `${Number.isFinite(count) ? count : 0} ${label}`;
  }

  function actorSubtitleText(actor) {
    const actorObject = safeObject(actor);
    const liveText = safeObject(actorObject.live_text);
    const speech = safeObject(actorObject.speech);
    return (
      text(liveText.assistant_subtitle) ||
      text(speech.assistant_subtitle) ||
      state.liveStatus.assistantSubtitle ||
      "waiting"
    );
  }

  function actorHeardText(actor) {
    const actorObject = safeObject(actor);
    const liveText = safeObject(actorObject.live_text);
    const activeListening = safeObject(actorObject.active_listening);
    const finalText = text(liveText.final_transcript) || state.liveStatus.lastFinalTranscript;
    if (finalText) {
      return finalText;
    }
    const partialText = text(liveText.partial_transcript) || state.liveStatus.lastPartialTranscript;
    if (partialText) {
      return partialText;
    }
    const finalChars = Number(activeListening.final_transcript_chars || 0);
    const partialChars = Number(activeListening.partial_transcript_chars || 0);
    if (finalChars > 0 || partialChars > 0) {
      return `${finalChars} final chars; ${partialChars} partial chars`;
    }
    return "not yet";
  }

  const listenerChipLabels = {
    heard_summary: { en: "I heard...", zh: "我听到..." },
    constraint_detected: { en: "constraint detected", zh: "检测到约束" },
    question_detected: { en: "question detected", zh: "检测到问题" },
    showing_object: { en: "showing object", zh: "正在看物体" },
    camera_limited: { en: "camera limited", zh: "视觉受限" },
    still_listening: { en: "still listening", zh: "继续听" },
    ready_to_answer: { en: "ready to answer", zh: "可以回答" },
  };

  function semanticListener(activeListening) {
    return safeObject(safeObject(activeListening).semantic_state_v3);
  }

  function listenerChipLabel(chip, language) {
    const object = safeObject(chip);
    const chipId = text(object.chip_id);
    const labels = listenerChipLabels[chipId] || null;
    if (language === "zh") {
      return text(object.localized_label) || (labels && labels.zh) || text(object.label);
    }
    return text(object.label) || (labels && labels.en) || chipId;
  }

  function listenerChipsText(activeListening, language) {
    const semantic = semanticListener(activeListening);
    const chips = safeArray(semantic.listener_chips)
      .map((chip) => listenerChipLabel(chip, language))
      .filter(Boolean)
      .slice(0, 7);
    if (chips.length) {
      return chips.join(" | ");
    }
    return language === "zh" ? "继续听" : "still listening";
  }

  function semanticListenerText(activeListening) {
    const semantic = semanticListener(activeListening);
    const summary = text(semantic.safe_live_summary);
    const intent = text(semantic.detected_intent);
    if (summary && intent) {
      return `${summary}; intent=${intent}`;
    }
    return summary || intent || "unknown";
  }

  function participantIsLocal(participant) {
    if (!participant || typeof participant !== "object") {
      return true;
    }
    return participant.local !== false;
  }

  function deviceErrorState(error) {
    const errorType = text(safeObject(error).type).toLowerCase();
    if (errorType.includes("permission") || errorType === "permissions") {
      return "permission_denied";
    }
    if (errorType.includes("not-found") || errorType.includes("not_found")) {
      return "device_not_found";
    }
    if (errorType.includes("in-use") || errorType.includes("constraints")) {
      return "error";
    }
    return "unavailable";
  }

  function mediaTrackSettingState(track, key) {
    try {
      if (!track || typeof track.getSettings !== "function") {
        return "unknown";
      }
      const value = safeObject(track.getSettings())[key];
      if (value === true) {
        return "enabled";
      }
      if (value === false) {
        return "disabled";
      }
      if (value == null) {
        return "unknown";
      }
      return text(value) || "unknown";
    } catch (_error) {
      return "unknown";
    }
  }

  function audioEchoStatus(track) {
    return {
      echoCancellationState: mediaTrackSettingState(track, "echoCancellation"),
      noiseSuppressionState: mediaTrackSettingState(track, "noiseSuppression"),
      autoGainControlState: mediaTrackSettingState(track, "autoGainControl"),
    };
  }

  function currentMediaMode() {
    const microphoneReady = ["ready", "receiving"].includes(state.liveStatus.microphoneState);
    const cameraReady = ["ready", "receiving"].includes(state.liveStatus.cameraState);
    if (microphoneReady && cameraReady) {
      return "camera_and_microphone";
    }
    if (microphoneReady) {
      return "audio_only";
    }
    if (
      state.liveStatus.microphoneState === "unknown" &&
      state.liveStatus.cameraState === "unknown"
    ) {
      return "unreported";
    }
    return "unavailable";
  }

  function publicMediaPayload() {
    return {
      mode: currentMediaMode(),
      camera_state: state.liveStatus.cameraState || "unknown",
      microphone_state: state.liveStatus.microphoneState || "unknown",
      echo_cancellation: state.liveStatus.echoCancellationState || "unknown",
      noise_suppression: state.liveStatus.noiseSuppressionState || "unknown",
      auto_gain_control: state.liveStatus.autoGainControlState || "unknown",
      reason_codes: [
        `browser_media:${currentMediaMode()}`,
        `browser_camera:${state.liveStatus.cameraState || "unknown"}`,
        `browser_microphone:${state.liveStatus.microphoneState || "unknown"}`,
        `browser_echo_cancellation:${state.liveStatus.echoCancellationState || "unknown"}`,
        `browser_noise_suppression:${state.liveStatus.noiseSuppressionState || "unknown"}`,
        `browser_auto_gain_control:${state.liveStatus.autoGainControlState || "unknown"}`,
      ],
    };
  }

  async function flushClientMediaReport(payload) {
    state.liveStatus.mediaReportInFlight = true;
    try {
      await fetch(clientMediaEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        cache: "no-store",
      });
    } catch (_error) {
      // The live UI must keep working even before the local runtime endpoint is ready.
    } finally {
      state.liveStatus.mediaReportInFlight = false;
      const queued = state.liveStatus.mediaReportQueued;
      state.liveStatus.mediaReportQueued = null;
      if (queued) {
        queueClientMediaReport(queued);
      }
    }
  }

  function queueClientMediaReport(payload) {
    const key = JSON.stringify(payload);
    if (key === state.liveStatus.mediaReportKey) {
      return;
    }
    state.liveStatus.mediaReportKey = key;
    if (state.liveStatus.mediaReportInFlight) {
      state.liveStatus.mediaReportQueued = payload;
      return;
    }
    flushClientMediaReport(payload);
  }

  function updateLiveStatus(next, options) {
    Object.assign(state.liveStatus, next || {});
    if (options && options.reportMedia) {
      queueClientMediaReport(publicMediaPayload());
    }
    render();
  }

  function clientCallbacks() {
    return {
      onConnected: () =>
        updateLiveStatus({
          connectionState: "connected",
          transportState: "connected",
          modeHint: "connected",
        }),
      onDisconnected: () =>
        updateLiveStatus(
          {
            cameraState: "unknown",
            connectionState: "waiting",
            echoCancellationState: "unknown",
            noiseSuppressionState: "unknown",
            autoGainControlState: "unknown",
            microphoneState: "unknown",
            activeListeningDurationMs: 0,
            activeListeningPhase: "idle",
            activeListeningStartedAt: 0,
            lastPartialTranscript: "",
            partialTranscriptAvailable: false,
            modeHint: "waiting",
            transportState: "disconnected",
          },
          { reportMedia: true },
        ),
      onTransportStateChanged: (transportState) =>
        updateLiveStatus({
          connectionState: text(transportState) || "waiting",
          transportState: text(transportState) || "waiting",
        }),
      onUserStartedSpeaking: () =>
        updateLiveStatus({
          activeListeningDurationMs: 0,
          activeListeningPhase: "speech_started",
          activeListeningStartedAt: Date.now(),
          lastFinalTranscript: "",
          lastPartialTranscript: "",
          partialTranscriptAvailable: false,
          modeHint: "listening",
        }),
      onUserStoppedSpeaking: () =>
        updateLiveStatus({
          activeListeningDurationMs: activeListeningDurationMs(),
          activeListeningPhase: "transcribing",
          modeHint: "heard",
        }),
      onUserTranscript: (payload) => {
        const final = transcriptIsFinal(payload);
        const transcript = transcriptText(payload);
        if (!final) {
          updateLiveStatus({
            activeListeningPhase: "partial_transcript",
            lastPartialTranscript: transcript,
            partialTranscriptAvailable: Boolean(transcript),
            modeHint: "listening",
          });
          return;
        }
        updateLiveStatus({
          activeListeningPhase: "final_transcript",
          activeListeningDurationMs: activeListeningDurationMs(),
          lastFinalTranscript: appendTranscriptText(
            state.liveStatus.lastFinalTranscript,
            payload,
            visibleTextLimit,
          ),
          lastPartialTranscript: "",
          partialTranscriptAvailable: false,
          modeHint: "heard",
        });
      },
      onBotLlmStarted: () =>
        updateLiveStatus({
          assistantSubtitle: "",
          modeHint: "thinking",
        }),
      onBotLlmText: (payload) =>
        updateLiveStatus({
          assistantSubtitle: appendVisibleText(
            state.liveStatus.assistantSubtitle,
            payload,
            visibleTextLimit,
          ),
          modeHint: "thinking",
        }),
      onBotOutput: (payload) =>
        updateLiveStatus({
          assistantSubtitle:
            visibleText(payload, visibleTextLimit) || state.liveStatus.assistantSubtitle,
        }),
      onBotTtsStarted: () =>
        updateLiveStatus({
          modeHint: "speaking",
          ttsState: "speaking",
        }),
      onBotTtsText: (payload) =>
        updateLiveStatus({
          assistantSubtitle:
            visibleText(payload, visibleTextLimit) || state.liveStatus.assistantSubtitle,
          modeHint: "speaking",
          ttsState: "speaking",
        }),
      onBotTtsStopped: () =>
        updateLiveStatus({
          modeHint: "listening",
          ttsState: "idle",
        }),
      onBotStartedSpeaking: () =>
        updateLiveStatus({
          modeHint: "speaking",
          ttsState: "speaking",
        }),
      onBotStoppedSpeaking: () =>
        updateLiveStatus({
          modeHint: "listening",
          ttsState: "idle",
        }),
      onTrackStarted: (track, participant) => {
        if (!participantIsLocal(participant)) {
          return;
        }
        const kind = text(track && track.kind);
        if (kind === "audio") {
          updateLiveStatus(
            { microphoneState: "receiving", ...audioEchoStatus(track) },
            { reportMedia: true },
          );
        } else if (kind === "video") {
          updateLiveStatus({ cameraState: "ready" }, { reportMedia: true });
        }
      },
      onTrackStopped: (track, participant) => {
        if (!participantIsLocal(participant)) {
          return;
        }
        const kind = text(track && track.kind);
        if (kind === "audio") {
          updateLiveStatus({ microphoneState: "stalled" }, { reportMedia: true });
        } else if (kind === "video") {
          updateLiveStatus({ cameraState: "stalled" }, { reportMedia: true });
        }
      },
      onDeviceError: (error) => {
        const devices = safeArray(safeObject(error).devices);
        const next = {
          lastDeviceError: deviceErrorState(error),
          modeHint: "error",
        };
        if (devices.includes("mic")) {
          next.microphoneState = deviceErrorState(error);
        }
        if (devices.includes("cam")) {
          next.cameraState = deviceErrorState(error);
        }
        updateLiveStatus(next, { reportMedia: true });
      },
      onError: () => updateLiveStatus({ modeHint: "error" }),
    };
  }

  window.BlinkLiveStatus = Object.assign({}, window.BlinkLiveStatus || {}, {
    clientCallbacks,
    snapshot: () => Object.assign({}, state.liveStatus),
  });

  function placePanel(panel) {
    const operatorPanel = document.getElementById("blink-operator-workbench");
    if (operatorPanel) {
      operatorPanel.appendChild(panel);
      return;
    }
    const root = document.getElementById("root");
    if (root && root.parentElement) {
      root.insertAdjacentElement("afterend", panel);
      return;
    }
    document.body.appendChild(panel);
  }

  function ensurePanel() {
    let panel = document.getElementById(panelId);
    if (panel) {
      return panel;
    }
    panel = document.createElement("aside");
    panel.id = panelId;
    panel.setAttribute("aria-label", "Blink runtime state");
    assignStyles(panel, {
      position: "relative",
      zIndex: "1",
      width: "calc(100% - 32px)",
      maxWidth: "none",
      maxHeight: "min(440px, 55vh)",
      overflow: "auto",
      boxSizing: "border-box",
      margin: "16px",
      padding: "10px 12px",
      borderRadius: "8px",
      border: "1px solid rgba(148, 163, 184, 0.35)",
      background: "rgba(15, 23, 42, 0.88)",
      color: "#E5E7EB",
      fontFamily:
        "Geist, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
      fontSize: "12px",
      lineHeight: "1.35",
      boxShadow: "0 12px 34px rgba(15, 23, 42, 0.28)",
      backdropFilter: "blur(12px)",
      pointerEvents: "auto",
    });
    panel.innerHTML = "";

    const header = document.createElement("button");
    header.type = "button";
    header.setAttribute("aria-expanded", "true");
    assignStyles(header, {
      appearance: "none",
      border: "0",
      background: "transparent",
      color: "#F8FAFC",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      width: "100%",
      padding: "0",
      margin: "0 0 8px",
      cursor: "pointer",
      fontSize: "12px",
      fontWeight: "650",
      letterSpacing: "0",
    });
    header.addEventListener("click", () => {
      state.collapsed = !state.collapsed;
      render();
      if (!state.collapsed) {
        refresh();
      }
    });

    const headerTitle = document.createElement("span");
    headerTitle.setAttribute("data-header-title", "");
    headerTitle.textContent = "Blink state";
    const headerHint = document.createElement("span");
    headerHint.setAttribute("data-collapse-hint", "");
    assignStyles(headerHint, {
      color: "#CBD5E1",
      fontWeight: "500",
    });
    header.append(headerTitle, headerHint);

    const body = document.createElement("div");
    body.setAttribute("data-state-body", "");

    const expressionSection = document.createElement("section");
    expressionSection.setAttribute("data-expression-body", "");

    const performanceSection = document.createElement("section");
    performanceSection.setAttribute("data-performance-body", "");
    assignStyles(performanceSection, {
      paddingBottom: "8px",
      marginBottom: "8px",
      borderBottom: "1px solid rgba(148, 163, 184, 0.25)",
    });

    const memoryDetails = document.createElement("details");
    memoryDetails.setAttribute("data-memory-details", "");
    assignStyles(memoryDetails, {
      marginTop: "8px",
      paddingTop: "8px",
      borderTop: "1px solid rgba(148, 163, 184, 0.25)",
    });
    const memorySummary = document.createElement("summary");
    memorySummary.textContent = "Memory";
    assignStyles(memorySummary, {
      cursor: "pointer",
      color: "#F8FAFC",
      fontWeight: "650",
      marginBottom: "6px",
    });
    const memoryBody = document.createElement("div");
    memoryBody.setAttribute("data-memory-body", "");
    memoryDetails.append(memorySummary, memoryBody);

    const advancedDetails = document.createElement("details");
    advancedDetails.setAttribute("data-advanced-details", "");
    assignStyles(advancedDetails, {
      marginTop: "8px",
      paddingTop: "8px",
      borderTop: "1px solid rgba(148, 163, 184, 0.18)",
    });
    const advancedSummary = document.createElement("summary");
    advancedSummary.textContent = "Advanced";
    assignStyles(advancedSummary, {
      cursor: "pointer",
      color: "#CBD5E1",
    });
    const advancedBody = document.createElement("div");
    advancedBody.setAttribute("data-advanced-body", "");
    advancedDetails.append(advancedSummary, advancedBody);

    body.append(performanceSection, expressionSection, memoryDetails, advancedDetails);
    panel.append(header, body);
    placePanel(panel);
    return panel;
  }

  function compactStatus(status) {
    if (!status || typeof status !== "object") {
      return "unavailable";
    }
    const persona = status.persona_expression || "unknown";
    const defaults = status.persona_defaults || "unknown";
    return `persona=${persona}; defaults=${defaults}`;
  }

  function compactVoicePolicy(policy) {
    if (!policy || typeof policy !== "object" || !policy.available) {
      return "unavailable";
    }
    const mode = policy.chunking_mode || "off";
    const maxChars = policy.max_spoken_chunk_chars || 0;
    const activeHints = Array.isArray(policy.active_hints) ? policy.active_hints : [];
    const unsupportedHints = Array.isArray(policy.unsupported_hints)
      ? policy.unsupported_hints
      : [];
    const activeText = activeHints.length ? activeHints.slice(0, 3).join(", ") : "none";
    const noopText = unsupportedHints.length ? unsupportedHints.slice(0, 3).join(", ") : "none";
    return `${mode}; max=${maxChars}; active=${activeText}; no-op=${noopText}`;
  }

  function setTextLine(parent, label, value) {
    const row = document.createElement("div");
    const labelNode = document.createElement("span");
    labelNode.textContent = `${label}: `;
    labelNode.style.color = "#CBD5E1";
    const valueNode = document.createElement("span");
    valueNode.textContent = value || "unavailable";
    valueNode.style.color = "#F8FAFC";
    row.append(labelNode, valueNode);
    parent.appendChild(row);
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
    if (Number.isFinite(number)) {
      return number.toFixed(2);
    }
    return String(value);
  }

  function memoryUseText(record) {
    if (record.used_in_current_turn) {
      return "used now";
    }
    if (record.last_used_reason) {
      return `last used: ${String(record.last_used_reason)}`;
    }
    if (record.last_used_at) {
      return "last used";
    }
    return "";
  }

  function normalizeMemoryAction(action) {
    return String(action || "").trim().replaceAll("_", "-");
  }

  function supportedRecordActions(record) {
    const actions = Array.isArray(record.user_actions) ? record.user_actions : [];
    const seen = new Set();
    return actions
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

  function pendingActionKey(memoryId, action) {
    return `${memoryId}:${action}`;
  }

  function safeMemoryActionText(result) {
    if (!result || typeof result !== "object") {
      return "";
    }
    const status = result.accepted ? "Accepted" : "Rejected";
    const action = normalizeMemoryAction(result.action) || "action";
    const applied = result.applied ? "applied" : "not applied";
    return `${status}: ${action} ${applied}.`;
  }

  function renderMemoryActionResult(parent, memoryId) {
    const text = safeMemoryActionText(state.actionResults[memoryId]);
    if (!text) {
      return;
    }
    const resultNode = document.createElement("div");
    resultNode.textContent = text;
    assignStyles(resultNode, {
      color: "#BFDBFE",
      marginTop: "5px",
    });
    parent.appendChild(resultNode);
  }

  async function postMemoryAction(memoryId, action, body) {
    try {
      const response = await fetch(memoryActionEndpoint(memoryId, action), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      if (!response.ok) {
        throw new Error("request failed");
      }
      return await response.json();
    } catch (_error) {
      return {
        accepted: false,
        applied: false,
        action,
        memory_id: memoryId,
        reason_codes: ["memory_action_request_failed"],
      };
    }
  }

  async function runMemoryAction(memoryId, action, body) {
    const key = pendingActionKey(memoryId, action);
    if (state.pendingActions[key]) {
      return;
    }
    state.pendingActions[key] = true;
    render();
    const result = await postMemoryAction(memoryId, action, body);
    delete state.pendingActions[key];
    state.actionResults[memoryId] = result;
    state.latestActionResult = result;
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
      gap: "4px",
      marginTop: "5px",
    });
    actions.forEach((action) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = supportedMemoryActions[action].label;
      button.disabled = Boolean(state.pendingActions[pendingActionKey(memoryId, action)]);
      assignStyles(button, {
        border: "1px solid rgba(148, 163, 184, 0.28)",
        borderRadius: "6px",
        background: "rgba(30, 41, 59, 0.84)",
        color: "#E2E8F0",
        cursor: button.disabled ? "wait" : "pointer",
        padding: "2px 6px",
        fontSize: "11px",
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
      gap: "5px",
      marginTop: "7px",
    });
    const input = document.createElement("input");
    input.required = true;
    input.type = "text";
    input.value = state.correctionDrafts[memoryId] || "";
    input.placeholder = "Replacement memory value";
    assignStyles(input, {
      border: "1px solid rgba(148, 163, 184, 0.35)",
      borderRadius: "6px",
      background: "rgba(15, 23, 42, 0.94)",
      color: "#F8FAFC",
      padding: "5px 6px",
      fontSize: "12px",
    });
    input.addEventListener("input", () => {
      state.correctionDrafts[memoryId] = input.value;
    });
    const controls = document.createElement("div");
    assignStyles(controls, {
      display: "flex",
      gap: "5px",
    });
    const submit = document.createElement("button");
    submit.type = "submit";
    submit.textContent = "Submit correction";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "Cancel";
    [submit, cancel].forEach((button) => {
      assignStyles(button, {
        border: "1px solid rgba(148, 163, 184, 0.28)",
        borderRadius: "6px",
        background: "rgba(30, 41, 59, 0.84)",
        color: "#E2E8F0",
        cursor: "pointer",
        padding: "2px 6px",
        fontSize: "11px",
      });
    });
    cancel.addEventListener("click", () => {
      state.correctionMemoryId = null;
      render();
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const replacementValue = input.value.trim();
      state.correctionDrafts[memoryId] = replacementValue;
      if (!replacementValue) {
        state.actionResults[memoryId] = {
          accepted: false,
          applied: false,
          action: "correct",
          memory_id: memoryId,
          reason_codes: ["replacement_value_required"],
        };
        render();
        return;
      }
      runMemoryAction(memoryId, "correct", { replacement_value: replacementValue });
    });
    controls.append(submit, cancel);
    form.append(input, controls);
    parent.appendChild(form);
  }

  function renderMemoryRecord(parent, record) {
    const item = document.createElement("div");
    assignStyles(item, {
      paddingTop: "7px",
      marginTop: "7px",
      borderTop: "1px solid rgba(148, 163, 184, 0.14)",
    });
    const title = document.createElement("div");
    title.textContent = record.summary || record.title || "Memory";
    assignStyles(title, {
      color: "#F8FAFC",
      fontWeight: "600",
    });
    const metaParts = [
      record.display_kind || "memory",
      statusText(record),
      confidenceText(record.confidence) ? `confidence ${confidenceText(record.confidence)}` : "",
      record.pinned ? "pinned" : "",
      memoryUseText(record),
    ].filter(Boolean);
    const meta = document.createElement("div");
    meta.textContent = metaParts.join(" | ");
    meta.style.color = "#CBD5E1";
    item.append(title, meta);
    if (record.safe_provenance_label) {
      const provenance = document.createElement("div");
      provenance.textContent = String(record.safe_provenance_label);
      assignStyles(provenance, {
        color: "#94A3B8",
        marginTop: "3px",
      });
      item.appendChild(provenance);
    }
    addMemoryActionControls(item, record);
    renderCorrectionForm(item, record);
    renderMemoryActionResult(item, String(record.memory_id || ""));
    parent.appendChild(item);
  }

  function summarizeReasonCodes(reasonCodes) {
    const codes = Array.isArray(reasonCodes) ? reasonCodes : [];
    if (codes.length === 0) {
      return "none";
    }
    const categories = Array.from(
      new Set(
        codes
          .map((code) => String(code || "").split(":")[0])
          .filter((category) => category.length > 0),
      ),
    ).slice(0, 4);
    return `${codes.length} codes${categories.length ? `; categories=${categories.join(", ")}` : ""}`;
  }

  function actorReasonCategories(reasonCodes) {
    const codes = safeArray(reasonCodes);
    return Array.from(
      new Set(
        codes
          .map((code) => text(code).split(":")[0])
          .filter(Boolean),
      ),
    ).slice(0, 4);
  }

  function actorEventKeyIsRenderable(key) {
    const lowered = text(key).toLowerCase();
    return (
      lowered &&
      !unsafeActorEventKeyFragments.some((fragment) => lowered.includes(fragment)) &&
      (lowered.endsWith("_count") ||
        lowered.endsWith("_counts") ||
        lowered.endsWith("_state") ||
        lowered.endsWith("_kind") ||
        lowered.endsWith("_label") ||
        lowered.endsWith("_mode") ||
        lowered.endsWith("_backend") ||
        lowered === "floor_state" ||
        lowered === "previous_floor_state" ||
        lowered === "input_type" ||
        lowered === "text_kind" ||
        lowered === "readiness_state" ||
        lowered === "grounding_mode")
    );
  }

  function actorEventValueText(value) {
    if (value === null || value === undefined) {
      return "";
    }
    if (typeof value === "boolean") {
      return value ? "true" : "false";
    }
    if (typeof value === "number") {
      return Number.isFinite(value) ? String(value) : "";
    }
    if (Array.isArray(value)) {
      return `${value.length} items`;
    }
    if (typeof value === "object") {
      return `${Object.keys(value).length} fields`;
    }
    const valueText = text(value);
    const lowered = valueText.toLowerCase();
    if (unsafeActorEventKeyFragments.some((fragment) => lowered.includes(fragment))) {
      return "";
    }
    return valueText.slice(0, 48);
  }

  function actorEventMetadataText(event) {
    const metadata = safeObject(safeObject(event).metadata);
    const parts = [];
    Object.keys(metadata)
      .filter(actorEventKeyIsRenderable)
      .slice(0, 6)
      .forEach((key) => {
        const value = actorEventValueText(metadata[key]);
        if (value) {
          parts.push(`${key}=${value}`);
        }
      });
    return parts.join("; ");
  }

  function renderActorPanel(parent, title, rows) {
    const section = document.createElement("section");
    assignStyles(section, {
      borderTop: "1px solid rgba(148, 163, 184, 0.16)",
      paddingTop: "7px",
      marginTop: "7px",
    });
    const heading = document.createElement("div");
    heading.textContent = title;
    assignStyles(heading, {
      color: "#F8FAFC",
      fontWeight: "700",
      marginBottom: "4px",
    });
    section.appendChild(heading);
    rows.forEach(([label, value]) => setTextLine(section, label, value));
    parent.appendChild(section);
  }

  function renderActorEventDrawer(parent, actor) {
    const details = document.createElement("details");
    details.setAttribute("data-actor-debug-timeline", "");
    assignStyles(details, {
      borderTop: "1px solid rgba(148, 163, 184, 0.16)",
      paddingTop: "7px",
      marginTop: "8px",
    });
    const summary = document.createElement("summary");
    summary.textContent = actorLabel(actor, "debugTimeline");
    assignStyles(summary, {
      cursor: "pointer",
      color: "#CBD5E1",
      fontWeight: "650",
    });
    details.appendChild(summary);
    const events = state.actorEvents.slice(-12).reverse();
    if (!events.length) {
      setTextLine(details, "events", "none");
    }
    events.forEach((event) => {
      const eventObject = safeObject(event);
      const item = document.createElement("div");
      assignStyles(item, {
        paddingTop: "6px",
        marginTop: "6px",
        borderTop: "1px solid rgba(148, 163, 184, 0.12)",
      });
      const eventTitle = document.createElement("div");
      const id = text(eventObject.event_id || "");
      const type = text(eventObject.event_type || "event");
      const mode = text(eventObject.mode || "unknown");
      eventTitle.textContent = id ? `#${id} ${type}; ${mode}` : `${type}; ${mode}`;
      eventTitle.style.color = "#F8FAFC";
      item.appendChild(eventTitle);
      const eventMeta = [
        text(eventObject.timestamp).slice(0, 19),
        text(eventObject.source || "runtime"),
        actorReasonCategories(eventObject.reason_codes).join(", "),
        actorEventMetadataText(eventObject),
      ].filter(Boolean);
      const eventDetail = document.createElement("div");
      eventDetail.textContent = eventMeta.join(" | ") || "public event";
      eventDetail.style.color = "#CBD5E1";
      item.appendChild(eventDetail);
      details.appendChild(item);
    });
    parent.appendChild(details);
  }

  function renderAdvanced(body, expression, memory) {
    body.replaceChildren();
    const expressionReasons = expression?.reason_codes || [];
    const voicePolicy = expression?.voice_policy || {};
    const voicePolicyReasons = voicePolicy.reason_codes || [];
    const voicePolicyNoops = voicePolicy.noop_reason_codes || [];
    const memoryReasons = memory?.reason_codes || [];
    setTextLine(body, "expression reasons", summarizeReasonCodes(expressionReasons));
    setTextLine(body, "voice policy reasons", summarizeReasonCodes(voicePolicyReasons));
    setTextLine(body, "voice policy no-ops", summarizeReasonCodes(voicePolicyNoops));
    setTextLine(body, "memory reasons", summarizeReasonCodes(memoryReasons));
    setTextLine(body, "actor state", summarizeReasonCodes(state.actor?.reason_codes || []));
    setTextLine(body, "actor events", latestActorEventText());
  }

  function latestPerformanceEventText() {
    const event = state.performanceEvents[state.performanceEvents.length - 1];
    if (!event || typeof event !== "object") {
      return "none";
    }
    const type = event.event_type || "event";
    const mode = event.mode || "unknown";
    return `${type}; mode=${mode}`;
  }

  function latestPerformanceEvent() {
    return safeObject(state.performanceEvents[state.performanceEvents.length - 1]);
  }

  function latestActorEventText() {
    const event = state.actorEvents[state.actorEvents.length - 1];
    if (!event || typeof event !== "object") {
      return "none";
    }
    return `${event.event_type || "event"}; mode=${event.mode || "unknown"}`;
  }

  function statusMode(performance) {
    const performanceMode = text(performance.mode);
    if (performanceMode && performanceMode !== "waiting") {
      return performanceMode;
    }
    return state.liveStatus.modeHint || performanceMode || "waiting";
  }

  function ttsStateText(performance) {
    const event = latestPerformanceEvent();
    if (event.event_type === "tts.error") {
      return "error";
    }
    if (event.event_type === "tts.speech_start" || state.liveStatus.ttsState === "speaking") {
      return "speaking";
    }
    if (event.event_type === "tts.speech_end") {
      return "idle";
    }
    return state.liveStatus.ttsState || "idle";
  }

  function echoStatusText(media) {
    const echo = safeObject(media.echo);
    const echoCancellation =
      echo.echo_cancellation || state.liveStatus.echoCancellationState || "unknown";
    const noiseSuppression =
      echo.noise_suppression || state.liveStatus.noiseSuppressionState || "unknown";
    const autoGainControl =
      echo.auto_gain_control || state.liveStatus.autoGainControlState || "unknown";
    return `aec ${echoCancellation}; ns ${noiseSuppression}; agc ${autoGainControl}`;
  }

  function interruptionStatusText(performance) {
    const interruption = safeObject(performance.interruption);
    const stateLabel = interruption.barge_in_state === "armed" ? "barge-in armed" : "protected";
    const decision = text(interruption.last_decision) || "none";
    const reason = text(interruption.last_reason);
    return reason && reason !== "none" ? `${stateLabel}; ${decision}; ${reason}` : `${stateLabel}; ${decision}`;
  }

  function speechStatusText(performance) {
    const speech = safeObject(performance.speech);
    const mode = text(speech.director_mode) || "unavailable";
    const subtitleMs = Number(speech.first_subtitle_latency_ms || 0);
    const audioMs = Number(speech.first_audio_latency_ms || 0);
    const queueDepth = Number(speech.speech_queue_depth_current || 0);
    const staleDrops = Number(speech.stale_chunk_drop_count || 0);
    const subtitle = subtitleMs > 0 ? `${Math.round(subtitleMs)}ms subtitle` : "subtitle pending";
    const audio = audioMs > 0 ? `${Math.round(audioMs)}ms audio` : "audio pending";
    return `${mode}; ${subtitle}; ${audio}; queue ${queueDepth}; stale ${staleDrops}`;
  }

  function cameraPresenceStatusText(performance, media) {
    const cameraPresence = safeObject(performance.camera_presence);
    const presenceState = text(cameraPresence.state) || text(media.camera_state) || "unknown";
    const trackState = text(cameraPresence.track_state) || "unknown";
    const fresh = cameraPresence.camera_fresh ? "fresh" : "not fresh";
    return `${presenceState}; track ${trackState}; ${fresh}`;
  }

  function cameraFrameAgeText(performance) {
    const cameraPresence = safeObject(performance.camera_presence);
    const ageMs = Number(cameraPresence.latest_frame_age_ms);
    const frameSeq = Number(cameraPresence.latest_frame_seq || 0);
    if (!Number.isFinite(ageMs) || ageMs < 0) {
      return frameSeq > 0 ? `frame ${frameSeq}; age pending` : "no frame";
    }
    const ageText = ageMs >= 1000 ? `${Math.round(ageMs / 1000)}s` : `${Math.round(ageMs)}ms`;
    return frameSeq > 0 ? `frame ${frameSeq}; ${ageText}` : ageText;
  }

  function cameraGroundingText(performance) {
    const cameraPresence = safeObject(performance.camera_presence);
    const mode = text(cameraPresence.grounding_mode) || "none";
    const result = text(cameraPresence.last_vision_result_state) || "none";
    const used = cameraPresence.current_answer_used_vision ? "used this answer" : "not used";
    return `${mode}; ${used}; ${result}`;
  }

  function activeListeningStatusText(performance) {
    const activeListening = safeObject(performance.active_listening);
    const phase = text(activeListening.phase) || state.liveStatus.activeListeningPhase || "idle";
    const durationMs =
      Number(activeListening.turn_duration_ms || 0) ||
      (phase === "speech_started" || phase === "speech_continuing"
        ? activeListeningDurationMs()
        : Number(state.liveStatus.activeListeningDurationMs || 0));
    const partialAvailable =
      activeListening.partial_available || state.liveStatus.partialTranscriptAvailable;
    const partialLabel = partialAvailable ? "partial available" : "no partials from STT";
    const durationLabel = durationMs > 0 ? `${Math.round(durationMs)}ms` : "duration pending";
    return `${phase}; ${durationLabel}; ${partialLabel}`;
  }

  function partialTranscriptText(performance) {
    const activeListening = safeObject(performance.active_listening);
    if (state.liveStatus.lastPartialTranscript) {
      return state.liveStatus.lastPartialTranscript;
    }
    if (activeListening.partial_available) {
      const chars = Number(activeListening.partial_transcript_chars || 0);
      return chars > 0 ? `${chars} chars` : "partial available";
    }
    return "no partials from STT";
  }

  function hintStatusText(items, fallback) {
    const values = safeArray(items)
      .map((item) => text(safeObject(item).value))
      .filter(Boolean)
      .slice(0, 3);
    return values.length ? values.join("; ") : fallback;
  }

  function degradedMessage(performance, media, vision) {
    const event = latestPerformanceEvent();
    const microphoneState = text(media.microphone_state);
    const cameraState = text(media.camera_state);
    const cameraPresence = safeObject(performance.camera_presence);
    const cameraPresenceState = text(cameraPresence.state);
    const ttsLabel = text(performance.tts || performance.tts_backend);
    if (microphoneState === "permission_denied") {
      return "Microphone permission denied";
    }
    if (microphoneState === "device_not_found" || microphoneState === "unavailable") {
      return "Microphone unavailable";
    }
    if (microphoneState === "stalled") {
      return "Microphone stalled";
    }
    if (vision.enabled) {
      if (cameraPresenceState === "error") {
        return "Camera vision error";
      }
      if (cameraPresenceState === "stale") {
        return "Camera stale";
      }
      if (cameraPresenceState === "stalled") {
        return "Camera stalled";
      }
      if (cameraState === "permission_denied") {
        return "Camera permission denied";
      }
      if (cameraState === "stale") {
        return "Camera stale";
      }
      if (cameraState === "stalled") {
        return "Camera stalled";
      }
      if (cameraState === "device_not_found" || cameraState === "unavailable") {
        return "Camera unavailable";
      }
    }
    if (event.event_type === "tts.error") {
      if (ttsLabel.includes("MeloTTS")) {
        return "MeloTTS unavailable";
      }
      if (ttsLabel.includes("Kokoro") || ttsLabel.includes("kokoro")) {
        return "Kokoro unavailable";
      }
      return "TTS unavailable";
    }
    if (event.event_type === "llm.error") {
      return "LLM unavailable";
    }
    if (performance.available === false) {
      return "Runtime unavailable";
    }
    if (state.liveStatus.lastDeviceError) {
      return `Media ${state.liveStatus.lastDeviceError}`;
    }
    return "none";
  }

  function memoryPersonaSummary(expression, memory) {
    const personaStatus = compactStatus(expression.memory_persona_section_status);
    const usedRecords = safeArray(memory.records).filter((record) => record.used_in_current_turn);
    const usedText = usedRecords.length ? `${usedRecords.length} memory used this turn` : "no memory used";
    return `${usedText}; ${personaStatus}`;
  }

  function actorMemoryPersonaSummary(actor, expression, memory) {
    const actorObject = safeObject(actor);
    const memoryPersona = safeObject(actorObject.memory_persona);
    const continuity = safeObject(memoryPersona.memory_continuity_trace);
    const performancePlan = safeObject(memoryPersona.performance_plan_v2);
    const usedRefs = safeArray(memoryPersona.used_in_current_reply);
    const personaRefs = safeArray(memoryPersona.persona_references);
    const styleSummary = text(performancePlan.style_summary || memoryPersona.summary);
    const memoryEffect = text(continuity.memory_effect || memoryPersona.memory_policy);
    const counts = [
      countText(memoryPersona.selected_memory_count || usedRefs.length, "memories"),
      countText(personaRefs.length || safeArray(performancePlan.persona_references_used).length, "references"),
    ];
    if (memoryEffect && memoryEffect !== "unavailable") {
      counts.push(memoryEffect);
    }
    if (styleSummary && styleSummary !== "Memory/persona performance unavailable.") {
      counts.push(styleSummary);
    } else {
      counts.push(memoryPersonaSummary(expression, memory));
    }
    return counts.join("; ");
  }

  function actorInterruptionText(actor, performance) {
    const actorObject = safeObject(actor);
    const interruption = safeObject(actorObject.interruption);
    const audioHealth = safeObject(actorObject.webrtc_audio_health);
    if (!Object.keys(interruption).length && !Object.keys(audioHealth).length) {
      return interruptionStatusText(performance);
    }
    const stateLabel =
      text(audioHealth.barge_in_state || interruption.barge_in_state) || "protected";
    const echoRisk = text(audioHealth.echo_risk_level || audioHealth.echo_risk_state) || "unknown";
    const decision = text(interruption.last_decision) || "none";
    const reason = text(interruption.last_reason || audioHealth.last_decision_reason);
    return [stateLabel, `echo ${echoRisk}`, decision, reason].filter(Boolean).join("; ");
  }

  function actorCameraSceneText(actor, performance) {
    const actorObject = safeObject(actor);
    const scene = safeObject(actorObject.camera_scene || safeObject(actorObject.camera).scene);
    const social = safeObject(scene.scene_social_state_v2);
    if (!Object.keys(scene).length) {
      return cameraPresenceStatusText(performance, safeObject(performance.browser_media));
    }
    const stateText = text(scene.state || scene.status) || "unknown";
    const freshness = text(scene.freshness_state) || (scene.camera_fresh ? "fresh" : "not fresh");
    const frameSeq = Number(scene.latest_frame_sequence || scene.latest_frame_seq || 0);
    const frameAge = formatAgeMs(scene.latest_frame_age_ms);
    const grounding = text(scene.grounding_mode) || "none";
    const used = scene.current_answer_used_vision ? "used vision" : "not used";
    const honesty = text(social.camera_honesty_state);
    const transition = text(social.scene_transition);
    return [
      stateText,
      freshness,
      `frame ${frameSeq}`,
      frameAge,
      grounding,
      used,
      honesty && `honesty ${honesty}`,
      transition && `scene ${transition}`,
    ]
      .filter(Boolean)
      .join("; ");
  }

  function actorMediaText(actor, performance) {
    const actorObject = safeObject(actor);
    const webrtc = safeObject(actorObject.webrtc);
    const media = safeObject(webrtc.media || performance.browser_media);
    const mediaMode = text(webrtc.media_mode || media.mode) || "unreported";
    const mic = text(media.microphone_state) || "unknown";
    const camera = text(media.camera_state) || "unknown";
    const echo = safeObject(media.echo);
    const echoCancellation = text(echo.echo_cancellation) || "unknown";
    const clientState = webrtc.client_active ? "client active" : "client disconnected";
    return [
      mediaMode,
      `mic ${mic}`,
      `camera ${camera}`,
      `echo ${echoCancellation}`,
      clientState,
    ].join("; ");
  }

  function renderActorSurface(performanceBody, actor, performance, expression, memory) {
    const actorObject = safeObject(actor);
    const media = safeObject(safeObject(actorObject.webrtc).media || performance.browser_media);
    const vision = safeObject(actorObject.vision || performance.vision);
    const tts = safeObject(actorObject.tts);
    const degradation = safeObject(actorObject.degradation);
    const activeListening = safeObject(actorObject.active_listening);
    const speech = safeObject(actorObject.speech);
    const scene = safeObject(actorObject.camera_scene || safeObject(actorObject.camera).scene);
    const interruption = safeObject(actorObject.interruption);
    const audioHealth = safeObject(actorObject.webrtc_audio_health);
    const mode = actorMode(actorObject, performance);
    const language = text(actorObject.language || "unknown");
    const ttsLabel = text(tts.label || performance.tts || performance.tts_backend || "unknown");
    const visionLabel = vision.enabled === false ? "none" : text(vision.backend || "moondream");

    performanceBody.replaceChildren();
    const headline = document.createElement("div");
    headline.textContent = modeText(mode);
    assignStyles(headline, {
      color: degradation.state === "error" || mode === "error" ? "#FCA5A5" : "#F8FAFC",
      fontSize: "16px",
      fontWeight: "760",
      lineHeight: "1.2",
      marginBottom: "6px",
      textTransform: "capitalize",
    });
    performanceBody.appendChild(headline);
    setTextLine(performanceBody, "profile", profileBadge(actorObject, performance));
    setTextLine(performanceBody, "language", language);
    setTextLine(performanceBody, "tts", `${ttsLabel}; ${ttsStateText(performance)}`);
    setTextLine(performanceBody, "media", actorMediaText(actorObject, performance));
    setTextLine(performanceBody, "camera/Moondream", actorCameraSceneText(actorObject, performance));
    setTextLine(performanceBody, "mode", mode);
    setTextLine(performanceBody, "interruption", actorInterruptionText(actorObject, performance));
    setTextLine(performanceBody, "degradation", text(degradation.state) || degradedMessage(performance, media, vision));

    renderActorPanel(performanceBody, actorLabel(actorObject, "heard"), [
      ["phase", text(activeListening.phase) || state.liveStatus.activeListeningPhase || "idle"],
      ["text", actorHeardText(actorObject)],
      ["chips", listenerChipsText(activeListening, language)],
      ["semantic", semanticListenerText(activeListening)],
      ["topics", hintListText(activeListening.topics, "none")],
      ["constraints", hintListText(activeListening.constraints, "none")],
    ]);
    renderActorPanel(performanceBody, actorLabel(actorObject, "saying"), [
      ["subtitle", actorSubtitleText(actorObject)],
      ["director", text(speech.director_mode) || "unavailable"],
      ["queue", String(Number(speech.speech_queue_depth_current || 0))],
      ["stale drops", String(Number(speech.stale_chunk_drop_count || 0))],
      ["backend", ttsLabel],
    ]);
    renderActorPanel(performanceBody, actorLabel(actorObject, "looking"), [
      ["state", text(scene.state || scene.status) || text(vision.state) || "unknown"],
      ["freshness", text(scene.freshness_state) || (scene.camera_fresh ? "fresh" : "unknown")],
      ["frame", `${Number(scene.latest_frame_sequence || scene.latest_frame_seq || 0)}; ${formatAgeMs(scene.latest_frame_age_ms)}`],
      ["grounding", text(scene.grounding_mode || vision.grounding_mode) || "none"],
      ["used", scene.current_answer_used_vision || vision.current_answer_used_vision ? "yes" : "no"],
      ["honesty", text(safeObject(scene.scene_social_state_v2).camera_honesty_state) || "unavailable"],
      ["scene", text(safeObject(scene.scene_social_state_v2).scene_transition) || "none"],
      ["presence", text(safeObject(scene.scene_social_state_v2).user_presence_hint) || "unknown"],
      ["object", text(safeObject(scene.scene_social_state_v2).object_hint) || "not_evaluated"],
    ]);
    renderActorPanel(performanceBody, actorLabel(actorObject, "memoryPersona"), [
      ["summary", actorMemoryPersonaSummary(actorObject, expression, memory)],
      [
        "style",
        text(
          safeObject(safeObject(actorObject.memory_persona).performance_plan_v2).style_summary ||
            safeObject(actorObject.memory_persona).summary,
        ) || "unavailable",
      ],
      [
        "continuity",
        text(safeObject(safeObject(actorObject.memory_persona).memory_continuity_trace).memory_effect) ||
          "unavailable",
      ],
    ]);
    renderActorPanel(performanceBody, actorLabel(actorObject, "interruption"), [
      ["state", text(audioHealth.barge_in_state || interruption.barge_in_state) || "protected"],
      ["echo risk", text(audioHealth.echo_risk_level || audioHealth.echo_risk_state) || "unknown"],
      ["last decision", text(interruption.last_decision) || "none"],
      [
        "false counts",
        text(
          interruption.false_interruption_count ||
            safeObject(interruption.false_interruption_counts).total ||
            0,
        ),
      ],
    ]);
    renderActorEventDrawer(performanceBody, actorObject);
  }

  function render() {
    const expression = state.expression || {};
    const memory = state.memory || {};
    const actor = state.actor || {};
    const performance = state.performance || {};
    const actorSurfaceEnabled = actorSurfaceV2Enabled();
    const panel = ensurePanel();
    const header = panel.querySelector("button");
    const headerTitle = panel.querySelector("[data-header-title]");
    const hint = panel.querySelector("[data-collapse-hint]");
    const body = panel.querySelector("[data-state-body]");
    const performanceBody = panel.querySelector("[data-performance-body]");
    const expressionBody = panel.querySelector("[data-expression-body]");
    const memoryBody = panel.querySelector("[data-memory-body]");
    const advancedBody = panel.querySelector("[data-advanced-body]");
    if (!body || !performanceBody || !expressionBody || !memoryBody || !advancedBody) {
      return;
    }
    if (header) {
      header.setAttribute("aria-expanded", state.collapsed ? "false" : "true");
    }
    if (hint) {
      hint.textContent = state.collapsed ? "show" : "hide";
    }
    if (headerTitle) {
      headerTitle.textContent = actorSurfaceEnabled ? actorLabel(actor, "actorTitle") : "Blink state";
    }
    body.style.display = state.collapsed ? "none" : "block";
    if (actorSurfaceEnabled) {
      renderActorSurface(performanceBody, actor, performance, expression, memory);
    } else {
      const media = performance.browser_media || {};
      const vision = performance.vision || {};
      const cameraPresence = safeObject(performance.camera_presence);
      const interruption = safeObject(performance.interruption);
      const activeListening = safeObject(performance.active_listening);
      const mode = statusMode(performance);
      performanceBody.replaceChildren();
      const headline = document.createElement("div");
      headline.textContent = modeText(mode);
      assignStyles(headline, {
        color: mode === "error" ? "#FCA5A5" : "#F8FAFC",
        fontSize: "16px",
        fontWeight: "760",
        lineHeight: "1.2",
        marginBottom: "6px",
        textTransform: "capitalize",
      });
      performanceBody.appendChild(headline);
      setTextLine(performanceBody, "mode", mode);
      setTextLine(performanceBody, "connection", state.liveStatus.connectionState || "waiting");
      setTextLine(performanceBody, "profile", performance.profile || "manual");
      setTextLine(
        performanceBody,
        "tts",
        `${performance.tts || performance.tts_backend || "unknown"}; ${ttsStateText(performance)}`,
      );
      setTextLine(performanceBody, "mic", media.microphone_state || "unknown");
      setTextLine(performanceBody, "camera", media.camera_state || "unknown");
      setTextLine(performanceBody, "camera presence", cameraPresenceStatusText(performance, media));
      setTextLine(performanceBody, "frame age", cameraFrameAgeText(performance));
      setTextLine(performanceBody, "grounding", cameraGroundingText(performance));
      setTextLine(
        performanceBody,
        "playback",
        performance.protected_playback === false ? "barge-in" : "protected",
      );
      setTextLine(performanceBody, "interruption", interruptionStatusText(performance));
      setTextLine(performanceBody, "echo", echoStatusText(media));
      setTextLine(performanceBody, "speech", speechStatusText(performance));
      setTextLine(performanceBody, "listening", activeListeningStatusText(performance));
      setTextLine(performanceBody, "partial", partialTranscriptText(performance));
      if (interruption.headphones_recommended) {
        setTextLine(performanceBody, "headphones", "headphones recommended");
      }
      setTextLine(
        performanceBody,
        "vision",
        vision.enabled
          ? `${vision.continuous_perception_enabled ? "continuous" : "on-demand"}; ${
              cameraPresence.grounding_mode || "none"
            }`
          : "off",
      );
      setTextLine(
        performanceBody,
        "heard",
        state.liveStatus.lastFinalTranscript || "not yet",
      );
      setTextLine(
        performanceBody,
        "topics",
        hintStatusText(activeListening.topics, "none"),
      );
      setTextLine(
        performanceBody,
        "constraints",
        hintStatusText(activeListening.constraints, "none"),
      );
      setTextLine(
        performanceBody,
        "subtitle",
        state.liveStatus.assistantSubtitle || "waiting",
      );
      setTextLine(
        performanceBody,
        "memory/persona",
        memoryPersonaSummary(expression, memory),
      );
      setTextLine(performanceBody, "degraded", degradedMessage(performance, media, vision));
      setTextLine(performanceBody, "event", latestPerformanceEventText());
    }

    expressionBody.replaceChildren();
    setTextLine(expressionBody, "profile", expression.persona_profile_id || "unavailable");
    setTextLine(expressionBody, "teaching", expression.teaching_mode_label || "unavailable");
    setTextLine(
      expressionBody,
      "memory/persona",
      compactStatus(expression.memory_persona_section_status),
    );
    setTextLine(expressionBody, "voice", expression.voice_style_summary || "unavailable");
    setTextLine(expressionBody, "voice policy", compactVoicePolicy(expression.voice_policy));

    memoryBody.replaceChildren();
    setTextLine(memoryBody, "summary", memory.summary || "Memory unavailable.");
    setTextLine(memoryBody, "health", memory.health_summary || "Memory health unavailable.");
    if (safeMemoryActionText(state.latestActionResult)) {
      setTextLine(memoryBody, "last action", safeMemoryActionText(state.latestActionResult));
    }
    const records = Array.isArray(memory.records) ? memory.records : [];
    if (records.length === 0) {
      setTextLine(memoryBody, "visible", "none");
    } else {
      records.slice(0, 5).forEach((record) => renderMemoryRecord(memoryBody, record));
      if (records.length > 5) {
        setTextLine(memoryBody, "more", String(records.length - 5));
      }
    }
    renderAdvanced(advancedBody, expression, memory);
  }

  async function fetchJson(endpoint, fallback) {
    try {
      const response = await fetch(endpoint, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`status ${response.status}`);
      }
      return await response.json();
    } catch (_error) {
      return fallback;
    }
  }

  async function refresh() {
    if (state.collapsed) {
      await refreshPerformance();
      return;
    }
    const now = Date.now();
    if (refreshState.fullInFlight) {
      return;
    }
    if (refreshState.lastFullRefreshAt && now - refreshState.lastFullRefreshAt < fullRefreshMinMs) {
      await refreshPerformance();
      return;
    }
    refreshState.fullInFlight = true;
    try {
      const [expression, memory] = await Promise.all([
        fetchJson(expressionEndpoint, {
          persona_profile_id: null,
          teaching_mode_label: "unavailable",
          memory_persona_section_status: { persona_expression: "unavailable" },
          voice_style_summary: "unavailable",
          voice_policy: {
            available: false,
            reason_codes: ["voice_policy:unavailable"],
            noop_reason_codes: ["voice_policy_noop:hardware_control_forbidden"],
          },
          reason_codes: ["expression_fetch_failed"],
        }),
        fetchJson(memoryEndpoint, {
          available: false,
          summary: "Memory unavailable.",
          records: [],
          health_summary: "Memory health unavailable.",
          reason_codes: ["memory_fetch_failed"],
        }),
      ]);
      state.expression = expression;
      state.memory = memory;
      refreshState.lastFullRefreshAt = Date.now();
      await refreshPerformance();
      render();
    } finally {
      refreshState.fullInFlight = false;
    }
  }

  async function refreshPerformance() {
    if (refreshState.performanceInFlight) {
      return;
    }
    refreshState.performanceInFlight = true;
    try {
    const afterId = Number(state.latestPerformanceEventId || 0);
    const eventEndpoint = `${performanceEventsEndpoint}?after_id=${afterId}&limit=20`;
    const actorAfterId = Number(state.latestActorEventId || 0);
    const actorEventEndpoint = `${actorEventsEndpoint}?after_id=${actorAfterId}&limit=20`;
    const actorStateFallback = {
      available: false,
      schema_version: 2,
      mode: state.liveStatus.modeHint || "waiting",
      profile: "manual",
      language: "unknown",
      tts: { backend: "unknown", label: "unknown" },
      vision: {
        enabled: false,
        backend: "none",
        state: "disabled",
        current_answer_used_vision: false,
      },
      protected_playback: true,
      interruption: { barge_in_state: "protected", last_decision: "none" },
      webrtc_audio_health: { barge_in_state: "protected", echo_risk_level: "unknown" },
      active_listening: {
        phase: state.liveStatus.activeListeningPhase || "idle",
        partial_available: state.liveStatus.partialTranscriptAvailable,
        final_available: Boolean(state.liveStatus.lastFinalTranscript),
        topics: [],
        constraints: [],
      },
      speech: {
        director_mode: "unavailable",
        speech_queue_depth_current: 0,
        stale_chunk_drop_count: 0,
        assistant_subtitle: state.liveStatus.assistantSubtitle || null,
      },
      camera_scene: {
        state: "waiting_for_frame",
        freshness_state: "unknown",
        latest_frame_sequence: 0,
        latest_frame_age_ms: null,
        grounding_mode: "none",
        current_answer_used_vision: false,
        scene_social_state_v2: {
          camera_honesty_state: "unavailable",
          scene_transition: "none",
          user_presence_hint: "unknown",
          object_hint: "not_evaluated",
        },
      },
      memory_persona: {
        selected_memory_count: 0,
        used_in_current_reply: [],
        persona_references: [],
        summary: "Memory/persona performance unavailable.",
      },
      degradation: { state: "ok", components: [], reason_codes: ["actor_state_fetch_failed"] },
      live_text: {
        partial_transcript: state.liveStatus.lastPartialTranscript || null,
        final_transcript: state.liveStatus.lastFinalTranscript || null,
        assistant_subtitle: state.liveStatus.assistantSubtitle || null,
      },
      reason_codes: ["actor_state_fetch_failed"],
    };
    const [performance, eventsPayload, actor, actorEventsPayload] = await Promise.all([
      fetchJson(performanceStateEndpoint, {
        available: false,
        mode: "waiting",
        profile: "manual",
        tts: "unknown",
        browser_media: {
          camera_state: "unknown",
          microphone_state: "unknown",
          echo: {
            echo_cancellation: "unknown",
            noise_suppression: "unknown",
            auto_gain_control: "unknown",
          },
        },
        interruption: {
          barge_in_state: "protected",
          last_decision: "none",
          headphones_recommended: false,
        },
        speech: {
          director_mode: "unavailable",
          first_subtitle_latency_ms: 0,
          first_audio_latency_ms: 0,
          speech_queue_depth_current: 0,
          stale_chunk_drop_count: 0,
        },
        active_listening: {
          available: false,
          phase: "idle",
          partial_available: false,
          partial_transcript_chars: 0,
          final_transcript_chars: 0,
          turn_duration_ms: 0,
          topics: [],
          constraints: [],
          reason_codes: ["active_listening_fetch_failed"],
        },
        camera_presence: {
          enabled: false,
          available: false,
          state: "disabled",
          camera_fresh: false,
          track_state: "unknown",
          latest_frame_seq: 0,
          latest_frame_age_ms: null,
          current_answer_used_vision: false,
          grounding_mode: "none",
          last_vision_result_state: "none",
          reason_codes: ["camera_presence_fetch_failed"],
        },
        vision: {
          enabled: false,
          continuous_perception_enabled: false,
        },
        reason_codes: ["performance_state_fetch_failed"],
      }),
      fetchJson(eventEndpoint, {
        latest_event_id: afterId,
        events: [],
        reason_codes: ["performance_events_fetch_failed"],
      }),
      actorSurfaceV2Enabled() ? fetchJson(actorStateEndpoint, actorStateFallback) : actorStateFallback,
      actorSurfaceV2Enabled()
        ? fetchJson(actorEventEndpoint, {
            latest_event_id: actorAfterId,
            events: [],
            reason_codes: ["actor_events_fetch_failed"],
          })
        : {
            latest_event_id: actorAfterId,
            events: [],
            reason_codes: ["actor_events_disabled"],
          },
    ]);
    state.performance = performance;
    state.actor = actor;
    const events = Array.isArray(eventsPayload.events) ? eventsPayload.events : [];
    if (events.length > 0) {
      state.performanceEvents = state.performanceEvents.concat(events).slice(-6);
    }
    const latestEventId = Number(eventsPayload.latest_event_id || performance.last_event_id || afterId);
    if (Number.isFinite(latestEventId)) {
      state.latestPerformanceEventId = Math.max(state.latestPerformanceEventId || 0, latestEventId);
    }
    const actorEvents = Array.isArray(actorEventsPayload.events) ? actorEventsPayload.events : [];
    if (actorEvents.length > 0) {
      state.actorEvents = state.actorEvents.concat(actorEvents).slice(-actorEventLimit);
    }
    const latestActorEventId = Number(
      actorEventsPayload.latest_event_id || actor.last_actor_event_id || actorAfterId,
    );
    if (Number.isFinite(latestActorEventId)) {
      state.latestActorEventId = Math.max(state.latestActorEventId || 0, latestActorEventId);
    }
    render();
    } finally {
      refreshState.performanceInFlight = false;
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", refresh, { once: true });
  } else {
    refresh();
  }
  window.setInterval(() => {
    if (!state.collapsed) {
      refresh();
    }
  }, 2500);
  window.setInterval(refreshPerformance, 1000);
})();
