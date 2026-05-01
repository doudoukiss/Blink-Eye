(function () {
  const trackedSignatures = new WeakMap();
  const trackedElements = new WeakSet();
  const MEDIA_SELECTOR = "audio[autoplay], video[autoplay]";

  function getTrackSignature(element) {
    const stream = element.srcObject;
    if (!stream || typeof stream.getTracks !== "function") {
      return "";
    }
    return stream
      .getTracks()
      .map((track) => `${track.kind}:${track.id}:${track.readyState}`)
      .join("|");
  }

  function attemptPlayback(element) {
    if (!(element instanceof HTMLMediaElement)) {
      return;
    }

    const signature = getTrackSignature(element);
    if (!signature) {
      trackedSignatures.delete(element);
      return;
    }

    if (trackedSignatures.get(element) === signature && !element.paused) {
      return;
    }

    trackedSignatures.set(element, signature);

    try {
      const playback = element.play();
      if (playback && typeof playback.catch === "function") {
        playback.catch(() => {});
      }
    } catch (_error) {
      // Ignore autoplay rejections and wait for the next readiness event.
    }
  }

  function attachMediaHooks(element) {
    if (!(element instanceof HTMLMediaElement) || trackedElements.has(element)) {
      return;
    }

    trackedElements.add(element);
    element.addEventListener("loadedmetadata", () => attemptPlayback(element));
    element.addEventListener("canplay", () => attemptPlayback(element));
    element.addEventListener("play", () =>
      trackedSignatures.set(element, getTrackSignature(element))
    );
    attemptPlayback(element);
  }

  function scanMediaElements() {
    document.querySelectorAll(MEDIA_SELECTOR).forEach((element) => attachMediaHooks(element));
    document.querySelectorAll(MEDIA_SELECTOR).forEach((element) => attemptPlayback(element));
  }

  const observer = new MutationObserver(() => scanMediaElements());
  observer.observe(document.documentElement, { childList: true, subtree: true });

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      scanMediaElements();
    }
  });

  window.addEventListener("pageshow", () => scanMediaElements());
  window.setInterval(scanMediaElements, 1000);
  scanMediaElements();
})();
