# Dogfooding prompts

Use these to test both primary paths manually.

## Chinese/Melo/Moondream

```text
我想继续研究 Blink。请先听我说完：我们现在有中文 MeloTTS 加浏览器摄像头，也有英文 Kokoro 加浏览器摄像头。我最担心的是用户说话时它像没在听、打断不自然、记忆和性格不明显。你先不要马上长篇回答，先告诉我你听到了哪些重点。
```

Then show an object to camera and ask:

```text
你现在能看一下我手里拿的是什么吗？如果你没有真正调用摄像头，请不要假装看到了。
```

Interrupt during answer:

```text
等一下，不是这个重点。
```

## English/Kokoro/Moondream

```text
I want to test whether Blink feels alive. Please listen first: the Chinese Melo path and English Kokoro path are equally important, camera grounding must be honest, memory has to visibly change behavior, and I care more about natural interaction than another hidden prompt. Before giving a plan, summarize what you heard.
```

Then show an object to camera and ask:

```text
Can you look at what I am holding? If you did not actually use the camera, say so.
```

Interrupt during answer:

```text
Wait, that is not the part I meant.
```

Score both sessions on state clarity, felt heard, voice pacing, interruption, camera honesty, memory, persona, enjoyment, and not fake-human.
