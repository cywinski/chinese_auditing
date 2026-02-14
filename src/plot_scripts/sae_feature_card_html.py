# ABOUTME: Generates a compact Anthropic-style HTML card for a single SAE feature.
# ABOUTME: Displays explanation, max activating examples with token highlights, and translated positive logits.

import asyncio
import json
import os
import re
import webbrowser
from pathlib import Path

import aiohttp
import fire
from dotenv import load_dotenv

from src.openrouter_client import sample_response_chat
from src.sae.prepare_features import contains_chinese

DEFAULT_CACHE_PATH = "output/sae_translation_cache/positive_logits_cache.json"
TRANSLATION_MODEL = "google/gemini-3-flash-preview"


def _load_cache(cache_path: str) -> dict[str, str]:
    path = Path(cache_path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, str], cache_path: str):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


async def _translate_tokens_async(
    tokens: list[str],
    api_key: str,
    model: str = TRANSLATION_MODEL,
) -> dict[str, str]:
    """Translate Chinese tokens to English via OpenRouter chat API."""
    translations = {}
    semaphore = asyncio.Semaphore(10)

    async def translate_one(session, token):
        async with semaphore:
            prompt = (
                f"Translate this Chinese text to English. "
                f"Reply with ONLY the translation, nothing else: {token}"
            )
            try:
                response = await sample_response_chat(
                    session=session,
                    user_message=prompt,
                    model=model,
                    api_key=api_key,
                    temperature=0.0,
                    max_tokens=50,
                )
                content = response["choices"][0]["message"]["content"].strip()
                translations[token] = content
            except Exception as e:
                print(f"  Warning: Translation failed for '{token}': {e}")

    async with aiohttp.ClientSession() as session:
        tasks = [translate_one(session, t) for t in tokens]
        await asyncio.gather(*tasks)

    return translations


def _translate_logits(
    positive_logits: list[dict],
    cache_path: str = DEFAULT_CACHE_PATH,
) -> list[dict]:
    """Fill in translations for Chinese logit tokens using cache, translating missing ones."""
    load_dotenv()
    cache = _load_cache(cache_path)

    tokens_to_translate = list({
        pl["token"]
        for pl in positive_logits
        if contains_chinese(pl["token"]) and pl["token"] not in cache
    })

    if tokens_to_translate:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Warning: OPENROUTER_API_KEY not set, skipping translation")
        else:
            print(f"Translating {len(tokens_to_translate)} new Chinese tokens...")
            new_translations = asyncio.run(
                _translate_tokens_async(tokens_to_translate, api_key)
            )
            cache.update(new_translations)
            _save_cache(cache, cache_path)

    for pl in positive_logits:
        token = pl["token"]
        if contains_chinese(token):
            pl["translation"] = cache.get(token)

    return positive_logits


def main(
    feature_idx: int,
    explanations_path: str = "output/sae_feature_explanations_no_chinese2/feature_explanations.json",
    translation_cache_path: str = DEFAULT_CACHE_PATH,
    output_path: str = None,
    open_browser: bool = False,
):
    """Generate an Anthropic-style HTML card for a single SAE feature."""
    with open(explanations_path) as f:
        data = json.load(f)

    config = data.get("config", {})
    features = data["features"]

    key = str(feature_idx)
    if key not in features:
        print(f"Feature {feature_idx} not found. Available: {list(features.keys())[:10]}...")
        return

    feat = features[key]
    feat["positive_logits"] = _translate_logits(
        feat.get("positive_logits", []), cache_path=translation_cache_path
    )

    if output_path is None:
        output_path = f"output/plots/sae_feature_{feature_idx}.html"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    html = _build_feature_html(feature_idx, feat, config)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved to {output_path}")

    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")


def _parse_examples(examples_str: str) -> list[dict]:
    """Parse the examples_str into structured data.

    Each example has the form:
        N. context<<tok1, tok2, tok3>>context
        Activations: ('tok1', val), ('tok2', val), ...
    """
    examples = []
    # Split into individual examples by numbered prefix
    parts = re.split(r"\n\n(?=\d+\.\s)", examples_str.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Split text and activations line
        act_match = re.search(r"\nActivations:\s*(.+)$", part, re.DOTALL)
        if not act_match:
            continue

        text_part = part[: act_match.start()].strip()
        act_str = act_match.group(1).strip()

        # Remove leading number "N. "
        text_part = re.sub(r"^\d+\.\s*", "", text_part)

        # Parse activations: ('token', value), ...
        activations = {}
        for m in re.finditer(r"\('([^']*)',\s*([\d.]+)\)", act_str):
            token = m.group(1)
            value = float(m.group(2))
            activations[token] = value

        # Parse text into segments: normal text and <<highlighted>> tokens
        segments = []
        last_end = 0
        for m in re.finditer(r"<<(.+?)>>", text_part):
            if m.start() > last_end:
                segments.append({"text": text_part[last_end : m.start()], "highlighted": False})
            # Highlighted tokens are comma-separated inside <<>>
            highlighted_tokens = m.group(1).split(", ")
            for tok in highlighted_tokens:
                act_val = activations.get(tok, 0)
                segments.append({"text": tok, "highlighted": True, "activation": act_val})
            last_end = m.end()
        if last_end < len(text_part):
            segments.append({"text": text_part[last_end:], "highlighted": False})

        # Get max activation for normalizing colors
        max_act = max(activations.values()) if activations else 1.0

        examples.append({
            "segments": segments,
            "activations": activations,
            "max_activation": max_act,
        })

    return examples


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_feature_html(feature_idx: int, feat: dict, config: dict) -> str:
    explanation = feat.get("explanation", "—")
    examples = _parse_examples(feat.get("examples_str", ""))
    positive_logits = feat.get("positive_logits", [])
    layer = config.get("sae_layer", "?")

    # Build examples HTML
    examples_html_parts = []
    for i, ex in enumerate(examples):
        tokens_html = ""
        for seg in ex["segments"]:
            if seg["highlighted"]:
                act = seg.get("activation", 0)
                max_act = ex["max_activation"]
                intensity = act / max_act if max_act > 0 else 0
                # Color from visible orange to deep orange-red
                r = 255
                g = int(200 - 100 * intensity)
                b = int(150 - 120 * intensity)
                a = 0.55 + 0.45 * intensity
                tokens_html += (
                    f'<span class="token-highlight" '
                    f'style="background:rgba({r},{g},{b},{a:.2f})" '
                    f'title="{_esc(seg["text"].strip())}: {act:.1f}">'
                    f"{_esc(seg['text'])}</span>"
                )
            else:
                tokens_html += _esc(seg["text"])

        examples_html_parts.append(
            f'<div class="example">'
            f'<div class="example-idx">{i + 1}</div>'
            f'<div class="example-body">'
            f'<div class="example-text">{tokens_html}</div>'
            f"</div></div>"
        )

    examples_html = "\n".join(examples_html_parts)

    # Build logits HTML as inline token chips
    logit_chips = []
    for pl in positive_logits:
        token = pl["token"]
        translation = pl.get("translation") or ""
        chip = f'<span class="logit-token-text">{_esc(token)}</span>'
        if translation:
            chip += f'<span class="logit-translation">[{_esc(translation)}]</span>'
        logit_chips.append(f'<div class="logit-chip">{chip}</div>')
    logits_html = "\n".join(logit_chips)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Feature {feature_idx}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f7f7f8;
    color: #1a1a1a;
    padding: 16px;
    display: flex;
    justify-content: center;
}}
.card {{
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    max-width: 960px;
    width: 100%;
    overflow: hidden;
}}
.body {{
    display: flex;
}}
.examples-col {{
    flex: 1;
    min-width: 0;
    padding: 10px 20px;
    border-right: 1px solid #e8e8eb;
}}
.logits-col {{
    width: 180px;
    flex-shrink: 0;
    padding: 10px 16px;
}}
.header {{
    padding: 14px 20px 10px;
    border-bottom: 1px solid #e8e8eb;
}}
.header-top {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}}
.feature-badge {{
    font-size: 11px;
    font-weight: 600;
    background: #e8e8eb;
    color: #555;
    padding: 2px 8px;
    border-radius: 5px;
    font-family: 'SF Mono', Menlo, monospace;
}}
.layer-badge {{
    font-size: 11px;
    background: #dbeafe;
    color: #1e40af;
    padding: 2px 8px;
    border-radius: 5px;
    font-weight: 500;
}}
.explanation {{
    font-size: 16px;
    font-weight: 600;
    color: #111;
    line-height: 1.3;
}}
.section-title {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #888;
    margin-bottom: 8px;
}}
.example {{
    display: flex;
    gap: 8px;
    margin-bottom: 6px;
    padding-bottom: 6px;
    border-bottom: 1px solid #f5f5f5;
}}
.example:last-child {{
    margin-bottom: 0;
    padding-bottom: 0;
    border-bottom: none;
}}
.example-idx {{
    font-size: 10px;
    font-weight: 600;
    color: #bbb;
    min-width: 16px;
    padding-top: 1px;
}}
.example-text {{
    font-size: 12px;
    line-height: 1.45;
    color: #333;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    word-break: break-word;
}}
.token-highlight {{
    border-radius: 2px;
    padding: 0 1px;
}}
.logit-chips {{
    display: flex;
    flex-direction: column;
    gap: 4px;
}}
.logit-chip {{
    background: #eef2ff;
    padding: 3px 8px;
    border-radius: 4px;
}}
.logit-token-text {{
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 12px;
    color: #3730a3;
    display: block;
}}
.logit-translation {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #888;
    font-size: 10px;
    display: block;
}}
</style>
</head>
<body>
<div class="card">
    <div class="header">
        <div class="header-top">
            <span class="feature-badge">Feature {feature_idx}</span>
            <span class="layer-badge">Layer {layer}</span>
        </div>
        <div class="explanation">{_esc(explanation)}</div>
    </div>
    <div class="body">
        <div class="examples-col">
            <div class="section-title">Max Activating Examples</div>
            {examples_html}
        </div>
        <div class="logits-col">
            <div class="section-title">Positive Logits</div>
            <div class="logit-chips">{logits_html}</div>
        </div>
    </div>
</div>
</body>
</html>"""


if __name__ == "__main__":
    fire.Fire(main)
