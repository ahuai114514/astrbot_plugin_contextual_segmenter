from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
import astrbot.api.message_components as Comp
from astrbot.api.star import Context, Star, register

PLUGIN_ID = "astrbot_plugin_contextual_segmenter"

DEFAULT_INPUT_DISABLE_KEYWORDS = [
    "作文",
    "写作",
    "简历",
    "个人介绍",
    "自我介绍",
    "个人简介",
    "怎么写",
    "结合教材",
    "展开讨论",
    "谈你的理解",
    "论述",
    "内在联系",
    "为什么说",
]

STRUCTURED_OUTPUT_KEYWORDS = (
    "id",
    "uid",
    "umo",
    "bot",
    "session",
    "stars",
    "forks",
    "issues",
    "watchers",
    "project",
    "简介",
    "核心指标",
)

ESSAY_HINTS = (
    "作文",
    "写一篇",
    "文章",
    "散文",
    "议论文",
    "说明文",
    "记叙文",
    "读后感",
    "观后感",
    "总结",
    "报告",
    "检讨",
    "致辞",
    "演讲稿",
    "长文",
)

PROTECTED_PATTERNS = (
    r"```[\s\S]*?```",
    r"`[^`]+`",
    r"^>\s+",
    r"^\s*[-*+]\s+",
    r"^\s*\d+[.)、]\s+",
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*")


@dataclass
class SegmentDecision:
    should_segment: bool
    paragraph_count: int


@register(PLUGIN_ID, "Codex", "根据语境智能控制回复分条", "1.0.0")
class ContextualSegmenterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config
        self._pending_segments: dict[str, list[str]] = {}

    @filter.on_decorating_result(priority=-100)
    async def segment_result(self, event: AstrMessageEvent) -> None:
        result = event.get_result()
        if result is None or not getattr(result, "chain", None):
            return

        cfg = self._config()
        if not cfg.get("enabled", True):
            return

        input_text = getattr(event, "message_str", "") or ""

        plain_components = [component for component in result.chain if isinstance(component, Comp.Plain)]
        non_plain_components = [component for component in result.chain if not isinstance(component, Comp.Plain)]
        if non_plain_components or len(plain_components) != 1:
            return

        component = plain_components[0]
        segments = self._segment_text(component.text, input_text, cfg)
        if len(segments) <= 1:
            component.text = segments[0] if segments else component.text
            return

        component.text = segments[0]
        pending_key = self._event_key(event)
        if pending_key:
            self._pending_segments[pending_key] = segments[1:]

    @filter.after_message_sent()
    async def send_follow_up_segments(self, event: AstrMessageEvent) -> None:
        pending_key = self._event_key(event)
        if not pending_key:
            return

        segments = self._pending_segments.pop(pending_key, None)
        if not segments:
            return

        delay = self._send_delay_seconds()
        for index, segment in enumerate(segments):
            if delay > 0 and index > 0:
                await asyncio.sleep(delay)
            try:
                await event.send(event.plain_result(segment))
            except Exception as exc:
                logger.warning("[%s] failed to send follow-up segment: %s", PLUGIN_ID, exc)
                break

    def _segment_text(self, text: str, input_text: str, cfg: dict[str, Any]) -> list[str]:
        original = text
        text = text.strip()
        if not text:
            return [original]

        if self._matches_keywords(
            input_text,
            cfg.get("input_disable_keywords", DEFAULT_INPUT_DISABLE_KEYWORDS),
        ):
            return [original]

        if self._should_keep_as_is(text, cfg):
            return [original]

        forced = self._split_by_existing_newlines(text, cfg)
        if forced is not None:
            return forced

        decision = self._decide(text, cfg)
        if decision.should_segment and decision.paragraph_count > 1:
            chunks = self._build_chunks(text, decision.paragraph_count)
            return chunks if chunks else [original]

        return [original]

    def _decide(self, text: str, cfg: dict[str, Any]) -> SegmentDecision:
        length = len(text)
        min_length = int(cfg.get("min_length_to_segment", 20))
        max_paragraphs = max(1, int(cfg.get("max_paragraphs", 4)))

        if length < min_length:
            return SegmentDecision(False, 1)

        score = 0
        score += 1 if length >= 120 else 0
        score += 1 if length >= 220 else 0
        score += 1 if length >= 360 else 0
        score += min(2, self._sentence_count(text) // 4)
        score += 1 if self._has_topic_shift(text) else 0
        score += 1 if self._has_enum_structure(text) else 0
        score -= 2 if self._looks_like_essay(text) else 0
        score -= 1 if self._has_dense_linebreaks(text) else 0

        if score <= 1:
            return SegmentDecision(False, 1)

        paragraph_count = min(max_paragraphs, max(2, min(4, score)))
        return SegmentDecision(True, paragraph_count)

    def _build_chunks(self, text: str, paragraph_count: int) -> list[str]:
        sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
        if len(sentences) <= paragraph_count:
            return sentences

        total = len(sentences)
        base = total // paragraph_count
        extra = total % paragraph_count
        chunks: list[str] = []
        index = 0
        for i in range(paragraph_count):
            size = base + (1 if i < extra else 0)
            if size <= 0:
                continue
            chunk = "".join(sentences[index : index + size]).strip()
            if chunk:
                chunks.append(chunk)
            index += size
        if index < total:
            tail = "".join(sentences[index:]).strip()
            if tail:
                chunks.append(tail)
        return chunks

    def _should_keep_as_is(self, text: str, cfg: dict[str, Any]) -> bool:
        if self._matches_keywords(text, cfg.get("output_disable_keywords", [])):
            return True
        if self._is_structured_multiline_output(text, cfg):
            return True
        if self._looks_like_essay(text):
            return True
        for pattern in PROTECTED_PATTERNS:
            if re.search(pattern, text, re.M):
                return True
        return False

    def _split_by_existing_newlines(self, text: str, cfg: dict[str, Any]) -> list[str] | None:
        force_newline = bool(cfg.get("force_segment_on_newline", True))
        if force_newline and "\n" in text:
            parts = [part.strip() for part in text.splitlines() if part.strip()]
            return parts if len(parts) >= 2 else None

        return None

    def _matches_keywords(self, text: str, keywords: Any) -> bool:
        if not text:
            return False
        if not isinstance(keywords, (list, tuple)):
            return False
        return any(isinstance(keyword, str) and keyword and keyword in text for keyword in keywords)

    def _is_structured_multiline_output(self, text: str, cfg: dict[str, Any]) -> bool:
        if not bool(cfg.get("protect_structured_multiline_output", True)):
            return False

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        min_lines = max(2, int(cfg.get("structured_multiline_min_lines", 5)))
        if len(lines) < min_lines:
            return False

        if self._looks_like_natural_prose(text, lines):
            return False

        kv_count = self._count_key_value_lines(lines)
        short_line_count = sum(1 for line in lines if len(line) <= 40)
        keyword_hits = self._structured_keyword_hits(lines)
        avg_len = sum(len(line) for line in lines) / len(lines)
        max_len = max(len(line) for line in lines)

        if kv_count >= 2 and kv_count / len(lines) >= 0.4:
            return True
        if short_line_count / len(lines) >= 0.7 and keyword_hits >= 2:
            return True
        if short_line_count / len(lines) >= 0.8 and avg_len < 28 and max_len < 80:
            return True
        if self._looks_like_titled_panel(lines) and short_line_count / len(lines) >= 0.6:
            return True
        return False

    def _count_key_value_lines(self, lines: list[str]) -> int:
        return sum(
            1
            for line in lines
            if re.match(r"^[A-Za-z0-9_\-\u4e00-\u9fff\s]{1,30}[:：]\s*.+$", line)
        )

    def _structured_keyword_hits(self, lines: list[str]) -> int:
        text = "\n".join(lines).lower()
        return sum(1 for keyword in STRUCTURED_OUTPUT_KEYWORDS if keyword in text)

    def _looks_like_titled_panel(self, lines: list[str]) -> bool:
        if len(lines) < 2:
            return False
        first = lines[0]
        if first.endswith(":") or first.endswith("："):
            return True
        return "information" in first.lower() or "信息" in first

    def _looks_like_natural_prose(self, text: str, lines: list[str]) -> bool:
        connective_hits = sum(
            1
            for marker in ("首先", "其次", "最后", "另外", "此外", "因此", "总之", "一方面", "另一方面")
            if marker in text
        )
        punctuation_hits = sum(text.count(mark) for mark in ("。", "？", "！", "；", ".", "?", "!"))
        avg_len = sum(len(line) for line in lines) / len(lines)
        return avg_len >= 32 and (connective_hits >= 2 or punctuation_hits >= len(lines))

    def _looks_like_essay(self, text: str) -> bool:
        hint_hits = sum(1 for hint in ESSAY_HINTS if hint in text)
        if hint_hits >= 2:
            return True
        if "第一段" in text or "第二段" in text or "结尾" in text:
            return True
        if "开头" in text and "结尾" in text and len(text) >= 150:
            return True
        return False

    def _has_topic_shift(self, text: str) -> bool:
        markers = ("不过", "另外", "同时", "再说", "总之", "首先", "其次", "最后", "此外", "换句话说")
        return sum(1 for marker in markers if marker in text) >= 2

    def _has_enum_structure(self, text: str) -> bool:
        return bool(re.search(r"(?:\n|^)\s*(?:\d+[.)、]|[-*+])\s+", text))

    def _has_dense_linebreaks(self, text: str) -> bool:
        return text.count("\n") >= 2

    def _sentence_count(self, text: str) -> int:
        parts = [part for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
        return len(parts) or 1

    def _config(self) -> dict[str, Any]:
        if self.config is None:
            return {}
        if hasattr(self.config, "get"):
            return dict(self.config)
        return {}

    def _event_key(self, event: AstrMessageEvent) -> str:
        message_obj = getattr(event, "message_obj", None)
        message_id = getattr(message_obj, "message_id", None)
        if message_id:
            return f"{event.unified_msg_origin}:{message_id}"
        return event.unified_msg_origin

    def _send_delay_seconds(self) -> float:
        cfg = self._config()
        try:
            return max(0.0, float(cfg.get("send_delay_seconds", 0.4)))
        except (TypeError, ValueError):
            return 0.4
