from __future__ import annotations

from preprocess.event_pipeline.schema import ReplyItem


def preserve_relative_delay_reply(reply: ReplyItem) -> ReplyItem:
    if reply.time.observation_kind == "relative_delay_observed":
        reply.extra["has_relative_delay"] = True
    return reply
