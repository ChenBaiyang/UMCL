from __future__ import annotations

from preprocess.event_pipeline.schema import ReplyItem


def mark_weibo_reply_semantics(reply: ReplyItem) -> ReplyItem:
    if reply.interaction_type == "repost" and reply.text.status in {"boilerplate", "empty"}:
        reply.is_semantically_substantive = False
        reply.is_boilerplate_text = reply.text.status == "boilerplate"
    elif reply.interaction_type == "quote_repost" and reply.text.status == "substantive":
        reply.is_semantically_substantive = True
    return reply
