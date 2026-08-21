from __future__ import annotations

from collections import defaultdict, deque

from preprocess.event_pipeline.schema import PropagationSummary, PropagationView, StanceSummary, StanceView
from utils.ids import stable_sort_key


def preprocess_propagation_view(view: PropagationView) -> PropagationView:
    nodes_by_id = {node.post_id: node for node in view.nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for edge in view.edges:
        if edge.parent_post_id in nodes_by_id and edge.child_post_id in nodes_by_id:
            children[edge.parent_post_id].append(edge.child_post_id)

    queue = deque([(view.source_post_id, 0)])
    visited: set[str] = set()
    stable_counter = 0
    while queue:
        node_id, depth = queue.popleft()
        if node_id in visited or node_id not in nodes_by_id:
            continue
        visited.add(node_id)
        node = nodes_by_id[node_id]
        node.depth = depth
        node.stable_order = stable_counter
        stable_counter += 1
        ordered_children = sorted(
            children.get(node_id, []),
            key=lambda cid: stable_sort_key(nodes_by_id[cid].time.time_from_source_sec, cid),
        )
        for child_id in ordered_children:
            queue.append((child_id, depth + 1))

    disconnected_depth = max((node.depth or 0) for node in view.nodes) + 1 if view.nodes else 0
    for node in sorted(view.nodes, key=lambda item: stable_sort_key(item.time.time_from_source_sec, item.post_id)):
        if node.post_id not in visited:
            node.depth = disconnected_depth
            node.stable_order = stable_counter
            stable_counter += 1
            disconnected_depth += 1

    replies = [node for node in view.nodes if not node.is_source]
    observed_secs = [node.time.time_from_source_sec for node in replies if node.time.time_from_source_sec is not None]
    view.summary = PropagationSummary(
        n_posts=len(view.nodes),
        n_replies=len(replies),
        max_depth=max((node.depth or 0) for node in view.nodes) if view.nodes else None,
        n_observed_timestamps=sum(1 for node in view.nodes if node.time.is_observed_time),
        first_reply_time_sec=min(observed_secs) if observed_secs else None,
        n_placeholders=sum(1 for node in view.nodes if node.is_placeholder),
        n_text_unobserved_nodes=sum(1 for node in view.nodes if not node.is_observed_text),
        n_boilerplate_nodes=sum(1 for node in view.nodes if node.text_status == "boilerplate"),
    )
    view.nodes.sort(key=lambda item: item.stable_order if item.stable_order is not None else 10**12)
    return view


def preprocess_stance_view(view: StanceView, propagation: PropagationView) -> StanceView:
    node_by_id = {node.post_id: node for node in propagation.nodes}
    source_id = view.source_post_id
    for reply in view.reply_items:
        node = node_by_id.get(reply.post_id)
        if node:
            reply.depth = node.depth
            reply.stable_order = node.stable_order
        reply.parent_is_source = reply.parent_post_id == source_id

    view.reply_items.sort(
        key=lambda item: (
            item.time.time_from_source_sec is None,
            item.time.time_from_source_sec if item.time.time_from_source_sec is not None else float("inf"),
            item.stable_order if item.stable_order is not None else 10**12,
            item.post_id,
        )
    )
    view.summary = StanceSummary(
        n_replies_total=len(view.reply_items),
        n_replies_text_observed=sum(1 for item in view.reply_items if item.is_observed_text),
        n_replies_semantically_substantive=sum(1 for item in view.reply_items if item.is_semantically_substantive),
        n_replies_stance_labeled=sum(1 for item in view.reply_items if item.stance_label is not None),
        n_placeholders=sum(1 for item in view.reply_items if item.is_placeholder),
        n_boilerplate_replies=sum(1 for item in view.reply_items if item.is_boilerplate_text),
    )
    return view
