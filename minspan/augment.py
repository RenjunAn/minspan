"""Phase 2 data augmentations: structured envelopes, wrapper templates, and
business-imperative hard negatives.

Envelopes wrap an already-attacked plain-text record into a randomly
generated structured document (JSON, XML, ...). The attack is inserted in
plain text first; the envelope then escapes the payload, so drop spans are
remapped through the format's escape function. Every escape function here is
character-wise (escape(a + b) == escape(a) + escape(b)), which makes the
remapping exact: new_offset = len(before) + len(escape(payload[:offset])).

YAML is deliberately held out of the training pool so the format_test split
can measure generalization to an unseen envelope format.
"""

from __future__ import annotations

import json
import random
import re


_PLACEHOLDER = "@@PAYLOAD@@"

_PAYLOAD_KEYS = ("body", "content", "text", "message", "description", "note")
_STATUSES = ("open", "closed", "pending", "resolved", "sent", "archived")
_PRIORITIES = ("low", "normal", "high")
_SENDERS = (
    "alice@example.com",
    "bob@example.com",
    "carol@northwind.test",
    "dave@contoso.test",
    "erin@fabrikam.test",
)
_NAMES = ("Alice", "Bob", "Carol", "Dave", "Erin", "Frank")
_SUBJECTS = (
    "Quarterly review",
    "Shipping update",
    "Meeting notes",
    "Invoice 4521",
    "Project status",
    "Support ticket",
)


def _hex_id(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(8))


def _timestamp(rng: random.Random) -> str:
    return (
        f"2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"
        f"T{rng.randrange(24):02d}:{rng.randrange(60):02d}:00Z"
    )


# ---------------------------------------------------------------------------
# Envelope renderers: return the full document with _PLACEHOLDER marking the
# payload location exactly once.
# ---------------------------------------------------------------------------

def _render_json(rng: random.Random) -> str:
    pairs = [
        ("id", _hex_id(rng)),
        ("status", rng.choice(_STATUSES)),
        ("timestamp", _timestamp(rng)),
        ("sender", rng.choice(_SENDERS)),
        ("priority", rng.choice(_PRIORITIES)),
    ]
    rng.shuffle(pairs)
    pairs = pairs[: rng.randrange(2, 5)]
    pairs.insert(rng.randrange(len(pairs) + 1), (rng.choice(_PAYLOAD_KEYS), _PLACEHOLDER))
    document: dict = dict(pairs)
    if rng.random() < 0.5:
        document = {rng.choice(("result", "data", "item", "record")): document}
    return json.dumps(document, ensure_ascii=False)


def _render_xml(rng: random.Random) -> str:
    tag = rng.choice(("message", "note", "document", "entry"))
    fields = [
        ("sender", rng.choice(_SENDERS)),
        ("status", rng.choice(_STATUSES)),
        ("timestamp", _timestamp(rng)),
    ]
    rng.shuffle(fields)
    fields = fields[: rng.randrange(1, 4)]
    children = [f"  <{key}>{value}</{key}>" for key, value in fields]
    payload_tag = rng.choice(_PAYLOAD_KEYS)
    children.insert(
        rng.randrange(len(children) + 1),
        f"  <{payload_tag}>{_PLACEHOLDER}</{payload_tag}>",
    )
    return "\n".join(
        [f'<{tag} id="{_hex_id(rng)}">'] + children + [f"</{tag}>"]
    )


def _render_kvlog(rng: random.Random) -> str:
    lines = [
        f"ID: {_hex_id(rng)}",
        f"FROM: {rng.choice(_SENDERS)}",
        f"STATUS: {rng.choice(_STATUSES)}",
    ]
    rng.shuffle(lines)
    lines.append(f"{rng.choice(_PAYLOAD_KEYS).upper()}: {_PLACEHOLDER}")
    return "\n".join(lines)


def _render_markdown(rng: random.Random) -> str:
    return (
        f"# {rng.choice(_SUBJECTS)}\n\n"
        f"**From:** {rng.choice(_NAMES)}  \n"
        f"**Status:** {rng.choice(_STATUSES)}\n\n"
        f"{_PLACEHOLDER}\n\n"
        f"---\n_{rng.choice(_NAMES)}_"
    )


def _render_email(rng: random.Random) -> str:
    return (
        f"From: {rng.choice(_SENDERS)}\n"
        f"To: {rng.choice(_SENDERS)}\n"
        f"Subject: {rng.choice(_SUBJECTS)}\n"
        f"Date: {_timestamp(rng)}\n\n"
        f"{_PLACEHOLDER}\n\n"
        f"--\n{rng.choice(_NAMES)}"
    )


def _render_yaml(rng: random.Random) -> str:
    return (
        f"message_id: {_hex_id(rng)}\n"
        f"status: {rng.choice(_STATUSES)}\n"
        f"{rng.choice(_PAYLOAD_KEYS)}: |\n"
        f"  {_PLACEHOLDER}\n"
        f"labels:\n  - inbox"
    )


# ---------------------------------------------------------------------------
# Character-wise escape functions
# ---------------------------------------------------------------------------

def _escape_json(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_yaml(text: str) -> str:
    return text.replace("\n", "\n  ")


def _identity(text: str) -> str:
    return text


_ENVELOPES = {
    "json": (_render_json, _escape_json),
    "xml": (_render_xml, _escape_xml),
    "kvlog": (_render_kvlog, _identity),
    "markdown": (_render_markdown, _identity),
    "email": (_render_email, _identity),
    "yaml": (_render_yaml, _escape_yaml),
}

ALL_ENVELOPE_FORMATS = tuple(_ENVELOPES)
HELD_OUT_ENVELOPE_FORMATS = ("yaml",)
TRAIN_ENVELOPE_WEIGHTS = {
    "json": 0.50,
    "xml": 0.15,
    "kvlog": 0.15,
    "markdown": 0.10,
    "email": 0.10,
}


def escape_for_envelope(text: str, format_name: str) -> str:
    if format_name not in _ENVELOPES:
        raise ValueError(f"unknown envelope format: {format_name}")
    return _ENVELOPES[format_name][1](text)


def choose_envelope_format(rng: random.Random) -> str:
    return rng.choices(
        list(TRAIN_ENVELOPE_WEIGHTS),
        weights=list(TRAIN_ENVELOPE_WEIGHTS.values()),
        k=1,
    )[0]


def wrap_in_envelope(
    attacked_data: str,
    clean_data: str,
    drop_spans: list[dict[str, int]],
    format_name: str,
    rng: random.Random,
) -> tuple[str, str, list[dict[str, int]]]:
    """Wrap an attacked/clean text pair into one structured envelope and
    remap the drop spans through the format's escape function."""
    if format_name not in _ENVELOPES:
        raise ValueError(f"unknown envelope format: {format_name}")
    render, escape = _ENVELOPES[format_name]
    parts = render(rng).split(_PLACEHOLDER)
    if len(parts) != 2:
        raise ValueError(f"{format_name} renderer must yield exactly one payload slot")
    before, after = parts

    attacked_env = before + escape(attacked_data) + after
    clean_env = before + escape(clean_data) + after
    new_spans = [
        {
            "start": len(before) + len(escape(attacked_data[: span["start"]])),
            "end": len(before) + len(escape(attacked_data[: span["end"]])),
        }
        for span in drop_spans
    ]
    return attacked_env, clean_env, new_spans


# ---------------------------------------------------------------------------
# Wrapper templates around injections
# ---------------------------------------------------------------------------

WRAPPER_TEMPLATES = {
    "train": (
        "[ATTENTION]\n{payload}\n[/ATTENTION]",
        "<<SYSTEM NOTICE>>\n{payload}\n<</SYSTEM NOTICE>>",
        "<!-- ADMIN NOTE: {payload} -->",
        "*** URGENT ***\n{payload}\n**************",
        "=== IMPORTANT UPDATE ===\n{payload}\n========================",
        "BEGIN PRIORITY DIRECTIVE\n{payload}\nEND PRIORITY DIRECTIVE",
        "### ACTION REQUIRED ###\n{payload}\n#######################",
        "[NOTICE] {payload} [/NOTICE]",
    ),
    "test": (
        "!!! READ FIRST !!!\n{payload}\n!!!!!!!!!!!!!!!!!!",
        "<priority-note>\n{payload}\n</priority-note>",
        "--- OFFICIAL NOTICE ---\n{payload}\n-----------------------",
        "~~ DIRECT REQUEST ~~\n{payload}\n~~~~~~~~~~~~~~~~~~~~",
    ),
}


def wrap_injection(injection: str, rng: random.Random, pool: str = "train") -> str:
    return rng.choice(WRAPPER_TEMPLATES[pool]).format(payload=injection)


# ---------------------------------------------------------------------------
# Business-imperative hard negatives: legitimate task-relevant imperatives
# addressed to humans inside the data, never to the model.
# ---------------------------------------------------------------------------

HARD_NEGATIVE_SENTENCES = (
    "Please reply to the customer by Friday at the latest.",
    "Remember to attach the updated invoice before sending.",
    "Make sure the final report reaches the client this week.",
    "Please review the attached contract and flag any concerns.",
    "Don't forget to submit your expense report by end of month.",
    "Schedule the follow-up call with the vendor for next Tuesday.",
    "Please confirm your attendance before Thursday noon.",
    "Send the revised slides to the whole team once approved.",
    "Double-check the shipping address before dispatching the order.",
    "Please escalate this ticket if there is no response within 24 hours.",
    "Update the project tracker after each milestone is completed.",
    "Make sure to back up the shared folder before the migration.",
    "Please forward this summary to the regional managers.",
    "Verify the payment details with finance before processing.",
    "Remember to renew the software license before it expires.",
    "Please book the conference room for the quarterly review.",
    "Submit the draft to legal for review by Wednesday.",
    "Keep the customer informed about the revised delivery date.",
    "Please archive last quarter's reports in the shared drive.",
    "Notify the warehouse team about the change in pickup time.",
    "Make sure every attendee receives the agenda in advance.",
    "Please sign and return the agreement at your earliest convenience.",
    "Check the inventory levels before confirming the bulk order.",
    "Remember to log your hours before the payroll cutoff.",
    "Please coordinate with marketing on the launch announcement.",
    "Ensure the invoices are numbered consecutively this time.",
    "Follow up with the supplier about the delayed shipment.",
    "Please add the new hires to the onboarding mailing list.",
    "Confirm the hotel reservations for the visiting delegation.",
    "Make sure the meeting minutes are circulated within two days.",
    "Please collect feedback from the pilot users by next week.",
    "Update the price list before the promotion goes live.",
    "Remember to rotate the on-call schedule for the holidays.",
    "Please reconcile the accounts before the audit begins.",
    "Share the customer survey results with the product team.",
    "Make sure the demo environment is ready before the call.",
    "Please label all boxes clearly before the office move.",
    "Get approval from the department head before purchasing.",
    "Remember to include the tracking number in the confirmation email.",
    "Please review the budget proposal and add your comments.",
    "Test the backup restore procedure before the weekend.",
    "Notify clients about the upcoming maintenance window.",
    "Please file the signed copies with the originals.",
    "Make sure the press release is proofread twice.",
    "Send a reminder to participants one day before the workshop.",
    "Please tag the urgent orders so the warehouse can prioritize.",
    "Confirm the catering headcount by Tuesday morning.",
    "Remember to update your emergency contact information.",
    "Please return the loaner equipment to the IT desk.",
    "Check that all figures in the deck match the latest forecast.",
    "Make sure the new policy is acknowledged by every team member.",
    "Please submit the timesheets before leaving on Friday.",
    "Invite the stakeholders to the kickoff meeting next Monday.",
    "Remember to lock the storage room after retrieving the samples.",
    "Please consolidate the regional numbers into one spreadsheet.",
    "Verify that the contractor has signed the safety waiver.",
    "Make sure the refund is processed within five business days.",
    "Please print three copies of the agreement for the signing.",
    "Reassign the open tickets before your vacation starts.",
    "Remember to mention the discount code in the newsletter.",
    "Please keep the receipts for anything purchased for the event.",
    "Confirm with the printer that the banners arrive by Thursday.",
    "Make sure the candidate receives directions to the office.",
    "Please brief the night shift about the revised procedure.",
    "Order replacement toner before the current stock runs out.",
    "Remember to congratulate the team on closing the deal.",
    "Please validate the data export before sharing it externally.",
    "Check whether the meeting clashes with the regional holiday.",
    "Make sure the trial accounts are extended for the beta testers.",
    "Please route all media inquiries through communications.",
    "Update the seating chart after the new desks are installed.",
    "Remember to charge the presentation remote before the keynote.",
    "Please confirm the translation is ready before the release.",
    "Collect the signed permission slips before the site visit.",
    "Make sure the vendor badge is returned at the end of the day.",
    "Please summarize the action items at the end of the meeting.",
    "Verify the emergency exits are unobstructed before the inspection.",
    "Remember to water the office plants during the long weekend.",
    "Please cross-check the serial numbers against the manifest.",
)


def _separator(left: str, right: str) -> str:
    if not left or not right or left[-1].isspace() or right[0].isspace():
        return ""
    return " "


def insert_hard_negative(text: str, rng: random.Random) -> str:
    """Insert one benign imperative sentence at a word boundary; the result
    is legitimate content with no drop span."""
    sentence = rng.choice(HARD_NEGATIVE_SENTENCES)
    if not text:
        return sentence
    boundaries = [match.start() for match in re.finditer(r"\S+", text)]
    boundaries = list(dict.fromkeys(boundaries + [len(text)]))
    offset = boundaries[rng.randrange(len(boundaries))]
    left, right = text[:offset], text[offset:]
    return (
        left
        + _separator(left, sentence)
        + sentence
        + _separator(sentence, right)
        + right
    )
