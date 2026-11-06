from __future__ import annotations


_INTRO = """Analyze the pose and composition in the supplied image. Use the automatic joint description and viewpoint hints only as unverified starting points.

Inspect the image independently before writing the answer. Check every automatic claim against what is visible. Correct or discard claims that do not match the image; do not simply copy or paraphrase them."""

_SUPPLEMENTARY_GUIDANCE = """The user supplementary pose note may be written in any language. Treat each concrete detail in it as a separate pose constraint. Preserve the specified finger, its action and contact point, the visibility or position of other named fingers, hand shape, body posture, and head-tilt direction. For example, do not compress "the index finger touches the cheek and the thumb is visible" into "the hand touches the cheek." Include finger details in the corresponding arm line and state the contact in the interactions line. Use the note to resolve details that are difficult to read in a simplified drawing. If the note clearly contradicts what is visible in the image, describe the visible image instead."""

_DETAIL_GUIDANCE = """Determine hand shapes, individual finger positions and actions, contact points, and head-tilt direction from the image when they are visible. Treat each `image-left arm` and `image-right arm` line as a description of the entire limb. Each of those two lines must use the internal structure `arm=...; hand=...; fingers=...`. In `fingers=`, name every identifiable finger separately and preserve its position, bend, extension, visibility, action, and contact point. A generic phrase such as `hand touching cheek`, `fist`, or `open hand` is not a substitute for the named finger details. Use `fingers=none visible` only when no fingers are visible or supported, and `fingers=unclear` only when they genuinely cannot be determined. Do not replace a specific visible action with a vague location: for example, if an index finger touches the cheek, state that exact index-finger contact rather than saying only that the hand is near the cheek. Replace an incorrect automatic claim rather than including both the incorrect claim and its correction."""

_OUTPUT_LABELS = """pose summary:
body view:
head view:
head tilt:
image-left arm:
image-right arm:
image-left leg:
image-right leg:
interactions:
framing:
camera angle:
figure scale:
screen position:
visible crop:"""

_CONTENT_SCOPE = """Describe only pose and composition. Do not add character design, clothing, identity, scenery, emotion, or facial-expression information. Use `none visible` when a body part is not shown and `unclear` when a detail cannot be determined."""


def build_pose_analysis_prompt(
    automatic_joint_description: str,
    pose_and_viewpoint_hints: str = "",
    user_supplementary_pose_note: str = "",
) -> str:
    """Build the complete VL prompt while preserving all supplied source text."""
    automatic = automatic_joint_description
    hints = pose_and_viewpoint_hints
    supplementary = user_supplementary_pose_note

    if not automatic.strip():
        raise ValueError("automatic_joint_description must not be empty")

    has_hints = bool(hints.strip())
    has_supplementary = bool(supplementary.strip())

    sections = [_INTRO]
    if has_supplementary:
        sections.append(_SUPPLEMENTARY_GUIDANCE)

    sections.append(_DETAIL_GUIDANCE)
    support = "the image and the supplementary note" if has_supplementary else "the image"
    sections.append(
        "Image-left and image-right always mean screen sides, not the subject's anatomical sides. "
        "Describe head tilt by which side of the head is higher or lower in the image; do not confuse "
        "tilt with the direction the face is turned. Do not invent hidden limbs or details that "
        f"{support} do not support."
    )
    sections.append(_CONTENT_SCOPE)
    output_instruction = (
        "Return exactly these labeled lines in natural English, with one complete description on "
        "the same line after each label. Format each arm line exactly as `arm=...; hand=...; "
        "fingers=...`, keeping all three fields on that labeled line. Be concise without omitting visible "
        "finger identities, positions, actions, visibility, hand shapes, or contacts."
    )
    if has_supplementary:
        output_instruction += (
            " Preserve every specific finger action, visible finger, hand shape, contact, and body "
            "posture stated in the supplementary note."
        )
    output_instruction += (
        " Do not return JSON, markdown, commentary, coordinates, confidence values, alternatives, "
        "or extra lines. Do not copy awkward phrasing from the automatic description."
    )
    sections.append(output_instruction + "\n\n" + _OUTPUT_LABELS)

    final_check = (
        "Before responding, check that no incorrect automatic claim remains. For each visible hand, "
        "verify that its arm line contains non-omitted `hand=` and `fingers=` fields and that every "
        "supported finger appears explicitly in `fingers=`. An answer that leaves "
        "finger information implied by a generic hand description is incomplete and must be rewritten."
    )
    if has_supplementary:
        final_check += (
            " Also check each concrete detail in the supplementary note separately: the named "
            "finger and its action, other visible fingers, the hand shape and contact, body posture, "
            "and head tilt must not be lost through summarization."
        )
    sections.append(final_check)
    sections.append(f"Automatic joint description:\n{automatic}")

    if has_hints:
        sections.append(f"Pose and viewpoint hints:\n{hints}")
    if has_supplementary:
        sections.append(
            "User supplementary pose note (may be in any language):\n" + supplementary
        )

    return "\n\n".join(sections)
