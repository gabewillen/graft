import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from isolation import fisher_two_sided, score  # noqa: E402

from graft.skills import World  # noqa: E402


def deck_world(title, slides, tone="professional, warm"):
    w = World()
    w.execute("slides_create", {"title": title, "audience": "Retirees in Columbus, Ohio", "tone": tone})
    for t, b in slides:
        w.execute("slides_add_slide", {"title": t, "bullets": b})
    return w


GOOD = [("Comfortable Cleanings", ["Gentle care.", "Thorough checks.", "Healthy gums."]),
        ("Dental Implants", ["Durable.", "Natural look.", "Restores function."]),
        ("Teeth Whitening", ["Brighter smile.", "Safe.", "Fast."])]


def test_tone_word_is_not_required():
    r = score(deck_world("Lakeview Dental Open House", GOOD), {"loaded": ["slides"]})
    assert r["correct"] and not r["old_formal_tone_check"]  # the old scorer would have failed this


def test_leakage_types_detected_separately():
    bad = [("comfy cleanings 🦷", ["SUPER gentle!", "#SmileBright", "Ready for a fresh smile?"]),
           ("Implants", ["Awesome results.", "Like our croissants.", "Ok."]),
           ("Whitening", ["a", "b", "c"])]
    w = deck_world("Open House", bad)
    w.execute("site_write", {"page": "x", "content": "y"})
    leaks = score(w, {"loaded": []})["leaks"]
    for k in ("emoji", "exclamation", "hashtag", "all_caps_words", "slang_casual", "question_to_reader",
              "lowercase_titles", "history_bleed", "foreign_tools"):
        assert leaks[k], k


def test_fisher():
    assert abs(fisher_two_sided(10, 20, 2, 20) - 0.0138) < 1e-3
    assert fisher_two_sided(5, 5, 5, 5) == 1.0


def test_world_tool_with_missing_args_is_an_error_not_a_crash():
    from graft.skills import World

    content, err = World().execute("site_write", {"page": "home"})
    assert err and "invalid arguments" in content
