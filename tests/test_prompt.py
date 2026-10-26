from hunkrisk.prompt import guide, messages, prompt_version


def test_prompt_contains_the_rules_but_not_the_guide_preamble():
    assert guide().startswith("## Task")
    system = messages("a.py", "@@ -1 +1 @@\n+x")[0]["content"]
    assert "## Categories" in system
    assert "source for the teacher prompt" not in system


def test_user_message_is_path_then_hunk():
    assert messages("a.py", "@@ -1 +1 @@\n+x")[1]["content"] == "a.py\n@@ -1 +1 @@\n+x"


def test_prompt_version_is_stable():
    assert prompt_version() == prompt_version()
    assert len(prompt_version()) == 12
