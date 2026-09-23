"""How the steps back and forward go through the tracks selected so far."""

from hessdalen.dashboard.track_history import History, stepped, visited


def test_a_history_of_nothing_stands_on_no_track() -> None:
    empty = History()

    assert empty.standing is None
    assert not empty.behind
    assert not empty.ahead


def test_the_history_stands_on_the_track_last_selected() -> None:
    history = visited(visited(History(), key="a"), key="b")

    assert history.standing == "b"
    assert history.behind
    assert not history.ahead


def test_a_step_back_reaches_the_track_selected_before() -> None:
    history = stepped(visited(visited(History(), key="a"), key="b"), offset=-1)

    assert history.standing == "a"
    assert history.ahead
    assert not history.behind


def test_a_step_forward_returns_to_the_track_stepped_back_from() -> None:
    history = visited(visited(History(), key="a"), key="b")

    assert stepped(stepped(history, offset=-1), offset=1).standing == "b"


def test_a_step_past_either_end_holds_where_it_is() -> None:
    history = visited(visited(History(), key="a"), key="b")

    assert stepped(history, offset=5).standing == "b"
    assert stepped(history, offset=-5).standing == "a"
    assert stepped(History(), offset=-1) == History()


def test_selecting_the_track_stood_on_leaves_the_history_as_it_is() -> None:
    history = visited(visited(History(), key="a"), key="b")

    assert visited(history, key="b") == history


def test_a_selection_after_a_step_back_drops_what_stood_ahead() -> None:
    """The step forward is about tracks already seen, so a track selected from
    the map takes the place of whatever was ahead."""
    history = stepped(visited(visited(History(), key="a"), key="b"), offset=-1)

    carried = visited(history, key="c")

    assert carried.keys == ("a", "c")
    assert carried.standing == "c"
    assert not carried.ahead


def test_a_track_selected_again_is_kept_where_it_was_selected_last() -> None:
    history = visited(visited(visited(History(), key="a"), key="b"), key="a")

    assert history.keys == ("a", "b", "a")
    assert stepped(history, offset=-1).standing == "b"
