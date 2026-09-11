import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import game as G  # noqa: E402


def room(*players):
    r = {"id": "room1", "seats": [], "names": {}, "turn": None}
    for p in players:
        G.seat(r, p, p.title())
    return r


def a_round(r, sender="ann", points=40, rounds=()):
    rnd = G.create_round(r, sender, "clip-1", "The Incredibles", points, rounds=rounds)
    rnd["id"] = "rd1"
    return rnd


class Seating(unittest.TestCase):
    def test_first_player_gets_the_first_turn(self):
        r = room("ann")
        self.assertEqual(r["turn"], "ann")

    def test_rooms_hold_more_than_two(self):
        r = room("ann", "bob", "cal", "dee", "eve")
        self.assertEqual(len(r["seats"]), 5)

    def test_room_cap_is_enforced(self):
        r = room(*[f"p{i}" for i in range(G.MAX_PLAYERS)])
        with self.assertRaises(G.RuleError):
            G.seat(r, "one-too-many")

    def test_seating_is_idempotent_and_updates_the_name(self):
        r = room("ann")
        G.seat(r, "ann", "Annabel")
        self.assertEqual(r["seats"], ["ann"])
        self.assertEqual(r["names"]["ann"], "Annabel")

    def test_cannot_send_alone(self):
        r = room("ann")
        with self.assertRaises(G.RuleError):
            G.create_round(r, "ann", "c", "X", 10, rounds=[])

    def test_leaving_passes_the_turn_on(self):
        r = room("ann", "bob")
        G.leave(r, "ann")
        self.assertEqual(r["seats"], ["bob"])
        self.assertEqual(r["turn"], "bob")


class Turns(unittest.TestCase):
    def test_only_the_player_on_turn_may_send(self):
        r = room("ann", "bob", "cal")
        with self.assertRaises(G.RuleError) as cm:
            G.create_round(r, "bob", "c", "X", 10, rounds=[])
        self.assertIn("Ann", str(cm.exception))

    def test_one_clip_in_play_at_a_time(self):
        r = room("ann", "bob", "cal")
        rnd = a_round(r)
        with self.assertRaises(G.RuleError):
            G.create_round(r, "ann", "c2", "Y", 10, rounds=[rnd])

    def test_turn_rotates_round_robin(self):
        r = room("ann", "bob", "cal")
        for expected in ("bob", "cal", "ann"):
            rnd = a_round(r, sender=r["turn"])
            for p in G.others(r, rnd["sender"]):
                G.submit_guess(rnd, p, "The Incredibles")
            G.close_round(rnd)
            G.advance_turn(r, rnd)
            self.assertEqual(r["turn"], expected)


class IndependentGuessing(unittest.TestCase):
    def setUp(self):
        self.r = room("ann", "bob", "cal", "dee")
        self.rnd = a_round(self.r, points=40)

    def test_everyone_but_the_sender_is_a_guesser(self):
        self.assertEqual(set(self.rnd["players"]), {"bob", "cal", "dee"})

    def test_the_sender_cannot_guess_their_own_clip(self):
        with self.assertRaises(G.RuleError):
            G.submit_guess(self.rnd, "ann", "The Incredibles")

    def test_one_person_solving_does_not_end_it_for_others(self):
        G.submit_guess(self.rnd, "bob", "The Incredibles")
        G.close_round(self.rnd)
        self.assertEqual(self.rnd["state"], G.OPEN)
        self.assertEqual(self.rnd["players"]["cal"]["status"], G.WAITING)

    def test_each_guesser_banks_their_own_points(self):
        G.submit_guess(self.rnd, "bob", "The Incredibles")
        G.submit_guess(self.rnd, "cal", "The Incredibles")
        totals = G.scoreboard(self.r["seats"], [self.rnd])
        self.assertEqual(totals["bob"], 40)
        self.assertEqual(totals["cal"], 40)
        self.assertEqual(totals["dee"], 0)
        self.assertEqual(totals["ann"], 0)

    def test_round_closes_when_everyone_is_done(self):
        G.submit_guess(self.rnd, "bob", "The Incredibles")
        G.submit_guess(self.rnd, "cal", "The Incredibles")
        G.reveal(self.rnd, "dee")
        G.close_round(self.rnd)
        self.assertEqual(self.rnd["state"], G.CLOSED)

    def test_sender_can_end_it_early_and_stragglers_score_nothing(self):
        G.submit_guess(self.rnd, "bob", "The Incredibles")
        G.close_round(self.rnd, by="ann")
        self.assertEqual(self.rnd["state"], G.CLOSED)
        self.assertEqual(G.scoreboard(self.r["seats"], [self.rnd])["cal"], 0)

    def test_a_guesser_cannot_end_the_round(self):
        with self.assertRaises(G.RuleError):
            G.close_round(self.rnd, by="bob")

    def test_no_double_guessing_after_solving(self):
        G.submit_guess(self.rnd, "bob", "The Incredibles")
        with self.assertRaises(G.RuleError):
            G.submit_guess(self.rnd, "bob", "The Incredibles")


class Hints(unittest.TestCase):
    def setUp(self):
        self.r = room("ann", "bob", "cal")
        self.rnd = a_round(self.r, points=80)

    def test_asking_costs_nothing(self):
        G.request_hint(self.rnd, "bob")
        self.assertEqual(G.reward_now(self.rnd), 80)

    def test_delivery_costs_25_percent(self):
        G.request_hint(self.rnd, "bob")
        G.deliver_hint(self.rnd, "ann", "hint-1")
        self.assertEqual(G.reward_now(self.rnd), 60)

    def test_two_hints_compound(self):
        for i in range(2):
            G.request_hint(self.rnd, "bob")
            G.deliver_hint(self.rnd, "ann", f"h{i}")
        self.assertEqual(G.reward_now(self.rnd), 45)

    def test_a_hint_does_not_claw_back_points_already_banked(self):
        G.submit_guess(self.rnd, "bob", "The Incredibles")   # banks 80
        G.request_hint(self.rnd, "cal")
        G.deliver_hint(self.rnd, "ann", "h")
        G.submit_guess(self.rnd, "cal", "The Incredibles")   # banks 60
        totals = G.scoreboard(self.r["seats"], [self.rnd])
        self.assertEqual(totals["bob"], 80)
        self.assertEqual(totals["cal"], 60)

    def test_anyone_still_guessing_may_ask(self):
        G.request_hint(self.rnd, "cal")
        self.assertIsNotNone(G.pending_hint(self.rnd))

    def test_cannot_stack_requests(self):
        G.request_hint(self.rnd, "bob")
        with self.assertRaises(G.RuleError):
            G.request_hint(self.rnd, "cal")

    def test_only_the_sender_delivers_and_only_audio(self):
        G.request_hint(self.rnd, "bob")
        with self.assertRaises(G.RuleError):
            G.deliver_hint(self.rnd, "bob", "h")
        with self.assertRaises(G.RuleError):
            G.deliver_hint(self.rnd, "ann", "")

    def test_reward_floor_is_one(self):
        rnd = a_round(self.r, points=2, rounds=[])
        for i in range(20):
            G.request_hint(rnd, "bob")
            G.deliver_hint(rnd, "ann", f"h{i}")
        self.assertEqual(G.reward_now(rnd), 1)

    def test_someone_who_gave_up_cannot_ask_for_a_hint(self):
        G.reveal(self.rnd, "bob")
        with self.assertRaises(G.RuleError):
            G.request_hint(self.rnd, "bob")


class TitleMatching(unittest.TestCase):
    def test_forgiving_matches(self):
        for guess, answer in [
            ("the office", "The Office"),
            ("office", "The Office"),
            ("The Office US", "The Office"),
            ("monsters inc", "Monsters, Inc."),
            ("monsters, inc. (2001)", "Monsters Inc"),
            ("road to el dorado", "The Road to El Dorado"),
            ("  THE   INCREDIBLES  ", "The Incredibles"),
        ]:
            self.assertTrue(G.titles_match(guess, answer), f"{guess!r}/{answer!r}")

    def test_rejects_different_titles(self):
        for guess, answer in [
            ("Finding Nemo", "Finding Dory"),
            ("Up", "The Incredibles"),
            ("", "The Office"),
            ("up", "Upgrade"),
        ]:
            self.assertFalse(G.titles_match(guess, answer), f"{guess!r}/{answer!r}")


class Disputes(unittest.TestCase):
    def setUp(self):
        self.r = room("ann", "bob", "cal")
        self.rnd = a_round(self.r, points=50)

    def test_sender_can_allow_a_rejected_guess(self):
        G.submit_guess(self.rnd, "bob", "that Pixar superhero one")
        G.dispute(self.rnd, "bob")
        G.resolve_dispute(self.rnd, "ann", "bob", True)
        self.assertEqual(self.rnd["players"]["bob"]["status"], G.SOLVED)
        self.assertEqual(self.rnd["players"]["bob"]["awarded"], 50)

    def test_sender_can_refuse_and_that_player_keeps_guessing(self):
        G.submit_guess(self.rnd, "bob", "Shrek")
        G.dispute(self.rnd, "bob")
        G.resolve_dispute(self.rnd, "ann", "bob", False)
        self.assertEqual(self.rnd["players"]["bob"]["status"], G.WAITING)
        self.assertFalse(self.rnd["players"]["bob"]["disputed"])
        G.submit_guess(self.rnd, "bob", "The Incredibles")
        self.assertEqual(self.rnd["players"]["bob"]["status"], G.SOLVED)

    def test_a_guesser_cannot_resolve_their_own_dispute(self):
        G.submit_guess(self.rnd, "bob", "Shrek")
        G.dispute(self.rnd, "bob")
        with self.assertRaises(G.RuleError):
            G.resolve_dispute(self.rnd, "bob", "bob", True)

    def test_a_third_player_cannot_resolve_it_either(self):
        G.submit_guess(self.rnd, "bob", "Shrek")
        G.dispute(self.rnd, "bob")
        with self.assertRaises(G.RuleError):
            G.resolve_dispute(self.rnd, "cal", "bob", True)


class Privacy(unittest.TestCase):
    """A player who has not solved it gets audio and points, nothing else."""

    def setUp(self):
        self.r = room("ann", "bob", "cal")
        self.rnd = a_round(self.r, points=40)
        # Fields a careless implementation might attach to a round.
        self.rnd["search_phrase"] = "you were this close to losing your job"
        self.rnd["transcript"] = "You were THIS close to losing your job"
        self.rnd["source_url"] = "https://coub.com/view/5cru0"
        self.rnd["source_page_title"] = "The Incredibles: You were this close..."
        self.rnd["source_filename"] = "the-incredibles-huph.mp3"
        self.names = self.r["names"]

    def test_guesser_view_hides_the_answer(self):
        self.assertNotIn("answer", G.round_view(self.rnd, "bob", self.names))

    def test_guesser_view_leaks_nothing_identifying(self):
        blob = repr(G.round_view(self.rnd, "bob", self.names)).lower()
        for leak in ("coub", "incredibles", "losing your job", "transcript",
                     ".mp3", "http", "source"):
            self.assertNotIn(leak, blob, f"leaked {leak!r}")

    def test_sender_sees_the_answer(self):
        self.assertEqual(
            G.round_view(self.rnd, "ann", self.names)["answer"], "The Incredibles"
        )

    def test_solver_sees_the_answer_immediately(self):
        G.submit_guess(self.rnd, "bob", "The Incredibles")
        self.assertEqual(
            G.round_view(self.rnd, "bob", self.names)["answer"], "The Incredibles"
        )

    def test_a_solver_does_not_leak_the_answer_to_everyone_else(self):
        G.submit_guess(self.rnd, "bob", "The Incredibles")
        self.assertNotIn("answer", G.round_view(self.rnd, "cal", self.names))

    def test_giving_up_reveals_it_to_that_player_only(self):
        G.reveal(self.rnd, "cal")
        self.assertIn("answer", G.round_view(self.rnd, "cal", self.names))
        self.assertNotIn("answer", G.round_view(self.rnd, "bob", self.names))

    def test_closing_reveals_it_to_the_room(self):
        G.close_round(self.rnd, by="ann")
        for who in ("bob", "cal"):
            self.assertIn("answer", G.round_view(self.rnd, who, self.names))

    def test_other_players_guess_text_is_not_shown(self):
        G.submit_guess(self.rnd, "bob", "Shrek Two Electric Boogaloo")
        view = G.round_view(self.rnd, "cal", self.names)
        self.assertNotIn("boogaloo", repr(view).lower())
        # ...but cal can see that bob is still going.
        statuses = {t["name"]: t["status"] for t in view["table"]}
        self.assertEqual(statuses["Bob"], G.WAITING)

    def test_the_sender_sees_disputed_guesses(self):
        G.submit_guess(self.rnd, "bob", "Shrek")
        G.dispute(self.rnd, "bob")
        view = G.round_view(self.rnd, "ann", self.names)
        self.assertEqual(view["disputes"][0]["guess"], "Shrek")

    def test_you_see_your_own_guesses(self):
        G.submit_guess(self.rnd, "bob", "Shrek")
        self.assertEqual(
            G.round_view(self.rnd, "bob", self.names)["your_guesses"][0]["text"], "Shrek"
        )

    def test_no_quote_field_exists_on_a_round(self):
        self.assertNotIn("quote", a_round(room("x", "y"), sender="x"))


class History(unittest.TestCase):
    def test_play_history_counts_sources(self):
        h = G.play_history([{"answer": "The Office"}, {"answer": "Monsters, Inc."}])
        self.assertEqual(h["office"], 1)
        self.assertEqual(h["monsters inc"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
