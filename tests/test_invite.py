import pytest

from companion.invite import PREFIX, Invite, looks_like_invite, make_invite, parse_invite


def test_an_invitation_round_trips_with_and_without_a_code():
    assert parse_invite(make_invite("https://relay.example.com", "123456")) == Invite("https://relay.example.com", "123456")
    assert parse_invite(make_invite("https://relay.example.com")) == Invite("https://relay.example.com", None)


def test_the_address_is_not_readable_in_the_invitation():
    assert "relay.example.com" not in make_invite("https://relay.example.com", "123456")


def test_surrounding_whitespace_from_pasting_is_ignored():
    assert parse_invite("  " + make_invite("https://relay.example.com", "1") + "\n").code == "1"


@pytest.mark.parametrize("text", ["", "123456", "AFK1", "AFK1.", "AFK2.aHR0cHM6Ly94", "AFK1.@@@", "AFK1.aaa.bbb.ccc"])
def test_anything_else_is_refused(text):
    with pytest.raises(ValueError):
        parse_invite(text)


def test_only_secure_addresses_are_accepted_except_for_local_development():
    with pytest.raises(ValueError):
        make_invite("http://relay.example.com")
    assert make_invite("http://localhost:8787")
    assert make_invite("http://127.0.0.1:8787")


def test_an_address_with_embedded_credentials_is_refused():
    with pytest.raises(ValueError):
        make_invite("https://user:pass@relay.example.com")


def test_looks_like_invite():
    assert looks_like_invite(make_invite("https://relay.example.com"))
    assert not looks_like_invite("123456")
    assert PREFIX == "AFK1"
