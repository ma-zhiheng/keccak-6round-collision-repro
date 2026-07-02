"""Verify the printed 6-round Keccak[1440,160,6,160] collision."""

from __future__ import annotations

from paper_collisions import TABLE18_KECCAK_1440_160_6_160, verify_collision


def main() -> None:
    ok, details = verify_collision(
        TABLE18_KECCAK_1440_160_6_160,
        rounds=6,
        digest_bits=160,
    )
    print(TABLE18_KECCAK_1440_160_6_160.name)
    print(f"  digest M1: {details['digest1']}")
    print(f"  digest M2: {details['digest2']}")
    print(f"  paper order M1: {details['digest1_paper_order']}")
    print(f"  paper:     {details['paper_digest']}")
    print(f"  collision: {details['digest_equal']}")
    print(f"  matches printed digest: {details['matches_paper_digest']}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
