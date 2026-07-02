"""Show the local equations used in the connector construction."""

from __future__ import annotations

from sbox_constraints import demo_transition


def main() -> None:
    print(demo_transition(0x03, 0x02))
    print()
    print(demo_transition(0x01, 0x01))


if __name__ == "__main__":
    main()
