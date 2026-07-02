"""Transcribed collision tables from Qiao et al. 2017.

The final collision tables in the PDF are rendered as images, so entries here
are kept in the same 5x5 lane layout for manual auditing against the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

from keccak_state import rounds_int, squeeze_digest
from trail_parser import parse_state_matrix


@dataclass(frozen=True)
class PaperCollision:
    name: str
    m1: str
    m2: str
    digest_hex: str


TABLE10_KECCAK_1440_160_5_160 = PaperCollision(
    name="Table 10: Collision for Keccak[1440,160,5,160]",
    m1="""
    |C09C5501A913CC3C|7406D907E6569334|89182C870A0387A0|980A9D8F82C40A90|9306194AEBBC1C17|
    |6D7DFE249ED35BB5|35C1981BFF84755C|37E7FA11AAD390EB|19485675C7530B8E|042893444D9EC364|
    |6D317B9B40DE874C|E2EC2A3613678DDA|3939A7F72AC29BF6|4FABBC80AE5192EA|AB50ABCBC7E5CC7|
    |0F152006D01F65AC|AEC5B4B7EEC068E1|58E287388571520A|569ED102CB7D2EFA|4AC1C2A0645D5B2C|
    |4C323DADBB2DAFC4|36F6BEEB558F2B22|0000000089F71BE8|0000000000000000|0000000000000000|
    """,
    m2="""
    |D634EAE0EF26F002|90371C35B5CFABC|7396C3D058D2F577|78CDF403D882B742|22ECA6BCBFC9501F|
    |2352A9667EB05FCD|4CA3FD90EFB8A2D3|8DDFF276C0B60599|4B4CCD54AD6B2646|A490FAFA55BF4E37|
    |234734EA58D9191D|3C580CA9664107ED|29E6AEB01815FB08|8FB33829BABDF8C2|48A21B6E764A7987|
    |D9FA24DCB0331C80|9272D67CEF52F8E3|0C82810B4BE7307A|CF164B325F4DEEBE|BA41517B4D315C3C|
    |99CD68FF39016FC4|AB018238479D9A8D|00000000E3233895|0000000000000000|0000000000000000|
    """,
    digest_hex="A6E173DCDFC3E8EF8242EAEA1EE736D5E33875A0",
)

TABLE18_KECCAK_1440_160_6_160 = PaperCollision(
    name="Table 18: Collision for Keccak[1440,160,6,160]",
    m1="""
    |DA27ABE5B7EC359D|328A2AB4CD0E256A|00DBDEECA184390E|3843F66481C745F4|DDF83BEF39D4F594|
    |46BA2A960272C97A|8CC8CE3E13185558|2D7C6CC662546532|4D8DCDC25DC7F4B8|574252F43F85BF94|
    |BDCFA2D6B04CBDEE|208D7A02168A7596|AFE7C652F0A68792|467C04748D85916F|F1BFEAF63C4B97C3|
    |C2B0AAEA35887CD4|72A3D23F9D84434D|97A5D9A090590B61|BBE1EC62DBD4327E|64284BCB9BE462C5|
    |8843CBC8B55E106A|DD3DD96A1AC48100|00000000E9151D67|0000000000000000|0000000000000000|
    """,
    m2="""
    |5A0C640730278910|32C1A7D724790C0B|8BCE75C46404A83A|7FCE23E92ECE7E31|1BEE08F9F932C785|
    |3969BA55EB6B17F9|E82948B06C21C6A8|AF42ACEF22202C1F|A9C1BD90BF96FB60|0F98E27C36B57BDA|
    |A02B26453D88C70F|5EC5F74DC919C7E6|31391D7A23A3C8DD|C0BECDAD0AC7F275|14FA28F6B2C9D390|
    |69F67EEAEF258217|159B7FEDCED37178|DA89C2B0291CCA7D|7BDDE79F989414AE|3088CBE192E15B4B|
    |138617865C48CEA9|2A917CE5E3AD1374|0000000098425E60|0000000000000000|0000000000000000|
    """,
    digest_hex="602133DD97109089611B5125914B0F05532B96C0",
)


def table_digest(state: int, rounds: int = 5, digest_bits: int = 160) -> int:
    return squeeze_digest(rounds_int(state, rounds), digest_bits)


def digest_hex_numeric(digest: int, digest_bits: int) -> str:
    width = (digest_bits + 3) // 4
    return f"{digest:0{width}x}"


def digest_hex_paper_order(digest: int, digest_bits: int) -> str:
    """Render digest lanes in the left-to-right order used by the paper tables."""
    parts: list[str] = []
    full_words = digest_bits // 64
    rem = digest_bits % 64
    for word in range(full_words):
        parts.append(f"{(digest >> (64 * word)) & ((1 << 64) - 1):016x}")
    if rem:
        parts.append(f"{(digest >> (64 * full_words)) & ((1 << rem) - 1):0{rem // 4}x}")
    return "".join(parts)


def verify_collision(
    collision: PaperCollision,
    rounds: int,
    digest_bits: int,
) -> tuple[bool, dict[str, bool | str]]:
    m1 = parse_state_matrix(collision.m1)
    m2 = parse_state_matrix(collision.m2)
    digest1 = table_digest(m1, rounds=rounds, digest_bits=digest_bits)
    digest2 = table_digest(m2, rounds=rounds, digest_bits=digest_bits)
    digest1_hex = digest_hex_numeric(digest1, digest_bits)
    digest2_hex = digest_hex_numeric(digest2, digest_bits)
    digest1_paper_hex = digest_hex_paper_order(digest1, digest_bits)
    paper_digest = collision.digest_hex.lower()
    details: dict[str, bool | str] = {
        "digest1": digest1_hex,
        "digest2": digest2_hex,
        "digest1_paper_order": digest1_paper_hex,
        "paper_digest": paper_digest,
        "digest_equal": digest1 == digest2,
        "matches_paper_digest": digest1_paper_hex == paper_digest,
    }
    ok = bool(details["digest_equal"] and details["matches_paper_digest"])
    return ok, details


def demo() -> None:
    collision = TABLE10_KECCAK_1440_160_5_160
    ok, details = verify_collision(collision, rounds=5, digest_bits=160)
    print(collision.name)
    print(f"  local digest M1: {details['digest1']}")
    print(f"  local digest M2: {details['digest2']}")
    print(f"  local collision: {details['digest_equal']}")
    print(f"  paper digest:    {details['paper_digest']}")
    if not ok:
        print("  WARNING: transcription or representation does not verify yet")


if __name__ == "__main__":
    demo()
