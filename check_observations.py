"""Check Section 4.1 observations from Qiao et al. 2017."""

from __future__ import annotations

from sbox_linearization import (
    DDT,
    affine_equations,
    contained_two_dim_linearizable_subspaces,
    format_subset,
    is_linearizable_affine_subspace,
    linearizable_affine_subspaces,
    value_solution_set,
)


def check_observation_1() -> None:
    planes = linearizable_affine_subspaces(2)
    three_dim = linearizable_affine_subspaces(3)

    print("Observation 1")
    print(f"  2-dimensional linearizable affine subspaces: {len(planes)}")
    print(f"  3-dimensional linearizable affine subspaces: {len(three_dim)}")
    print("  first 10 planes:")
    for plane in planes[:10]:
        print(f"    {format_subset(plane)}")

    assert len(planes) == 80
    assert len(three_dim) == 0


def check_observation_2() -> None:
    counts = {2: 0, 4: 0, 8: 0}
    ddt8_examples: list[tuple[int, int, int, str]] = []

    for delta_in in range(32):
        for delta_out in range(32):
            entry = DDT[delta_in][delta_out]
            if entry == 0:
                continue
            values = value_solution_set(delta_in, delta_out)
            if entry in counts:
                counts[entry] += 1

            if entry in (2, 4):
                assert is_linearizable_affine_subspace(values), (
                    delta_in,
                    delta_out,
                    entry,
                    values,
                )
            elif entry == 8:
                planes = contained_two_dim_linearizable_subspaces(values)
                assert len(planes) == 6, (delta_in, delta_out, values, planes)
                if len(ddt8_examples) < 3:
                    ddt8_examples.append(
                        (delta_in, delta_out, len(planes), format_subset(values))
                    )

    example_values = value_solution_set(0x03, 0x02)
    example_equations = affine_equations(example_values)

    print("\nObservation 2")
    print(f"  nonzero DDT entries with value 2: {counts[2]}")
    print(f"  nonzero DDT entries with value 4: {counts[4]}")
    print(f"  nonzero DDT entries with value 8: {counts[8]}")
    print("  all DDT=2/4 value solution sets are linearizable affine subspaces")
    print("  every DDT=8 value solution set contains exactly six linearizable planes")
    print(f"  paper example V(03,02): {format_subset(example_values)}")
    print("  one equation form for V(03,02):")
    for mask, rhs in example_equations:
        print(f"    mask={mask:05b}, rhs={rhs}")
    print("  sample DDT=8 cases:")
    for delta_in, delta_out, count, values in ddt8_examples:
        print(f"    din={delta_in:02X}, dout={delta_out:02X}, planes={count}, V={values}")


def main() -> None:
    check_observation_1()
    check_observation_2()
    print("\nAll Section 4.1 checks passed.")


if __name__ == "__main__":
    main()
