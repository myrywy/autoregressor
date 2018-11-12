import pytest
from utils import parallel_nested_tuples_apply

@pytest.mark.parametrize("input, expected_output, fn, a, k", 
    [
        (
            [1, 2],
            3,
            lambda x, y: x + y,
            [],
            {},
        ),
        (
            [1, 2],
            6,
            lambda x, y, z: x + y + z,
            [3],
            {},
        ),
        (
            [1, 2],
            6,
            lambda x, y, z, u=0: x + y + z + u,
            [3],
            {},
        ),
        (
            [1, 2],
            10,
            lambda x, y, z, u=0: x + y + z + u,
            [3],
            {"u": 4},
        ),
        (
            [(1,), (2,)],
            (3,),
            lambda x, y: x + y,
            [],
            {},
        ),
        (
            [(1,), (2,)],
            (10,),
            lambda x, y, z, u=0: x + y + z + u,
            [3],
            {"u": 4},
        ),
        (
            [(1,9), (2,10)],
            (3,19),
            lambda x, y: x + y,
            [],
            {},
        ),
        (
            [(1,9), (2,10)],
            (10,26),
            lambda x, y, z, u=0: x + y + z + u,
            [3],
            {"u": 4},
        ),
        (
            [(1,(9,20)), (2,(10,30))],
            (10,(26,57)),
            lambda x, y, z, u=0: x + y + z + u,
            [3],
            {"u": 4},
        ),
        (
            [(1,(9,20)), (2,(10,30))],
            (3,(19,50)),
            lambda x, y: x + y,
            [],
            {},
        ),
        (
            [],
            -1,
            lambda: -1,
            [],
            {},
        ),
    ]
)
def test_parallel_nested_tuples_apply(input, expected_output, fn, a, k):
    assert expected_output == parallel_nested_tuples_apply(input, fn, *a, **k)


@pytest.mark.parametrize("input, expected_output, fn, a, k", 
    [
        (
            [(), ()],
            1,
            lambda: 1,
            [],
            {},
        ),
        (
            [(1,), 2],
            (10,),
            lambda x, y, z, u=0: x + y + z + u,
            [3],
            {"u": 4},
        ),
        (
            [(1,9), (2,)],
            (10,26),
            lambda x, y, z, u=0: x + y + z + u,
            [3],
            {"u": 4},
        ),
        (
            [(1,(9,20)), (2,(10,(30,)))],
            (10,(26,57)),
            lambda x, y, z, u=0: x + y + z + u,
            [3],
            {"u": 4},
        ),
    ]
)
def test_parallel_nested_tuples_apply_raises_value_error(input, expected_output, fn, a, k):
    with pytest.raises(ValueError):
        parallel_nested_tuples_apply(input, fn, *a, **k)