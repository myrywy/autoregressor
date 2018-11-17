import pytest
from pytest import approx 
import tensorflow as tf
from utils import parallel_nested_tuples_apply, batched_top_k_from_2d_tensor, repeat_in_ith_dimension

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


@pytest.mark.parametrize("input, k, expected_values, expected_indices_0, expected_indices_1",[
    (
        # input
        [
            [
                [0.1, 0.2],
                [0.3, 0.4]
            ],
        ],
        # k
        1,
        # expected values
        [
            [0.4],
        ],
        [
            [1],
        ],
        [
            [1]
        ],
    ),
    (
        # input
        [
            [
                [0.1, 0.2],
                [0.3, 0.4]
            ],
        ],
        # k
        2,
        # expected values
        [
            [0.4, 0.3],
        ],
        [
            [1, 1],
        ],
        [
            [1, 0]
        ],
    ),
    (
        # input
        [
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            [
                [0.1, 0.2],
                [0.5, 0.4],
            ],
        ],
        # k
        3,
        # expected values
        [
            [0.4, 0.3, 0.2],
            [0.5, 0.4, 0.2],
        ],
        [
            [1, 1, 0],
            [1, 1, 0],
        ],
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
    ),
    (
        # input
        [
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.1, 0.1]
            ],
            [
                [0.1, 0.2],
                [0.5, 0.4],
                [0.1, 0.1],
            ],
        ],
        # k
        3,
        # expected values
        [
            [0.4, 0.3, 0.2],
            [0.5, 0.4, 0.2],
        ],
        [
            [1, 1, 0],
            [1, 1, 0],
        ],
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
    ),
    (
        # input
        [
            [
                [0.1, 0.2, 0.1, 0.0],
                [0.3, 0.4, 0.0, 0.1],
            ],
            [
                [0.1, 0.2, 0.0, 0.1],
                [0.5, 0.4, 0.1, 0.0],
            ],
        ],
        # k
        3,
        # expected values
        [
            [0.4, 0.3, 0.2],
            [0.5, 0.4, 0.2],
        ],
        [
            [1, 1, 0],
            [1, 1, 0],
        ],
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
    ),
    (
        # input
        [
            [
                [0.2, 3.2],
                [0.4, 0.34],
                [5.5, 2.4],
                [0.5, 0.4]
            ],
            [
                [0.2, 3.2],
                [0.4, 5.5],
                [5.5, 2.4],
                [0.5, 0.4]
            ],
            [
                [0.2, 3.2],
                [5.5, 0.34],
                [5.5, 2.4],
                [0.5, 0.4]
            ],
            [
                [0.2, 3.2],
                [0.4, 0.34],
                [5.5, 5.5],
                [0.5, 0.4]
            ],
        ],
        # k
        2,
        # expected values
        [
            [5.5, 3.2],
            [5.5, 5.5],
            [5.5, 5.5],
            [5.5, 5.5],
        ],
        [
            [2, 0],
            [1, 2],
            [1, 2],
            [2, 2],
        ],
        [
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 1],
        ],
    ),
])
def test_batched_top_k_from_2d_tensor(input, k, expected_values, expected_indices_0, expected_indices_1):
    values, (indices_0, indices_1) = batched_top_k_from_2d_tensor(input, k)
    
    with tf.Session() as sess:
        r_vailues, r_indices_0, r_indices_1 = sess.run((values, indices_0, indices_1))

    assert r_vailues == approx(expected_values)
    assert r_indices_0 == approx(expected_indices_0)
    assert r_indices_1 == approx(expected_indices_1)

@pytest.fixture
def tensor3d():
    return tf.constant(
        [
            [
                [1,2,3],
                [4,5,6],
            ],
            [
                [7,8,9],
                [10,11,12],
            ],
            [
                [13,14,15],
                [16,17,18],
            ],
        ]
    )

@pytest.mark.parametrize("i, k",
    [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3), (3, 3),
    ]
)
def test_repeat_in_ith_dimension(tensor3d, i, k):
    t_ouptut = repeat_in_ith_dimension(tensor3d, i, k)
    with tf.Session() as sess:
        r_output, r_input = sess.run((t_ouptut, tensor3d))
    for j in range(k):
        input_equal_slice = [slice(None) for _ in tensor3d.shape]
        input_equal_slice.insert(i, j)
        assert r_output[input_equal_slice] == approx(r_input)