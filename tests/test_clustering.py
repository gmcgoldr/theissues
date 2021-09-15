import numpy as np

from theissues import clustering, utils


def test_calculate_pca():
    # two dimensions of variance: along x and along y
    points = np.array(
        [
            [+2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, +1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    # rotate 45 deg. (each column projects to the new coordinate)
    rt2 = 2 ** -0.5
    rotation = np.array(
        [
            [+rt2, +rt2, 0],
            [-rt2, +rt2, 0],
            [0, 0, 0],
        ]
    )
    transformed = np.dot(points, rotation)
    offset = np.array([1.0, 2.0, 3.0])
    transformed = transformed + offset

    bias, basis = clustering.calculate_pca(transformed, 2)

    np.testing.assert_allclose(bias, -offset)

    # should recover the original points, though the overall sign is arbitrary
    # and is -1 in this case
    np.testing.assert_allclose(
        np.dot(transformed + bias, basis), -points[:, :2], rtol=1e-6, atol=1e-6
    )


def test_build_topic_groups():
    texts_topics = [
        ("CERB", "CERB"),
        ("Canadian Emergency Response Benefit", "CERB"),
        ("Residential School", "Residential Schools"),
        ("Residential Schools", "Residential Schools"),
        ("Truth and Reconciliation", "Residential Schools"),
        ("Carbon Tax", "Carbon Tax"),
        ("Price on Carbon", "Carbon Tax"),
        ("Vaccine", "Vaccine"),
        ("Vaccines", "Vaccine"),
        ("Pfizer", "Vaccine"),
        ("Moderna", "Vaccine"),
        ("AstraZeneca", "Vaccine"),
        ("Reopening", "Reopening"),
        ("Reopen", "Reopening"),
    ]

    topic_groups = clustering.build_topic_groups([t for t, _ in texts_topics])

    num_grouped = sum([len(g) for g in topic_groups.values()])
    assert num_grouped == len(texts_topics)
    for i, (_, topic) in enumerate(texts_topics):
        assert i in topic_groups[topic]


def test_build_intra_topic_pairs():
    embeddings = np.array(
        [
            # group 0
            [0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 3.0, 1.0],
            # group 1
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
        ]
    )
    groups = [[0, 1, 2], [3, 4]]
    rng = np.random.Generator(np.random.PCG64())
    pairs = clustering.build_intra_topic_pairs(groups, embeddings, 3, rng)

    # 2 groups x 3 pairs per group, 2 to a pair, 3 dims
    assert pairs.shape == (2 * 3, 2, 3)

    # pairs in group 0 start with 0, in group 1 start with 1
    np.testing.assert_equal(pairs.reshape((2, -1, 3))[0, :, 0], 0.0)
    np.testing.assert_equal(pairs.reshape((2, -1, 3))[1, :, 0], 1.0)
    # all remaining dims have nonzero values
    assert np.all(pairs[:, :, 1:] != 0)


def test_build_inter_topic_pairs():
    embeddings = np.array(
        [
            # group 0
            [0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 3.0, 1.0],
            # group 1
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
        ]
    )
    groups = [[0, 1, 2], [3, 4]]
    rng = np.random.Generator(np.random.PCG64())
    pairs = clustering.build_inter_topic_pairs(groups, embeddings, 3, rng)

    # 2 groups x 3 pairs per group, 2 to a pair, 3 dims
    assert pairs.shape == (2 * 3, 2, 3)

    # pairs should have different indices on the first axis
    assert np.all(pairs[:, 0, 0] != pairs[:, 1, 0])
    # all remaining dims have nonzero values
    assert np.all(pairs[:, :, 1:] != 0)
