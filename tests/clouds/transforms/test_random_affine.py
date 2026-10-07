import pytest
import torch

from clouds.transforms.random_affine import (
    RandomRotate,
    RandomScale,
    RandomScaleAndRotate,
    random_rotation_matrix,
    random_scaling_matrix,
    transform_normals,
)


@pytest.mark.parametrize('dim', [2, 3])
def test_random_scaling_matrix_shape_and_diagonal(dim):
    matrix = random_scaling_matrix(dim, scales=(0.5, 1.5))

    assert matrix.shape == (dim, dim)
    assert torch.equal(matrix, torch.diag(torch.diagonal(matrix)))


def test_random_scaling_matrix_within_bounds():
    diag = torch.diagonal(random_scaling_matrix(3, scales=(0.5, 1.5)))

    assert torch.all(diag >= 0.5) and torch.all(diag <= 1.5)


def test_random_scaling_matrix_uniform_scaling_has_equal_diagonal():
    diag = torch.diagonal(random_scaling_matrix(3, scales=(0.5, 1.5), uniform_scaling=True))

    assert torch.allclose(diag, diag[0].expand_as(diag))


@pytest.mark.parametrize('dim,axis', [(2, [2]), (3, [0]), (3, [1]), (3, [2]), (3, [0, 1])])
def test_random_rotation_matrix_is_orthogonal_with_unit_determinant(dim, axis):
    matrix = random_rotation_matrix(dim, degrees=(0.0, 360.0), axis=axis)

    assert torch.allclose(matrix @ matrix.T, torch.eye(dim), atol=1e-6)
    assert torch.allclose(torch.linalg.det(matrix), torch.tensor(1.0), atol=1e-5)


def test_random_rotation_matrix_zero_degrees_is_identity():
    assert torch.allclose(random_rotation_matrix(3, degrees=(0.0, 0.0), axis=[2]), torch.eye(3), atol=1e-6)


def test_random_rotation_matrix_degrees_as_list_picks_from_list():
    matrix = random_rotation_matrix(3, degrees=[180.0], axis=[2])

    expected = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

    assert torch.allclose(matrix, expected, atol=1e-6)


def test_transform_normals_rejects_mismatched_shapes():
    with pytest.raises(AssertionError):
        transform_normals(torch.rand(3, 2), torch.eye(3))


def test_transform_normals_identity_leaves_normals_unchanged():
    normals = torch.tensor([[1.0, 1.0, 0.0]])
    normals = normals / normals.norm(dim=-1, keepdim=True)

    assert torch.allclose(transform_normals(normals, torch.eye(3)), normals, atol=1e-6)


def test_transform_normals_output_is_unit_norm():
    normals = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    normals = normals / normals.norm(dim=-1, keepdim=True)
    transform = torch.tensor([[2.0, 0.5, 0.0], [0.3, 1.5, 0.0], [0.0, 0.0, 1.0]])

    assert torch.allclose(transform_normals(normals, transform).norm(dim=-1), torch.ones(2), atol=1e-6)


def test_transform_normals_scaling_matches_hand_computed_direction():
    # For a diagonal (symmetric) scaling transform, the correct normal is normal @ inv(transform), renormalized.
    normals = torch.tensor([[1.0, 1.0, 0.0]]) / torch.tensor(2.0).sqrt()
    transform = torch.diag(torch.tensor([2.0, 1.0, 1.0]))

    expected = torch.tensor([[0.5, 1.0, 0.0]])
    expected = expected / expected.norm(dim=-1, keepdim=True)

    assert torch.allclose(transform_normals(normals, transform), expected, atol=1e-6)


def test_transform_normals_round_trips_through_inverse_transform():
    normals = torch.tensor([[1.0, 1.0, 0.0]])
    normals = normals / normals.norm(dim=-1, keepdim=True)
    transform = torch.tensor([[2.0, 0.5, 0.0], [0.3, 1.5, 0.0], [0.0, 0.0, 1.0]])

    back = transform_normals(transform_normals(normals, transform), torch.linalg.inv(transform))

    assert torch.allclose(back, normals, atol=1e-5)


def test_random_scale_scales_positions(make_point_cloud):
    data = make_point_cloud(num_nodes=3, dim=3)

    assert torch.allclose(RandomScale(scales=(2.0, 2.0), uniform_scaling=True, p=1.0)(data).pos, data.pos * 2.0, atol=1e-5)


def test_random_scale_uniform_scaling_preserves_normal_direction(make_point_cloud):
    data = make_point_cloud(num_nodes=3, dim=3, with_norm=True)

    assert torch.allclose(
        RandomScale(scales=(2.0, 2.0), uniform_scaling=True, correct_norm=True, p=1.0)(data).norm, data.norm, atol=1e-5
    )


def test_random_scale_correct_norm_false_leaves_norm_untouched(make_point_cloud):
    data = make_point_cloud(num_nodes=3, dim=3, with_norm=True)
    original_norm = data.norm.clone()

    assert torch.equal(
        RandomScale(scales=(2.0, 0.5), uniform_scaling=False, correct_norm=False, p=1.0)(data).norm, original_norm
    )


def test_random_scale_p_zero_never_applies(make_point_cloud):
    data = make_point_cloud(num_nodes=3, dim=3)

    assert torch.equal(RandomScale(scales=(2.0, 2.0), p=0.0)(data).pos, data.pos)


def test_random_rotate_preserves_vector_norms(make_point_cloud):
    data = make_point_cloud(num_nodes=3, dim=3)

    assert torch.allclose(
        RandomRotate(degrees=(0.0, 360.0), axis=2, p=1.0)(data).pos.norm(dim=-1), data.pos.norm(dim=-1), atol=1e-5
    )


def test_random_rotate_zero_degrees_is_identity(make_point_cloud):
    data = make_point_cloud(num_nodes=3, dim=3)

    assert torch.allclose(RandomRotate(degrees=0.0, axis=2, p=1.0)(data).pos, data.pos, atol=1e-5)


def test_random_rotate_p_zero_never_applies(make_point_cloud):
    data = make_point_cloud(num_nodes=3, dim=3)

    assert torch.equal(RandomRotate(degrees=360.0, p=0.0)(data).pos, data.pos)


@pytest.mark.parametrize(('axis_in', 'expected_axis'), [(2, [2]), ([0, 1], [0, 1])])
def test_random_rotate_normalizes_axis_to_list(axis_in, expected_axis):
    assert RandomRotate(axis=axis_in).axis == expected_axis


def test_random_rotate_repr():
    assert repr(RandomRotate(degrees=180.0, axis=2, correct_norm=True, p=0.5)) == (
        'RandomRotate(degrees=(-180.0, 180.0), correct_norm=True, axis=[2], p=0.5)'
    )


def test_random_scale_and_rotate_scale_only(make_point_cloud):
    data = make_point_cloud(num_nodes=3, dim=3)

    assert torch.allclose(
        RandomScaleAndRotate(scales=(2.0, 2.0), uniform_scaling=True, scale_prob=1.0, rotate_prob=0.0)(data).pos,
        data.pos * 2.0,
        atol=1e-5,
    )


def test_random_scale_and_rotate_rotate_only_preserves_norms(make_point_cloud):
    data = make_point_cloud(num_nodes=3, dim=3)

    assert torch.allclose(
        RandomScaleAndRotate(degrees=(0.0, 360.0), scale_prob=0.0, rotate_prob=1.0)(data).pos.norm(dim=-1),
        data.pos.norm(dim=-1),
        atol=1e-5,
    )


def test_random_scale_and_rotate_neither_applies_when_both_probs_zero(make_point_cloud):
    data = make_point_cloud(num_nodes=3, dim=3)

    assert torch.equal(RandomScaleAndRotate(scale_prob=0.0, rotate_prob=0.0)(data).pos, data.pos)
