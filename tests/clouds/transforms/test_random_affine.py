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


class TestScalingMatrix:
    @pytest.mark.parametrize('dim', [2, 3])
    def test_random_scaling_matrix_shape_and_diagonal(self, dim):
        matrix = random_scaling_matrix(dim, scales=(0.5, 1.5))

        assert matrix.shape == (dim, dim)
        assert torch.equal(matrix, torch.diag(torch.diagonal(matrix)))

    def test_random_scaling_matrix_within_bounds(self):
        low, high = 0.5, 1.5
        matrix = random_scaling_matrix(3, scales=(low, high))

        diag = torch.diagonal(matrix)
        assert torch.all(diag >= low) and torch.all(diag <= high)

    def test_random_scaling_matrix_uniform_scaling_has_equal_diagonal(self):
        matrix = random_scaling_matrix(3, scales=(0.5, 1.5), uniform_scaling=True)

        diag = torch.diagonal(matrix)
        assert torch.allclose(diag, diag[0].expand_as(diag))


class TestRotationMatrix:
    @pytest.mark.parametrize('dim,axis', [(2, [2]), (3, [0]), (3, [1]), (3, [2]), (3, [0, 1])])
    def test_random_rotation_matrix_is_orthogonal_with_unit_determinant(self, dim, axis):
        matrix = random_rotation_matrix(dim, degrees=(0.0, 360.0), axis=axis)

        identity = torch.eye(dim)
        assert torch.allclose(matrix @ matrix.T, identity, atol=1e-6)
        assert torch.allclose(torch.linalg.det(matrix), torch.tensor(1.0), atol=1e-5)

    def test_random_rotation_matrix_zero_degrees_is_identity(self):
        matrix = random_rotation_matrix(3, degrees=(0.0, 0.0), axis=[2])

        assert torch.allclose(matrix, torch.eye(3), atol=1e-6)

    def test_random_rotation_matrix_degrees_as_list_picks_from_list(self):
        matrix = random_rotation_matrix(3, degrees=[180.0], axis=[2])

        expected = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert torch.allclose(matrix, expected, atol=1e-6)


class TestTransformNormals:
    def test_transform_normals_rejects_mismatched_shapes(self):
        normals = torch.rand(3, 2)
        transform = torch.eye(3)

        with pytest.raises(AssertionError):
            transform_normals(normals, transform)

    def test_transform_normals_identity_leaves_normals_unchanged(self):
        normals = torch.tensor([[1.0, 1.0, 0.0]])
        normals = normals / normals.norm(dim=-1, keepdim=True)

        out = transform_normals(normals, torch.eye(3))

        assert torch.allclose(out, normals, atol=1e-6)

    def test_transform_normals_output_is_unit_norm(self):
        normals = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        normals = normals / normals.norm(dim=-1, keepdim=True)
        transform = torch.tensor([[2.0, 0.5, 0.0], [0.3, 1.5, 0.0], [0.0, 0.0, 1.0]])

        out = transform_normals(normals, transform)

        assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-6)

    def test_transform_normals_scaling_matches_hand_computed_direction(self):
        # For a diagonal (symmetric) scaling transform, the correctly-transformed
        # normal is normal @ inv(transform), renormalized.
        normals = torch.tensor([[1.0, 1.0, 0.0]]) / torch.tensor(2.0).sqrt()
        transform = torch.diag(torch.tensor([2.0, 1.0, 1.0]))

        out = transform_normals(normals, transform)

        expected = torch.tensor([[0.5, 1.0, 0.0]])
        expected = expected / expected.norm(dim=-1, keepdim=True)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_transform_normals_round_trips_through_inverse_transform(self):
        # Applying a transform and then its inverse should recover the original
        normals = torch.tensor([[1.0, 1.0, 0.0]])
        normals = normals / normals.norm(dim=-1, keepdim=True)
        transform = torch.tensor([[2.0, 0.5, 0.0], [0.3, 1.5, 0.0], [0.0, 0.0, 1.0]])

        forward = transform_normals(normals, transform)
        back = transform_normals(forward, torch.linalg.inv(transform))

        assert torch.allclose(back, normals, atol=1e-5)


class TestRandomScale:
    def test_random_scale_scales_positions(self, make_point_cloud):
        data = make_point_cloud(num_nodes=3, dim=3)
        transform = RandomScale(scales=(2.0, 2.0), uniform_scaling=True, p=1.0)

        out = transform(data)

        assert torch.allclose(out.pos, data.pos * 2.0, atol=1e-5)

    def test_random_scale_uniform_scaling_preserves_normal_direction(self, make_point_cloud):
        data = make_point_cloud(num_nodes=3, dim=3, with_norm=True)
        transform = RandomScale(scales=(2.0, 2.0), uniform_scaling=True, correct_norm=True, p=1.0)

        out = transform(data)

        # Uniform scaling doesn't change directions
        assert torch.allclose(out.norm, data.norm, atol=1e-5)

    def test_random_scale_correct_norm_false_leaves_norm_untouched(self, make_point_cloud):
        data = make_point_cloud(num_nodes=3, dim=3, with_norm=True)
        original_norm = data.norm.clone()
        transform = RandomScale(scales=(2.0, 0.5), uniform_scaling=False, correct_norm=False, p=1.0)

        out = transform(data)

        assert torch.equal(out.norm, original_norm)

    def test_random_scale_p_zero_never_applies(self, make_point_cloud):
        data = make_point_cloud(num_nodes=3, dim=3)
        transform = RandomScale(scales=(2.0, 2.0), p=0.0)

        out = transform(data)

        assert torch.equal(out.pos, data.pos)


class TestRandomRotate:
    def test_random_rotate_preserves_vector_norms(self, make_point_cloud):
        data = make_point_cloud(num_nodes=3, dim=3)
        transform = RandomRotate(degrees=(0.0, 360.0), axis=2, p=1.0)

        out = transform(data)

        assert torch.allclose(out.pos.norm(dim=-1), data.pos.norm(dim=-1), atol=1e-5)

    def test_random_rotate_zero_degrees_is_identity(self, make_point_cloud):
        data = make_point_cloud(num_nodes=3, dim=3)
        transform = RandomRotate(degrees=0.0, axis=2, p=1.0)

        out = transform(data)

        assert torch.allclose(out.pos, data.pos, atol=1e-5)

    def test_random_rotate_p_zero_never_applies(self, make_point_cloud):
        data = make_point_cloud(num_nodes=3, dim=3)
        transform = RandomRotate(degrees=360.0, p=0.0)

        out = transform(data)

        assert torch.equal(out.pos, data.pos)

    @pytest.mark.parametrize(('axis_in', 'expected_axis'), [(2, [2]), ([0, 1], [0, 1])])
    def test_random_rotate_normalizes_axis_to_list(self, axis_in, expected_axis):
        transform = RandomRotate(axis=axis_in)

        assert transform.axis == expected_axis

    def test_random_rotate_repr(self):
        transform = RandomRotate(degrees=180.0, axis=2, correct_norm=True, p=0.5)

        assert repr(transform) == ('RandomRotate(degrees=(-180.0, 180.0), correct_norm=True, axis=[2], p=0.5)')


class TestRandomScaleAndRotate:
    def test_random_scale_and_rotate_scale_only(self, make_point_cloud):
        data = make_point_cloud(num_nodes=3, dim=3)
        transform = RandomScaleAndRotate(
            scales=(2.0, 2.0),
            uniform_scaling=True,
            scale_prob=1.0,
            rotate_prob=0.0,
        )

        out = transform(data)

        assert torch.allclose(out.pos, data.pos * 2.0, atol=1e-5)

    def test_random_scale_and_rotate_rotate_only_preserves_norms(self, make_point_cloud):
        data = make_point_cloud(num_nodes=3, dim=3)
        transform = RandomScaleAndRotate(
            degrees=(0.0, 360.0),
            scale_prob=0.0,
            rotate_prob=1.0,
        )

        out = transform(data)

        assert torch.allclose(out.pos.norm(dim=-1), data.pos.norm(dim=-1), atol=1e-5)

    def test_random_scale_and_rotate_neither_applies_when_both_probs_zero(self, make_point_cloud):
        data = make_point_cloud(num_nodes=3, dim=3)
        transform = RandomScaleAndRotate(scale_prob=0.0, rotate_prob=0.0)

        out = transform(data)

        assert torch.equal(out.pos, data.pos)
