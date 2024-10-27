import numpy as np
import torch


def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([
            points,
            torch.ones(
                (*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
        ],
                         dim=-1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def euclidean_to_homogeneous_norm(points):
    """Converts euclidean points to homogeneous and ||point||=1

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    results = euclidean_to_homogeneous(points)
    if isinstance(points, np.ndarray):
        norm1 = np.linalg.norm(results, ord=1, axis=1, keepdims=True)
        norm1 = np.where(norm1 == 0, 1, norm1)
        results = results / norm1
        return results
    elif torch.is_tensor(points):
        norm1 = torch.norm(points, p=1, dim=1)
        norm1 = torch.where(norm1 == 0, 1, norm1)
        results = results / norm1
        return results
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(-1, 0)[:-1] /
                points.transpose(-1, 0)[-1]).transpose(-1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def project_3d_points_to_image_plane_without_distortion(
        proj_matrix, points_3d, convert_back_to_euclidean=True):
    """Project 3D points to image plane not taking into account distortion
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                        NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    if isinstance(proj_matrix, np.ndarray) and isinstance(
            points_3d, np.ndarray):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def triangulate_point_from_multiple_views_linear(proj_matricies, points):
    """Triangulates one point from multiple (N) views using direct linear transformation (DLT).
    For more information look at "Multiple view geometry in computer vision",
    Richard Hartley and Andrew Zisserman, 12.2 (p. 312).

    Args:
        proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
        points numpy array of shape (N, 2): sequence of points' coordinates

    Returns:
        point_3d numpy array of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)
    A = np.zeros((2 * n_views, 4))
    for j in range(len(proj_matricies)):
        A[j * 2 +
          0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
        A[j * 2 +
          1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

    u, s, vh = np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]

    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d


def triangulate_point_from_multiple_views_linear_torch(proj_matricies,
                                                       points,
                                                       confidences=None):
    """Similar as triangulate_point_from_multiple_views_linear() but for PyTorch.
    For more information see its documentation.
    Args:
        proj_matricies torch tensor of shape (N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (N, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (N,): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (3,): triangulated point
    """

    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)

    if confidences is None:
        confidences = torch.ones(n_views,
                                 dtype=torch.float32,
                                 device=points.device)

    A = proj_matricies[:, 2:3].expand(n_views, 2, 4) * points.view(
        n_views, 2, 1)
    A -= proj_matricies[:, :2]
    A *= confidences.view(-1, 1, 1)
    # singular value decomposition
    u, s, vh = torch.svd(A.view(-1, 4))
    point_3d_homo = -vh[:, 3]

    point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]

    return point_3d


def triangulate_withprior_point_from_multiple_views_linear_torch(
        proj_matricies,
        points,
        point_3d_root,
        confidences=None,
        pca_matrix=None,
        pca_mean=None,
        pca_matrix_kcs_1=None,
        kcs_trans_1=None,
        pca_matrix_kcs_2=None,
        kcs_trans_2=None,
        O_0=None,
        O_1=None,
        O_2=None,
        pca_group=None,
        weight_0=None,
        weight_1=None,
        weight_2=None,
        rotations=None):
    """Similar as triangulate_point_from_multiple_views_linear() but for PyTorch and with prior.
    For more information see its documentation.
    Args:
        proj_matricies torch tensor of shape (3, 4): projection matricies (3x4)
        points torch tensor of shape (M, 2): points' coordinates
        confidences None or torch tensor of shape (N, M, ): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
        pca_matrix torch tensor of shape (M*3, L): matrix of pca
        pca_mean torch tensor of shape (M*3, ): matrix of pca normalization
        kcs_trans torch tensor of shape (X, M*3): matrix of kcs tranformation
        points_3d_ori torch tensor of shape (M*3): sequence of points' coordinates using learnable triangulation
        rotations torch tensor of shap (3,3): rotation to make ori normalized
    Returns:
        point_3d numpy torch tensor of shape (M, 3,): triangulated point
    """

    n_views, n_joints = points.shape[:2]

    assert n_views == proj_matricies.shape[0]
    # print("Start triangualtion with prior, nviews:{}, n_joints:{}".format(n_views, n_joints))
    A = torch.zeros([2 * n_views * n_joints, 3 * n_joints],
                    device=points.device)
    b = torch.zeros([2 * n_views * n_joints], device=points.device)

    if confidences is None:
        confidences = torch.ones([n_views, n_joints],
                                 dtype=torch.float32,
                                 device=points.device)
    if rotations is not None:
        rotations_batch = torch.zeros([n_joints * 3, n_joints * 3],
                                      device=points.device)
    for i in range(n_joints):
        A_i = proj_matricies[:, 2:3].expand(
            n_views, 2, 4) * points[:, i, :].view(n_views, 2, 1)
        A_i -= proj_matricies[:, :2]
        A_i *= confidences[:, i].view(-1, 1, 1)
        A_i = A_i.view(-1, 4)
        row_start = 2 * n_views * i
        row_end = 2 * n_views * (i + 1)
        A[row_start:row_end, (3 * i):(3 * (i + 1))] = A_i[:, :3]
        b[row_start:row_end] = A_i[:, 3]
        if rotations is not None:
            row_start = 3 * i
            row_end = 3 * (i + 1)
            rotations_batch[row_start:row_end, row_start:row_end] = rotations
            # print("rotation_batch:", rotations_batch)

    # print("start least-square")
    A_T = torch.transpose(A, 1, 0)
    X = A_T @ A
    y = -A_T @ b
    # print("X_0:", X)
    # print("A.shape:{}, b.shape:{}".format(A.shape, b.shape))

    if pca_matrix is not None:
        M = torch.matmul(pca_matrix, torch.transpose(pca_matrix, 1, 0))
        # print("M:", M.shape)
        if O_0 is not None:
            # print("O_0:", O_0)
            M += 0.0001 * O_0

        I = torch.eye(3 * n_joints, device=points.device)
        M = I - M
        M_T = torch.transpose(M, 1, 0)
        if rotations is None:
            if weight_0 is not None:
                X += weight_0 * M_T @ M
            else:
                # X += 5000 * M_T @ M
                X += 6000 * M_T @ M
        else:
            rotations_batch = rotations_batch.type(torch.float64)
            if weight_0 is not None:
                X += weight_0 * M_T @ M @ rotations_batch
            else:
                X += 10 * M_T @ M @ rotations_batch
        # print("X:", X)
        # print(100000*M_T @ M)
        x_root = point_3d_root[:].expand(n_joints, 3)
        if rotations is not None:
            x_root = torch.einsum('ij,mj->mi', rotations, x_root)
        x_root = x_root.contiguous().view(-1, 1).squeeze()
        # print("M.dtype:", M.dtype)
        # print("x_root.dtype:", x_root.dtype)
        # print("(x_root + pca_mean).dtype:", (x_root + pca_mean).dtype)
        if weight_0 is not None:
            y += (weight_0 * M_T @ M) @ (x_root + pca_mean)
        else:
            # print("6000dis0")
            # y += (5000 * M_T @ M) @ (x_root + pca_mean)
            y += (6000 * M_T @ M) @ (x_root + pca_mean)
        # print("A.shape:{}, b.shape:{}, M.shape:{}, pca_mean.shape:{}, x_root.shape:{}".format(A.shape, b.shape, M.shape, pca_mean.shape, x_root.shape))

    if pca_matrix_kcs_1 is not None:
        # print("pca_matrix_kcs_1.shape:{}, pca_mean.shape:{}".format(pca_matrix_kcs_1.shape, pca_mean.shape))
        M_k = torch.matmul(pca_matrix_kcs_1,
                           torch.transpose(pca_matrix_kcs_1, 1, 0))
        if O_1 is not None:
            # print("O_1:", O_1)
            M_k += 0.0001 * O_1

        I_k = torch.eye(M_k.shape[0], device=points.device)
        M_k = I_k - M_k
        M_k_T = torch.transpose(M_k, 1, 0)
        C = kcs_trans_1
        C_pinv = torch.pinverse(C)
        if rotations is None:
            if weight_1 is not None:
                X += weight_1 * M_T @ M
            else:
                # print("2000dis1")
                # X += 400 * (C_pinv @ (M_k_T @ M_k) @ C)
                X += 4000 * (C_pinv @ (M_k_T @ M_k) @ C)
        else:
            rotations_batch = rotations_batch.type(torch.float64)
            if weight_1 is not None:
                X += weight_1 * M_T @ M
            else:
                # print("1800dis1")
                X += 10 * (C_pinv @ (M_k_T @ M_k) @ C) @ rotations_batch
        # print("X_2:", X)
        # print(12000*(C_pinv @ (M_k_T @ M_k) @C))
        x_root = point_3d_root[:].expand(n_joints, 3)
        if rotations is not None:
            x_root = torch.einsum('ij,mj->mi', rotations, x_root)
        x_root = x_root.contiguous().view(-1, 1).squeeze()
        if weight_1 is not None:
            y += (weight_1 *
                  (C_pinv @ (M_k_T @ M_k) @ C)) @ (x_root + pca_mean)
        else:
            # y += (400 * (C_pinv @ (M_k_T @ M_k) @ C)) @ (x_root + pca_mean)
            y += (4000 * (C_pinv @ (M_k_T @ M_k) @ C)) @ (x_root + pca_mean)
        # print("A.shape:{}, b.shape:{}, M.shape:{}, pca_mean.shape:{}, x_root.shape:{}".format(A.shape, b.shape, M_k.shape, pca_mean.shape, x_root.shape))

    if pca_matrix_kcs_2 is not None:
        M_k = torch.matmul(pca_matrix_kcs_2,
                           torch.transpose(pca_matrix_kcs_2, 1, 0))
        if O_2 is not None:
            # print("O_2:", O_2)
            M_k += 0.0001 * O_2

        I_k = torch.eye(M_k.shape[0], device=points.device)
        M_k = I_k - M_k
        M_k_T = torch.transpose(M_k, 1, 0)
        C = kcs_trans_2
        C_pinv = torch.pinverse(C)
        if rotations is None:
            if weight_2 is not None:
                X += weight_2 * M_T @ M
            else:
                # print("1200dis2")
                # X += 800 * (C_pinv @ (M_k_T @ M_k) @ C) 
                X += 2000 * (C_pinv @ (M_k_T @ M_k) @ C)  
        else:
            rotations_batch = rotations_batch.type(torch.float64)
            if weight_2 is not None:
                X += weight_2 * M_T @ M
            else:
                # print("1800dis1")
                X += 10 * (C_pinv @ (M_k_T @ M_k) @ C) @ rotations_batch
                
        # print(C_pinv @ (M_k_T @ M_k) @C)
        # print(12000*(C_pinv @ (M_k_T @ M_k) @C))
        x_root = point_3d_root[:].expand(n_joints, 3)
        if rotations is not None:
            x_root = torch.einsum('ij,mj->mi', rotations, x_root)
        x_root = x_root.contiguous().view(-1, 1).squeeze()
        if weight_2 is not None:
            y += (weight_2 *
                  (C_pinv @ (M_k_T @ M_k) @ C)) @ (x_root + pca_mean)
        else:
            # y += (800 * (C_pinv @ (M_k_T @ M_k) @ C)) @ (x_root + pca_mean)
            y += (2000 * (C_pinv @ (M_k_T @ M_k) @ C)) @ (x_root + pca_mean)
        # print("A.shape:{}, b.shape:{}, M.shape:{}, pca_mean.shape:{}, x_root.shape:{}".format(A.shape, b.shape, M_k.shape, pca_mean.shape, x_root.shape))

    if pca_group is not None:
        group_num = pca_group['num']
        for i in range(group_num):
            pca_matrix_group = pca_group[i]['pca']
            trans_matrix_group = pca_group[i]['trans']
            M_g = torch.matmul(pca_matrix_group,
                               torch.transpose(pca_matrix_group, 1, 0))
            # print("M_g:", M_g.shape)
            # print("trans_matrix_group:", trans_matrix_group.shape)
            I_g = torch.eye(M_g.shape[0], device=points.device)
            M_g = I_g - M_g
            M_g = M_g @ trans_matrix_group
            M_g_T = torch.transpose(M_g, 1, 0)
            C = kcs_trans_1
            C_pinv = torch.pinverse(C)
            X += 600 * (C_pinv @ (M_g_T @ M_g) @ C)
            x_root = point_3d_root[:].expand(n_joints, 3)
            x_root = x_root.contiguous().view(-1, 1).squeeze()
            # print("x_root:", x_root.shape)
            # print("pca_mean:", pca_mean.shape)
            y += (600 * (C_pinv @ (M_g_T @ M_g) @ C)) @ (x_root + pca_mean)

    # print("X.shape:", X.shape)
    # print("X:", X)
    # print("y.shape:", y.shape)
    # print("y:", y)
    points_3d = torch.linalg.solve(X, y)
    points_3d = points_3d.view(n_joints, -1)
    # print("end least-square")
    # print("points_3d.shape:{}".format(points_3d.shape))

    return points_3d


def triangulate_batch_of_points(proj_matricies_batch,
                                points_batch,
                                confidences_batch=None):
    batch_size, n_views, n_joints = points_batch.shape[:3]
    point_3d_batch = torch.zeros(batch_size,
                                 n_joints,
                                 3,
                                 dtype=torch.float32,
                                 device=points_batch.device)

    for batch_i in range(batch_size):
        for joint_i in range(n_joints):
            points = points_batch[batch_i, :, joint_i, :]

            confidences = confidences_batch[
                batch_i, :, joint_i] if confidences_batch is not None else None
            point_3d = triangulate_point_from_multiple_views_linear_torch(
                proj_matricies_batch[batch_i], points, confidences=confidences)
            point_3d_batch[batch_i, joint_i] = point_3d

    return point_3d_batch


def calc_reprojection_error_batch_of_points(proj_matricies_batch,
                                            keypoints_2d_batch,
                                            keypoints_3d_batch):
    batch_size, n_views, n_joints = keypoints_2d_batch.shape[:3]
    reproj_errors = torch.zeros(batch_size,
                                n_views,
                                n_joints,
                                dtype=torch.float32,
                                device=keypoints_2d_batch.device)
    keypoints_2d_proj_batch = torch.ones_like(keypoints_2d_batch)
    for batch_i in range(batch_size):
        keypoints_2d = keypoints_2d_batch[batch_i, :, :, :]

        for view_i in range(n_views):
            keypoints_2d_proj_batch[
                batch_i,
                view_i] = project_3d_points_to_image_plane_without_distortion(
                    proj_matricies_batch[batch_i, view_i],
                    keypoints_3d_batch[batch_i])
        keypoints_2d_proj = keypoints_2d_proj_batch[batch_i, :, :, :]
        reproj_error = torch.sqrt(
            torch.sum((keypoints_2d - keypoints_2d_proj)**2, dim=2))

        reproj_errors[batch_i] = reproj_error

    return reproj_errors, keypoints_2d_proj_batch


def calc_reprojection_error_matrix(keypoints_3d, keypoints_2d_list,
                                   proj_matricies):
    reprojection_error_matrix = []
    for keypoints_2d, proj_matrix in zip(keypoints_2d_list, proj_matricies):
        keypoints_2d_projected = project_3d_points_to_image_plane_without_distortion(
            proj_matrix, keypoints_3d)
        reprojection_error = 1 / 2 * np.sqrt(
            np.sum((keypoints_2d - keypoints_2d_projected)**2, axis=1))
        reprojection_error_matrix.append(reprojection_error)

    return np.vstack(reprojection_error_matrix).T


# 3D proj 2D joints
def projto_2D(keypoints_2d_pred_batch, keypoints_3d_pred_batch,
              proj_matricies_batch, keypoints_binary_validity, batch_size,
              nviews):
    keypoints_2d_proj_batch = torch.zeros(
        batch_size,
        nviews,
        keypoints_3d_pred_batch.shape[1],
        2,
        dtype=torch.float32,
        device=keypoints_2d_pred_batch.device)
    # print("proj:",proj_matricies_batch.shape)
    for batch in range(batch_size):
        keypoints_3d_pred = keypoints_3d_pred_batch[batch]
        for view in range(nviews):
            proj_matricies = proj_matricies_batch[batch, view]
            # print(proj_matricies)
            keypoints_2d_proj = project_3d_points_to_image_plane_without_distortion(
                proj_matricies, keypoints_3d_pred)
            keypoints_2d_proj_batch[batch, view] = keypoints_2d_proj
    return keypoints_2d_proj_batch

def proj_3d_to_2d(x_3d,  proj_matrixes):
    # project 3d point into 2d plane
    b, j = x_3d.shape[0], x_3d.shape[1]
    v = proj_matrixes.shape[1]
    x_2d = torch.zeros((b, v, j, 2), device=x_3d.device)
    for b_i in range(b):
        for v_i in range(v):
            x_2d[b_i, v_i] = project_3d_points_to_image_plane_without_distortion(proj_matrixes[b_i, v_i], x_3d[b_i])
    x_2d = x_2d.permute(0,2,1,3)
    x_2d = x_2d.reshape(b,j,-1)
    return x_2d


def unproj_2d_to_3d(x_2d, proj_matrixes):
    # project 2d points with shape [B, J, V*2] into 3d
    b, j = x_2d.shape[0], x_2d.shape[1]
    x_2d = x_2d.reshape(b,j,-1,2)
    v = x_2d.shape[2]
    x_2d = x_2d.permute(0,2,1,3) # shape(b,v,j,2)
    x_3d = triangulate_batch_of_points(
                    proj_matrixes, x_2d,
                    confidences_batch=None
                )

    return x_3d, x_2d