import keras

def custom_lstsq(A, B):

    # QR decomposition of A
    Q, R = keras.ops.qr(A)  # A = Q @ R

    # Compute Q^T @ B (project B onto the subspace spanned by A's columns)
    QTB = keras.ops.transpose(Q) @ B

    # Solve R @ X = Q^T @ B using torch.linalg.solve
    X = keras.ops.solve(R, QTB)
    return X
