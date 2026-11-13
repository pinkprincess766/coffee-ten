void golden_matmul(const float (*A)[4], const float (*B)[4], float (*restrict out)[4]) {
    for (long i = 0; i < 4; i++) {
        for (long j = 0; j < 4; j++) {
            out[i][j] = 0.0f;
        }
    }
    for (long i = 0; i < 4; i++) {
        for (long k = 0; k < 4; k++) {
            for (long j = 0; j < 4; j++) {
                out[i][j] = out[i][j] + (A[i][k] * B[k][j]);
            }
        }
    }
}
