void golden_matmul(const float (*A)[4], const float (*B)[4], float (*restrict out)[4]) {
    for (long i = 0; i < 4; i++) {
        for (long j = 0; j < 4; j++) {
            out[i][j] = 0.0f;
        }
    }
    for (long i = 0; i < 4; i++) {
        for (long jo = 0; jo < 2; jo++) {
            for (long k = 0; k < 4; k++) {
                for (long ji = 0; ji < 2; ji++) {
                    out[i][2*jo + ji] = out[i][2*jo + ji] + (A[i][k] * B[k][2*jo + ji]);
                }
            }
        }
    }
}
