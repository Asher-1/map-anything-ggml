// Minimal semantics check for the ggml ops used by the VGGT-Omega graph.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>

int main() {
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_init_params ip{ggml_tensor_overhead() * 8192, nullptr, true};
    ggml_context* ctx = ggml_init(ip);

    // x {4, 2, 1, 1} = [[1,2],[3,4],[5,6],[7,8]] (ne0 fastest)
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 2, 1, 1);
    float xd[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    ggml_tensor* rows = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 1, 1, 1);
    float rd[4] = {2, 2, 2, 2};

    // 1) mul broadcast {4,1,1,1} over {4,2,1,1}
    ggml_tensor* m = ggml_mul(ctx, x, rows);
    ggml_cgraph* g = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(g, m);
    ggml_gallocr_alloc_graph(al, g);
    ggml_backend_tensor_set(x, xd, 0, sizeof(xd));
    ggml_backend_tensor_set(rows, rd, 0, sizeof(rd));
    ggml_backend_graph_compute(be, g);
    float out[8];
    ggml_backend_tensor_get(m, out, 0, sizeof(out));
    printf("mul broadcast: %g %g %g %g %g %g %g %g (expect 2..16)\n",
           out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);

    // 2) view offset: view ne {2,2,1,1} of x offset 2*4 bytes -> elements 3,4
    ggml_tensor* v = ggml_view_4d(ctx, x, 2, 2, 1, 1, x->nb[1], x->nb[2],
                                  x->nb[3], 2 * sizeof(float));
    ggml_tensor* v2 = ggml_cont(ctx, v);
    ggml_cgraph* g2 = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(g2, v2);
    ggml_gallocr_alloc_graph(al, g2);
    ggml_backend_graph_compute(be, g2);
    (void)v;
    float o2[4];
    ggml_backend_tensor_get(v2, o2, 0, sizeof(o2));
    printf("view offset (expect 3 4 5 6): %g %g %g %g\n", o2[0], o2[1], o2[2], o2[3]);

    // 3) concat along dim0 of {2,2,1,1} + {2,2,1,1}
    ggml_tensor* ca = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 1, 1);
    float cad[4] = {10, 11, 12, 13};
    ggml_backend_tensor_set(ca, cad, 0, sizeof(cad));
    ggml_tensor* cb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 1, 1);
    float cbd[4] = {20, 21, 22, 23};
    ggml_backend_tensor_set(cb, cbd, 0, sizeof(cbd));
    ggml_tensor* cc = ggml_concat(ctx, ca, cb, 0);
    ggml_cgraph* g3 = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(g3, cc);
    ggml_gallocr_alloc_graph(al, g3);
    ggml_backend_graph_compute(be, g3);
    float o3[8];
    ggml_backend_tensor_get(cc, o3, 0, sizeof(o3));
    printf("concat dim0 (expect 10 11 12 13 20 21 22 23): %g %g %g %g %g %g %g %g\n",
           o3[0], o3[1], o3[2], o3[3], o3[4], o3[5], o3[6], o3[7]);

    // 4) mul with b {4,1,2,1} (broadcast over ne1 AND repeat ne2)
    ggml_tensor* rows3 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 1, 2, 1);
    float r3[8] = {1, 1, 1, 1, 3, 3, 3, 3};
    ggml_backend_tensor_set(rows3, r3, 0, sizeof(r3));
    ggml_tensor* x4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 2, 2, 1);
    float x4d[16];
    for (int i = 0; i < 16; i++) x4d[i] = i + 1;
    ggml_backend_tensor_set(x4, x4d, 0, sizeof(x4d));
    ggml_tensor* m4 = ggml_mul(ctx, x4, rows3);
    ggml_cgraph* g4 = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(g4, m4);
    ggml_gallocr_alloc_graph(al, g4);
    ggml_backend_graph_compute(be, g4);
    float o4[16];
    ggml_backend_tensor_get(m4, o4, 0, sizeof(o4));
    printf("mul4 ne2-bcast rows: %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g\n",
           o4[0], o4[1], o4[2], o4[3], o4[4], o4[5], o4[6], o4[7],
           o4[8], o4[9], o4[10], o4[11], o4[12], o4[13], o4[14], o4[15]);

    ggml_gallocr_free(al);
    ggml_free(ctx);
    return 0;
}
