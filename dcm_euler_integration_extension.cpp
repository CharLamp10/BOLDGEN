#include <torch/extension.h>
#include <vector>
#include <cmath>

// Original inner function converted to work with torch::Tensor
std::vector<torch::Tensor> dcm_euler_integration(
    torch::Tensor A,
    torch::Tensor C,
    torch::Tensor U,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor rho,
    double alphainv,
    torch::Tensor tau,
    double gamma,
    torch::Tensor kappa,
    std::vector<double> paramList
) {
    // ---- Retrieve sizes ----
    int nStates = paramList[2];
    int nTime   = paramList[1];
    int nInputs = paramList[3];
    double dt   = paramList[0];
    int dcmTypeB = paramList[5];
    int dcmTypeD = paramList[6];

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(A.device());
    auto x_out = torch::zeros({nTime, nStates}, options);
    auto s_out = torch::zeros_like(x_out);
    auto f1_out = torch::ones_like(x_out);
    auto v1_out = torch::ones_like(x_out);
    auto q1_out = torch::ones_like(x_out);

    std::vector<double> oldf1(nStates, 0.0);
    std::vector<double> oldv1(nStates, 0.0);
    std::vector<double> oldq1(nStates, 0.0);

    // ---- Main integration loop ----
    auto Aacc = A.accessor<double,2>();
    auto Cacc = C.accessor<double,2>();
    auto Uacc = U.accessor<double,2>();
    auto Bacc = B.accessor<double,3>();
    auto Dacc = D.accessor<double,3>();
    auto rhoacc = rho.accessor<double,1>();
    auto tauacc = tau.accessor<double,1>();
    auto kappacc = kappa.accessor<double,1>();

    auto xacc = x_out.accessor<double,2>();
    auto sacc = s_out.accessor<double,2>();
    auto facc = f1_out.accessor<double,2>();
    auto vacc = v1_out.accessor<double,2>();
    auto qacc = q1_out.accessor<double,2>();

    for (int t = 0; t < nTime - 1; ++t) {
        // --- neuronal update ---
        for (int j = 0; j < nStates; ++j) {
            double Ax = 0.0;
            for (int k = 0; k < nStates; ++k)
                Ax += xacc[t][k] * Aacc[k][j];
            double update = Ax + Cacc[t][j];

            if (dcmTypeB != 0) {
                for (int k = 0; k < nInputs; ++k) {
                    double tmp = 0.0;
                    for (int m = 0; m < nStates; ++m)
                        tmp += xacc[t][m] * Bacc[m][j][k];
                    update += Uacc[t][k] * tmp;
                }
            }
            if (dcmTypeD != 0) {
                for (int k = 0; k < nStates; ++k) {
                    double tmp = 0.0;
                    for (int m = 0; m < nStates; ++m)
                        tmp += xacc[t][m] * Dacc[m][j][k];
                    update += xacc[t][k] * tmp;
                }
            }
            xacc[t+1][j] = xacc[t][j] + dt * update;
        }

        // --- hemodynamic update (vectorized per state) ---
        for (int j = 0; j < nStates; ++j) {
            double s = sacc[t][j];
            double f = facc[t][j];
            double v = vacc[t][j];
            double q = qacc[t][j];

            sacc[t+1][j] = s + dt * (xacc[t][j] - kappacc[j]*s - gamma*(f-1));
            oldf1[j] += dt * (s / f);
            double temp1 = pow(v, alphainv);
            double temp2 = (1.0 - pow(1.0 - rhoacc[j], 1.0/f)) / rhoacc[j];
            oldv1[j] += dt * ((f - temp1) / (tauacc[j]*v));
            oldq1[j] += dt * ((f*temp2 - temp1*q/v) / (tauacc[j]*q));

            facc[t+1][j] = exp(oldf1[j]);
            vacc[t+1][j] = exp(oldv1[j]);
            qacc[t+1][j] = exp(oldq1[j]);
        }
    }

    return {x_out, s_out, f1_out, v1_out, q1_out};
}

// ---- PyTorch module definition ----
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dcm_euler_integration", &dcm_euler_integration, "DCM Euler integration (C++ fast version)");
}

