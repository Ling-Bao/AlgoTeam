#include <iostream>
#include <random>

#include "src/kalman_filter.h"

#define DELTA_T 0.2
#define MEAN 0.0        // mean of gaussian noise
#define STDDEV 0.01     // standard deviation of gaussian noise
#define CAR_VX 0.50     // car velocity of x-axis
#define CAR_VY 0.51     // car velocity of y-axis


void GetLidarData(double &mx, double &my, double &timestamp) {
    double delta_t = DELTA_T;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(MEAN, STDDEV);
    double noise = distribution(generator);

    mx = mx + delta_t * (CAR_VX + noise);
    my = my + delta_t * (CAR_VY + noise);
    timestamp += delta_t;
}


int main() {
    double m_x = 0.0;
    double m_y = 0.0;

    double last_timestamp = 0.0;
    double now_timestamp = 0.0;

    KalmanFilter kf;

    for (int i = 0; i < 10000; ++i) {
        GetLidarData(m_x, m_y, now_timestamp);

        double delat_t = now_timestamp - last_timestamp;
        last_timestamp = now_timestamp;

        // initialize kalman filter
        if (!kf.IsInitialization()) {

            Eigen::VectorXd x_in(4, 1);
            x_in << m_x, m_y, 0.0, 0.0;

            kf.Initialization(x_in);

            // set state covariance matrix
            Eigen::MatrixXd P_in(4, 4);
            P_in << 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 100.0, 0.0,
                    0.0, 0.0, 0.0, 100.0;
            kf.SetP(P_in);

            // set process covariance matrix
            Eigen::MatrixXd Q_in(4, 4);
            Q_in << 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0;
            kf.SetQ(Q_in);

            // measurement matrix
            Eigen::MatrixXd H_in(2, 4);
            H_in << 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0;
            kf.SetH(H_in);

            // measurement convairance matrix
            Eigen::MatrixXd R_in(2, 2);
            R_in << 0.00225, 0.0,
                    0.0, 0.00225;
            kf.SetR(R_in);
        }

        // state transition matrix
        Eigen::MatrixXd F_in(4, 4);
        F_in << 1.0, 0.0, delat_t, 0.0,
                0.0, 1.0, 0.0, delat_t,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0;
        kf.SetF(F_in);

        kf.Predict();

        // measurement value
        Eigen::VectorXd z(2, 1);
        z << m_x, m_y;
        kf.MeasurementUpdate(z);

        // get results
        Eigen::VectorXd x_out = kf.GetX();
        std::cout << "kalman error x: " << x_out(0) - i * delat_t * CAR_VX <<
                     "  error y: " << x_out(1) - i * delat_t *CAR_VY <<
                     "  v_x: " << x_out(2) << "  v_y: " << x_out(3) << std::endl;
    }

    return 0;
}