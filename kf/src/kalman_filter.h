//
// Created by bl on 19-3-10.
//

#ifndef KF_KALMAN_FILTER_H
#define KF_KALMAN_FILTER_H


#include "Eigen/Dense"


class KalmanFilter {
public:
    KalmanFilter();

    ~KalmanFilter();

    void Initialization(Eigen::VectorXd x_in);

    void SetF(Eigen::MatrixXd F_in);

    void SetP(Eigen::MatrixXd P_in);

    void SetQ(Eigen::MatrixXd Q_in);

    void Predict();

private:
    // flag of initialization
    bool b_initialized_;

    // state vector
    Eigen::VectorXd x_;

    // state transition matrix
    Eigen::MatrixXd F_;

    // state covariance matrix
    Eigen::MatrixXd P_;

    // process covariance matrix
    Eigen::MatrixXd Q_;
};


#endif //KF_KALMAN_FILTER_H
