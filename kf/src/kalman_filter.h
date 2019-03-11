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

    void SetH(Eigen::MatrixXd H_in);

    void SetR(Eigen::MatrixXd R_in);

    void Predict();

    void MeasurementUpdate(const Eigen::VectorXd &z);

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

    // measurement matrix
    Eigen::MatrixXd H_;

    // measurement covariance matrix
    Eigen::MatrixXd R_;
};


#endif //KF_KALMAN_FILTER_H
