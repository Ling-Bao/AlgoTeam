//
// Created by bl on 19-3-10.
//


#include "kalman_filter.h"


KalmanFilter::KalmanFilter() {
    b_initialized_ = false;
}


void KalmanFilter::Initialization(Eigen::VectorXd x_in) {
    x_ = x_in;
}


void KalmanFilter::SetF(Eigen::MatrixXd F_in) {
    F_ = F_in;
}


void KalmanFilter::SetP(Eigen::MatrixXd P_in) {
    P_ = P_in;
}


void KalmanFilter::SetQ(Eigen::MatrixXd Q_in) {
    Q_ = Q_in;
}


void KalmanFilter::Predict() {
    x_ = F_ * x_;

    Eigen::VectorXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}