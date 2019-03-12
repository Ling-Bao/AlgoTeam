//
// Created by bl on 19-3-10.
//


#include "kalman_filter.h"


KalmanFilter::KalmanFilter() {
    b_initialized_ = false;
}


KalmanFilter::~KalmanFilter() {

}


bool KalmanFilter::IsInitialization() {
    return b_initialized_;
}

void KalmanFilter::Initialization(Eigen::VectorXd x_in) {
    x_ = x_in;
    b_initialized_ = true;
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


void KalmanFilter::SetH(Eigen::MatrixXd H_in) {
    H_ = H_in;
}


void KalmanFilter::SetR(Eigen::MatrixXd R_in) {
    R_ = R_in;
}


void KalmanFilter::Predict() {
    x_ = F_ * x_;

    Eigen::MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}


void KalmanFilter::MeasurementUpdate(const Eigen::VectorXd &z) {
    Eigen::VectorXd y = z - H_ * x_;
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ = x_ + K * y;

    long size = x_.size();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(size, size);
    P_ = (I - K * H_) * P_;
}


Eigen::VectorXd KalmanFilter::GetX() {
    return x_;
}