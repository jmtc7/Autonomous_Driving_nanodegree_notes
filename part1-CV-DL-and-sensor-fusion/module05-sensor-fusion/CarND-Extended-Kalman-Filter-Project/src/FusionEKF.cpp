#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF()
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;


  // DONE: Finish initializing the FusionEKF
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0; // Laser can only measure the position

  Hj_<< 1, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 1, 1; //For the non-linear RADAR measurements

  // NOTE: Hj_ will be properly assigned before the "update" step using "tools.cpp" later in this file
  
  // DONE: Set the state covariance (low in position, high in velocity)
  ekf_.P_ = MatrixXd(4,4);
  ekf_.P_<< 1, 0, 0,    0,
            0, 1, 0,    0,
            0, 0, 1000, 0,
            0, 0, 0,    1000;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
  /**
   * Initialization
   */
  if (!is_initialized_)
  {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix Q.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
      // DONE: Unpack measurement
      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);

      // DONE: Convert from polar to cartesian coordinates
      float px = rho*cos(phi);
      float py = rho*sin(phi);
      float vx = rho_dot*cos(phi);
      float vy = rho_dot*sin(phi);

      // DONE: Initialize state
      // Using radial vel as a better estimation of the total vel
       ekf_.x_ << px, py, vx, vy;
       std::cout << "[i] EKF initialised with a RADAR measure (known target radial velocity)" << std::endl;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
    {
      // DONE: Unpack measurement
      float px = measurement_pack.raw_measurements_(0);
      float py = measurement_pack.raw_measurements_(1);

      // DONE: Initialize state (unknown velocity, so it keeps the one of the initialization)
      ekf_.x_(0) = px;
      ekf_.x_(1) = py;
      std::cout << "[i] EKF initialised with a LIDAR measure (unknown target velocity)" << std::endl;
    }

    // DONE: Update last timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    
    return;
  }


  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix Q
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // DONE: Compute elapsed time (in seconds) and update last timestamp
  long long dt = measurement_pack.timestamp_ - previous_timestamp_; // Elapsed time in microseconds
  dt /= 1000000.0; //Elapsed time in seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  
  // DONE: Update state transmition matrix F (using new elapsed time in seconds)
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_<< 1, 0, dt, 0,
            0, 1, 0,  dt,
            0, 0, 1,  0,
            0, 0, 0,  1;

  // DONE: Update the process covariance matrix Q (using 9 for the acceleration noise)
  float noise_ax = 9.0;
  float noise_ay = 9.0;
  
  long long dt_2 = dt * dt;
  long long dt_3 = dt_2 * dt;
  long long dt_4 = dt_3 * dt;
  
  double dt_44 = dt_4/4.0;
  double dt_32 = dt_3/2.0;
  
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_<< dt_44*noise_ax, 0,              dt_32*noise_ax,   0,
            0,              dt_44*noise_ay, 0,                dt_32*noise_ay,
            dt_32*noise_ax, 0,              dt_2*noise_ax,    0,
            0,              dt_32*noise_ay, 0,                dt_2*noise_ay;

  
  ekf_.Predict();
  
  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step (will change if the measure is from the RADAR or from the LIDAR).
   * - Update the state and covariance matrices.
   */

  // DONE: Perform "update" step using linear H or linearized Hj depending on the sensor type
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    // DONE: Build the jacobian Hj of the perception matrix H
    Tools tools;
    Hj_ = tools.CalculateJacobian(ekf_.x_);

    // DONE: Update EKF's measurement matrix H and measurement covariance matrix R
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;

    // DONE: Use the EKF update step (RADAR measurements are non-linear)
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else
  {
    // DONE: Laser's measurement matrix (H_laser_) was already build in the class initialisation

    // DONE: Update EKF's measurement matrix H and measurement covariance matrix R
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    // DONE: Use the KF update step (LIDAR measurements are linear)
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;

  return;
}
