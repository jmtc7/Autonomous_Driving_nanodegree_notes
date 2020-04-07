#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth)
{
  /**
   * DONE: Calculate the Root Mean Square Error (RMSE)
   */

  VectorXd rmse(4);
  rmse << 0,0,0,0;


  // Check the validity of the following inputs
  //// Check that the estimations are not empty
  if(estimations.empty())
  {
      std::cout << "[!] ERROR! CalculateRMSE () - The estimations vector is empty" << std::endl;
      return rmse;
  }
  //// Check that the estimations and the GT have the same size
  else if (estimations.size() != ground_truth.size())
  {
      std::cout << "[!] ERROR! CalculateRMSE () - The estimations and the ground truth vector have different sizes" << std::endl;
      return rmse;
  }
  

  // Accumulate squared residuals
  VectorXd residual;
  for (unsigned int i=0; i<estimations.size(); i++)
  {
    // Compute the residual
    residual = estimations[i] - ground_truth[i];

    // Accumulate its square
    //// RMSE w/o index bc each estimation/ground_truth are the 4 state variables...
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // Calculate the mean
  rmse = rmse/estimations.size();

  // Calculate the squared root of the mean error (converting vectorXd to array)
  //// The array class is for general-purpose. Matrix and Vector are intended for linear algebra
  rmse = rmse.array().sqrt();

  // Return the result
  return rmse;
}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
  /**
   * DONE:
   * Calculate a Jacobian here.
   */
   MatrixXd Hj(3,4);
   Hj << 1, 1, 0, 0,
         1, 1, 0, 0,
         1, 1, 1, 1;

   // Recover state parameters
   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);

   // Pre-computing
   float sum = px*px + py*py;
   float squareRoot = sqrt(sum);
   float sum32 = sum*squareRoot;

   // Check division by zero
   if(sum==0)
   {
      std::cout << "[!] ERROR! CalculateJacobian () - Division by Zero" << std::endl;
      return Hj;
   }
   
   // Compute the Jacobian matrix
   Hj << px/squareRoot,          py/squareRoot,          0,             0,
         -py/sum,                px/sum,                 0,             0,
         py*(vx*py-vy*px)/sum32, px*(vy*px-vx*py)/sum32, px/squareRoot, py/squareRoot;
   
   return Hj;
}
