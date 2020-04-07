#include <iostream>
#include <vector>
#include "Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

MatrixXd CalculateJacobian(const VectorXd& x_state);

int main()
{
  /**
   * Compute the Jacobian Matrix
   */

  // predicted state example
  // px = 1, py = 2, vx = 0.2, vy = 0.4
  VectorXd x_predicted(4);
  x_predicted << 1, 2, 0.2, 0.4;

  MatrixXd Hj = CalculateJacobian(x_predicted);

  cout << "Hj:" << endl << Hj << endl;

  return 0;
}

MatrixXd CalculateJacobian(const VectorXd& x_state)
{

  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // DONE: YOUR CODE HERE 
  // Pre-computing
  float sum = px*px + py*py;
  float squareRoot = sqrt(sum);
  float sum32 = sum*squareRoot;

  // check division by zero
  if(sum==0)
  {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }
  
  // compute the Jacobian matrix
  Hj << px/squareRoot,          py/squareRoot,          0,             0,
        -py/sum,                px/sum,                 0,             0,
        py*(vx*py-vy*px)/sum32, px*(vy*px-vx*py)/sum32, px/squareRoot, py/squareRoot;
  
  return Hj;
}
