#include "PID.h"

/**
 * DONE: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_)
{
  /**
   * DONE: Initialize PID coefficients (and errors, if needed)
   */
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;

  return;
}

void PID::UpdateError(double cte)
{
  /**
   * DONE: Update PID errors based on cte.
   */

  // Differential error -> d(cte(t))/dt -> (current_cte-last_cte)/(current_time-last_time)
  //// p_error is still storing the last cte
  //// dt = 1 assumed (no time reference, 1 time step)
  d_error = cte-p_error;

  // Proportional error -> CTE
  p_error = cte;

  // Integral error -> integral(cte(t)*dt) -> sum(cte(t))
  i_error += cte;

  return;
}

double PID::TotalError()
{
  /**
   * DONE: Calculate and return the total error
   */

  return -d_error-p_error-i_error;  
}