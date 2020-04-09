#include <iostream>
#include "multiv_gauss.h"

int main() {
  // define inputs
  double sig_x, sig_y, x_obs, y_obs, mu_x, mu_y;
  // define outputs for observations
  double weight1, weight2, weight3;
  // final weight
  double final_weight;
    
  // OBS1 values
  sig_x = 0.3;
  sig_y = 0.3;
  x_obs = 6;
  y_obs = 3;
  mu_x = 5;
  mu_y = 3;
  // Calculate OBS1 weight
  weight1 = multiv_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
  // should be around 0.00683644777551 rounding to 6.84E-3
  std::cout << "Weight1: " << weight1 << std::endl;
    
  // OBS2 values
  sig_x = 0.3;
  sig_y = 0.3;
  x_obs = 2;
  y_obs = 2;
  mu_x = 2;
  mu_y = 1;
  // Calculate OBS2 weight
  weight2 = multiv_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
  // should be around 0.00683644777551 rounding to 6.84E-3
  std::cout << "Weight2: " << weight2 << std::endl;
    
  // OBS3 values
  sig_x = 0.3;
  sig_y = 0.3;
  x_obs = 0;
  y_obs = 5;
  mu_x = 2;
  mu_y = 1;
  // Calculate OBS3 weight
  weight3 = multiv_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
  // should be around 9.83184874151e-49 rounding to 9.83E-49
  std::cout << "Weight3: " << weight3 << std::endl;
    
  // Output final weight
  final_weight = weight1 * weight2 * weight3;
  // 4.60E-53
  std::cout << "Final weight: " << final_weight << std::endl;
    
  return 0;
}
