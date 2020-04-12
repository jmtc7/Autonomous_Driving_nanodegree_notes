#include "cost.h"

double inefficiency_cost(int target_speed, int intended_lane, int final_lane, const std::vector<int> &lane_speeds)
{
  // Cost becomes higher for trajectories with intended lane and final lane 
  //   that have traffic slower than target_speed.

  // Target speed is the speed we want to travel at
  // Intended lane is the lane we are preparing to turn to
  // Final lane is the lane we are going to turn to
  // Lane speeds contain the average speed of each lane

  /**
   * DONE: Replace cost = 0 with an appropriate cost function.
   */
  
  double speed_intended = lane_speeds[intended_lane];
  double speed_final = lane_speeds[final_lane];
  double cost = (2.0*target_speed - speed_intended - speed_final)/target_speed;
    
  return cost;
}
